#!/usr/bin/env python3
"""
P1: fused persistent kernel with REAL FA + REAL O_proj (single rank, AR identity).

Validates the FA -> O_proj path end-to-end inside the dispatch loop:
  * C_sym partial (written by real O_proj across all CTAs/tasks) matches the fp32
    oproj_reference (concat_h(FA_out) @ W_o), per (row_tile, out_n_tile) tile with
    valid_m / valid_n predication,
  * scheduler still exactly-once (fa/oproj/ar exec all == 1) + terminates.

O_scratch is produced by real FA; C_sym is produced by real O_proj reading that
O_scratch. AR stays identity (tp_size=1), so this isolates the O_proj numerics.

    python tests/fused/test_fused_oproj_path.py
"""
import cuda.bindings.driver as cuda
import numpy as np
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from mega_attention.kernels.sm90.fused_fa_oproj_ar import FusedFaOprojAr, NUM_CTRL
from mega_attention.metadata.row_desc import (
    build_row_desc, oproj_task_counts, cdiv)
from mega_attention.reference.fused import fa_reference, oproj_reference

DT = torch.bfloat16
DEV = "cuda:0"
SENT = -7.0          # bf16-exact sentinel: O_proj must write ONLY valid elements


def _u32(n, dev):
    return torch.zeros(n, dtype=torch.uint32, device=dev)


def _i32(a, dev):
    return torch.tensor(np.asarray(a), dtype=torch.int32, device=dev)


def run_case(seqlens, H_local, D=128, hidden=512, N_TILE=128, super_group_n_tiles=4,
             num_ctas=8, seed=0):
    torch.manual_seed(seed)
    dev = torch.device(DEV)
    meta = build_row_desc(seqlens)
    R = meta.num_row_tiles
    K_local = H_local * D
    num_fa = R * H_local
    num_out_n_tiles, num_super_groups, total_oproj = oproj_task_counts(
        R, hidden, N_TILE, super_group_n_tiles)
    tot = int(sum(seqlens))
    hidden_pad = num_out_n_tiles * N_TILE

    Q = (torch.randn(tot, H_local, D, device=dev, dtype=DT) * 0.2)
    K = (torch.randn(tot, H_local, D, device=dev, dtype=DT) * 0.2)
    V = (torch.randn(tot, H_local, D, device=dev, dtype=DT) * 0.2)
    W_o = (torch.randn(K_local, hidden, device=dev, dtype=DT) * (K_local ** -0.5))
    W_o_pad = torch.zeros(K_local, hidden_pad, device=dev, dtype=DT)
    W_o_pad[:, :hidden] = W_o

    Oscr = torch.zeros(R, 128, H_local, D, device=dev, dtype=DT)
    C_sym = torch.full((R, 128, num_out_n_tiles, N_TILE), SENT, device=dev, dtype=DT)

    ctrl = _u32(NUM_CTRL, dev)
    head_ready = _u32(R, dev)
    oproj_queue = _u32(total_oproj, dev)
    tp_size = 1
    owner_slots = (total_oproj + tp_size - 1) // tp_size
    owner_words = (owner_slots + 63) // 64
    ready_count_owner = _u32(owner_slots, dev)
    ar_ready_bits = torch.zeros(owner_words, dtype=torch.int64, device=dev)
    ar_done_bits = torch.zeros(owner_words, dtype=torch.int64, device=dev)
    fa_exec = _u32(num_fa, dev)
    oproj_exec = _u32(total_oproj, dev)
    ar_exec = _u32(total_oproj, dev)
    partial_check = _u32(total_oproj, dev)
    cu_q = _i32(meta.cu_seqlens_q, dev)
    cu_k = _i32(meta.cu_seqlens_k, dev)
    fa_b = _i32(meta.batch_idx, dev)
    fa_mb = _i32(meta.m_block, dev)

    cts = [from_dlpack(t, assumed_align=4) for t in (ctrl, head_ready, oproj_queue,
                                                     ready_count_owner)]
    cts += [from_dlpack(t, assumed_align=8) for t in (ar_ready_bits, ar_done_bits)]
    cts += [from_dlpack(t, assumed_align=4) for t in (fa_exec, oproj_exec, ar_exec,
                                                      partial_check)]
    cts += [from_dlpack(t, assumed_align=16) for t in (Q, K, V, Oscr, W_o_pad, C_sym)]
    cts += [from_dlpack(t, assumed_align=16) for t in (cu_q, cu_k, fa_b, fa_mb)]

    ker = FusedFaOprojAr(num_fa=num_fa, num_row_tiles=R, H_local=H_local, D=D,
                         num_super_groups=num_super_groups, total_oproj=total_oproj,
                         num_ctas=num_ctas, hidden=hidden, tp_size=1, N_TILE=N_TILE,
                         super_group_n_tiles=super_group_n_tiles)
    ts = torch.cuda.Stream(); st = cuda.CUstream(ts.cuda_stream)
    compiled = cute.compile(ker, *cts, st)
    with torch.cuda.stream(ts):
        compiled(*cts, st)
    torch.cuda.synchronize()

    O_ref = fa_reference(Q, K, V, meta)                  # [tot, H, D] fp32
    Y_ref = oproj_reference(O_ref, W_o, meta)            # [tot, hidden] fp32
    C = C_sym.cpu()

    # gather each (row_tile, out_n_tile) valid block and compare to Y_ref
    err = 0.0
    leak = 0.0
    for t in range(R):
        vm = meta.valid_m(t)
        qstart = meta.q_tile_start(t)
        for o in range(num_out_n_tiles):
            vn = min(N_TILE, hidden - o * N_TILE)
            got = C[t, :vm, o, :vn].float()
            exp = Y_ref[qstart:qstart + vm, o * N_TILE: o * N_TILE + vn].cpu()
            err = max(err, (got - exp).abs().max().item())
            # tail rows/cols of a written tile must remain sentinel (not overwritten)
            if vm < 128:
                leak = max(leak, (C[t, vm:, o, :] != SENT).float().max().item())
            if vn < N_TILE:
                leak = max(leak, (C[t, :, o, vn:] != SENT).float().max().item())

    return dict(
        err=err, leak=leak, R=R, num_fa=num_fa, total_oproj=total_oproj,
        fa_exec=fa_exec.cpu(), oproj_exec=oproj_exec.cpu(), ar_exec=ar_exec.cpu(),
        order_err=int(ctrl[8].item()), fa_done=int(ctrl[1].item()),
        op_done=int(ctrl[5].item()), ar_done=int(ctrl[6].item()),
    )


def _check(name, r, tol=3e-2):
    ok = True
    msgs = []
    if not (r["err"] < tol):
        ok = False; msgs.append(f"C_sym err={r['err']:.4g}")
    if r["leak"] != 0.0:
        ok = False; msgs.append("wrote masked tail (sentinel overwritten)")
    if not bool((r["fa_exec"] == 1).all()):
        ok = False; msgs.append("fa_exec != 1")
    if not bool((r["oproj_exec"] == 1).all()):
        ok = False; msgs.append("oproj_exec != 1")
    if not bool((r["ar_exec"] == 1).all()):
        ok = False; msgs.append("ar_exec != 1")
    if r["order_err"] != 0:
        ok = False; msgs.append(f"order_err={r['order_err']}")
    if r["fa_done"] != r["num_fa"] or r["op_done"] != r["total_oproj"] or r["ar_done"] != r["total_oproj"]:
        ok = False; msgs.append(f"done fa={r['fa_done']}/{r['num_fa']} op={r['op_done']}/{r['total_oproj']} ar={r['ar_done']}/{r['total_oproj']}")
    print(f"{'PASS' if ok else 'FAIL'} {name}: err={r['err']:.4g} R={r['R']} "
          f"op={r['total_oproj']}" + ("" if ok else "  ||  " + "; ".join(msgs)), flush=True)
    return ok


def main():
    cases = [
        ("uniform_128",   [128],          4, 512),
        ("vm44_300",      [300],          4, 512),   # valid_m=44 tail (P0 regression too)
        ("multi_seq",     [200, 64, 300], 4, 768),
        ("ragged_hidden", [200, 130],     4, 640),   # 640 -> 5 out_n_tiles, ragged super_group
    ]
    failed = 0
    for name, seqlens, H, hidden in cases:
        try:
            r = run_case(seqlens, H, hidden=hidden, seed=hash(name) % 1000)
            if not _check(name, r):
                failed += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {name}: {e}", flush=True)
            traceback.print_exc()
    print(f"\n{'ALL PASS' if failed == 0 else f'{failed} FAILED'}")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(1 if main() else 0)
