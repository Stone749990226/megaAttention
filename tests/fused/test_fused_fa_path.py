#!/usr/bin/env python3
"""
5b-1b: single-rank fused persistent kernel with REAL FA (O_proj/AR still stubs).

Validates the FA path end-to-end inside the dispatch loop:
  * O_scratch (written by real FA across all CTAs/tasks) matches o_scratch_reference,
  * scheduler still exactly-once (fa/oproj/ar exec all == 1) + ordered (order_err==0,
    via head_ready check in the O_proj stub) + terminates.

    python -m pytest tests/fused/test_fused_fa_path.py
"""
import cuda.bindings.driver as cuda
import numpy as np
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from mega_attention.kernels.sm90.fused_fa_oproj_ar import FusedFaOprojAr, NUM_CTRL
from mega_attention.metadata.row_desc import build_row_desc, oproj_task_counts
from mega_attention.reference.fused import fa_reference, o_scratch_reference

DT = torch.bfloat16
DEV = "cuda:0"


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
    num_fa = R * H_local
    _, num_super_groups, total_oproj = oproj_task_counts(R, hidden, N_TILE, super_group_n_tiles)
    tot = int(sum(seqlens))

    Q = (torch.randn(tot, H_local, D, device=dev, dtype=DT) * 0.2)
    K = (torch.randn(tot, H_local, D, device=dev, dtype=DT) * 0.2)
    V = (torch.randn(tot, H_local, D, device=dev, dtype=DT) * 0.2)
    Oscr = torch.zeros(R, 128, H_local, D, device=dev, dtype=DT)

    ctrl = _u32(NUM_CTRL, dev)
    head_ready = _u32(R, dev)
    oproj_queue = _u32(total_oproj, dev)
    ready_count_owner = _u32(total_oproj, dev)
    ar_probe = _u32(total_oproj, dev)
    ar_done_flag = _u32(total_oproj, dev)
    fa_exec = _u32(num_fa, dev)
    oproj_exec = _u32(total_oproj, dev)
    ar_exec = _u32(total_oproj, dev)
    partial_check = _u32(total_oproj, dev)
    cu_q = _i32(meta.cu_seqlens_q, dev)
    cu_k = _i32(meta.cu_seqlens_k, dev)
    fa_b = _i32(meta.batch_idx, dev)
    fa_mb = _i32(meta.m_block, dev)

    u32s = [ctrl, head_ready, oproj_queue, ready_count_owner, ar_probe, ar_done_flag,
            fa_exec, oproj_exec, ar_exec, partial_check]
    c_u32 = [from_dlpack(t, assumed_align=4) for t in u32s]
    c_data = [from_dlpack(t, assumed_align=16) for t in (Q, K, V, Oscr)]
    c_meta = [from_dlpack(t, assumed_align=16) for t in (cu_q, cu_k, fa_b, fa_mb)]
    cts = c_u32 + c_data + c_meta

    ker = FusedFaOprojAr(num_fa=num_fa, num_row_tiles=R, H_local=H_local, D=D,
                         num_super_groups=num_super_groups, total_oproj=total_oproj,
                         num_ctas=num_ctas, tp_size=1)
    ts = torch.cuda.Stream(); st = cuda.CUstream(ts.cuda_stream)
    compiled = cute.compile(ker, *cts, st)
    with torch.cuda.stream(ts):
        compiled(*cts, st)
    torch.cuda.synchronize()

    O_ref = fa_reference(Q, K, V, meta)                  # [tot, H, D] fp32
    Oscr_ref = o_scratch_reference(O_ref, meta)          # [R, 128, H, D] fp32
    got = Oscr.float()
    err = (got - Oscr_ref).abs().max().item()
    tail = 0.0
    for t in range(R):
        vm = meta.valid_m(t)
        if vm < 128:
            tail = max(tail, got[t, vm:].abs().max().item())

    return dict(
        err=err, tail=tail, R=R, num_fa=num_fa, total_oproj=total_oproj,
        fa_exec=fa_exec.cpu(), oproj_exec=oproj_exec.cpu(), ar_exec=ar_exec.cpu(),
        order_err=int(ctrl[8].item()), fa_done=int(ctrl[1].item()),
        op_done=int(ctrl[5].item()), ar_done=int(ctrl[6].item()),
    )


def _check(name, r, tol=2e-2):
    ok = True
    msgs = []
    if not (r["err"] < tol):
        ok = False; msgs.append(f"O_scratch err={r['err']:.4g}")
    if r["tail"] != 0.0:
        ok = False; msgs.append(f"tail={r['tail']:.4g}")
    if not bool((r["fa_exec"] == 1).all()):
        ok = False; msgs.append(f"fa_exec min={int(r['fa_exec'].min())} max={int(r['fa_exec'].max())}")
    if not bool((r["oproj_exec"] == 1).all()):
        ok = False; msgs.append("oproj_exec != 1")
    if not bool((r["ar_exec"] == 1).all()):
        ok = False; msgs.append("ar_exec != 1")
    if r["order_err"] != 0:
        ok = False; msgs.append(f"order_err={r['order_err']}")
    if r["fa_done"] != r["num_fa"] or r["op_done"] != r["total_oproj"] or r["ar_done"] != r["total_oproj"]:
        ok = False; msgs.append(f"done fa={r['fa_done']}/{r['num_fa']} op={r['op_done']}/{r['total_oproj']} ar={r['ar_done']}/{r['total_oproj']}")
    print(f"{'PASS' if ok else 'FAIL'} {name}: err={r['err']:.4g} R={r['R']} "
          f"fa={r['num_fa']} op={r['total_oproj']}" + ("" if ok else "  ||  " + "; ".join(msgs)),
          flush=True)
    return ok


def main():
    cases = [
        ("uniform_128",   [128],          4, 512),
        ("varlen_200",    [200],          4, 512),
        # 单序列 [300]: 末 tile valid_m=44 (300%128, 非 8 的倍数) -> finalize 的 warp
        # collective shuffle 若放在 valid_m 发散分支内会死锁。显式回归该 bug。
        ("vm44_300",      [300],          4, 512),
        ("multi_seq",     [200, 64, 300], 4, 768),
        ("multiseq_h8",   [128, 384, 64], 8, 512),
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
