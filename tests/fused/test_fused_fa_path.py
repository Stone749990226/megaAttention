#!/usr/bin/env python3
"""
5b-1b: single-rank fused persistent kernel with REAL FA (O_proj/AR still stubs).

Validates the FA path end-to-end inside the dispatch loop:
  * O_scratch (written by real FA across all CTAs/tasks) matches o_scratch_reference,
  * scheduler terminates with fa/oproj/ar done counts at their task totals.

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
             num_ctas=8, seed=0, w_fa=4, w_oproj=1, w_ar=1, q_per_kv=1, seqlens_k=None):
    torch.manual_seed(seed)
    dev = torch.device(DEV)
    meta = build_row_desc(seqlens, seqlens_k=seqlens_k)
    R = meta.num_row_tiles
    num_fa = R * H_local                       # FA task 仍按 Q head 数
    assert H_local % q_per_kv == 0, (H_local, q_per_kv)
    H_kv = H_local // q_per_kv
    _, num_super_groups, total_oproj = oproj_task_counts(R, hidden, N_TILE, super_group_n_tiles)
    tot = int(sum(seqlens))                     # tot_q (O_scratch/O_proj 按 Q 计量)
    tot_k = int(sum(seqlens if seqlens_k is None else seqlens_k))

    Q = (torch.randn(tot, H_local, D, device=dev, dtype=DT) * 0.2)
    K = (torch.randn(tot_k, H_kv, D, device=dev, dtype=DT) * 0.2)
    V = (torch.randn(tot_k, H_kv, D, device=dev, dtype=DT) * 0.2)
    Oscr = torch.zeros(R, 128, H_local, D, device=dev, dtype=DT)
    # O_proj is now real: provide W_o / C_sym buffers (this test only checks O_scratch
    # + scheduler, but the fused kernel runs the full FA -> O_proj path).
    num_out_n_tiles = (hidden + N_TILE - 1) // N_TILE
    hidden_pad = num_out_n_tiles * N_TILE
    W_o = torch.zeros(H_local * D, hidden_pad, device=dev, dtype=DT)
    W_o[:, :hidden] = torch.randn(H_local * D, hidden, device=dev, dtype=DT) * ((H_local * D) ** -0.5)
    C_sym = torch.zeros(R, 128, num_out_n_tiles, N_TILE, device=dev, dtype=DT)

    ctrl = _u32(NUM_CTRL, dev)
    head_ready = _u32(R, dev)
    oproj_queue = _u32(total_oproj, dev)
    tp_size = 1
    owner_slots = (total_oproj + tp_size - 1) // tp_size
    owner_words = (owner_slots + 63) // 64
    ready_count_owner = _u32(owner_slots, dev)
    ar_ready_bits = torch.zeros(owner_words, dtype=torch.int64, device=dev)
    ar_done_bits = torch.zeros(owner_words, dtype=torch.int64, device=dev)
    cu_q = _i32(meta.cu_seqlens_q, dev)
    cu_k = _i32(meta.cu_seqlens_k, dev)
    fa_b = _i32(meta.batch_idx, dev)
    fa_mb = _i32(meta.m_block, dev)

    cts = [from_dlpack(t, assumed_align=4) for t in (ctrl, head_ready, oproj_queue,
                                                     ready_count_owner)]
    cts += [from_dlpack(t, assumed_align=8) for t in (ar_ready_bits, ar_done_bits)]
    cts += [from_dlpack(t, assumed_align=16) for t in (Q, K, V, Oscr, W_o, C_sym)]
    cts += [from_dlpack(t, assumed_align=16) for t in (cu_q, cu_k, fa_b, fa_mb)]

    ker = FusedFaOprojAr(num_fa=num_fa, num_row_tiles=R, H_local=H_local, D=D,
                         num_super_groups=num_super_groups, total_oproj=total_oproj,
                         num_ctas=num_ctas, hidden=hidden, tp_size=1, N_TILE=N_TILE,
                         super_group_n_tiles=super_group_n_tiles, q_per_kv=q_per_kv,
                         w_fa=w_fa, w_oproj=w_oproj, w_ar=w_ar)
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
        fa_done=int(ctrl[1].item()),
        op_done=int(ctrl[5].item()), ar_done=int(ctrl[6].item()),
    )


def _check(name, r, tol=2e-2):
    ok = True
    msgs = []
    if not (r["err"] < tol):
        ok = False; msgs.append(f"O_scratch err={r['err']:.4g}")
    if r["tail"] != 0.0:
        ok = False; msgs.append(f"tail={r['tail']:.4g}")
    if r["fa_done"] != r["num_fa"] or r["op_done"] != r["total_oproj"] or r["ar_done"] != r["total_oproj"]:
        ok = False; msgs.append(f"done fa={r['fa_done']}/{r['num_fa']} op={r['op_done']}/{r['total_oproj']} ar={r['ar_done']}/{r['total_oproj']}")
    print(f"{'PASS' if ok else 'FAIL'} {name}: err={r['err']:.4g} R={r['R']} "
          f"fa={r['num_fa']} op={r['total_oproj']}" + ("" if ok else "  ||  " + "; ".join(msgs)),
          flush=True)
    return ok


def main():
    # 每个 case: (name, seqlens_q, H, hidden, q_per_kv, seqlens_k)。seqlens_k=None -> q==k。
    cases = [
        ("uniform_128",   [128],          4, 512, 1, None),
        ("varlen_200",    [200],          4, 512, 1, None),
        # 单序列 [300]: 末 tile valid_m=44 (300%128, 非 8 的倍数) -> finalize 的 warp
        # collective shuffle 若放在 valid_m 发散分支内会死锁。显式回归该 bug。
        ("vm44_300",      [300],          4, 512, 1, None),
        ("multi_seq",     [200, 64, 300], 4, 768, 1, None),
        ("multiseq_h8",   [128, 384, 64], 8, 512, 1, None),
        # GQA: 8 Q head 共享 2 KV head (q_per_kv=4)，跨多序列 + 含 tail tile
        ("gqa_q4_h8",     [200, 64, 300], 8, 512, 4, None),
        # GQA: q_per_kv=2，验证非整 tile 末尾 (300%128=44) 下的 K/V 复用
        ("gqa_q2_h8_vm44", [300],         8, 512, 2, None),
        # ---- q_len < k_len: contiguous-KV chunked/append prefill (offset = k_len-q_len) ----
        # offset 对齐 128: offset=256, 单 tile q=128 attends 384 KV (3 blocks, 仅末块 mask)
        ("chunk_aligned",   [128],        4, 512, 1, [384]),
        # offset 不对齐 128: offset=188 -> 右侧多个 block 需 causal mask
        ("chunk_unaligned", [128],        4, 512, 1, [316]),
        # 多 tile q + 长 KV 前缀 + tail tile (q=300 vm44, k=700, offset=400)
        ("chunk_multitile", [300],        4, 512, 1, [700]),
        # varlen 多序列混合 q<k 与 q==k，含 tail
        ("chunk_multiseq",  [200, 64, 300], 4, 768, 1, [512, 64, 460]),
        # GQA + q<k + offset 不对齐 + tail
        ("chunk_gqa_q4",    [200, 300],   8, 512, 4, [328, 700]),
    ]
    failed = 0
    for name, seqlens, H, hidden, q_per_kv, seqlens_k in cases:
        try:
            r = run_case(seqlens, H, hidden=hidden, seed=hash(name) % 1000,
                         q_per_kv=q_per_kv, seqlens_k=seqlens_k)
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
