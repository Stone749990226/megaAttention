#!/usr/bin/env python3
"""
单卡综合测试: fused persistent kernel 的 REAL FA + REAL O_proj (tp_size=1, AR 恒等)。

在一条覆盖完整 prefill / chunk q<k / GQA / tail tile 的 case 矩阵上, 一次启动同时校验
fused kernel 的三层产物与一个调度不变量:
  * O_scratch (real FA 跨所有 CTA/task 写出) == o_scratch_reference;
  * C_sym (real O_proj 读 O_scratch 写出) == oproj_reference, 按 (row_tile, out_n_tile)
    做 valid_m / valid_n predication;
  * sentinel-leak: C_sym 预填 SENT, masked tail 行/列必须保持 SENT (O_proj 只写 valid 元素);
  * exit-clean: scheduler 跑完后 exit cleaner 把 task_ctrl 清零 (done 计数全 0)。

tp_size=1 -> AR 恒等, 所以 C_sym 即本 rank 的 O_proj partial, 这里隔离 FA+O_proj 数值;
真正的跨 rank NVLS AR 由 8 卡 test_fused_production.py 覆盖。

    python tests/fused/test_fused_single_card.py        # 直接跑
    python -m pytest tests/fused/test_fused_single_card.py
"""
import cuda.bindings.driver as cuda
import numpy as np
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from mega_attention.kernels.sm90.fused_fa_oproj_ar import FusedFaOprojAr, NUM_CTRL, NUM_SYNC
from mega_attention.metadata.row_desc import build_row_desc, oproj_task_counts, active_counts
from mega_attention.reference.fused import fa_reference, o_scratch_reference, oproj_reference

DT = torch.bfloat16
DEV = "cuda:0"
SENT = -7.0          # bf16-exact sentinel: O_proj 必须只写 valid 元素


def _u32(n, dev):
    return torch.zeros(n, dtype=torch.uint32, device=dev)


def _i32(a, dev):
    return torch.tensor(np.asarray(a), dtype=torch.int32, device=dev)


def run_case(seqlens, H_local, D=128, hidden=512, N_TILE=128, super_group_n_tiles=4,
             num_ctas=8, seed=0, w_fa=4, w_oproj=1, w_ar=1, q_per_kv=1, seqlens_k=None):
    torch.manual_seed(seed)
    dev = torch.device(DEV)
    assert H_local % q_per_kv == 0, (H_local, q_per_kv)
    H_kv = H_local // q_per_kv
    meta = build_row_desc(seqlens, seqlens_k=seqlens_k)
    R = meta.num_row_tiles
    K_local = H_local * D
    num_fa = R * H_local                                  # FA task 按 Q head 数
    num_out, num_super_groups, total_oproj = oproj_task_counts(
        R, hidden, N_TILE, super_group_n_tiles)
    tot = int(sum(seqlens))                               # tot_q
    tot_k = int(sum(seqlens if seqlens_k is None else seqlens_k))
    hidden_pad = num_out * N_TILE

    Q = (torch.randn(tot, H_local, D, device=dev, dtype=DT) * 0.2)
    K = (torch.randn(tot_k, H_kv, D, device=dev, dtype=DT) * 0.2)
    V = (torch.randn(tot_k, H_kv, D, device=dev, dtype=DT) * 0.2)
    W_o = (torch.randn(K_local, hidden, device=dev, dtype=DT) * (K_local ** -0.5))
    W_o_pad = torch.zeros(K_local, hidden_pad, device=dev, dtype=DT)
    W_o_pad[:, :hidden] = W_o

    Oscr = torch.zeros(R, 128, H_local, D, device=dev, dtype=DT)
    C_sym = torch.full((R, 128, num_out, N_TILE), SENT, device=dev, dtype=DT)

    ctrl = _u32(NUM_CTRL, dev)
    sync_ctrl = _u32(NUM_SYNC, dev)         # grid_sync (init/exit) + nvl counter
    head_ready = _u32(R, dev)
    oproj_queue = _u32(total_oproj, dev)
    tp_size = 1
    owner_slots = (total_oproj + tp_size - 1) // tp_size
    owner_words = (owner_slots + 63) // 64
    actv = _i32(active_counts(R, H_local, num_super_groups, tp_size, 0), dev)
    ready_count_owner = _u32(owner_slots, dev)
    ar_ready_bits = torch.zeros(owner_words, dtype=torch.int64, device=dev)
    ar_done_bits = torch.zeros(owner_words, dtype=torch.int64, device=dev)
    cu_q = _i32(meta.cu_seqlens_q, dev)
    cu_k = _i32(meta.cu_seqlens_k, dev)
    fa_b = _i32(meta.batch_idx, dev)
    fa_mb = _i32(meta.m_block, dev)

    cts = [from_dlpack(t, assumed_align=4) for t in (ctrl, sync_ctrl, actv, head_ready,
                                                     oproj_queue, ready_count_owner)]
    cts += [from_dlpack(t, assumed_align=8) for t in (ar_ready_bits, ar_done_bits)]
    cts += [from_dlpack(t, assumed_align=16) for t in (Q, K, V, Oscr, W_o_pad, C_sym)]
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
    Y_ref = oproj_reference(O_ref, W_o, meta)            # [tot, hidden] fp32

    # ---- 1) O_scratch (FA) ----
    o_got = Oscr.float()
    o_err = (o_got - Oscr_ref).abs().max().item()
    o_tail = 0.0
    for t in range(R):
        vm = meta.valid_m(t)
        if vm < 128:
            o_tail = max(o_tail, o_got[t, vm:].abs().max().item())

    # ---- 2) C_sym (O_proj) + 3) sentinel-leak ----
    C = C_sym.cpu()
    c_err = 0.0
    leak = 0.0
    for t in range(R):
        vm = meta.valid_m(t); qs = meta.q_tile_start(t)
        for o in range(num_out):
            vn = min(N_TILE, hidden - o * N_TILE)
            got = C[t, :vm, o, :vn].float()
            exp = Y_ref[qs:qs + vm, o * N_TILE: o * N_TILE + vn].cpu()
            c_err = max(c_err, (got - exp).abs().max().item())
            if vm < 128:                                  # masked tail 行仍是 SENT
                leak = max(leak, (C[t, vm:, o, :] != SENT).float().max().item())
            if vn < N_TILE:                               # masked tail 列仍是 SENT
                leak = max(leak, (C[t, :, o, vn:] != SENT).float().max().item())

    return dict(
        o_err=o_err, o_tail=o_tail, c_err=c_err, leak=leak,
        R=R, num_fa=num_fa, total_oproj=total_oproj,
        fa_done=int(ctrl[1].item()), op_done=int(ctrl[5].item()), ar_done=int(ctrl[6].item()),
    )


def _check(name, r, o_tol=2e-2, c_tol=3e-2):
    ok = True
    msgs = []
    if not (r["o_err"] < o_tol):
        ok = False; msgs.append(f"O_scratch err={r['o_err']:.4g}")
    if r["o_tail"] != 0.0:
        ok = False; msgs.append(f"O_scratch tail={r['o_tail']:.4g}")
    if not (r["c_err"] < c_tol):
        ok = False; msgs.append(f"C_sym err={r['c_err']:.4g}")
    if r["leak"] != 0.0:
        ok = False; msgs.append("wrote masked tail (sentinel overwritten)")
    # exit cleaner zeroes task_ctrl on the way out; reaching it (all_done) leaves 0。
    if r["fa_done"] != 0 or r["op_done"] != 0 or r["ar_done"] != 0:
        ok = False; msgs.append(
            f"exit-clean expected 0, got fa={r['fa_done']} op={r['op_done']} ar={r['ar_done']}")
    print(f"{'PASS' if ok else 'FAIL'} {name}: o_err={r['o_err']:.3g} c_err={r['c_err']:.3g} "
          f"R={r['R']} op={r['total_oproj']}" + ("" if ok else "  ||  " + "; ".join(msgs)),
          flush=True)
    return ok


def main():
    # (name, seqlens_q, H, hidden, q_per_kv, seqlens_k)。seqlens_k=None -> q==k 完整 prefill。
    cases = [
        # ---- q==k 完整 prefill ----
        ("uniform_128",     [128],          4, 512, 1, None),
        ("varlen_200",      [200],          4, 512, 1, None),
        # 单序列 [300]: 末 tile valid_m=44 (300%128, 非 8 的倍数) -> finalize 的 warp
        # collective shuffle 若放在 valid_m 发散分支内会死锁。显式回归该 bug。
        ("vm44_300",        [300],          4, 512, 1, None),
        ("multi_seq",       [200, 64, 300], 4, 768, 1, None),
        ("multiseq_h8",     [128, 384, 64], 8, 512, 1, None),
        # 640 -> 5 out_n_tiles, ragged super_group (sg=4 -> 4+1), 压 O_proj 多 super-group 路径
        ("ragged_hidden",   [200, 130],     4, 640, 1, None),
        # ---- GQA ----
        ("gqa_q4_h8",       [200, 64, 300], 8, 512, 4, None),   # 8 Q head 共享 2 KV head
        ("gqa_q2_h8_vm44",  [300],          8, 512, 2, None),   # 非整 tile 末尾 K/V 复用
        # ---- q_len < k_len: contiguous-KV chunked/append prefill (offset = k_len-q_len) ----
        ("chunk_aligned",   [128],          4, 512, 1, [384]),  # offset=256 对齐 128
        ("chunk_unaligned", [128],          4, 512, 1, [316]),  # offset=188 不对齐
        ("chunk_multitile", [300],          4, 512, 1, [700]),  # 多 tile q + 长 KV + tail
        ("chunk_multiseq",  [200, 64, 300], 4, 768, 1, [512, 64, 460]),  # varlen 混合 q<k/q==k
        ("chunk_gqa_q4",    [200, 300],     8, 512, 4, [328, 700]),      # GQA + q<k + tail
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
