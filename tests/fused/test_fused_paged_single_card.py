#!/usr/bin/env python3
"""单卡 paged-KV TMA-128 测试: fused persistent kernel 的 paged FA + REAL O_proj (tp=1)。

对照设计文档 §19 (Paged KV TMA-128 实现设计) 与 §18 验证矩阵。paged 路径只改变 FA 读取
K/V 的地址映射: logical n_block -> physical page (page_table 查表), k_len 来自 cache_seqlens;
O_scratch / O_proj / C_sym 协议与 contiguous 完全一致。

数值参考: 把 logical 连续 K/V 作为 ground truth (与 contiguous 测试同一套 fa_reference),
再 scatter 到乱序 physical pages 喂给 kernel。两条路径必须逐元素一致。

覆盖 (设计 §18):
  * page_size = 128, causal, D = 128;
  * MHA / GQA;
  * q_len == k_len 完整 prompt prefill / q_len < k_len chunked-append prefill;
  * 最后一个 Q tile 不满 128 (vm44) / 最后一个 K page 不满 128 (k_len % 128 != 0);
  * batch 内不同 sequence 长度;
  * page_table 使用非连续、乱序 physical pages (shuffle=True, 含多余 physical page);
  * wrapper 对 page_size != 128 拒绝进入 paged variant。

    python tests/fused/test_fused_paged_single_card.py
    python -m pytest tests/fused/test_fused_paged_single_card.py
"""
import cuda.bindings.driver as cuda
import numpy as np
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from mega_attention.kernels.sm90.fused_fa_oproj_ar import FusedFaOprojAr, NUM_CTRL, NUM_SYNC
from mega_attention.metadata.row_desc import build_row_desc, oproj_task_counts, active_counts
from mega_attention.reference.fused import (
    fa_reference, o_scratch_reference, oproj_reference, make_paged_kv)

DT = torch.bfloat16
DEV = "cuda:0"
SENT = -7.0


def _u32(n, dev):
    return torch.zeros(n, dtype=torch.uint32, device=dev)


def _i32(a, dev):
    return torch.tensor(np.asarray(a), dtype=torch.int32, device=dev)


def run_case(seqlens_q, seqlens_k, H_local, D=128, hidden=512, N_TILE=128,
             super_group_n_tiles=4, num_ctas=8, seed=0, q_per_kv=1,
             extra_pages=2, shuffle=True):
    torch.manual_seed(seed)
    dev = torch.device(DEV)
    assert H_local % q_per_kv == 0, (H_local, q_per_kv)
    H_kv = H_local // q_per_kv
    # paged 路径: k_len 来自 cache_seqlens (== seqlens_k); meta 仍按 (q, k) 构造,
    # 其 cu_seqlens_k 只供 reference 还原 logical 连续 K/V, kernel 不读它。
    meta = build_row_desc(seqlens_q, seqlens_k=seqlens_k)
    R = meta.num_row_tiles
    K_local = H_local * D
    num_fa = R * H_local
    num_out, num_super_groups, total_oproj = oproj_task_counts(
        R, hidden, N_TILE, super_group_n_tiles)
    tot = int(sum(seqlens_q))
    tot_k = int(sum(seqlens_k))
    hidden_pad = num_out * N_TILE

    Q = (torch.randn(tot, H_local, D, device=dev, dtype=DT) * 0.2)
    # logical 连续 K/V: ground truth, 同时被 scatter 进 paged cache。
    K_logical = (torch.randn(tot_k, H_kv, D, device=dev, dtype=DT) * 0.2)
    V_logical = (torch.randn(tot_k, H_kv, D, device=dev, dtype=DT) * 0.2)
    W_o = (torch.randn(K_local, hidden, device=dev, dtype=DT) * (K_local ** -0.5))
    W_o_pad = torch.zeros(K_local, hidden_pad, device=dev, dtype=DT)
    W_o_pad[:, :hidden] = W_o

    total_logical_pages = int(sum((int(s) + 127) // 128 for s in seqlens_k))
    K_cache, V_cache, page_table, cache_seqlens = make_paged_kv(
        K_logical, V_logical, seqlens_k, page_size=N_TILE,
        num_pages=total_logical_pages + extra_pages, shuffle=shuffle, seed=seed + 7)

    Oscr = torch.zeros(R, 128, H_local, D, device=dev, dtype=DT)
    C_sym = torch.full((R, 128, num_out, N_TILE), SENT, device=dev, dtype=DT)

    ctrl = _u32(NUM_CTRL, dev)
    sync_ctrl = _u32(NUM_SYNC, dev)
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
    cu_k = _i32(meta.cu_seqlens_k, dev)         # paged 路径不读, 占位
    fa_b = _i32(meta.batch_idx, dev)
    fa_mb = _i32(meta.m_block, dev)

    cts = [from_dlpack(t, assumed_align=4) for t in (ctrl, sync_ctrl, actv, head_ready,
                                                     oproj_queue, ready_count_owner)]
    cts += [from_dlpack(t, assumed_align=8) for t in (ar_ready_bits, ar_done_bits)]
    # mK/mV 位置传 paged cache (4D); 其余 16B 对齐张量同 contiguous。
    cts += [from_dlpack(t, assumed_align=16) for t in (Q, K_cache, V_cache, Oscr, W_o_pad, C_sym)]
    cts += [from_dlpack(t, assumed_align=16) for t in (cu_q, cu_k, fa_b, fa_mb)]
    pt_ct = from_dlpack(page_table, assumed_align=4)
    cs_ct = from_dlpack(cache_seqlens, assumed_align=4)

    ker = FusedFaOprojAr(num_fa=num_fa, num_row_tiles=R, H_local=H_local, D=D,
                         num_super_groups=num_super_groups, total_oproj=total_oproj,
                         num_ctas=num_ctas, hidden=hidden, tp_size=1, N_TILE=N_TILE,
                         super_group_n_tiles=super_group_n_tiles, q_per_kv=q_per_kv,
                         paged=True, page_size=N_TILE)
    ts = torch.cuda.Stream(); st = cuda.CUstream(ts.cuda_stream)
    compiled = cute.compile(ker, *cts, st, pt_ct, cs_ct)
    with torch.cuda.stream(ts):
        compiled(*cts, st, pt_ct, cs_ct)
    torch.cuda.synchronize()

    O_ref = fa_reference(Q, K_logical, V_logical, meta)
    Oscr_ref = o_scratch_reference(O_ref, meta)
    Y_ref = oproj_reference(O_ref, W_o, meta)

    o_got = Oscr.float()
    o_err = (o_got - Oscr_ref).abs().max().item()
    o_tail = 0.0
    for t in range(R):
        vm = meta.valid_m(t)
        if vm < 128:
            o_tail = max(o_tail, o_got[t, vm:].abs().max().item())

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
            if vm < 128:
                leak = max(leak, (C[t, vm:, o, :] != SENT).float().max().item())
            if vn < N_TILE:
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
    if r["fa_done"] != 0 or r["op_done"] != 0 or r["ar_done"] != 0:
        ok = False; msgs.append(
            f"exit-clean expected 0, got fa={r['fa_done']} op={r['op_done']} ar={r['ar_done']}")
    print(f"{'PASS' if ok else 'FAIL'} {name}: o_err={r['o_err']:.3g} c_err={r['c_err']:.3g} "
          f"R={r['R']} op={r['total_oproj']}" + ("" if ok else "  ||  " + "; ".join(msgs)),
          flush=True)
    return ok


def _test_reject_bad_page_size():
    """wrapper 对 page_size != 128 必须拒绝进入 paged variant (设计 §16/§19.5)。"""
    try:
        FusedFaOprojAr(num_fa=1, num_row_tiles=1, H_local=4, D=128,
                       num_super_groups=1, total_oproj=1, num_ctas=8, hidden=512,
                       tp_size=1, paged=True, page_size=64)
    except AssertionError:
        print("PASS reject_page_size_64", flush=True)
        return True
    print("FAIL reject_page_size_64: page_size=64 未被拒绝", flush=True)
    return False


def main():
    # (name, seqlens_q, seqlens_k, H, hidden, q_per_kv)。
    cases = [
        # ---- q==k 完整 prompt prefill (paged) ----
        ("paged_uniform_128",   [128],          [128],          4, 512, 1),
        ("paged_varlen_200",    [200],          [200],          4, 512, 1),  # 尾 page 不满
        ("paged_vm44_300",      [300],          [300],          4, 512, 1),  # 尾 Q tile vm44
        ("paged_multi_seq",     [200, 64, 300], [200, 64, 300], 4, 768, 1),
        ("paged_ragged_hidden", [200, 130],     [200, 130],     4, 640, 1),
        # ---- GQA ----
        ("paged_gqa_q4_h8",     [200, 64, 300], [200, 64, 300], 8, 512, 4),
        ("paged_gqa_q2_vm44",   [300],          [300],          8, 512, 2),
        # ---- q<k chunked / append prefill (offset = k_len - q_len) ----
        ("paged_chunk_aligned", [128],          [384],          4, 512, 1),  # offset=256 对齐
        ("paged_chunk_unalign", [128],          [316],          4, 512, 1),  # offset=188 不对齐
        ("paged_chunk_multitile", [300],        [700],          4, 512, 1),  # 多 tile q + 长 k
        ("paged_chunk_multiseq", [200, 64, 300], [512, 64, 460], 4, 768, 1),
        ("paged_chunk_gqa_q4",  [200, 300],     [328, 700],     8, 512, 4),
        # ---- 长 KV (多 page) + 尾 page 不满 ----
        ("paged_long_k_1000",   [256],          [1000],         4, 512, 1),
    ]
    failed = 0
    for name, sq, sk, H, hidden, q_per_kv in cases:
        try:
            r = run_case(sq, sk, H, hidden=hidden, seed=hash(name) % 1000, q_per_kv=q_per_kv)
            if not _check(name, r):
                failed += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {name}: {e}", flush=True)
            traceback.print_exc()
    if not _test_reject_bad_page_size():
        failed += 1
    print(f"\n{'ALL PASS' if failed == 0 else f'{failed} FAILED'}")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(1 if main() else 0)
