#!/usr/bin/env python3
"""多卡 paged-KV TMA-128 生产场景: TP fused FA(paged) + O_proj + real NVLS AR。

    torchrun --nproc_per_node=4 tests/fused/test_fused_paged_production.py

与 test_fused_production.py 同构, 但 FA 走 paged-KV TMA-128 路径 (设计 §19): 一个
FusedFaOprojArWorkspace(paged=True) 在 bucket capacity 上编译 ONCE, 之后 NO host reset
连续吞各种 shape 的 layer。每层把 logical 连续 K/V scatter 进乱序 physical pages 喂 kernel,
数值参考仍用 logical K/V 的 fa_reference, 跨 rank NVLS AR 真实求和。

每层校验:
  * C_sym[valid] == Y_final = all_reduce_sum_rank(O_paged @ W_o);
  * exit-clean: done 计数清 0。
末尾负例验证 paged set_layer 对超 page 容量干净拒绝, 且拒绝后 workspace 仍可用。
"""
import os
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import cutlass  # noqa: F401

from mega_attention.runtime import FusedFaOprojArWorkspace
from mega_attention.metadata.row_desc import build_row_desc
from mega_attention.reference.fused import fa_reference, oproj_reference, make_paged_kv

DT = torch.bfloat16

# 固定模型层参数。hidden=640 -> 5 out tiles -> 2 super groups (ragged)。
H_LOCAL, D, HIDDEN, Q_PER_KV = 4, 128, 640, 2
H_KV = H_LOCAL // Q_PER_KV
K_LOCAL = H_LOCAL * D
MAX_RT = 12                   # num_row_tiles 上界
MAX_BATCH = 12
MAX_TOT_K = 2048
MAX_NUM_PAGES = 16            # 物理 page 容量上界 (设计 §14 paged K/V capacity)
MAX_PAGES_PER_SEQ = 8         # page_table 第二维上界 (>= ceil(max k_len/128))

# (tag, seqlens_q, seqlens_k, seed)。paged 路径 k_len 必须显式 (cache_seqlens)。
LAYERS = [
    ("P00_qk_varlen",    [200, 64, 300],      [200, 64, 300],   0),   # 6 pages
    ("P01_chunk_mixed",  [200, 64, 300],      [512, 64, 460],   1),   # 9 pages, q<k
    ("P02_single_vm44",  [300],               [300],            2),   # vm44 tail
    ("P03_full_cap",     [128] * 12,          [128] * 12,       3),   # R=12, 12 pages
    ("P04_tiny_after_big", [128],             [128],            4),
    ("P05_chunk_aligned", [128],              [384],            5),   # offset=256 对齐
    ("P06_chunk_unalign", [128],              [316],            6),   # offset=188, 尾 page 不满
    ("P07_long_k",       [256],               [1024],           7),   # 8 pages 单序列
    ("P08_gqa_chunk",    [200, 300],          [328, 700],       8),   # GQA + q<k uneven
    ("P09_repeat_P00",   [200, 64, 300],      [200, 64, 300],   0),   # 复跑
    ("P10_chunk_multi",  [200, 64, 300],      [512, 64, 460],   1),   # 复跑 P01
]

# 越界负例: 单序列需要 10 pages > MAX_PAGES_PER_SEQ=8, paged set_layer 必须 assert。
NEG_LAYERS = [
    ("N0_over_pages_per_seq", [256], [1200]),     # ceil(1200/128)=10 > 8
]


def _gen_logical(dev, rank, seed, tot, tot_k):
    g = torch.Generator(device=dev).manual_seed(1234 + rank + 1000 * seed)
    Q = torch.randn(tot, H_LOCAL, D, device=dev, dtype=DT, generator=g) * 0.2
    K = torch.randn(tot_k, H_KV, D, device=dev, dtype=DT, generator=g) * 0.2
    V = torch.randn(tot_k, H_KV, D, device=dev, dtype=DT, generator=g) * 0.2
    W_o = torch.randn(K_LOCAL, HIDDEN, device=dev, dtype=DT, generator=g) * (K_LOCAL ** -0.5)
    return Q, K, V, W_o


def main():
    lr = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
    dev = torch.device(f"cuda:{lr}")
    dist.init_process_group("nccl")
    rank, ws_size = dist.get_rank(), dist.get_world_size()
    gname = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(gname)

    ws = FusedFaOprojArWorkspace.create(
        gname, max_num_row_tiles=MAX_RT, hidden=HIDDEN, H_local=H_LOCAL, D=D,
        tp_size=ws_size, rank=rank, q_per_kv=Q_PER_KV, max_num_batch=MAX_BATCH,
        max_tot_k=MAX_TOT_K, dtype=DT, device=dev,
        paged=True, page_size=128, max_num_pages=MAX_NUM_PAGES,
        max_num_pages_per_seq=MAX_PAGES_PER_SEQ)
    ws.compile(num_ctas=8)
    num_out = ws.num_out

    def run_layer(tag, sq, sk, seed):
        meta = build_row_desc(sq, seqlens_k=sk)
        R = meta.num_row_tiles
        tot = int(sum(sq)); tot_k = int(sum(sk))
        Q, K, V, W_o = _gen_logical(dev, rank, seed, tot, tot_k)
        total_logical_pages = int(sum((int(s) + 127) // 128 for s in sk))
        extra = min(2, MAX_NUM_PAGES - total_logical_pages)
        # 物理 page 乱序 + 多余 page; 各 rank 物理布局不同 (seed+rank), 但长度一致。
        K_cache, V_cache, page_table, cache_seqlens = make_paged_kv(
            K, V, sk, page_size=128, num_pages=total_logical_pages + max(extra, 0),
            shuffle=True, seed=seed * 100 + rank + 1)

        ws.set_layer(meta, Q, K_cache, V_cache, W_o,
                     page_table=page_table, cache_seqlens=cache_seqlens)
        dist.barrier()
        ws.launch()
        torch.cuda.synchronize(); dist.barrier()

        # full-chain reference: paged 还原为 logical K/V 即原 K/V -> 直接复用 fa_reference。
        O_ref = fa_reference(Q, K, V, meta)
        Yp = oproj_reference(O_ref, W_o, meta)
        Yf = Yp.clone(); dist.all_reduce(Yf, op=dist.ReduceOp.SUM)
        C = ws.csym.float().cpu()
        err = 0.0
        for t in range(R):
            vm = meta.valid_m(t); qs = meta.q_tile_start(t)
            for o in range(num_out):
                vn = min(128, HIDDEN - o * 128)
                got = C[t, :vm, o, :vn]
                exp = Yf[qs:qs + vm, o * 128:o * 128 + vn].cpu()
                err = max(err, (got - exp).abs().max().item())
        fa_d, op_d, ar_d = int(ws.ctrl[1]), int(ws.ctrl[5]), int(ws.ctrl[6])
        ok = (err < 5e-2 and fa_d == 0 and op_d == 0 and ar_d == 0)
        print(f"[rank{rank}][{tag}] err={err:.4g} R={R}/{MAX_RT} pages={total_logical_pages} "
              f"done(fa,op,ar)=({fa_d},{op_d},{ar_d})[->0] {'OK' if ok else 'FAIL'}", flush=True)
        dist.barrier()
        return ok

    allok = True
    for tag, sq, sk, seed in LAYERS:
        allok = run_layer(tag, sq, sk, seed) and allok

    # ---- 越界负例: paged set_layer 应抛 AssertionError, 且不污染 workspace ----
    for tag, sq, sk in NEG_LAYERS:
        meta = build_row_desc(sq, seqlens_k=sk)
        tot = int(sum(sq)); tot_k = int(sum(sk))
        Q, K, V, W_o = _gen_logical(dev, rank, 99, tot, tot_k)
        total_logical_pages = int(sum((int(s) + 127) // 128 for s in sk))
        Kc, Vc, pt, cs = make_paged_kv(K, V, sk, page_size=128,
                                       num_pages=total_logical_pages, shuffle=True, seed=99)
        raised = False
        try:
            ws.set_layer(meta, Q, Kc, Vc, W_o, page_table=pt, cache_seqlens=cs)
        except AssertionError:
            raised = True
        allok = allok and raised
        if rank == 0:
            print(f"[neg][{tag}] over-capacity rejected={raised} {'OK' if raised else 'FAIL'}",
                  flush=True)
    dist.barrier()

    # ---- 拒绝后 workspace 仍可用: 复跑 P00 ----
    allok = run_layer("P00_after_neg", [200, 64, 300], [200, 64, 300], 0) and allok

    if rank == 0:
        print(f"\n{'ALL PASS' if allok else 'SOME FAILED'}", flush=True)
    dist.barrier(); dist.destroy_process_group()
    return allok


if __name__ == "__main__":
    main()
