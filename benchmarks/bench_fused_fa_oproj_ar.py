#!/usr/bin/env python3
"""P3c benchmark: fused FA+O_proj+NVLS-AR (one persistent kernel) vs a non-fused
best-of-breed baseline (flash_attn_varlen + cuBLAS O_proj + NVLS multimem AllReduce),
8xH200.

    # 单 shape:
    torchrun --nproc_per_node=8 benchmarks/bench_fused_fa_oproj_ar.py [--iters N --warmup W]
    # 跑 README 全表并生成 Markdown+JSON:
    torchrun --nproc_per_node=8 benchmarks/bench_fused_fa_oproj_ar.py --cases readme

计时用 torch.profiler (kineto): 按 kernel 的 self device time 取纯 GPU 时间, CPU launch
skew 不影响测量。baseline 的 FA / O_proj / AR 三段分别独立计时, 用来推导 overlap 理论值:
    t_compute = t_fa + t_oproj          # 共享 tensor core, 串行相加
    t_serial  = t_fa + t_oproj + t_ar   # 完全不重叠下界
    t_ideal   = max(t_compute, t_ar)    # 完美重叠下界 (compute 与 NVLink 通信并行)
    overlap%  = (t_serial - t_fused) / (t_serial - t_ideal)
注意: 三段用的是外部 best-of-breed 实现 (官方 flash_attn + cuBLAS), 不是融合 kernel 内部
的 FA/O_proj。所以 overlap% 是 "融合 vs 完美重叠的 best-of-breed", overlap%>100% 表示融合
连完美重叠的最强基线都打赢; 在内部 FA 较慢的 shape 上 overlap% 可能偏低甚至为负, 这是该口径
的定义, 不是 bug。

吞吐列全部除以 t_fused (融合实测): compute 和 comm 共享同一段墙钟, 偏低是重叠在起作用的信号。
    TFLOPS  = (fa_flops + oproj_flops) / t_fused
    NVLink  = ar_bytes * 2(n-1)/n / t_fused   # bus BW, NCCL 惯例, 可对标 NVLink 峰值
              (algorithm BW = ar_bytes / t_fused 进 JSON; NVLS multimem 物理约搬 2*bytes,
               2(n-1)/n 是便于对标的约定而非物理线速)
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack

from mega_attention.kernels.sm90.fused_fa_oproj_ar import FusedFaOprojAr, NUM_CTRL
from mega_attention.metadata.row_desc import build_row_desc, oproj_task_counts

from flash_attn import flash_attn_varlen_func     # baseline FA: 官方包, 不接受 SDPA fallback

DT = torch.bfloat16

# H200 (SXM) 峰值, 仅用于 utilization 注释; 跑前用拓扑确认。
H200_BF16_TFLOPS = 989.5          # dense bf16 tensor core
H200_NVLINK_GBPS = 900.0          # NVLink4 单卡双向聚合

# profiler 里需要从 self device time 求和中排除的噪声 kernel (reset 的 nccl barrier、
# 控制位清零的 fill、L2 flush 的 memset、rank 对齐的 _sleep spin 等)。真实算子
# (FusedFaOprojAr / flash_fwd_kernel / nvjet gemm / multimem_all_reduce) 都不含这些子串。
_NOISE = ("nccl", "fill", "elementwise", "memset", "memcpy", "copy_", "sleep", "spin")


def _self_dev_us(e):
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        if hasattr(e, attr):
            return float(getattr(e, attr))
    return 0.0


def bench_kineto(body, iters, warmup, barrier=False, align=False, flush_l2=True,
                 dump=False, tag=""):
    """用 kineto 测 `body` 的纯 GPU self device time (ms/iter)。

    对窗口内所有非噪声 kernel 的 self device time 求和后除以 iters。对单 kernel, CPU launch
    skew 不影响测量; 但持久 kernel 内部有跨 rank 自旋 barrier, launch skew 会变成被计入的
    自旋时间, 故 align=True 时每次 launch 前用 `_sleep + barrier` 把各 rank 起点对齐
    (MegaMoE 同款), 让 in-kernel barrier 不空转。
    """
    flush = torch.empty(int(8e9 // 4), dtype=torch.int, device="cuda") if flush_l2 else None

    def _pre():
        if flush is not None: flush.zero_()
        if align:
            torch.cuda._sleep(int(2e7))   # ~10ms GPU 延迟, 抵消 barrier 释放后的启动抖动
            dist.barrier()
        elif barrier:
            dist.barrier()

    for _ in range(warmup):
        _pre(); body()
    torch.cuda.synchronize()
    if barrier or align: dist.barrier()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            _pre(); body()
        torch.cuda.synchronize()
    evts = prof.key_averages()
    if dump and dist.get_rank() == 0:
        print(f"\n----- kineto dump [{tag}] -----", flush=True)
        print(prof.key_averages().table(sort_by="self_cuda_time_total",
                                         max_name_column_width=80, row_limit=10), flush=True)
    total_us = sum(_self_dev_us(e) for e in evts
                   if not any(n in e.key.lower() for n in _NOISE))
    return total_us / iters / 1e3     # ms/iter


# ── theoretical FLOP / byte counts (per rank) ────────────────────────────────
def fa_flops(seqlens_q, H_local, D, seqlens_k=None):
    """bottom-right causal QK^T+PV FLOP。per seq per head = 2 * pairs * (2*D),
    pairs = sum_{i=0..lq-1} (i + offset + 1), offset = lk - lq (>=0)。
    q==k 时 pairs = lq*(lq+1)/2, 退化为原完整 prefill 公式。seqlens_k=None -> q==k。"""
    sk = seqlens_q if seqlens_k is None else seqlens_k
    total = 0.0
    for lq, lk in zip(seqlens_q, sk):
        off = lk - lq
        pairs = lq * off + lq * (lq + 1) / 2
        total += H_local * 2 * pairs * (2 * D)
    return float(total)


def oproj_flops(tot, K_local, hidden):
    return float(2 * tot * K_local * hidden)


def ar_bytes(tot, hidden, dtype_bytes=2):
    return float(tot * hidden * dtype_bytes)


def _tok(n):
    return f"{n/1024:.1f}K".replace(".0K", "K") if n >= 1024 else str(n)


def shape_label(seqlens, H_local, hidden, q_per_kv=1):
    if len(seqlens) == 1:
        body = f"[{_tok(seqlens[0])}]"
    elif len(set(seqlens)) == 1:
        body = f"[{_tok(seqlens[0])}]x{len(seqlens)}"
    else:                                    # 混合长尾 varlen: 紧凑摘要, 完整 seqlens 进 JSON
        body = f"varlen(B={len(seqlens)},tot={_tok(sum(seqlens))},max={_tok(max(seqlens))})"
    hd = f"H{H_local}" if q_per_kv == 1 else f"H{H_local}/kv{H_local // q_per_kv}"
    return f"{body} {hd} hid{hidden}"


def bench_one(seqlens, H_local, hidden, w_fa, w_oproj, w_ar, sg,
              ws, rank, dev, iters, warmup, dump=False, q_per_kv=1, seqlens_k=None):
    """Build buffers, compile fused kernel, time fused + 3 baseline segments, return row dict.

    q_per_kv: 标准 GQA 分组比 (per rank)。K/V 用 H_kv_local = H_local // q_per_kv 个 head；
    q_per_kv == 1 即 MHA。baseline flash_attn_varlen_func 原生支持 GQA。
    seqlens_k: q_len<k_len contiguous-KV chunked/append prefill 的 KV 长度 (None -> q==k)。
    """
    gname = dist.group.WORLD.group_name

    D, N_TILE = 128, 128
    assert H_local % q_per_kv == 0, (H_local, q_per_kv)
    H_kv = H_local // q_per_kv
    meta = build_row_desc(seqlens, seqlens_k=seqlens_k)
    R = meta.num_row_tiles
    K_local = H_local * D
    num_fa = R * H_local
    num_out, num_super_groups, total_oproj = oproj_task_counts(R, hidden, N_TILE, sg)
    tot = int(sum(seqlens)); hidden_pad = num_out * N_TILE
    tot_k = int(sum(seqlens if seqlens_k is None else seqlens_k))
    owner_slots = (total_oproj + ws - 1) // ws
    owner_words = (owner_slots + 63) // 64

    g = torch.Generator(device=dev).manual_seed(1234 + rank)
    Q = torch.randn(tot, H_local, D, device=dev, dtype=DT, generator=g) * 0.2
    K = torch.randn(tot_k, H_kv, D, device=dev, dtype=DT, generator=g) * 0.2
    V = torch.randn(tot_k, H_kv, D, device=dev, dtype=DT, generator=g) * 0.2
    W_o = torch.randn(K_local, hidden, device=dev, dtype=DT, generator=g) * (K_local ** -0.5)
    W_o_pad = torch.zeros(K_local, hidden_pad, device=dev, dtype=DT); W_o_pad[:, :hidden] = W_o
    Oscr = torch.zeros(R, 128, H_local, D, device=dev, dtype=DT)

    def _u32(n): return torch.zeros(n, dtype=torch.uint32, device=dev)
    def _i32(a): return torch.tensor(np.asarray(a), dtype=torch.int32, device=dev)

    C_sym = symm_mem.empty(R, 128, num_out, N_TILE, device=dev, dtype=DT); C_sym.zero_()
    hC = symm_mem.rendezvous(C_sym, gname)
    rco = symm_mem.empty(owner_slots, device=dev, dtype=torch.uint32); rco.zero_()
    hRC = symm_mem.rendezvous(rco, gname)
    rbits = symm_mem.empty(owner_words, device=dev, dtype=torch.int64); rbits.zero_()
    hRB = symm_mem.rendezvous(rbits, gname)
    nvl = symm_mem.empty(8, device=dev, dtype=torch.uint32); nvl.zero_()
    hN = symm_mem.rendezvous(nvl, gname)
    ar_done_bits = torch.zeros(owner_words, dtype=torch.int64, device=dev)
    ctrl = _u32(NUM_CTRL); head_ready = _u32(R); oproj_queue = _u32(total_oproj)
    cu_q, cu_k = _i32(meta.cu_seqlens_q), _i32(meta.cu_seqlens_k)
    fa_b, fa_mb = _i32(meta.batch_idx), _i32(meta.m_block)

    cts = [from_dlpack(t, assumed_align=4) for t in (ctrl, head_ready, oproj_queue, rco)]
    cts += [from_dlpack(t, assumed_align=8) for t in (rbits, ar_done_bits)]
    cts += [from_dlpack(t, assumed_align=16) for t in (Q, K, V, Oscr, W_o_pad, C_sym)]
    cts += [from_dlpack(t, assumed_align=16) for t in (cu_q, cu_k, fa_b, fa_mb)]

    ker = FusedFaOprojAr(
        num_fa=num_fa, num_row_tiles=R, H_local=H_local, D=D, q_per_kv=q_per_kv,
        num_super_groups=num_super_groups, total_oproj=total_oproj, num_ctas=132,
        hidden=hidden, tp_size=ws, rank=rank, N_TILE=N_TILE, super_group_n_tiles=sg,
        csym_mc_ptr=hC.multicast_ptr, nvl_mc_ptr=hN.multicast_ptr,
        nvl_local_ptr=hN.buffer_ptrs[rank],
        rc_ptrs=[hRC.buffer_ptrs[r] for r in range(ws)],
        rb_ptrs=[hRB.buffer_ptrs[r] for r in range(ws)],
        w_fa=w_fa, w_oproj=w_oproj, w_ar=w_ar)
    ts = torch.cuda.current_stream(); st = cuda.CUstream(ts.cuda_stream)
    dist.barrier()
    compiled = cute.compile(ker, *cts, st)

    def reset_fused():
        for t in (ctrl, head_ready, oproj_queue, rco):
            t.zero_()
        rbits.zero_(); ar_done_bits.zero_(); nvl.zero_()
        torch.cuda.synchronize(); dist.barrier()

    def run_fused():
        reset_fused(); compiled(*cts, st)

    # ---- best-of-breed non-fused baseline: FlashAttention + GEMM + NVLS AllReduce ----
    Y_sym = symm_mem.empty(tot, hidden, device=dev, dtype=DT)
    symm_mem.rendezvous(Y_sym, gname)
    cu = cu_q.to(torch.int32); max_s = max(seqlens)
    cuk = cu_k.to(torch.int32); max_sk = max(seqlens if seqlens_k is None else seqlens_k)
    O_buf = torch.empty(tot, H_local, D, device=dev, dtype=DT)

    def run_fa():
        nonlocal O_buf
        O_buf = flash_attn_varlen_func(Q, K, V, cu, cuk, max_s, max_sk, causal=True)

    def run_oproj():
        torch.matmul(O_buf.reshape(tot, K_local), W_o, out=Y_sym)

    def run_ar():
        torch.ops.symm_mem.multimem_all_reduce_(Y_sym, "sum", gname)

    def run_baseline():
        run_fa(); run_oproj(); run_ar(); return Y_sym

    # ---- correctness cross-check: fused C_sym vs baseline ----
    reset_fused(); run_fused(); torch.cuda.synchronize(); dist.barrier()
    Cf = C_sym.float().cpu()
    Yb = run_baseline().float().cpu(); torch.cuda.synchronize(); dist.barrier()
    err_abs, ref_max = 0.0, 0.0
    for t in range(R):
        vm = meta.valid_m(t); qs = meta.q_tile_start(t)
        for o in range(num_out):
            vn = min(N_TILE, hidden - o * N_TILE)
            gf = Cf[t, :vm, o, :vn]; gb = Yb[qs:qs + vm, o * N_TILE:o * N_TILE + vn]
            err_abs = max(err_abs, (gf - gb).abs().max().item())
            ref_max = max(ref_max, gb.abs().max().item())
    err_rel = err_abs / max(ref_max, 1e-6)

    # ---- timing (kineto) ----
    t_fused = bench_kineto(run_fused, iters, warmup, align=True, dump=dump, tag="fused")
    t_fa = bench_kineto(run_fa, iters, warmup, dump=dump, tag="fa")
    run_fa()  # ensure O_buf valid for oproj timing
    t_oproj = bench_kineto(run_oproj, iters, warmup, dump=dump, tag="oproj")
    t_ar = bench_kineto(run_ar, iters, warmup, align=True, dump=dump, tag="ar")
    t_base = t_fa + t_oproj + t_ar

    # ---- derived metrics ----
    t_compute = t_fa + t_oproj
    t_ideal = max(t_compute, t_ar)
    denom = t_base - t_ideal
    overlap_pct = (t_base - t_fused) / denom * 100.0 if denom > 1e-9 else float("nan")

    flop = fa_flops(seqlens, H_local, D, seqlens_k=seqlens_k) + oproj_flops(tot, K_local, hidden)
    tflops = flop / 1e12 / (t_fused / 1e3)
    s_bytes = ar_bytes(tot, hidden)
    bus_factor = 2.0 * (ws - 1) / ws
    nvl_busbw = s_bytes * bus_factor / 1e9 / (t_fused / 1e3)
    nvl_algbw = s_bytes / 1e9 / (t_fused / 1e3)

    row = dict(
        shape=shape_label(seqlens, H_local, hidden, q_per_kv), seqlens=seqlens, H_local=H_local,
        hidden=hidden, tp=ws, tot=tot, q_per_kv=q_per_kv, H_kv_local=H_kv,
        fused_ms=t_fused, t_fa=t_fa, t_oproj=t_oproj, t_ar=t_ar,
        serial_ms=t_base, ideal_ms=t_ideal, overlap_pct=overlap_pct,
        tflops=tflops, tflops_pct=tflops / H200_BF16_TFLOPS * 100.0,
        nvl_busbw=nvl_busbw, nvl_algbw=nvl_algbw,
        nvl_busbw_pct=nvl_busbw / H200_NVLINK_GBPS * 100.0,
        ratio=t_base / t_fused, err_rel=err_rel,
        w=(w_fa, w_oproj, w_ar), sg=sg,
    )
    if rank == 0:
        print(f"[bench] {row['shape']:28s} fused={t_fused:.3f} fa={t_fa:.3f} "
              f"oproj={t_oproj:.3f} ar={t_ar:.3f} serial={t_base:.3f} ideal={t_ideal:.3f} "
              f"overlap={overlap_pct:5.0f}% {tflops:4.0f}TF NVL={nvl_busbw:4.0f}GB/s "
              f"ratio={row['ratio']:.2f}x err_rel={err_rel:.1e}", flush=True)
    return row


def print_table(rows):
    cols = ["shape", "tp", "fused", "FA", "O_proj", "AR", "serial", "ideal",
            "overlap%", "TFLOPS", "NVLink", "ratio", "err_rel"]
    print("\n| " + " | ".join(cols) + " |")
    print("| " + " | ".join("---" for _ in cols) + " |")
    for r in rows:
        print(f"| {r['shape']} | {r['tp']} | {r['fused_ms']:.3f} | {r['t_fa']:.3f} | "
              f"{r['t_oproj']:.3f} | {r['t_ar']:.3f} | {r['serial_ms']:.3f} | "
              f"{r['ideal_ms']:.3f} | {r['overlap_pct']:.0f}% | "
              f"{r['tflops']:.0f} ({r['tflops_pct']:.0f}%) | "
              f"{r['nvl_busbw']:.0f} ({r['nvl_busbw_pct']:.0f}%) | "
              f"{r['ratio']:.2f}x | {r['err_rel']:.1e} |")
    print("\n说明: 时间单位 ms (kineto self device time)。TFLOPS/NVLink 均除以 t_fused;"
          " NVLink 为 bus BW=2(n-1)/n 口径 (algorithm BW 见 JSON)。"
          " overlap% = (serial-fused)/(serial-ideal), 基于外部 best-of-breed 分段, "
          ">100% 表示融合优于完美重叠的最强基线。", flush=True)


# 真实大模型 (8 卡 TP, head_dim=128) + 一个合成 GQA stress 配置, 锚定 chunk-prefill 场景。
# kernel 支持标准 GQA: K/V 按 kv_head = q_head // q_per_kv 复用。每个 case 是
# (seqlens, H_local, hidden, q_per_kv), 均为 per-rank (TP8) 值。
#
# 旗舰 GQA 模型 num_kv_heads<=8, TP8 下每 rank 只剩 1 个 KV head (kv<8 跨 rank 复制),
# 即 per-rank MQA (H_kv_local=1, q_per_kv=H_local) —— 这是真实部署形态。
# 额外加一个合成 num_kv_heads=16 配置: TP8 下每 rank=2 个 KV head (H_kv_local=2),
# 真正压多-KV-head 寻址路径 (kv_head = head // q_per_kv 取到 0/1)。
#
# chunk-prefill: 不对超长单序列一次 prefill。用多段不规整 varlen, 每段<=8K, 总量中等批。
_CHUNK_BIG = [7680, 5376, 3968, 2560, 1664, 1024, 640, 256]    # ~23.2K tokens, 8 段不规整
_CHUNK_SMALL = [6912, 3200, 1792, 1088, 512, 256]             # ~13.8K tokens, 6 段不规整
README_CASES = [
    # Qwen3-235B-A22B: q64/kv4/hidden4096 -> TP8 H_local=8, kv->1, q_per_kv=8
    (_CHUNK_BIG, 8, 4096, 8),
    (_CHUNK_SMALL, 8, 4096, 8),
    # Qwen3-Coder-480B-A35B: q96/kv8/hidden6144 -> TP8 H_local=12, kv->1, q_per_kv=12
    (_CHUNK_BIG, 12, 6144, 12),
    (_CHUNK_SMALL, 12, 6144, 12),
    # GLM-4.6: q96/kv8/hidden5120 -> TP8 H_local=12, kv->1, q_per_kv=12
    (_CHUNK_BIG, 12, 5120, 12),
    (_CHUNK_SMALL, 12, 5120, 12),
    # Llama-3.1-405B: q128/kv8/hidden16384 -> TP8 H_local=16, kv->1, q_per_kv=16
    (_CHUNK_BIG, 16, 16384, 16),
    (_CHUNK_SMALL, 16, 16384, 16),
    # 合成 GQA stress: q128/kv16/hidden8192 -> TP8 H_local=16, H_kv_local=2, q_per_kv=8
    (_CHUNK_BIG, 16, 8192, 8),
    (_CHUNK_SMALL, 16, 8192, 8),
]

# ---- q_len < k_len 比例扫描: 固定 Q chunk, 扫 k_len/q_len ∈ {1,2,4} (offset=0/q/3q) ----
# 5-tuple: (seqlens_q, H_local, hidden, q_per_kv, seqlens_k)。ratio=1 即 q==k 回归基准。
RATIO_CASES = [
    (_CHUNK_SMALL, 8, 4096, 8, [s * 1 for s in _CHUNK_SMALL]),   # ratio 1x: offset=0
    (_CHUNK_SMALL, 8, 4096, 8, [s * 2 for s in _CHUNK_SMALL]),   # ratio 2x: KV 前缀=2x chunk
    (_CHUNK_SMALL, 8, 4096, 8, [s * 4 for s in _CHUNK_SMALL]),   # ratio 4x: KV 前缀=4x chunk
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seqlens", type=str, default="2048,2048")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--h_local", type=int, default=8)
    ap.add_argument("--q_per_kv", type=int, default=1,
                    help="标准 GQA 分组比 (per rank); 1=MHA。H_kv_local = h_local // q_per_kv")
    ap.add_argument("--w_fa", type=int, default=4)
    ap.add_argument("--w_oproj", type=int, default=1)
    ap.add_argument("--w_ar", type=int, default=1)
    ap.add_argument("--sg", type=int, default=4)
    ap.add_argument("--auto", action="store_true",
                    help="用 choose_launch_config 自动选 (w_fa,w_oproj,w_ar,sg)")
    ap.add_argument("--cases", type=str, default="",
                    help="'readme' 跑内置全表; 'ratio' 跑 q<k 比例扫描; 留空跑单 --seqlens")
    ap.add_argument("--json", type=str, default="", help="把行结果写入该 JSON 路径")
    ap.add_argument("--dump-kernels", action="store_true",
                    help="打印每段 profiler kernel 表 (校验名字匹配)")
    args = ap.parse_args()

    lr = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
    dev = torch.device(f"cuda:{lr}")
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    gname = dist.group.WORLD.group_name

    if args.cases == "readme":
        cases = README_CASES
    elif args.cases == "ratio":
        cases = RATIO_CASES
    else:
        cases = [([int(x) for x in args.seqlens.split(",")], args.h_local, args.hidden,
                  args.q_per_kv)]

    rows = []
    for case in cases:
        seqlens, H_local, hidden, q_per_kv = case[0], case[1], case[2], case[3]
        seqlens_k = case[4] if len(case) > 4 else None
        meta = build_row_desc(seqlens, seqlens_k=seqlens_k)
        if args.auto:
            from mega_attention.metadata.launch_heuristic import choose_launch_config
            cfg = choose_launch_config(meta, hidden, tp_size=ws)
            w_fa, w_oproj, w_ar, sg = cfg.w_fa, cfg.w_oproj, cfg.w_ar, cfg.sg
        else:
            w_fa, w_oproj, w_ar, sg = args.w_fa, args.w_oproj, args.w_ar, args.sg
        row = bench_one(seqlens, H_local, hidden, w_fa, w_oproj, w_ar, sg,
                        ws, rank, dev, args.iters, args.warmup, dump=args.dump_kernels,
                        q_per_kv=q_per_kv, seqlens_k=seqlens_k)
        rows.append(row)
        dist.barrier()

    if rank == 0:
        print_table(rows)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=2)
            print(f"\n[json] wrote {len(rows)} rows -> {args.json}", flush=True)

    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
