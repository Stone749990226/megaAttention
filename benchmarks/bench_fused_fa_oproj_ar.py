#!/usr/bin/env python3
"""P3c benchmark: fused FA+O_proj+NVLS-AR (one persistent kernel) vs a non-fused
baseline (per-batch SDPA + matmul O_proj + NCCL all_reduce), 8xH200.

    torchrun --nproc_per_node=8 benchmarks/bench_fused_fa_oproj_ar.py [--iters N --warmup W]

Honest first-version numbers: the fused do_ar is a correctness-first single-pass
multimem reduce (no comm/compute overlap), so this mainly establishes the baseline
and locates the bottleneck for later optimization.
"""
import argparse
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed._symmetric_memory as symm_mem
from torch.nn.attention import SDPBackend, sdpa_kernel
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack

from mega_attention.kernels.sm90.fused_fa_oproj_ar import FusedFaOprojAr, NUM_CTRL
from mega_attention.metadata.row_desc import build_row_desc, oproj_task_counts

try:
    from flash_attn import flash_attn_varlen_func
    _HAS_FA = True
except Exception:
    flash_attn_varlen_func = None
    _HAS_FA = False

DT = torch.bfloat16


def bench(body, iters, warmup, setup=None):
    """Time `body` (GPU) per iter; `setup` (untimed) runs before each iter."""
    for _ in range(warmup):
        if setup: setup()
        body()
    torch.cuda.synchronize(); dist.barrier()
    total = 0.0
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        if setup: setup()
        ev0.record(); body(); ev1.record(); torch.cuda.synchronize()
        total += ev0.elapsed_time(ev1)
    return total / iters     # ms/iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seqlens", type=str, default="2048,2048")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--h_local", type=int, default=8)
    args = ap.parse_args()

    lr = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
    dev = torch.device(f"cuda:{lr}")
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    gname = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(gname)

    seqlens = [int(x) for x in args.seqlens.split(",")]
    H_local, D, hidden = args.h_local, 128, args.hidden
    N_TILE, sg = 128, 4
    meta = build_row_desc(seqlens)
    R = meta.num_row_tiles
    K_local = H_local * D
    num_fa = R * H_local
    num_out, num_super_groups, total_oproj = oproj_task_counts(R, hidden, N_TILE, sg)
    tot = int(sum(seqlens)); hidden_pad = num_out * N_TILE
    owner_slots = (total_oproj + ws - 1) // ws
    owner_words = (owner_slots + 63) // 64
    if rank == 0:
        print(f"[bench] seqlens={seqlens} tot={tot} H_local={H_local} hidden={hidden} "
              f"R={R} num_fa={num_fa} total_oproj={total_oproj} ws={ws}", flush=True)

    g = torch.Generator(device=dev).manual_seed(1234 + rank)
    Q = torch.randn(tot, H_local, D, device=dev, dtype=DT, generator=g) * 0.2
    K = torch.randn(tot, H_local, D, device=dev, dtype=DT, generator=g) * 0.2
    V = torch.randn(tot, H_local, D, device=dev, dtype=DT, generator=g) * 0.2
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
    fa_exec = _u32(num_fa); oproj_exec = _u32(total_oproj); ar_exec = _u32(total_oproj)
    partial_check = _u32(total_oproj)
    cu_q, cu_k = _i32(meta.cu_seqlens_q), _i32(meta.cu_seqlens_k)
    fa_b, fa_mb = _i32(meta.batch_idx), _i32(meta.m_block)

    cts = [from_dlpack(t, assumed_align=4) for t in (ctrl, head_ready, oproj_queue, rco)]
    cts += [from_dlpack(t, assumed_align=8) for t in (rbits, ar_done_bits)]
    cts += [from_dlpack(t, assumed_align=4) for t in (fa_exec, oproj_exec, ar_exec, partial_check)]
    cts += [from_dlpack(t, assumed_align=16) for t in (Q, K, V, Oscr, W_o_pad, C_sym)]
    cts += [from_dlpack(t, assumed_align=16) for t in (cu_q, cu_k, fa_b, fa_mb)]

    ker = FusedFaOprojAr(
        num_fa=num_fa, num_row_tiles=R, H_local=H_local, D=D,
        num_super_groups=num_super_groups, total_oproj=total_oproj, num_ctas=132,
        hidden=hidden, tp_size=ws, rank=rank, N_TILE=N_TILE, super_group_n_tiles=sg,
        csym_mc_ptr=hC.multicast_ptr, nvl_mc_ptr=hN.multicast_ptr,
        nvl_local_ptr=hN.buffer_ptrs[rank],
        rc_ptrs=[hRC.buffer_ptrs[r] for r in range(ws)],
        rb_ptrs=[hRB.buffer_ptrs[r] for r in range(ws)])
    ts = torch.cuda.current_stream(); st = cuda.CUstream(ts.cuda_stream)
    dist.barrier()
    compiled = cute.compile(ker, *cts, st)

    def reset_fused():
        # per-iter control state must be re-zeroed (counters/queue/bitsets/nvl signal);
        # sync + cross-rank barrier so the monotonic nvl_barrier sees clean slots.
        for t in (ctrl, head_ready, oproj_queue, rco, fa_exec, oproj_exec, ar_exec,
                  partial_check):
            t.zero_()
        rbits.zero_(); ar_done_bits.zero_(); nvl.zero_()
        torch.cuda.synchronize(); dist.barrier()

    def run_fused():
        compiled(*cts, st)

    # ---- best-of-breed non-fused baseline: FlashAttention + GEMM + NVLS AllReduce ----
    # FA: flash_attn_varlen_func (official pkg) if available, else SDPA forced to the
    #     FlashAttention backend. O_proj: cuBLAS matmul. AR: NVLS multimem_all_reduce_
    #     (real NVLS over a symmetric buffer, not NCCL).
    Y_sym = symm_mem.empty(tot, hidden, device=dev, dtype=DT)
    symm_mem.rendezvous(Y_sym, gname)
    cu = cu_q.to(torch.int32); max_s = max(seqlens)

    def run_baseline():
        if _HAS_FA:
            O = flash_attn_varlen_func(Q, K, V, cu, cu, max_s, max_s, causal=True)  # [tot,H,D]
        else:
            O = torch.empty(tot, H_local, D, device=dev, dtype=DT)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                for b in range(meta.num_batch):
                    s = int(meta.cu_seqlens_q[b]); e = int(meta.cu_seqlens_q[b + 1])
                    o = F.scaled_dot_product_attention(
                        Q[s:e].transpose(0, 1).unsqueeze(0), K[s:e].transpose(0, 1).unsqueeze(0),
                        V[s:e].transpose(0, 1).unsqueeze(0), is_causal=True)
                    O[s:e] = o.squeeze(0).transpose(0, 1)
        torch.matmul(O.reshape(tot, K_local), W_o, out=Y_sym)
        torch.ops.symm_mem.multimem_all_reduce_(Y_sym, "sum", gname)
        return Y_sym

    fa_name = "flash_attn_varlen" if _HAS_FA else "SDPA-FLASH"

    # ---- correctness cross-check: fused C_sym vs baseline (FA+GEMM+NVLS), both paths
    #      independent; agreement within bf16 tol confirms the fused result is correct ----
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

    t_fused = bench(run_fused, args.iters, args.warmup, setup=reset_fused)
    t_base = bench(run_baseline, args.iters, args.warmup)
    if rank == 0:
        ok = "OK" if err_rel < 0.05 else "MISMATCH"
        print(f"[bench] fused={t_fused:.4f} ms  baseline({fa_name}+GEMM+NVLS-AR)={t_base:.4f} ms  "
              f"ratio={t_base / t_fused:.3f}x  err_abs={err_abs:.3g} err_rel={err_rel:.3g} "
              f"[{ok} vs baseline]", flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
