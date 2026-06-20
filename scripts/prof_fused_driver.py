#!/usr/bin/env python3
"""Minimal ncu profiling driver for the fused FA + O_proj + NVLS-AllReduce
persistent kernel (fused_fa_oproj_ar.py :: FusedFaOprojAr).

Unlike benchmarks/bench_fused_fa_oproj_ar.py this does NOT run the baseline
(flash_attn / SDPA + GEMM + NVLS AR), the correctness cross-check, or the timing
loop. It just builds the buffers, compiles the kernel once, runs a couple of
warm launches, then ONE nvtx-marked "prof" launch, and exits. That keeps each
ncu replay pass cheap and -- crucially -- this driver only ever instantiates the
REAL FusedFaOprojAr kernel (never the skeleton), so `-k regex:FusedFaOprojAr`
cannot accidentally match anything else.

Launch order of the fused kernel is fixed:  prime(1) + warmup(W) + prof(1).
So ncu can pin the steady-state prof launch with `--launch-skip $((1+W)) --launch-count 1`.

Multi-GPU profiling (single node, all ranks under one ncu via torchrun) per the
official NsightCompute multi-process guidance for mandatory-concurrent (NCCL /
NVSHMEM-style) kernels -- our kernel does cross-rank multimem AllReduce + nvl
spin-lock barriers, so the profiler MUST keep ranks in lockstep or it deadlocks:

  ncu --communicator shmem --communicator-shmem-num-peers 8 --lockstep-kernel-launch \
      -k regex:FusedFaOprojAr --launch-skip 5 --launch-count 1 \
      --section SpeedOfLight -f -o report \
      torchrun --nnodes=1 --nproc_per_node=8 scripts/prof_fused_driver.py

shmem communicator supports kernel/range replay only (NOT application replay),
which is fine -- kernel replay re-runs just the kernel (no whole-app relaunch).

Single GPU (nproc_per_node=1): tp_size==1, the nvl_barrier / multimem AR paths
are skipped (AR degenerates to identity); profiles the FA + O_proj compute with
ordinary kernel replay, `--set full` in one go.
"""
import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.cuda.nvtx as nvtx

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack

from mega_attention.kernels.sm90.fused_fa_oproj_ar import FusedFaOprojAr, NUM_CTRL
from mega_attention.metadata.row_desc import build_row_desc, oproj_task_counts

os.environ.setdefault("NCCL_NVLS_ENABLE", "1")
os.environ.setdefault("NCCL_ALGO", "NVLS")

DT = torch.bfloat16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlens", type=str, default="2048,2048")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--h_local", type=int, default=8)
    ap.add_argument("--w_fa", type=int, default=4)
    ap.add_argument("--w_oproj", type=int, default=1)
    ap.add_argument("--w_ar", type=int, default=1)
    ap.add_argument("--sg", type=int, default=4)
    ap.add_argument("--auto", action="store_true",
                    help="用 choose_launch_config 自动选 (w_fa,w_oproj,w_ar,sg)")
    ap.add_argument("--warmup", type=int, default=4,
                    help="prof launch 前的热身次数 (ncu 用 --launch-skip 跳过)；"
                         "总 launch 顺序 = prime(1)+warmup(W)+prof(1)")
    args = ap.parse_args()

    lr = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lr)
    dev = torch.device(f"cuda:{lr}")
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    gname = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(gname)

    seqlens = [int(x) for x in args.seqlens.split(",")]
    H_local, hidden = args.h_local, args.hidden
    meta = build_row_desc(seqlens)
    if args.auto:
        from mega_attention.metadata.launch_heuristic import choose_launch_config
        cfg = choose_launch_config(meta, hidden, tp_size=ws)
        w_fa, w_oproj, w_ar, sg = cfg.w_fa, cfg.w_oproj, cfg.w_ar, cfg.sg
    else:
        w_fa, w_oproj, w_ar, sg = args.w_fa, args.w_oproj, args.w_ar, args.sg

    # ---- buffers (mirrors bench_fused_fa_oproj_ar.py::bench_one) ----
    D, N_TILE = 128, 128
    R = meta.num_row_tiles
    K_local = H_local * D
    num_fa = R * H_local
    num_out, num_super_groups, total_oproj = oproj_task_counts(R, hidden, N_TILE, sg)
    tot = int(sum(seqlens)); hidden_pad = num_out * N_TILE
    owner_slots = (total_oproj + ws - 1) // ws
    owner_words = (owner_slots + 63) // 64

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
        rb_ptrs=[hRB.buffer_ptrs[r] for r in range(ws)],
        w_fa=w_fa, w_oproj=w_oproj, w_ar=w_ar)
    ts = torch.cuda.current_stream(); st = cuda.CUstream(ts.cuda_stream)

    if rank == 0:
        print(f"[prof] seqlens={seqlens} tot={tot} H_local={H_local} hidden={hidden} "
              f"R={R} num_fa={num_fa} total_oproj={total_oproj} ws={ws} "
              f"w=({w_fa},{w_oproj},{w_ar}) sg={sg} warmup={args.warmup} compiling ...", flush=True)
    dist.barrier()
    compiled = cute.compile(ker, *cts, st)

    def reset_fused():
        for t in (ctrl, head_ready, oproj_queue, rco, fa_exec, oproj_exec, ar_exec,
                  partial_check):
            t.zero_()
        rbits.zero_(); ar_done_bits.zero_(); nvl.zero_()
        torch.cuda.synchronize(); dist.barrier()

    def run_fused():
        compiled(*cts, st)

    # prime (1 launch) ----------------------------------------------------
    reset_fused(); run_fused(); torch.cuda.synchronize(); dist.barrier()
    if rank == 0:
        print("[prof] primed; warmup ...", flush=True)

    # warmup (W launches, ncu --launch-skip'd) ---------------------------
    for _ in range(args.warmup):
        reset_fused(); run_fused()
    torch.cuda.synchronize(); dist.barrier()

    # prof launch (the one ncu profiles) ---------------------------------
    if rank == 0:
        print("[prof] profiled launch ...", flush=True)
    nvtx.range_push("prof")
    reset_fused(); run_fused(); torch.cuda.synchronize()
    nvtx.range_pop()

    dist.barrier()
    if rank == 0:
        print("[prof] done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
