#!/usr/bin/env python3
"""
Driver / benchmark for the fused O_proj GEMM + one-shot NVLS AllReduce SM90 kernel.

  torchrun --nproc_per_node=8 megaattn/bench.py
  torchrun --nproc_per_node=8 megaattn/bench.py --iters 50 --check

Reuses the shape + fp32 shadow reference from reference.py and the symmetric-memory
init style of /workspace/tp_overlap.py. Reports:
  * numeric error of fused `out` vs fp32 all_reduce reference
  * fused kernel time (GEMM+AR in one launch)
  * baseline: torch GEMM time + NVLS all_reduce time (the un-fused path)
  * exposed-AR = t_fused - t_gemm_only  (how much AR is *not* hidden)
"""
import os
import argparse

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack

import reference as ref
from oproj_ar_sm90 import OProjARFusedKernelSM90

# Match tp_attention/tp_overlap NVLS setup.
os.environ.setdefault("NCCL_NVLS_ENABLE", "1")
os.environ.setdefault("NCCL_ALGO", "NVLS")

BLOCK_M, BLOCK_N = 128, 128


def _t3(x):
    """(.,.) torch tensor -> (.,.,1) 3D for the (M,N,L=1) cute convention."""
    return x.unsqueeze(-1)


def cdiv(a, b):
    return (a + b - 1) // b


def bench_cuda(fn, warmup, iters, torch_stream=None):
    """Time `fn` on `torch_stream` (or current stream). Events must be recorded
    on the SAME stream the work is issued to, else they capture nothing."""
    ctx = torch.cuda.stream(torch_stream) if torch_stream is not None else _null_ctx()
    with ctx:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
    torch.cuda.synchronize()
    dist.barrier()
    return s.elapsed_time(e) / iters  # ms


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    # Profiling warm-up: number of fully-initialised run_fused() launches to fire
    # BEFORE the (profiled) correctness launch, so caches / clocks / symmetric-mem
    # first-touch are warm. Pair with ncu `--launch-skip <prof_warmup>` so ncu skips
    # these and profiles a steady-state launch instead of the cold first one.
    ap.add_argument("--prof-warmup", type=int, default=0)
    ap.add_argument("--check", action="store_true", default=True)
    ap.add_argument("--tol_abs", type=float, default=0.5)
    ap.add_argument("--tol_rel", type=float, default=5e-2)
    # MegaAttn: cross-rank handshake granularity G. Each SM does one flag bump +
    # spin per batch of G tiles instead of per tile (plan: per-tile -> per-batch).
    # G=1 reproduces the per-tile behaviour (numerical regression baseline).
    ap.add_argument("--comm-batch-tiles", type=int, default=8)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    ws = int(os.environ.get("WORLD_SIZE", 1))
    lr = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(lr)
    device = torch.device(f"cuda:{lr}")

    dist.init_process_group("nccl")
    group = dist.group.WORLD
    gname = group.group_name
    symm_mem.enable_symm_mem_for_group(gname)

    M, K, N = ref.shapes(ws)
    num_m_tiles = cdiv(M, BLOCK_M)
    num_n_tiles = cdiv(N, BLOCK_N)
    num_tiles = num_m_tiles * num_n_tiles

    if rank == 0:
        print(f"[megaattn] M={M} K={K} N={N} W={ws} "
              f"tiles={num_tiles} ({num_m_tiles}x{num_n_tiles}) dtype={ref.DTYPE}")

    # ---- inputs (per rank) ----
    O_local, W_o = ref.make_inputs(rank, ws, device)         # [M,K], [K,N] bf16
    B = W_o.t().contiguous()                                 # [N,K] k-major for GEMM

    # ---- symmetric C (= partial / AR input) and per-tile flag ----
    C = symm_mem.empty(M, N, device=device, dtype=ref.DTYPE)
    C.zero_()
    hC = symm_mem.rendezvous(C, gname)
    assert hC.has_multicast_support, "symmetric memory has no multicast support"
    c_mc_ptr = hC.multicast_ptr

    # MegaAttn per-batch handshake (plan: per-tile -> per-batch). flag layout:
    #   [0, num_batch_slots)              per-batch readiness
    #   [num_batch_slots, +nSM)           per-SM completion barrier (RS+AG)
    # nSM = physical SM count (>= grid size). The batch-slot count must match the
    # device-side formula exactly: a "batch" = G consecutive tiles in an SM's
    # persistent execution order, so each SM owns up to max_batches_per_sm slots.
    #   grid_size          = min(num_tiles, max_active_clusters)   [cluster=(1,1)]
    #   max_batches_per_sm = cdiv(cdiv(num_tiles, grid_size), G)
    #   num_batch_slots    = grid_size * max_batches_per_sm
    G = args.comm_batch_tiles
    hw = cutlass.utils.HardwareInfo()
    max_active_clusters = hw.get_max_active_clusters(1)
    grid_size = min(num_tiles, int(max_active_clusters))
    max_batches_per_sm = cdiv(cdiv(num_tiles, grid_size), G)
    num_batch_slots = grid_size * max_batches_per_sm
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    flag = symm_mem.empty(num_batch_slots + num_sms, device=device, dtype=torch.int32)
    flag.zero_()
    if rank == 0:
        print(f"[megaattn] comm_batch_tiles(G)={G} grid_size={grid_size} "
              f"max_batches_per_sm={max_batches_per_sm} "
              f"num_batch_slots={num_batch_slots} (nSM={num_sms})")
    hF = symm_mem.rendezvous(flag, gname)
    flag_mc_ptr = hF.multicast_ptr

    # RS+AG writes the AllReduce result in place into the symmetric C (no `out`).

    # ---- cute tensors (M,*,L=1) ----
    a_ct = from_dlpack(_t3(O_local), assumed_align=16)
    b_ct = from_dlpack(_t3(B), assumed_align=16)
    c_ct = from_dlpack(_t3(C), assumed_align=16)
    flag_ct = from_dlpack(flag, assumed_align=4)

    kernel = OProjARFusedKernelSM90(
        cutlass.Float32, (BLOCK_M, BLOCK_N), (1, 1), swizzle_size=1, raster_along_m=True,
        comm_batch_tiles=G,
    )

    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    if rank == 0:
        print("[megaattn] compiling fused kernel ...")
    compiled = cute.compile(
        kernel, a_ct, b_ct, c_ct, flag_ct,
        c_mc_ptr, flag_mc_ptr, rank, ws, max_active_clusters, stream,
    )

    def run_fused():
        C.zero_()
        flag.zero_()
        dist.barrier(device_ids=[lr])
        compiled(a_ct, b_ct, c_ct, flag_ct, stream)

    # ---- profiling warm-up ----
    # Fire N fully-initialised launches so the launch ncu actually profiles
    # (skipped to via --launch-skip N) runs with warm caches / settled clocks.
    # Each run_fused() re-zeroes C/flag and barriers, so state stays correct.
    for _ in range(args.prof_warmup):
        run_fused()
    if args.prof_warmup:
        torch.cuda.synchronize()

    # ---- correctness ----
    if args.check:
        run_fused()
        torch.cuda.synchronize()
        ref_out = ref.reference_out(O_local, W_o, group=group)
        max_abs, max_rel, mean_abs = ref.compare(C, ref_out)
        # bf16 reduction: gate on absolute error (relative error is meaningless
        # for the many near-zero output entries).
        ok = max_abs <= args.tol_abs
        msg = (f"[rank{rank}] check max_abs={max_abs:.4f} max_rel={max_rel:.4f} "
               f"mean_abs={mean_abs:.5f} ref_absmax={ref_out.abs().max().item():.3f} "
               f"-> {'PASS' if ok else 'FAIL'}")
        print(msg)
        flags = [None] * ws
        dist.all_gather_object(flags, ok)
        if rank == 0:
            print(f"[megaattn] correctness: {'ALL PASS' if all(flags) else 'FAIL'}")

    # ---- timing ----
    # fused kernel launches on `stream` (== torch_stream); record events there.
    t_fused = bench_cuda(lambda: compiled(a_ct, b_ct, c_ct, flag_ct, stream),
                         args.warmup, args.iters, torch_stream=torch_stream)

    partial = torch.empty(M, N, device=device, dtype=ref.DTYPE)
    t_gemm = bench_cuda(lambda: torch.matmul(O_local, W_o, out=partial),
                        args.warmup, args.iters)
    ar_buf = torch.empty(M, N, device=device, dtype=ref.DTYPE)
    t_ar = bench_cuda(lambda: dist.all_reduce(ar_buf, op=dist.ReduceOp.SUM),
                      args.warmup, args.iters)

    if rank == 0:
        baseline = t_gemm + t_ar
        print("\n==================== megaattn O_proj fused AR ====================")
        print(f"  fused GEMM+AR (1 kernel)   : {t_fused:.4f} ms")
        print(f"  torch GEMM only            : {t_gemm:.4f} ms")
        print(f"  NVLS all_reduce only       : {t_ar:.4f} ms")
        print(f"  un-fused baseline (sum)    : {baseline:.4f} ms")
        print(f"  exposed-AR (t_fused-t_gemm): {t_fused - t_gemm:.4f} ms")
        if baseline > 0:
            print(f"  speedup vs baseline        : {baseline / t_fused:.3f}x "
                  f"({100*(1-t_fused/baseline):.1f}% faster)")
        print("==================================================================\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
