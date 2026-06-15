#!/usr/bin/env python3
"""
Minimal ncu profiling driver for the fused O_proj GEMM + one-shot NVLS AllReduce
kernel. Unlike bench.py it does NOT run the torch-GEMM / NCCL-AR baseline loops —
it just builds the symmetric buffers, compiles the kernel once, runs a couple of
warmup launches, then one nvtx-marked "prof" launch, and exits.

Profiled under `ncu --replay-mode application` so each replay pass re-runs the whole
8-rank app (host dist.barrier()s included) and the collective stays in lockstep —
the cross-rank spin in the kernel completes every pass, so there is no profiler
deadlock (which kernel-replay isolation would cause).

  ncu --target-processes all --replay-mode application \
      --kernel-name regex:OProjARFused --launch-skip 2 --launch-count 1 \
      --section SpeedOfLight --section WarpStateStats -f -o report \
      /usr/bin/python -m torch.distributed.run --nproc_per_node=8 megaattn/prof_driver.py
"""
import os
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.cuda.nvtx as nvtx

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack

import reference as ref
from oproj_ar_sm90 import OProjARFusedKernelSM90

os.environ.setdefault("NCCL_NVLS_ENABLE", "1")
os.environ.setdefault("NCCL_ALGO", "NVLS")

BLOCK_M, BLOCK_N = 128, 128


def _t3(x):
    return x.unsqueeze(-1)


def cdiv(a, b):
    return (a + b - 1) // b


def main():
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
    num_tiles = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)

    O_local, W_o = ref.make_inputs(rank, ws, device)
    B = W_o.t().contiguous()

    C = symm_mem.empty(M, N, device=device, dtype=ref.DTYPE)
    C.zero_()
    hC = symm_mem.rendezvous(C, gname)
    assert hC.has_multicast_support, "symmetric memory has no multicast support"
    c_mc_ptr = hC.multicast_ptr

    flag = symm_mem.empty(num_tiles, device=device, dtype=torch.int32)
    flag.zero_()
    hF = symm_mem.rendezvous(flag, gname)
    flag_mc_ptr = hF.multicast_ptr

    out = torch.zeros(M, N, device=device, dtype=ref.DTYPE)
    out_i32 = out.view(torch.int32)

    a_ct = from_dlpack(_t3(O_local), assumed_align=16)
    b_ct = from_dlpack(_t3(B), assumed_align=16)
    c_ct = from_dlpack(_t3(C), assumed_align=16)
    out_ct = from_dlpack(_t3(out_i32), assumed_align=16)
    flag_ct = from_dlpack(flag, assumed_align=4)

    kernel = OProjARFusedKernelSM90(
        cutlass.Float32, (BLOCK_M, BLOCK_N), (1, 1), swizzle_size=1, raster_along_m=True
    )
    hw = cutlass.utils.HardwareInfo()
    max_active_clusters = hw.get_max_active_clusters(1)

    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    if rank == 0:
        print(f"[prof] M={M} K={K} N={N} W={ws} tiles={num_tiles} compiling ...")
    compiled = cute.compile(
        kernel, a_ct, b_ct, c_ct, out_ct, flag_ct,
        c_mc_ptr, flag_mc_ptr, rank, ws, max_active_clusters, stream,
    )

    # One reset+barrier launch to prime (mirrors bench.run_fused), then bare
    # compiled() launches inside the stream context (the comm warps self-reset
    # each flag slot to 0, so no per-launch zeroing is needed) — this is exactly
    # bench.py's working timing path.
    C.zero_()
    flag.zero_()
    dist.barrier(device_ids=[lr])
    compiled(a_ct, b_ct, c_ct, out_ct, flag_ct, stream)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print("[prof] primed; warmup ...", flush=True)

    with torch.cuda.stream(torch_stream):
        for _ in range(2):  # warmups, --launch-skip'd
            compiled(a_ct, b_ct, c_ct, out_ct, flag_ct, stream)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print("[prof] profiled launch ...", flush=True)
        nvtx.range_push("prof")
        compiled(a_ct, b_ct, c_ct, out_ct, flag_ct, stream)
        torch.cuda.synchronize()
        nvtx.range_pop()

    dist.barrier()
    if rank == 0:
        print("[prof] done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
