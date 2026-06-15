#!/usr/bin/env python3
"""
Phase 1 DSL-plumbing smoke test (single GPU, no torch.distributed).

Validates, in isolation, the riskiest CuTe DSL idioms the scheduler skeleton
needs BEFORE we build the full kernel:
  * launch a persistent grid of G CTAs x T threads
  * global atomic_add task-claim from a workspace counter (exactly-once coverage)
  * Mega-MoE-style device-wide grid_sync (phase via top-bit flip), looped N times
  * read workspace buffers back on the host

    /usr/bin/python megaAttention/smoke_persistent.py
"""
import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

FINISH_TAG = 0x80000000  # top bit of uint32


@cute.jit
def _smoke(
    task_counter: cute.Tensor,   # [1] u32  global task dispenser
    exec_count: cute.Tensor,     # [num_tasks] u32  per-task execution tally
    gsc: cute.Tensor,            # [1] u32  grid_sync counter
    sentinel: cute.Tensor,       # [1] u32  written once after grid_sync
    num_tasks: cutlass.Constexpr,
    num_ctas: cutlass.Constexpr,
    num_sync_rounds: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    _smoke_kernel(task_counter, exec_count, gsc, sentinel,
                  num_tasks, num_ctas, num_sync_rounds).launch(
        grid=[num_ctas, 1, 1], block=[128, 1, 1], stream=stream,
    )


@cute.kernel
def _smoke_kernel(
    task_counter: cute.Tensor,
    exec_count: cute.Tensor,
    gsc: cute.Tensor,
    sentinel: cute.Tensor,
    num_tasks: cutlass.Constexpr,
    num_ctas: cutlass.Constexpr,
    num_sync_rounds: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # --- persistent task claim: every CTA's thread 0 races the same counter ---
    if tidx == 0:
        done = False
        while not done:
            tid = cute.arch.atomic_add(task_counter.iterator + 0,
                                       cutlass.Int32(1), sem="relaxed", scope="gpu")
            if tid >= num_tasks:
                done = True
            else:
                cute.arch.atomic_add(exec_count.iterator + tid,
                                     cutlass.Int32(1), sem="relaxed", scope="gpu")

    # --- grid_sync, looped to stress the phase/top-bit-flip protocol ---
    for _ in cutlass.range_constexpr(num_sync_rounds):
        _grid_sync(gsc, num_ctas, bidx, tidx)

    # --- one writer after the barrier (proves all CTAs reached the sync) ---
    if bidx == 0 and tidx == 0:
        cute.arch.atomic_add(sentinel.iterator + 0,
                             cutlass.Int32(1), sem="relaxed", scope="gpu")


@cute.jit
def _grid_sync(gsc: cute.Tensor, num_ctas: cutlass.Constexpr, bidx, tidx):
    """Mega-MoE device barrier: last arriver flips FINISH_TAG; others spin on it.

    gsc is Uint32 so the top-bit (FINISH_TAG) flip and the modular add are clean.
    """
    cute.arch.sync_threads()
    if tidx == 0:
        # init before the dynamic branch (DSL: vars used after control flow
        # must have a value before it).
        delta = cutlass.Uint32(1)
        if bidx == 0:
            delta = cutlass.Uint32(FINISH_TAG - (num_ctas - 1))
        old = cute.arch.atomic_add(gsc.iterator + 0, delta,
                                   sem="release", scope="gpu")
        # spin until the top bit changes relative to our snapshot `old`
        spinning = True
        while spinning:
            cur = cute.arch.atomic_add(gsc.iterator + 0, cutlass.Uint32(0),
                                       sem="acquire", scope="gpu")
            if ((cur ^ old) & cutlass.Uint32(FINISH_TAG)) != 0:
                spinning = False
    cute.arch.sync_threads()


def main():
    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    num_tasks = 5000
    # one CTA per SM (persistent)
    num_ctas = torch.cuda.get_device_properties(dev).multi_processor_count
    num_sync_rounds = 8

    task_counter = torch.zeros(1, dtype=torch.int32, device=dev)
    exec_count = torch.zeros(num_tasks, dtype=torch.int32, device=dev)
    gsc = torch.zeros(1, dtype=torch.uint32, device=dev)
    sentinel = torch.zeros(1, dtype=torch.int32, device=dev)

    tc = from_dlpack(task_counter, assumed_align=4)
    ec = from_dlpack(exec_count, assumed_align=4)
    gs = from_dlpack(gsc, assumed_align=4)
    se = from_dlpack(sentinel, assumed_align=4)

    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    print(f"[smoke] num_ctas(SMs)={num_ctas} num_tasks={num_tasks} "
          f"sync_rounds={num_sync_rounds} compiling ...")
    compiled = cute.compile(_smoke, tc, ec, gs, se,
                            num_tasks, num_ctas, num_sync_rounds, stream)
    with torch.cuda.stream(torch_stream):
        compiled(tc, ec, gs, se, stream)
    torch.cuda.synchronize()

    cov = exec_count.cpu()
    ok_cov = bool((cov == 1).all())
    ok_counter = int(task_counter.item()) == num_tasks + num_ctas  # each CTA over-reads once
    ok_sentinel = int(sentinel.item()) == 1
    print(f"[smoke] coverage exactly-once: {ok_cov} "
          f"(min={int(cov.min())} max={int(cov.max())})")
    print(f"[smoke] counter={int(task_counter.item())} "
          f"expected={num_tasks + num_ctas} -> {ok_counter}")
    print(f"[smoke] sentinel={int(sentinel.item())} (==1 after {num_sync_rounds} "
          f"grid_syncs) -> {ok_sentinel}")
    ok = ok_cov and ok_counter and ok_sentinel
    print(f"[smoke] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
