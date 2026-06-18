#!/usr/bin/env python3
"""P3a smoke: device-side NVLink barrier (monotonic add1 + spin) across TP ranks.

torchrun --nproc_per_node=8 scratch/_probe_nvlbarrier.py
Validates: multimem_red_add1 on the multicast signal bumps ALL ranks; each rank
spins on its LOCAL signal slot until it reaches tp_size; kernel returns (no hang)
and both barrier slots == tp_size. Foundation for the P3 nvl_barrier(init/exit).
"""
import os
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import cutlass
import cutlass.cute as cute
import cutlass.utils.distributed as cda
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack


class NvlBarrierSmoke:
    def __init__(self, tp_size, mc_ptr):
        self.tp_size = tp_size
        self.mc_ptr = mc_ptr

    @cute.jit
    def __call__(self, sig_local: cute.Tensor, stream: cuda.CUstream):
        sig_mc_iter = cute.make_ptr(sig_local.element_type, self.mc_ptr,
                                    cute.AddressSpace.gmem, assumed_align=4)
        sig_mc = cute.make_tensor(sig_mc_iter, cute.make_layout(2))
        self.kernel(sig_local, sig_mc).launch(grid=[1, 1, 1], block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, sig_local: cute.Tensor, sig_mc: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        tp = cutlass.const_expr(self.tp_size)
        # ---- init barrier (slot 0): bump all ranks, spin until local == tp ----
        cute.arch.sync_threads()
        if tidx == 0:
            cda.multimem_red_add1(sig_mc.iterator + 0, order="release", scope="sys")
            cda.spin_lock_ld_lt_relaxed_wait(sig_local.iterator + 0,
                                             expected_val=cutlass.Int32(tp), scope="sys")
        cute.arch.sync_threads()
        # ---- exit barrier (slot 1) ----
        if tidx == 0:
            cda.multimem_red_add1(sig_mc.iterator + 1, order="release", scope="sys")
            cda.spin_lock_ld_lt_relaxed_wait(sig_local.iterator + 1,
                                             expected_val=cutlass.Int32(tp), scope="sys")
        cute.arch.sync_threads()


def main():
    lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    gname = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(gname)

    sig = symm_mem.empty(1024, device=f"cuda:{lr}", dtype=torch.int32); sig.zero_()
    h = symm_mem.rendezvous(sig, gname)
    assert h.has_multicast_support
    if rank == 0:
        print(f"[rank0] multicast_ptr={h.multicast_ptr:#x} buffer_ptr={h.buffer_ptrs[0]:#x}", flush=True)
    c_local = from_dlpack(sig, assumed_align=4)

    ker = NvlBarrierSmoke(ws, h.multicast_ptr)
    ts = torch.cuda.Stream(); st = cuda.CUstream(ts.cuda_stream)
    dist.barrier()
    compiled = cute.compile(ker, c_local, st)
    with torch.cuda.stream(ts):
        compiled(c_local, st)
    torch.cuda.synchronize()
    dist.barrier()
    got = sig[:2].cpu().tolist()
    ok = (got == [ws, ws])
    print(f"[rank{rank}] signal[:2]={got} expect=[{ws},{ws}] {'OK' if ok else 'FAIL'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
