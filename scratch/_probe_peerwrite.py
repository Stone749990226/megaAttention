#!/usr/bin/env python3
"""P3b smoke: cross-rank directed atomic write to buffer_ptrs[owner_rank] (RUNTIME
owner) via a constexpr-pointer if-ladder. torchrun --nproc_per_node=8.

Each rank, for a runtime dst in [0,tp), atomicAdds 1 to rank dst's ctrl[my_rank]
(peer write over NVLink). Here every rank writes to ALL ranks (dst loops 0..tp-1
read from a device tensor to force runtime selection), so each rank's ctrl[r] ends
up == tp (written once by every rank). Validates the owner-directed peer write that
P3b's publish_ar_ready needs.
"""
import os
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack


class PeerWriteSmoke:
    def __init__(self, tp_size, rank, buf_ptrs):
        self.tp_size = tp_size
        self.rank = rank
        self.buf_ptrs = list(buf_ptrs)          # tp_size peer base addresses (int)

    @cute.jit
    def __call__(self, ctrl_local: cute.Tensor, dsts: cute.Tensor, stream: cuda.CUstream):
        self.kernel(ctrl_local, dsts).launch(grid=[1, 1, 1], block=[32, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, ctrl_local: cute.Tensor, dsts: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        tp = cutlass.const_expr(self.tp_size)
        my = cutlass.const_expr(self.rank)
        if tidx == 0:
            # for each target slot d (runtime, read from dsts), peer-add 1 to rank d's ctrl[my]
            for i in cutlass.range_constexpr(tp):
                dst = dsts[i]                    # runtime value in [0, tp)
                for r in cutlass.range_constexpr(tp):
                    if dst == cutlass.Int32(r):
                        peer = cute.make_ptr(ctrl_local.element_type, self.buf_ptrs[r],
                                             cute.AddressSpace.gmem, assumed_align=4)
                        cute.arch.atomic_add(peer + cutlass.Int32(my),
                                             cutlass.Int32(1), sem="release", scope="sys")


def main():
    lr = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    gname = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(gname)

    ctrl = symm_mem.empty(max(ws, 64), device=f"cuda:{lr}", dtype=torch.int32); ctrl.zero_()
    h = symm_mem.rendezvous(ctrl, gname)
    buf_ptrs = [h.buffer_ptrs[r] for r in range(ws)]
    dsts = torch.arange(ws, dtype=torch.int32, device=f"cuda:{lr}")  # write to every rank

    c_local = from_dlpack(ctrl, assumed_align=4)
    c_dsts = from_dlpack(dsts, assumed_align=4)
    ker = PeerWriteSmoke(ws, rank, buf_ptrs)
    ts = torch.cuda.Stream(); st = cuda.CUstream(ts.cuda_stream)
    dist.barrier()
    compiled = cute.compile(ker, c_local, c_dsts, st)
    with torch.cuda.stream(ts):
        compiled(c_local, c_dsts, st)
    torch.cuda.synchronize()
    dist.barrier()
    got = ctrl[:ws].cpu().tolist()
    ok = all(v == 1 for v in got)   # each rank wrote ctrl[my] on every rank -> each slot==1
    print(f"[rank{rank}] ctrl[:ws]={got} expect all 1 {'OK' if ok else 'FAIL'}", flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
