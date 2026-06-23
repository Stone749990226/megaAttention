#!/usr/bin/env python3
"""P3: 8-rank fused FA + O_proj + real NVLS AR, end-to-end vs full_chain_reference.

torchrun --nproc_per_node=8 tests/fused/test_fused_full_chain.py

Each rank runs the full fused kernel on its OWN Q/K/V/W_o (per-rank seeds); the AR
owner protocol reduces every rank's C_sym partial in place via multimem over C_sym_mc.
After the kernel every rank's C_sym[valid] must equal Y_final = sum_rank(O@W_o).
"""
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack

from mega_attention.kernels.sm90.fused_fa_oproj_ar import FusedFaOprojAr, NUM_CTRL, NUM_SYNC
from mega_attention.metadata.row_desc import build_row_desc, oproj_task_counts, active_counts
from mega_attention.reference.fused import fa_reference, oproj_reference

DT = torch.bfloat16


def run_one(rank, ws, dev, gname, tag, seqlens, seqlens_k, H_local, hidden, q_per_kv):
    D = 128
    H_kv = H_local // q_per_kv
    N_TILE, sg = 128, 4
    meta = build_row_desc(seqlens, seqlens_k=seqlens_k)
    R = meta.num_row_tiles
    K_local = H_local * D
    num_fa = R * H_local
    num_out, num_super_groups, total_oproj = oproj_task_counts(R, hidden, N_TILE, sg)
    tot = int(sum(seqlens))                                    # tot_q
    tot_k = int(sum(seqlens if seqlens_k is None else seqlens_k))
    hidden_pad = num_out * N_TILE
    owner_slots = (total_oproj + ws - 1) // ws
    owner_words = (owner_slots + 63) // 64

    # per-rank inputs (different seeds -> AR is a real cross-rank sum)
    g = torch.Generator(device=dev).manual_seed(1234 + rank)
    Q = torch.randn(tot, H_local, D, device=dev, dtype=DT, generator=g) * 0.2
    K = torch.randn(tot_k, H_kv, D, device=dev, dtype=DT, generator=g) * 0.2
    V = torch.randn(tot_k, H_kv, D, device=dev, dtype=DT, generator=g) * 0.2
    W_o = torch.randn(K_local, hidden, device=dev, dtype=DT, generator=g) * (K_local ** -0.5)
    W_o_pad = torch.zeros(K_local, hidden_pad, device=dev, dtype=DT); W_o_pad[:, :hidden] = W_o
    Oscr = torch.zeros(R, 128, H_local, D, device=dev, dtype=DT)

    def _u32(n): return torch.zeros(n, dtype=torch.uint32, device=dev)
    def _i32(a): return torch.tensor(np.asarray(a), dtype=torch.int32, device=dev)

    # ---- symmetric buffers (C_sym + cross-rank control + nvl signal) ----
    C_sym = symm_mem.empty(R, 128, num_out, N_TILE, device=dev, dtype=DT); C_sym.zero_()
    hC = symm_mem.rendezvous(C_sym, gname); assert hC.has_multicast_support
    rco = symm_mem.empty(owner_slots, device=dev, dtype=torch.uint32); rco.zero_()
    hRC = symm_mem.rendezvous(rco, gname)
    rbits = symm_mem.empty(owner_words, device=dev, dtype=torch.int64); rbits.zero_()
    hRB = symm_mem.rendezvous(rbits, gname)
    nvl = symm_mem.empty(2, device=dev, dtype=torch.int32); nvl.zero_()  # phase/sign signal[2]
    hN = symm_mem.rendezvous(nvl, gname)

    ar_done_bits = torch.zeros(owner_words, dtype=torch.int64, device=dev)  # owner-local
    ctrl = _u32(NUM_CTRL); head_ready = _u32(R); oproj_queue = _u32(total_oproj)
    sync_ctrl = _u32(NUM_SYNC)          # grid_sync (init/exit) + nvl counter; never reset
    actv = _i32(active_counts(R, H_local, num_super_groups, ws, rank))  # runtime active counts
    cu_q, cu_k = _i32(meta.cu_seqlens_q), _i32(meta.cu_seqlens_k)
    fa_b, fa_mb = _i32(meta.batch_idx), _i32(meta.m_block)

    cts = [from_dlpack(t, assumed_align=4)
           for t in (ctrl, sync_ctrl, actv, head_ready, oproj_queue, rco)]
    cts += [from_dlpack(t, assumed_align=8) for t in (rbits, ar_done_bits)]
    cts += [from_dlpack(t, assumed_align=16) for t in (Q, K, V, Oscr, W_o_pad, C_sym)]
    cts += [from_dlpack(t, assumed_align=16) for t in (cu_q, cu_k, fa_b, fa_mb)]

    ker = FusedFaOprojAr(
        num_fa=num_fa, num_row_tiles=R, H_local=H_local, D=D, q_per_kv=q_per_kv,
        num_super_groups=num_super_groups, total_oproj=total_oproj, num_ctas=8,
        hidden=hidden, tp_size=ws, rank=rank, N_TILE=N_TILE, super_group_n_tiles=sg,
        csym_mc_ptr=hC.multicast_ptr, nvl_mc_ptr=hN.multicast_ptr,
        nvl_local_ptr=hN.buffer_ptrs[rank],
        nvl_peer_ptrs=[hN.buffer_ptrs[r] for r in range(ws)],
        rc_ptrs=[hRC.buffer_ptrs[r] for r in range(ws)],
        rb_ptrs=[hRB.buffer_ptrs[r] for r in range(ws)])
    ts = torch.cuda.Stream(); st = cuda.CUstream(ts.cuda_stream)
    dist.barrier()
    compiled = cute.compile(ker, *cts, st)

    # reference: Y_partial per rank -> all_reduce -> Y_final (same on all ranks)
    O_ref = fa_reference(Q, K, V, meta)
    Yp = oproj_reference(O_ref, W_o, meta)              # [tot, hidden] fp32
    Yf = Yp.clone(); dist.all_reduce(Yf, op=dist.ReduceOp.SUM)

    # Reuse path: launch the same layer several times with NO host reset between
    # launches. The kernel-start cleaner + phase/sign barriers must reproduce the
    # exact same result every iteration (cross-layer workspace reuse).
    # First-layer bootstrap: the kernel has no start cleaner / init barrier, so all
    # ranks' buffer zeroing must be complete and cross-rank visible before the first
    # launch issues a remote control write (mirrors workspace.create's barrier).
    torch.cuda.synchronize(); dist.barrier()
    NREUSE = 3
    err = 0.0
    for _ in range(NREUSE):
        with torch.cuda.stream(ts):
            compiled(*cts, st)
        torch.cuda.synchronize(); dist.barrier()
        C = C_sym.float().cpu()
        for t in range(R):
            vm = meta.valid_m(t); qs = meta.q_tile_start(t)
            for o in range(num_out):
                vn = min(N_TILE, hidden - o * N_TILE)
                got = C[t, :vm, o, :vn]
                exp = Yf[qs:qs + vm, o * N_TILE:o * N_TILE + vn].cpu()
                err = max(err, (got - exp).abs().max().item())
        dist.barrier()
    fa_done, op_done, ar_done = int(ctrl[1]), int(ctrl[5]), int(ctrl[6])
    local_owned = (max(total_oproj - rank, 0) + ws - 1) // ws
    # exit cleaner zeroes task_ctrl on the way out; reaching it (all_done) leaves 0.
    ok = (err < 5e-2 and fa_done == 0 and op_done == 0 and ar_done == 0)
    print(f"[rank{rank}][{tag}] err={err:.4g} done(fa,op,ar)=({fa_done},{op_done},{ar_done}) "
          f"[exit-clean->0] local_owned={local_owned} {'OK' if ok else 'FAIL'}", flush=True)
    dist.barrier()
    return ok


def main():
    lr = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
    dev = torch.device(f"cuda:{lr}")
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    gname = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(gname)
    torch.manual_seed(1000 + rank)

    # (tag, seqlens_q, seqlens_k, H_local, hidden, q_per_kv)
    configs = [
        # q==k 完整 prompt prefill 回归 (offset=0, GQA q_per_kv=2)
        ("qk_eq",  [200, 64, 300], None,           4, 512, 2),
        # q<k contiguous-KV chunked/append prefill: 混合 offset 对齐/不对齐 + tail + GQA
        ("chunk",  [200, 64, 300], [512, 64, 460], 4, 512, 2),
    ]
    allok = True
    for tag, sq, sk, H, hidden, qpk in configs:
        allok = run_one(rank, ws, dev, gname, tag, sq, sk, H, hidden, qpk) and allok
    dist.barrier(); dist.destroy_process_group()
    return allok


if __name__ == "__main__":
    main()
