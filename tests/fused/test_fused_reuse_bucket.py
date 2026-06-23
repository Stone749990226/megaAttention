#!/usr/bin/env python3
"""Phase 4: bucket-capacity compile + runtime active counts reuse (single rank).

Compile ONE FusedFaOprojAr at a bucket capacity, then launch it for several smaller
active shapes WITHOUT recompiling. Every kernel argument is allocated at capacity; each
active batch fills only the prefix. The kernel reads its active task counts from `actv`,
so scheduling / directed cleaner / AR claim all bound to the active range. C_sym[active]
must match the O_proj reference each time (tp_size=1 -> AR is identity).

    python tests/fused/test_fused_reuse_bucket.py
"""
import cuda.bindings.driver as cuda
import numpy as np
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from mega_attention.kernels.sm90.fused_fa_oproj_ar import FusedFaOprojAr, NUM_CTRL, NUM_SYNC
from mega_attention.metadata.row_desc import (
    build_row_desc, oproj_task_counts, active_counts)
from mega_attention.reference.fused import fa_reference, oproj_reference

DT = torch.bfloat16
DEV = "cuda:0"


def main():
    torch.manual_seed(0)
    dev = torch.device(DEV)
    H_local, D, hidden, N_TILE, sg = 4, 128, 512, 128, 4
    K_local = H_local * D

    # ---- bucket capacity (compile once against these shapes) ----
    MAX_RT = 6                                   # max num_row_tiles
    MAX_BATCH = MAX_RT                            # each tile could be its own seq
    tot_cap = MAX_RT * 128
    num_out, num_super_groups, max_total_oproj = oproj_task_counts(MAX_RT, hidden, N_TILE, sg)
    max_owner_slots = max_total_oproj            # tp_size = 1
    max_owner_words = (max_owner_slots + 63) // 64
    hidden_pad = num_out * N_TILE

    def _u32(n): return torch.zeros(n, dtype=torch.uint32, device=dev)
    def _i32(a): return torch.tensor(np.asarray(a), dtype=torch.int32, device=dev)

    # All kernel args are capacity-shaped; active launches fill prefixes only.
    Q = torch.zeros(tot_cap, H_local, D, device=dev, dtype=DT)
    K = torch.zeros(tot_cap, H_local, D, device=dev, dtype=DT)
    V = torch.zeros(tot_cap, H_local, D, device=dev, dtype=DT)
    W_o = torch.zeros(K_local, hidden_pad, device=dev, dtype=DT)
    W_o[:, :hidden] = torch.randn(K_local, hidden, device=dev, dtype=DT) * (K_local ** -0.5)
    Oscr = torch.zeros(MAX_RT, 128, H_local, D, device=dev, dtype=DT)
    C_sym = torch.zeros(MAX_RT, 128, num_out, N_TILE, device=dev, dtype=DT)

    ctrl = _u32(NUM_CTRL); sync_ctrl = _u32(NUM_SYNC)
    head_ready = _u32(MAX_RT); oproj_queue = _u32(max_total_oproj)
    rco = _u32(max_owner_slots)
    rbits = torch.zeros(max_owner_words, dtype=torch.int64, device=dev)
    ar_done_bits = torch.zeros(max_owner_words, dtype=torch.int64, device=dev)
    actv = _i32(np.zeros(6, dtype=np.int32))
    cu_q = _i32(np.zeros(MAX_BATCH + 1)); cu_k = _i32(np.zeros(MAX_BATCH + 1))
    fa_b = _i32(np.zeros(MAX_RT)); fa_mb = _i32(np.zeros(MAX_RT))

    cts = [from_dlpack(t, assumed_align=4) for t in (ctrl, sync_ctrl, actv, head_ready,
                                                     oproj_queue, rco)]
    cts += [from_dlpack(t, assumed_align=8) for t in (rbits, ar_done_bits)]
    cts += [from_dlpack(t, assumed_align=16) for t in (Q, K, V, Oscr, W_o, C_sym)]
    cts += [from_dlpack(t, assumed_align=16) for t in (cu_q, cu_k, fa_b, fa_mb)]

    # Compile at capacity: num_fa / num_row_tiles / total_oproj are the bucket maxima.
    ker = FusedFaOprojAr(num_fa=MAX_RT * H_local, num_row_tiles=MAX_RT, H_local=H_local,
                         D=D, num_super_groups=num_super_groups,
                         total_oproj=max_total_oproj, num_ctas=8, hidden=hidden,
                         tp_size=1, N_TILE=N_TILE, super_group_n_tiles=sg)
    ts = torch.cuda.Stream(); st = cuda.CUstream(ts.cuda_stream)
    compiled = cute.compile(ker, *cts, st)

    # ---- active shapes (all <= capacity), reusing the one compiled kernel ----
    shapes = [
        ("rt2_256",     [256]),
        ("rt4_128_300", [128, 300]),
        ("rt1_128",     [128]),
        ("rt6_full",    [128, 128, 128, 128, 128, 128]),
    ]
    failed = 0
    for name, seqlens in shapes:
        meta = build_row_desc(seqlens)
        R = meta.num_row_tiles
        assert R <= MAX_RT
        tot = int(sum(seqlens))
        # fill active input prefix
        Q[:tot] = torch.randn(tot, H_local, D, device=dev, dtype=DT) * 0.2
        K[:tot] = torch.randn(tot, H_local, D, device=dev, dtype=DT) * 0.2
        V[:tot] = torch.randn(tot, H_local, D, device=dev, dtype=DT) * 0.2
        # active metadata prefixes
        nb = meta.num_batch
        cu_q[:nb + 1] = torch.tensor(meta.cu_seqlens_q, device=dev, dtype=torch.int32)
        cu_k[:nb + 1] = torch.tensor(meta.cu_seqlens_k, device=dev, dtype=torch.int32)
        fa_b[:R] = torch.tensor(meta.batch_idx, device=dev, dtype=torch.int32)
        fa_mb[:R] = torch.tensor(meta.m_block, device=dev, dtype=torch.int32)
        actv.copy_(torch.tensor(active_counts(R, H_local, num_super_groups, 1, 0),
                                device=dev))

        with torch.cuda.stream(ts):
            compiled(*cts, st)
        torch.cuda.synchronize()

        O_ref = fa_reference(Q[:tot], K[:tot], V[:tot], meta)
        Y_ref = oproj_reference(O_ref, W_o[:, :hidden], meta)
        C = C_sym.float().cpu()
        err = 0.0
        for t in range(R):
            vm = meta.valid_m(t); qs = meta.q_tile_start(t)
            for o in range(num_out):
                vn = min(N_TILE, hidden - o * N_TILE)
                got = C[t, :vm, o, :vn]
                exp = Y_ref[qs:qs + vm, o * N_TILE:o * N_TILE + vn].cpu()
                err = max(err, (got - exp).abs().max().item())
        fa_d, op_d, ar_d = int(ctrl[1]), int(ctrl[5]), int(ctrl[6])
        # exit cleaner zeroes task_ctrl; reaching it (all_done) leaves done counters at 0.
        ok = (err < 2e-2 and fa_d == 0 and op_d == 0 and ar_d == 0)
        if not ok:
            failed += 1
        print(f"{'PASS' if ok else 'FAIL'} {name}: err={err:.4g} R={R}/{MAX_RT} "
              f"done(fa,op,ar)=({fa_d},{op_d},{ar_d}) [exit-clean->0]", flush=True)

    print(f"\n{'ALL PASS' if failed == 0 else f'{failed} FAILED'}")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(1 if main() else 0)
