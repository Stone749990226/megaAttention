#!/usr/bin/env python3
"""Phase 5: multi-rank, multi-layer workspace reuse via FusedFaOprojArWorkspace.

torchrun --nproc_per_node=8 tests/fused/test_fused_workspace_reuse.py

Create ONE workspace at a bucket capacity, compile ONCE, then run several "layers"
(different per-rank Q/K/V/W_o, and different active shapes <= capacity) back to back
with NO host reset between launches. Each layer's C_sym[valid] must equal the full-chain
reference Y_final = all_reduce_sum_rank(O @ W_o). This is the sglang reuse path: the
kernel-start directed cleaner + phase/sign barriers carry correctness across layers.
"""
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import cutlass  # noqa: F401  (ensures DSL runtime import)

from mega_attention.runtime import FusedFaOprojArWorkspace
from mega_attention.metadata.row_desc import build_row_desc
from mega_attention.reference.fused import fa_reference, oproj_reference

DT = torch.bfloat16


def main():
    lr = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
    dev = torch.device(f"cuda:{lr}")
    dist.init_process_group("nccl")
    rank, ws_size = dist.get_rank(), dist.get_world_size()
    gname = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(gname)

    H_local, D, hidden, q_per_kv = 4, 128, 512, 2
    H_kv = H_local // q_per_kv
    K_local = H_local * D
    MAX_RT = 6                                   # bucket capacity

    ws = FusedFaOprojArWorkspace.create(
        gname, max_num_row_tiles=MAX_RT, hidden=hidden, H_local=H_local, D=D,
        tp_size=ws_size, rank=rank, q_per_kv=q_per_kv, max_tot_k=2048, dtype=DT, device=dev)
    ws.compile(num_ctas=8)
    num_out = ws.num_out

    # Several "layers": (tag, seqlens_q, seqlens_k). Different shapes <= capacity.
    layers = [
        ("L0_qk_eq",  [200, 64, 300], None),
        ("L1_chunk",  [200, 64, 300], [512, 64, 460]),
        ("L2_small",  [128],          [400]),
        ("L3_full6",  [128] * 6,      None),
        ("L0_again",  [200, 64, 300], None),         # repeat L0 to confirm determinism
    ]
    allok = True
    for li, (tag, sq, sk) in enumerate(layers):
        meta = build_row_desc(sq, seqlens_k=sk)
        R = meta.num_row_tiles
        tot = int(sum(sq)); tot_k = int(sum(sq if sk is None else sk))
        g = torch.Generator(device=dev).manual_seed(1234 + rank + 100 * li)
        Q = torch.randn(tot, H_local, D, device=dev, dtype=DT, generator=g) * 0.2
        K = torch.randn(tot_k, H_kv, D, device=dev, dtype=DT, generator=g) * 0.2
        V = torch.randn(tot_k, H_kv, D, device=dev, dtype=DT, generator=g) * 0.2
        W_o = torch.randn(K_local, hidden, device=dev, dtype=DT, generator=g) * (K_local ** -0.5)

        ws.set_layer(meta, Q, K, V, W_o)
        dist.barrier()
        ws.launch()
        torch.cuda.synchronize(); dist.barrier()

        # reference full chain
        O_ref = fa_reference(Q, K, V, meta)
        Yp = oproj_reference(O_ref, W_o, meta)
        Yf = Yp.clone(); dist.all_reduce(Yf, op=dist.ReduceOp.SUM)
        C = ws.csym.float().cpu()
        err = 0.0
        for t in range(R):
            vm = meta.valid_m(t); qs = meta.q_tile_start(t)
            for o in range(num_out):
                vn = min(128, hidden - o * 128)
                got = C[t, :vm, o, :vn]
                exp = Yf[qs:qs + vm, o * 128:o * 128 + vn].cpu()
                err = max(err, (got - exp).abs().max().item())
        ok = err < 5e-2
        allok = allok and ok
        print(f"[rank{rank}][{tag}] err={err:.4g} R={R}/{MAX_RT} {'OK' if ok else 'FAIL'}",
              flush=True)
        dist.barrier()

    dist.barrier(); dist.destroy_process_group()
    return allok


if __name__ == "__main__":
    main()
