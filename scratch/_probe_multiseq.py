#!/usr/bin/env python3
"""临时探针：单独跑 fused FA-path 的 multi_seq 用例，区分 compile vs execute 是否 hang。"""
import time, sys
import numpy as np
import torch
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack

from mega_attention.kernels.sm90.fused_fa_oproj_ar import FusedFaOprojAr, NUM_CTRL
from mega_attention.metadata.row_desc import build_row_desc, oproj_task_counts

DT = torch.bfloat16
DEV = "cuda:0"
def _u32(n): return torch.zeros(n, dtype=torch.uint32, device=DEV)
def _i32(a): return torch.tensor(np.asarray(a), dtype=torch.int32, device=DEV)

seqlens, H_local, hidden = [200, 64, 300], 4, 768
D, N_TILE, super_group_n_tiles, num_ctas = 128, 128, 4, 8
torch.manual_seed(1)
meta = build_row_desc(seqlens)
R = meta.num_row_tiles
num_fa = R * H_local
_, num_super_groups, total_oproj = oproj_task_counts(R, hidden, N_TILE, super_group_n_tiles)
tot = int(sum(seqlens))
print(f"R={R} num_fa={num_fa} total_oproj={total_oproj}", flush=True)

Q = torch.randn(tot, H_local, D, device=DEV, dtype=DT) * 0.2
K = torch.randn(tot, H_local, D, device=DEV, dtype=DT) * 0.2
V = torch.randn(tot, H_local, D, device=DEV, dtype=DT) * 0.2
Oscr = torch.zeros(R, 128, H_local, D, device=DEV, dtype=DT)
u32s = [_u32(NUM_CTRL), _u32(R), _u32(total_oproj), _u32(total_oproj), _u32(total_oproj),
        _u32(total_oproj), _u32(num_fa), _u32(total_oproj), _u32(total_oproj), _u32(total_oproj)]
c_u32 = [from_dlpack(t, assumed_align=4) for t in u32s]
c_data = [from_dlpack(t, assumed_align=16) for t in (Q, K, V, Oscr)]
c_meta = [from_dlpack(t, assumed_align=16) for t in
          (_i32(meta.cu_seqlens_q), _i32(meta.cu_seqlens_k), _i32(meta.batch_idx), _i32(meta.m_block))]
cts = c_u32 + c_data + c_meta

ker = FusedFaOprojAr(num_fa=num_fa, num_row_tiles=R, H_local=H_local, D=D,
                     num_super_groups=num_super_groups, total_oproj=total_oproj,
                     num_ctas=num_ctas, tp_size=1)
ts = torch.cuda.Stream(); st = cuda.CUstream(ts.cuda_stream)
t0 = time.time()
compiled = cute.compile(ker, *cts, st)
print(f"COMPILE_DONE {time.time()-t0:.1f}s", flush=True)
t1 = time.time()
with torch.cuda.stream(ts):
    compiled(*cts, st)
print(f"LAUNCH_RETURNED {time.time()-t1:.1f}s", flush=True)
torch.cuda.synchronize()
print(f"SYNC_DONE {time.time()-t1:.1f}s  EXEC_OK", flush=True)
