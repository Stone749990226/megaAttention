#!/usr/bin/env python3
"""参数化探针：跑单个 fused FA-path 用例，区分 hang 触发条件。
用法: python3 scratch/_probe_param.py "200,64,300" H hidden num_ctas
外层用 timeout 包裹；正常会打印 RESULT，hang 则被 timeout 杀掉。"""
import sys, time
import numpy as np
import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.runtime import from_dlpack

from mega_attention.kernels.sm90.fused_fa_oproj_ar import FusedFaOprojAr, NUM_CTRL
from mega_attention.metadata.row_desc import build_row_desc, oproj_task_counts
from mega_attention.reference.fused import fa_reference, o_scratch_reference

DT, DEV = torch.bfloat16, "cuda:0"
def _u32(n): return torch.zeros(n, dtype=torch.uint32, device=DEV)
def _i32(a): return torch.tensor(np.asarray(a), dtype=torch.int32, device=DEV)

seqlens = [int(x) for x in sys.argv[1].split(",")]
H_local = int(sys.argv[2]) if len(sys.argv) > 2 else 4
hidden  = int(sys.argv[3]) if len(sys.argv) > 3 else 512
num_ctas= int(sys.argv[4]) if len(sys.argv) > 4 else 8
kv_stages = int(sys.argv[5]) if len(sys.argv) > 5 else 2
D, N_TILE, sg = 128, 128, 4
torch.manual_seed(0)
meta = build_row_desc(seqlens)
R = meta.num_row_tiles
num_fa = R * H_local
_, nsg, total_oproj = oproj_task_counts(R, hidden, N_TILE, sg)
tot = int(sum(seqlens))
nblks = [int(meta.m_block[t]) + 1 for t in range(R)]
print(f"CASE seqlens={seqlens} H={H_local} hidden={hidden} ctas={num_ctas} "
      f"R={R} num_fa={num_fa} total_oproj={total_oproj} max_nblk={max(nblks)} nblks={nblks}", flush=True)

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
                     num_super_groups=nsg, total_oproj=total_oproj, num_ctas=num_ctas,
                     tp_size=1, kv_stages=kv_stages)
ts = torch.cuda.Stream(); st = cuda.CUstream(ts.cuda_stream)
compiled = cute.compile(ker, *cts, st)
print("COMPILE_DONE", flush=True)
with torch.cuda.stream(ts):
    compiled(*cts, st)
torch.cuda.synchronize()
O_ref = fa_reference(Q, K, V, meta)
Oscr_ref = o_scratch_reference(O_ref, meta)
err = (Oscr.float() - Oscr_ref).abs().max().item()
ctrl = u32s[0].cpu()
print(f"RESULT err={err:.4g} fa_done={int(ctrl[1])}/{num_fa} "
      f"op_done={int(ctrl[5])}/{total_oproj} ar_done={int(ctrl[6])}/{total_oproj} "
      f"order_err={int(ctrl[8])} EXEC_OK", flush=True)
