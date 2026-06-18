#!/usr/bin/env python3
"""
Packed-varlen FA payload test (fa_varlen.py FaWsAttnPacked).

Proves the fused FA payload: ONE compiled kernel reads packed varlen Q/K/V
[tot,H,D] at runtime (fa_row_tile, head) decoded from cu_seqlens + fa_row_desc,
and writes O_scratch[fa_row_tile,:,head,:]. Loops over EVERY (row_tile, head)
task of a real varlen batch through the same compiled kernel and compares the
full O_scratch to o_scratch_reference (torch causal SDPA).

    python -m pytest tests/fused/test_fa_packed.py
"""
import cuda.bindings.driver as cuda
import numpy as np
import torch

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from mega_attention.kernels.sm90.fa_varlen import FaWsAttnPacked
from mega_attention.metadata.row_desc import build_row_desc
from mega_attention.reference.fused import fa_reference, o_scratch_reference

DT = torch.bfloat16
DEV = "cuda:0"


def _i32(a, dev):
    return torch.tensor(np.asarray(a), dtype=torch.int32, device=dev)


def run_case(seqlens, H_local, D=128, seed=0):
    torch.manual_seed(seed)
    dev = torch.device(DEV)
    meta = build_row_desc(seqlens)
    R = meta.num_row_tiles
    tot = int(sum(seqlens))

    Q = (torch.randn(tot, H_local, D, device=dev, dtype=DT) * 0.2)
    K = (torch.randn(tot, H_local, D, device=dev, dtype=DT) * 0.2)
    V = (torch.randn(tot, H_local, D, device=dev, dtype=DT) * 0.2)
    Oscr = torch.zeros(R, 128, H_local, D, device=dev, dtype=DT)

    cu_q = _i32(meta.cu_seqlens_q, dev)
    cu_k = _i32(meta.cu_seqlens_k, dev)
    fa_b = _i32(meta.batch_idx, dev)
    fa_mb = _i32(meta.m_block, dev)
    task = torch.zeros(1, dtype=torch.int32, device=dev)

    ker = FaWsAttnPacked(128, 128, D, H_local)
    cQ = from_dlpack(Q, assumed_align=16)
    cK = from_dlpack(K, assumed_align=16)
    cV = from_dlpack(V, assumed_align=16)
    cO = from_dlpack(Oscr, assumed_align=16)
    cCuQ = from_dlpack(cu_q, assumed_align=16)
    cCuK = from_dlpack(cu_k, assumed_align=16)
    cFaB = from_dlpack(fa_b, assumed_align=16)
    cFaMb = from_dlpack(fa_mb, assumed_align=16)
    cTask = from_dlpack(task, assumed_align=16)
    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    compiled = cute.compile(ker, cQ, cK, cV, cO, cCuQ, cCuK, cFaB, cFaMb, cTask, stream)

    # run every (row_tile, head) task through the SAME compiled kernel
    for tid in range(R * H_local):
        task.copy_(torch.tensor([tid], dtype=torch.int32))
        with torch.cuda.stream(torch_stream):
            compiled(cQ, cK, cV, cO, cCuQ, cCuK, cFaB, cFaMb, cTask, stream)
    torch.cuda.synchronize()

    # reference O_scratch [R, 128, H, D]
    O_ref = fa_reference(Q, K, V, meta)                  # [tot, H, D] fp32
    Oscr_ref = o_scratch_reference(O_ref, meta)          # [R, 128, H, D] fp32

    got = Oscr.float()
    err = (got - Oscr_ref).abs().max().item()
    # per-tile tail rows (m >= valid_m) must be zero
    tail = 0.0
    for t in range(R):
        vm = meta.valid_m(t)
        if vm < 128:
            tail = max(tail, got[t, vm:].abs().max().item())
    return err, tail, R, H_local


def main():
    cases = [
        ("uniform_128",   [128],          4),
        ("varlen_200",    [200],          4),       # tiles valid_m 128, 72
        ("multi_seq",     [200, 64, 300], 4),       # several seqs, varlen tails
        ("tiny_and_big",  [40, 512, 1],   8),       # valid_m 40 (WG2 masked), big seq
    ]
    failed = 0
    for name, seqlens, H in cases:
        try:
            err, tail, R, H_local = run_case(seqlens, H, seed=hash(name) % 1000)
            ok = err < 2e-2 and tail == 0.0
            print(f"{'PASS' if ok else 'FAIL'} {name}: err={err:.4g} tail={tail:.4g} "
                  f"(R={R} H={H_local}, {R*H_local} tasks)", flush=True)
            if not ok:
                failed += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {name}: {e}", flush=True)
            traceback.print_exc()
    print(f"\n{'ALL PASS' if failed == 0 else f'{failed} FAILED'}")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(1 if main() else 0)
