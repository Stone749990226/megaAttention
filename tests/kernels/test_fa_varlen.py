#!/usr/bin/env python3
"""
Dynamic varlen FA tile test (fa_varlen.py FaWsAttnDyn).

Proves ONE compiled kernel handles tiles of different RUNTIME shape: compile once,
then run several (q_start, valid_m, k_len, nblk) cases through the SAME compiled
kernel, comparing each to a torch causal SDPA reference. Exercises:
  * single-block (nblk=1, full valid_m=128),
  * multi-block causal (nblk=4 > kv_stages -> stage reuse), diagonal in last block,
  * varlen tail valid_m=72 (WG2 partial) and valid_m=40 (WG2 fully masked),
  * runtime k_len tail (k_len not a multiple of 128).

    python -m pytest tests/kernels/test_fa_varlen.py
"""
import cuda.bindings.driver as cuda
import torch
import torch.nn.functional as F

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from mega_attention.kernels.sm90.fa_varlen import FaWsAttnDyn

DT = torch.bfloat16
DEV = "cuda:0"
M, D, LK_MAX = 128, 128, 512        # compile-time variant


def _ref_tile(Qf, Kf, Vf, q_start, valid_m, k_len):
    """torch causal SDPA on the full single-head sequence; return tile rows."""
    q = Qf[None, :q_start + valid_m].float()      # [1, q_start+vm, D]
    k = Kf[None, :k_len].float()
    v = Vf[None, :k_len].float()
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=D ** -0.5)
    return o[0, q_start:q_start + valid_m]         # [valid_m, D]


def run():
    torch.manual_seed(0)
    dev = torch.device(DEV)
    # fixed-shape device tensors (compile once, mutate in place per case)
    Q = torch.zeros(M, D, device=dev, dtype=DT)
    K = torch.zeros(LK_MAX, D, device=dev, dtype=DT)
    V = torch.zeros(LK_MAX, D, device=dev, dtype=DT)
    O = torch.zeros(M, D, device=dev, dtype=DT)
    params = torch.zeros(4, device=dev, dtype=torch.int32)

    ker = FaWsAttnDyn(M, D, D, LK_MAX)
    cQ = from_dlpack(Q.unsqueeze(-1), assumed_align=16)
    cK = from_dlpack(K.unsqueeze(-1), assumed_align=16)
    cV = from_dlpack(V.unsqueeze(-1), assumed_align=16)
    cO = from_dlpack(O.unsqueeze(-1), assumed_align=16)
    cP = from_dlpack(params, assumed_align=16)
    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    compiled = cute.compile(ker, cQ, cK, cV, cO, cP, stream)

    # (name, full_seqlen L, q_start, valid_m, k_len)  [q_len == k_len == L]
    cases = [
        ("full_m128_nblk1",   128, 0,   128, 128),
        ("multiblock_nblk4",  512, 384, 128, 512),
        ("varlen_tail_m72",   200, 128, 72,  200),
        ("wg2_masked_m40",    40,  0,   40,  40),
        ("klen_tail_nblk3",   320, 256, 64,  320),
    ]
    failed = 0
    for name, L, q_start, valid_m, k_len in cases:
        g = torch.Generator(device=dev).manual_seed(100 + L + q_start)
        Qf = torch.randn(L, D, device=dev, dtype=DT, generator=g) * 0.2
        Kf = torch.randn(L, D, device=dev, dtype=DT, generator=g) * 0.2
        Vf = torch.randn(L, D, device=dev, dtype=DT, generator=g) * 0.2
        nblk = (k_len + 127) // 128
        Q.zero_(); K.zero_(); V.zero_(); O.zero_()
        Q[:valid_m] = Qf[q_start:q_start + valid_m]
        K[:k_len] = Kf[:k_len]
        V[:k_len] = Vf[:k_len]
        params.copy_(torch.tensor([q_start, valid_m, k_len, nblk], dtype=torch.int32))
        with torch.cuda.stream(torch_stream):
            compiled(cQ, cK, cV, cO, cP, stream)
        torch.cuda.synchronize()

        O_ref = _ref_tile(Qf, Kf, Vf, q_start, valid_m, k_len)
        err = (O[:valid_m].float() - O_ref).abs().max().item()
        tail = O[valid_m:].abs().max().item() if valid_m < M else 0.0
        ok = err < 2e-2 and tail == 0.0
        print(f"{'PASS' if ok else 'FAIL'} {name}: err={err:.4g} tail={tail:.4g} "
              f"(nblk={nblk})", flush=True)
        if not ok:
            failed += 1
    return failed


if __name__ == "__main__":
    import sys
    f = run()
    print(f"\n{'ALL PASS' if f == 0 else f'{f} FAILED'}")
    sys.exit(1 if f else 0)
