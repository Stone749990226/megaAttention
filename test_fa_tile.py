#!/usr/bin/env python3
"""
Phase 2 (a/b/c) validation for the standalone single-tile FA microkernel
(fa_tile_sm90.py) vs torch. Single GPU. Covers the full per-tile correctness
surface needed for varlen causal prefill BEFORE integration into the persistent
scheduler (Phase 2d):

  * QK^T WGMMA tile               (QKTileSm90)
  * online softmax + P@V          (FaTileSm90, non-causal)
  * causal mask at several q_start (FaTileSm90, causal)
  * varlen predication            (valid_m partial Q tile, k_len partial kv block)

    /usr/bin/python megaAttention/test_fa_tile.py
"""
import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from fa_tile_sm90 import QKTileSm90, FaTileSm90


def _t3(x):
    return x.unsqueeze(-1)


def _compile_run(ker, *tensors):
    cts = [from_dlpack(_t3(t), assumed_align=16) for t in tensors]
    ts = torch.cuda.Stream()
    st = cuda.CUstream(ts.cuda_stream)
    comp = cute.compile(ker, *cts, st)
    with torch.cuda.stream(ts):
        comp(*cts, st)
    torch.cuda.synchronize()


def test_qkt():
    dev = "cuda:0"
    M, N, D = 64, 128, 128
    torch.manual_seed(0)
    Q = torch.randn(M, D, device=dev, dtype=torch.bfloat16) * 0.1
    K = torch.randn(N, D, device=dev, dtype=torch.bfloat16) * 0.1
    S = torch.zeros(M, N, device=dev, dtype=torch.float32)
    _compile_run(QKTileSm90(M, N, D), Q, K, S)
    err = (S - Q.float() @ K.float().t()).abs().max().item()
    assert err < 1e-2, f"QK^T err={err}"


def _fa_ref(Q, K, V, scale, causal, q_start, valid_m, k_len):
    Qv, Kv, Vv = Q[:valid_m].float(), K[:k_len].float(), V[:k_len].float()
    S = (Qv @ Kv.t()) * scale
    if causal:
        qpos = torch.arange(q_start, q_start + valid_m, device=Q.device)
        kpos = torch.arange(k_len, device=Q.device)
        S = S.masked_fill(~(kpos[None, :] <= qpos[:, None]), float("-inf"))
    return torch.softmax(S, dim=-1) @ Vv


def _run_fa(causal, q_start, valid_m, k_len, M=64, N=128, D=128, Lk=256, seed=0):
    dev = "cuda:0"
    torch.manual_seed(seed)
    Q = torch.randn(M, D, device=dev, dtype=torch.bfloat16) * 0.2
    K = torch.randn(Lk, D, device=dev, dtype=torch.bfloat16) * 0.2
    V = torch.randn(Lk, D, device=dev, dtype=torch.bfloat16) * 0.2
    O = torch.zeros(M, D, device=dev, dtype=torch.bfloat16)
    scale = D ** -0.5
    ref = _fa_ref(Q, K, V, scale, causal, q_start, valid_m, k_len)
    _compile_run(FaTileSm90(M, N, D, Lk, causal=causal, q_start=q_start,
                            valid_m=valid_m, k_len=k_len), Q, K, V, O)
    err = (O[:valid_m].float() - ref).abs().max().item()
    tail = O[valid_m:].abs().max().item() if valid_m < M else 0.0
    assert err < 2e-2 and tail == 0.0, f"FA err={err} tail={tail}"


def test_fa_noncausal():
    _run_fa(causal=False, q_start=0, valid_m=64, k_len=256)


def test_fa_causal():
    for qs in (0, 128, 192):
        _run_fa(causal=True, q_start=qs, valid_m=64, k_len=256)


def test_fa_varlen_predication():
    _run_fa(causal=False, q_start=0, valid_m=40, k_len=200)
    _run_fa(causal=True, q_start=160, valid_m=40, k_len=200)
    _run_fa(causal=True, q_start=0, valid_m=40, k_len=200)


if __name__ == "__main__":
    import sys
    cases = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in cases:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    sys.exit(1 if failed else 0)
