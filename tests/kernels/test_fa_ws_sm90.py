#!/usr/bin/env python3
"""
Focused tests for the in-flight FA warp-specialized SM90 kernels.

Step 3 of the execution plan continues fa_ws.py, not the deleted
fa_tile_sm90.py. These tests pin the first required surface for the 128-row
tile path before adding causal/varlen FA math.

    python -m pytest tests/kernels/test_fa_ws.py
"""
import cuda.bindings.driver as cuda
import torch

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from mega_attention.kernels.sm90.fa_ws import FaWsAttnKV
from mega_attention.reference.fused import build_row_desc, fa_reference


def test_attn_kv_m128_uses_three_warp_groups():
    ker = FaWsAttnKV(128, 128, 128, 128, causal=True)
    assert ker.num_dma_threads == 128
    assert ker.num_mma_threads == 256
    assert ker.threads == 384
    assert ker.mma_atom_layout_mnk == (2, 1, 1)


def _t3(x):
    return x.unsqueeze(-1)


def _compile_run(ker, *tensors):
    cts = [from_dlpack(_t3(t), assumed_align=16) for t in tensors]
    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    compiled = cute.compile(ker, *cts, stream)
    with torch.cuda.stream(torch_stream):
        compiled(*cts, stream)
    torch.cuda.synchronize()


def test_attn_kv_noncausal_m64_matches_torch():
    dev = "cuda:0"
    M, N, D, Lk = 64, 128, 128, 256
    torch.manual_seed(1)
    Q = torch.randn(M, D, device=dev, dtype=torch.bfloat16) * 0.2
    K = torch.randn(Lk, D, device=dev, dtype=torch.bfloat16) * 0.2
    V = torch.randn(Lk, D, device=dev, dtype=torch.bfloat16) * 0.2
    O = torch.zeros(M, D, device=dev, dtype=torch.bfloat16)

    _compile_run(FaWsAttnKV(M, N, D, Lk), Q, K, V, O)

    ref = torch.softmax((Q.float() @ K.float().t()) * (D ** -0.5), dim=-1) @ V.float()
    err = (O.float() - ref).abs().max().item()
    assert err < 2e-2, f"FA noncausal M64 err={err}"


def test_attn_kv_causal_full_m128_matches_reference():
    dev = "cuda:0"
    M, N, D, Lk = 128, 128, 128, 128
    seqlens = [128]
    meta = build_row_desc(seqlens)
    torch.manual_seed(3)
    Q = torch.randn(M, D, device=dev, dtype=torch.bfloat16) * 0.2
    K = torch.randn(Lk, D, device=dev, dtype=torch.bfloat16) * 0.2
    V = torch.randn(Lk, D, device=dev, dtype=torch.bfloat16) * 0.2
    O = torch.zeros(M, D, device=dev, dtype=torch.bfloat16)

    _compile_run(FaWsAttnKV(M, N, D, Lk, causal=True), Q, K, V, O)

    O_ref = fa_reference(Q[:, None, :], K[:, None, :], V[:, None, :], meta)[:, 0, :]
    err = (O.float() - O_ref).abs().max().item()
    assert err < 2e-2, f"FA causal full M128 err={err}"


def test_attn_kv_causal_varlen_tail_m128_matches_reference_slice():
    dev = "cuda:0"
    M, N, D, Lk = 128, 128, 128, 256
    seqlens = [200]
    meta = build_row_desc(seqlens)
    torch.manual_seed(2)
    Q_full = torch.randn(seqlens[0], D, device=dev, dtype=torch.bfloat16) * 0.2
    K_full = torch.randn(seqlens[0], D, device=dev, dtype=torch.bfloat16) * 0.2
    V_full = torch.randn(seqlens[0], D, device=dev, dtype=torch.bfloat16) * 0.2
    Q = torch.zeros(M, D, device=dev, dtype=torch.bfloat16)
    K = torch.zeros(Lk, D, device=dev, dtype=torch.bfloat16)
    V = torch.zeros(Lk, D, device=dev, dtype=torch.bfloat16)
    Q[:72] = Q_full[128:200]
    K[:200] = K_full
    V[:200] = V_full
    O = torch.zeros(M, D, device=dev, dtype=torch.bfloat16)

    _compile_run(FaWsAttnKV(M, N, D, Lk, causal=True, q_start=128,
                            valid_m=72, k_len=200), Q, K, V, O)

    meta_ref = build_row_desc(seqlens)
    O_ref = fa_reference(Q_full[:, None, :], K_full[:, None, :], V_full[:, None, :], meta_ref)[:, 0, :]
    err = (O[:72].float() - O_ref[128:200]).abs().max().item()
    tail = O[72:].abs().max().item()
    assert err < 2e-2, f"FA causal/varlen tail err={err}"
    assert tail == 0.0, f"tail rows not masked: tail={tail}"


def test_attn_kv_causal_wg2_fully_masked_m128():
    """valid_m <= 64 => WG2 (rows 64..127) is fully store-masked (设计稿: 最后一个
    row tile 如果 valid_m <= 64, WG2 的 store 全部被谓词屏蔽). Rows 0..39 match the
    reference, rows 40..127 must be exactly zero."""
    dev = "cuda:0"
    M, N, D, Lk = 128, 128, 128, 128
    seqlens = [40]
    meta = build_row_desc(seqlens)
    torch.manual_seed(5)
    Q_full = torch.randn(40, D, device=dev, dtype=torch.bfloat16) * 0.2
    K_full = torch.randn(40, D, device=dev, dtype=torch.bfloat16) * 0.2
    V_full = torch.randn(40, D, device=dev, dtype=torch.bfloat16) * 0.2
    Q = torch.zeros(M, D, device=dev, dtype=torch.bfloat16)
    K = torch.zeros(Lk, D, device=dev, dtype=torch.bfloat16)
    V = torch.zeros(Lk, D, device=dev, dtype=torch.bfloat16)
    Q[:40] = Q_full
    K[:40] = K_full
    V[:40] = V_full
    O = torch.zeros(M, D, device=dev, dtype=torch.bfloat16)

    _compile_run(FaWsAttnKV(M, N, D, Lk, causal=True, q_start=0,
                            valid_m=40, k_len=40), Q, K, V, O)

    O_ref = fa_reference(Q_full[:, None, :], K_full[:, None, :], V_full[:, None, :],
                         meta)[:, 0, :]
    err = (O[:40].float() - O_ref).abs().max().item()
    tail = O[40:].abs().max().item()
    assert err < 2e-2, f"FA causal valid_m<=64 err={err}"
    assert tail == 0.0, f"WG2 not fully masked: tail={tail}"


def test_attn_kv_causal_multiblock_last_tile_m128():
    """Last 128-row tile of a 512-len causal sequence: 4 KV blocks exercise the
    online-softmax cross-block correction repeatedly at M=128 (WG1 rows 0..63,
    WG2 rows 64..127), with the causal diagonal landing in the final KV block."""
    dev = "cuda:0"
    M, N, D, Lk = 128, 128, 128, 512
    L = 512
    seqlens = [L]
    meta = build_row_desc(seqlens)
    torch.manual_seed(7)
    Q_full = torch.randn(L, D, device=dev, dtype=torch.bfloat16) * 0.2
    K_full = torch.randn(L, D, device=dev, dtype=torch.bfloat16) * 0.2
    V_full = torch.randn(L, D, device=dev, dtype=torch.bfloat16) * 0.2
    # tile covering q rows 384..511 (the 4th/last FA tile); keys 0..511.
    q_start = 384
    Q = torch.zeros(M, D, device=dev, dtype=torch.bfloat16)
    Q[:128] = Q_full[q_start:q_start + 128]
    K = K_full.clone()
    V = V_full.clone()
    O = torch.zeros(M, D, device=dev, dtype=torch.bfloat16)

    _compile_run(FaWsAttnKV(M, N, D, Lk, causal=True, q_start=q_start,
                            valid_m=128, k_len=L), Q, K, V, O)

    O_ref = fa_reference(Q_full[:, None, :], K_full[:, None, :], V_full[:, None, :],
                         meta)[:, 0, :]
    err = (O.float() - O_ref[q_start:q_start + 128]).abs().max().item()
    assert err < 2e-2, f"FA causal multiblock last-tile M128 err={err}"


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
