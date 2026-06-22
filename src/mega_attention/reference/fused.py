#!/usr/bin/env python3
"""
Full-chain fp32 torch reference for the fused FA + O_proj + NVLS AllReduce kernel.

Chain (per TP rank, then AllReduce across ranks):

    O[t, h, :]   = softmax(Q[t,h] K[h]^T / sqrt(D)) V[h]      # causal varlen FA
    O_row_tile   = view O as [row_tile_id, M_TILE, H_local, D] # O_scratch layout
    Y_partial    = concat_h(O)[token, H_local*D] @ W_o         # O_proj, [tokens, hidden]
    Y_final      = sum_rank Y_partial                          # tensor-parallel AllReduce

This isolates the kernel's numerics (bf16 in, fp32 accumulate, NVLS bf16 reduce)
from layout/scheduling. We expose intermediate references so each phase has its
own gate:
    * Phase 2 (FA only)      -> `o_scratch_reference`        ([row_tile,M_TILE,H,D])
    * Phase 3 (single-rank)  -> `oproj_reference`            ([tokens, hidden])
    * Phase 4 (multi-rank)   -> `allreduce_reference`        ([tokens, hidden])

Shapes mirror reference.py (DeepSeek-V3 dense MHA, W=8): NUM_HEADS=128,
HEAD_DIM=128, HIDDEN=7168. K_local = H_local * D = (128/ws) * 128.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from mega_attention.metadata.row_desc import build_row_desc, RowDescMeta

HIDDEN = 7168
NUM_HEADS = 128
HEAD_DIM = 128
DTYPE = torch.bfloat16


def shapes(world_size: int):
    """Return (H_local, D, hidden, K_local) for one TP rank."""
    h_local = NUM_HEADS // world_size
    return h_local, HEAD_DIM, HIDDEN, h_local * HEAD_DIM


# ---------------------------------------------------------------- inputs ----
def make_qkv_inputs(rank: int, world_size: int, seqlens_q, device,
                    dtype=DTYPE, seqlens_k=None, seed_base: int = 1234,
                    q_per_kv: int = 1):
    """Per-rank varlen-packed Q/K/V and the O_proj weight.

    Returns
    -------
    Q : [tot_q, H_local, D] bf16；K, V : [tot_k, H_kv_local, D] bf16，
        H_kv_local = H_local // q_per_kv（q_per_kv == 1 即 MHA）。
    W_o : [K_local, hidden] bf16，K_local = H_local*D（O_proj K 维按 Q head 数，
          不随 q_per_kv 变；同 reference.py 的 per-rank seed 约定，确保各 rank 不同）。
    """
    seqlens_q = np.asarray(seqlens_q, dtype=np.int64)
    if seqlens_k is None:
        seqlens_k = seqlens_q
    seqlens_k = np.asarray(seqlens_k, dtype=np.int64)
    h_local, d, hidden, k_local = shapes(world_size)
    assert h_local % q_per_kv == 0, (h_local, q_per_kv)
    h_kv_local = h_local // q_per_kv
    tot_q = int(seqlens_q.sum())
    tot_k = int(seqlens_k.sum())

    gq = torch.Generator(device=device).manual_seed(seed_base + 200 + rank)
    gk = torch.Generator(device=device).manual_seed(seed_base + 300 + rank)
    gv = torch.Generator(device=device).manual_seed(seed_base + 400 + rank)
    gw = torch.Generator(device=device).manual_seed(seed_base + rank)

    Q = (torch.randn(tot_q, h_local, d, device=device, dtype=dtype, generator=gq) * 0.1)
    K = (torch.randn(tot_k, h_kv_local, d, device=device, dtype=dtype, generator=gk) * 0.1)
    V = (torch.randn(tot_k, h_kv_local, d, device=device, dtype=dtype, generator=gv) * 0.1)
    W_o = (torch.randn(k_local, hidden, device=device, dtype=dtype, generator=gw)
           * (k_local ** -0.5))
    return Q.contiguous(), K.contiguous(), V.contiguous(), W_o.contiguous()


# ----------------------------------------------------- FA (causal varlen) ----
def fa_reference(Q, K, V, meta: RowDescMeta, softmax_scale=None):
    """Causal varlen prefill attention in fp32.

    Q : [tot_q, H_local, D], K/V : [tot_k, H_kv_local, D] (varlen packed).
    支持标准 GQA：q_per_kv = H_local / H_kv_local，连续分组 kv_head = q_head // q_per_kv。
    q_per_kv == 1 即退化为 MHA。Returns O [tot_q, H_local, D].
    """
    h_local = Q.shape[1]
    h_kv = K.shape[1]
    assert h_local % h_kv == 0, (h_local, h_kv)
    q_per_kv = h_local // h_kv
    d = Q.shape[2]
    scale = softmax_scale if softmax_scale is not None else d ** -0.5
    O = torch.zeros_like(Q, dtype=torch.float32)
    for b in range(meta.num_batch):
        qs = int(meta.cu_seqlens_q[b]); qe = int(meta.cu_seqlens_q[b + 1])
        ks = int(meta.cu_seqlens_k[b]); ke = int(meta.cu_seqlens_k[b + 1])
        if qe == qs:
            continue
        # [H, Lq, D] / [H_kv, Lk, D] for batched SDPA；GQA 时把 K/V 沿 head 维展开到 H
        q = Q[qs:qe].float().transpose(0, 1)
        k = K[ks:ke].float().transpose(0, 1)
        v = V[ks:ke].float().transpose(0, 1)
        if q_per_kv > 1:
            k = k.repeat_interleave(q_per_kv, dim=0)
            v = v.repeat_interleave(q_per_kv, dim=0)
        # causal mask aligned to the bottom-right (seqlen_q may != seqlen_k).
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        O[qs:qe] = o.transpose(0, 1)
    return O


def o_scratch_reference(O, meta: RowDescMeta):
    """Pack FA output O [tot_q, H, D] into O_scratch [num_row_tiles, M_TILE, H, D].

    Invalid (tail) rows are left as zero; the kernel must not read past valid_m,
    so only m < valid_m is compared in Phase 2.
    """
    num_row_tiles = meta.num_row_tiles
    M_TILE = meta.M_TILE
    h, d = O.shape[1], O.shape[2]
    scratch = torch.zeros(num_row_tiles, M_TILE, h, d,
                          device=O.device, dtype=torch.float32)
    for t in range(num_row_tiles):
        vm = meta.valid_m(t)
        qstart = meta.q_tile_start(t)
        scratch[t, :vm] = O[qstart:qstart + vm]
    return scratch


# ----------------------------------------------------------- O_proj + AR ----
def oproj_reference(O, W_o, meta: RowDescMeta):
    """Per-rank O_proj partial: Y_partial[token, hidden] = concat_h(O) @ W_o.

    O : [tot_q, H, D] (FA output). Returns fp32 [tot_q, hidden].
    """
    tot_q, h, d = O.shape
    A = O.float().reshape(tot_q, h * d)        # [tokens, K_local]
    return A @ W_o.float()                      # [tokens, hidden]


def allreduce_reference(Y_partial, group=None):
    """Sum partials across TP ranks (mimics the NVLS AllReduce). Returns bf16."""
    out = Y_partial.float().clone()
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
    return out.to(DTYPE)


def full_chain_reference(Q, K, V, W_o, meta: RowDescMeta, group=None,
                         softmax_scale=None):
    """Convenience: FA -> O_proj -> AllReduce. Returns (O, Y_partial, Y_final)."""
    O = fa_reference(Q, K, V, meta, softmax_scale=softmax_scale)
    Y_partial = oproj_reference(O, W_o, meta)
    Y_final = allreduce_reference(Y_partial, group=group)
    return O, Y_partial, Y_final


def compare(got: torch.Tensor, ref: torch.Tensor):
    """(max_abs, max_rel, mean_abs) of got vs ref in fp32."""
    g = got.float(); r = ref.float()
    abs_err = (g - r).abs()
    denom = r.abs().clamp_min(1e-3)
    return abs_err.max().item(), (abs_err / denom).max().item(), abs_err.mean().item()


if __name__ == "__main__":
    # Smoke test on CPU/GPU: shapes line up and the chain runs end to end.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ws = 8
    seqlens = [200, 64, 130, 1]
    meta = build_row_desc(seqlens)
    Q, K, V, W_o = make_qkv_inputs(0, ws, seqlens, device=dev)
    O, Yp, Yf = full_chain_reference(Q, K, V, W_o, meta)
    h_local, d, hidden, k_local = shapes(ws)
    assert O.shape == (sum(seqlens), h_local, d), O.shape
    assert Yp.shape == (sum(seqlens), hidden), Yp.shape
    scratch = o_scratch_reference(O, meta)
    assert scratch.shape == (meta.num_row_tiles, meta.M_TILE, h_local, d), scratch.shape
    print(f"[reference_fused] OK on {dev}: tot_q={sum(seqlens)} H_local={h_local} "
          f"K_local={k_local} hidden={hidden} num_row_tiles={meta.num_row_tiles}")
    print(f"  O absmax={O.abs().max():.4f}  Y_partial absmax={Yp.abs().max():.4f}")
