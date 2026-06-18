#!/usr/bin/env python3
"""
fp32 shadow reference for the fused O_proj GEMM + AllReduce.

partial_r = O_local_r @ W_o_r            # [M, N], each rank's partial sum
out       = sum_r partial_r              # [M, N], AllReduce result (every rank)

The fused SM90 kernel (megaattn_oproj_ar_sm90.py) must match `out` to within a
bf16-accumulation tolerance. We compute the reference in fp32 and reduce with
torch.distributed.all_reduce so the comparison isolates the kernel's numerics
(bf16 inputs, fp32 WGMMA accumulate, NVLS bf16 reduction) from layout/scheduling.

Shapes (DeepSeek-V3 dense MHA, bf16, W=8) -- mirrors /workspace/tp_attention.py:
    M = BATCH*SEQ      = 2*4096 = 8192       (tokens)
    K = local_heads*HEAD_DIM = 16*128 = 2048 (per-rank attention output width)
    N = HIDDEN         = 7168                 (model hidden)
"""
import torch
import torch.distributed as dist

# ---- problem shape (keep in sync with /workspace/tp_attention.py) ----
# DeepSeek-V3 attention scale: hidden 7168, 128 heads x 128 head_dim.
HIDDEN = 7168
NUM_HEADS = 128
HEAD_DIM = 128
SEQ = 4096
BATCH = 2
DTYPE = torch.bfloat16

M = BATCH * SEQ          # 8192


def shapes(world_size: int):
    """Return (M, K, N) for the O_proj GEMM given the TP world size."""
    local_heads = NUM_HEADS // world_size      # 16 for W=8
    K = local_heads * HEAD_DIM                  # 2048
    N = HIDDEN                                  # 7168
    return M, K, N


def make_inputs(rank: int, world_size: int, device, dtype=DTYPE, seed_base: int = 1234):
    """Per-rank O_local[M,K] and W_o[K,N], row-major bf16.

    W_o uses the same per-rank seed convention as tp_attention.init_block_weights
    (manual_seed(1234 + rank)); O_local is drawn from a separate stream so the two
    ranks' partials genuinely differ and the AllReduce is exercised.
    """
    m, k, n = shapes(world_size)
    gw = torch.Generator(device=device).manual_seed(seed_base + rank)
    go = torch.Generator(device=device).manual_seed(seed_base + 100 + rank)
    # match tp_attention scaling so magnitudes are realistic
    W_o = (torch.randn(k, n, device=device, dtype=dtype, generator=gw) * (k ** -0.5))
    O_local = (torch.randn(m, k, device=device, dtype=dtype, generator=go) * 0.1)
    return O_local.contiguous(), W_o.contiguous()


def reference_out(O_local: torch.Tensor, W_o: torch.Tensor, group=None) -> torch.Tensor:
    """fp32 partial @ then SUM-AllReduce across ranks -> bf16 [M,N] on every rank."""
    partial = (O_local.float() @ W_o.float())          # [M, N] fp32
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=group)
    return partial.to(O_local.dtype)


def compare(got: torch.Tensor, ref: torch.Tensor):
    """Return (max_abs, max_rel, mean_abs) of got vs ref in fp32."""
    g = got.float()
    r = ref.float()
    abs_err = (g - r).abs()
    denom = r.abs().clamp_min(1e-3)
    return (
        abs_err.max().item(),
        (abs_err / denom).max().item(),
        abs_err.mean().item(),
    )
