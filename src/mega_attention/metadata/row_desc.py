#!/usr/bin/env python3
"""
Host-side metadata for the fused FA + O_proj + NVLS AllReduce kernel.

Tile-size model (causal_varlen_prefill_persistent_fa_oproj_ar_plan_zh.md):
FA, O_proj and AR all share ONE 128-row row tile.

    ROW_M_TILE = FA_M_TILE = OPROJ_M_TILE = AR_M_TILE = 128

The two consumer warp groups split a tile along M (WG1 rows 0..63, WG2 rows
64..127); softmax/accumulators stay within each WG's 64 rows, so no sub-tile
descriptor is needed. A single flattened `row_tile_id` indexes O_scratch,
C_sym and every control array:

    cu_m_blocks[b] = sum_{i<b} ceil(seqlen_q[i] / ROW_M_TILE)
    row_desc[t]    = {batch_idx, m_block}          # task id -> varlen coords
    num_row_tiles  = cu_m_blocks[num_batch]

O_proj / AR task identity is a single `slot_id` (no descriptor):

    slot_id        = row_tile_id * num_super_groups + n_super_group
    row_tile_id    = slot_id // num_super_groups
    n_super_group  = slot_id %  num_super_groups

Everything else (q_start/k_start/k_len/valid_m) stays derivable from cu_seqlens
on the hot path and is intentionally NOT cached. First version requires every
sequence to satisfy q_len == k_len (complete prompt prefill); build_row_desc
asserts this when seqlens_k is given.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# FA, O_proj and AR share one 128-row row tile (设计稿: Tile 尺寸约定).
ROW_M_TILE = 128
FA_M_TILE = ROW_M_TILE
OPROJ_M_TILE = ROW_M_TILE
AR_M_TILE = ROW_M_TILE


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def cu_seqlens_from_seqlens(seqlens: np.ndarray) -> np.ndarray:
    """[B] sequence lengths -> [B+1] exclusive prefix sum (int32), cu[0]=0."""
    seqlens = np.asarray(seqlens, dtype=np.int64)
    cu = np.zeros(seqlens.shape[0] + 1, dtype=np.int32)
    cu[1:] = np.cumsum(seqlens, dtype=np.int64).astype(np.int32)
    return cu


def _build_tile_desc(seqlens_q: np.ndarray, m_tile: int):
    """Return (cu_m_blocks[B+1], batch_idx[T], m_block[T]) for tile size m_tile."""
    num_batch = int(seqlens_q.shape[0])
    nblk = np.array([cdiv(int(s), m_tile) for s in seqlens_q], dtype=np.int64)
    cu = np.zeros(num_batch + 1, dtype=np.int32)
    cu[1:] = np.cumsum(nblk, dtype=np.int64).astype(np.int32)
    total = int(cu[num_batch])
    batch_idx = np.empty(total, dtype=np.int32)
    m_block = np.empty(total, dtype=np.int32)
    for b in range(num_batch):
        s = int(cu[b])
        for j in range(int(nblk[b])):
            batch_idx[s + j] = b
            m_block[s + j] = j
    return cu, batch_idx, m_block


@dataclass
class RowDescMeta:
    """Unified 128-row schedule metadata for one varlen batch.

    `t` denotes a flattened row_tile_id in [0, num_row_tiles). FA task id is
    `t * H_local + head`，其中 H_local 是 Q head 数；GQA 下 K/V 在 kernel 内按
    kv_head = head // q_per_kv 复用（H_kv_local = H_local // q_per_kv），FA task 数
    与 O_scratch 仍按 Q head 数 H_local 组织。O_proj/AR slot id 是 `t * num_super_groups + nsg`。
    """

    num_batch: int
    M_TILE: int
    cu_seqlens_q: np.ndarray
    cu_seqlens_k: np.ndarray
    num_row_tiles: int
    cu_m_blocks: np.ndarray           # [B+1] prefix of ceil(seqlen_q / M_TILE)
    batch_idx: np.ndarray             # row_desc[t].batch_idx
    m_block: np.ndarray               # row_desc[t].m_block (tile index within seq)

    # ---- per-tile derivations (host mirror of the kernel's hot-path math) ----
    def q_len(self, t: int) -> int:
        b = int(self.batch_idx[t])
        return int(self.cu_seqlens_q[b + 1] - self.cu_seqlens_q[b])

    def k_len(self, t: int) -> int:
        b = int(self.batch_idx[t])
        return int(self.cu_seqlens_k[b + 1] - self.cu_seqlens_k[b])

    def q_tile_start(self, t: int) -> int:
        b = int(self.batch_idx[t])
        return int(self.cu_seqlens_q[b]) + int(self.m_block[t]) * self.M_TILE

    def valid_m(self, t: int) -> int:
        return min(self.M_TILE, self.q_len(t) - int(self.m_block[t]) * self.M_TILE)


def build_row_desc(seqlens_q, M_TILE: int = ROW_M_TILE, seqlens_k=None) -> RowDescMeta:
    """Build unified 128-row schedule metadata from per-sequence lengths.

    contiguous-KV prefill: 前置条件 `0 < q_len <= k_len`。`q_len == k_len` 是完整
    prompt prefill 的退化情形；`q_len < k_len` 表示当前 Q chunk 关注同一 sequence 的
    完整连续 KV 前缀（bottom-right aligned causal, offset = k_len - q_len）。row tile
    调度与 O_scratch/C_sym 容量只按 Q token 计量，与 k_len 无关。当未提供 `seqlens_k`
    时默认 k == q（完整 prompt prefill）。
    """
    assert int(M_TILE) == ROW_M_TILE, (
        "first version requires unified 128-row tiles "
        "(FA_M_TILE == OPROJ_M_TILE == AR_M_TILE == 128)")
    seqlens_q = np.asarray(seqlens_q, dtype=np.int64)
    assert seqlens_q.ndim == 1 and seqlens_q.shape[0] > 0
    assert (seqlens_q > 0).all(), "contiguous-KV prefill requires q_len > 0"
    if seqlens_k is None:
        seqlens_k = seqlens_q
    seqlens_k = np.asarray(seqlens_k, dtype=np.int64)
    assert seqlens_k.shape == seqlens_q.shape
    assert (seqlens_k >= seqlens_q).all(), (
        "contiguous-KV prefill requires 0 < q_len <= k_len "
        "(bottom-right aligned causal); q_len > k_len is not supported")

    cu, b, mb = _build_tile_desc(seqlens_q, M_TILE)
    return RowDescMeta(
        num_batch=int(seqlens_q.shape[0]),
        M_TILE=int(M_TILE),
        cu_seqlens_q=cu_seqlens_from_seqlens(seqlens_q),
        cu_seqlens_k=cu_seqlens_from_seqlens(seqlens_k),
        num_row_tiles=int(cu[-1]),
        cu_m_blocks=cu,
        batch_idx=b,
        m_block=mb,
    )


def oproj_task_counts(num_row_tiles: int, hidden: int, N_TILE: int,
                      super_group_n_tiles: int):
    """O_proj / AR task-space sizes (设计稿: O_proj 规模估算).

    Returns (num_out_n_tiles, num_super_groups, total_oproj_tasks).
    """
    num_out_n_tiles = cdiv(hidden, N_TILE)
    num_super_groups = cdiv(num_out_n_tiles, super_group_n_tiles)
    total_oproj_tasks = num_row_tiles * num_super_groups
    return num_out_n_tiles, num_super_groups, total_oproj_tasks


def decode_oproj_slot(slot_id: int, num_super_groups: int,
                      super_group_n_tiles: int, hidden: int, N_TILE: int):
    """slot_id -> (row_tile_id, n_super_group, base_out_n_tile, valid_n_tiles)."""
    row_tile_id = slot_id // num_super_groups
    n_super_group = slot_id % num_super_groups
    base_out_n_tile = n_super_group * super_group_n_tiles
    num_out_n_tiles = cdiv(hidden, N_TILE)
    valid_n_tiles = min(super_group_n_tiles, num_out_n_tiles - base_out_n_tile)
    return row_tile_id, n_super_group, base_out_n_tile, valid_n_tiles


# --------------------------------------------- tile-padded workspace sizing ---
def oscratch_numel(num_row_tiles: int, H_local: int, D: int,
                   M_TILE: int = ROW_M_TILE) -> int:
    """O_scratch_local element count: [num_row_tiles, M_TILE, H_local, D]."""
    return num_row_tiles * M_TILE * H_local * D


def csym_numel(num_row_tiles: int, hidden: int, N_TILE: int,
               M_TILE: int = ROW_M_TILE) -> int:
    """C_sym in-place partial/final element count, tile-padded:

        [num_row_tiles, M_TILE, num_out_n_tiles, N_TILE]

    Sized by physical tile padding (NOT logical T*hidden): tail rows
    (valid_m < M_TILE) and tail hidden (valid_n < N_TILE) still occupy address
    space; stores/reduces mask them with valid_m / valid_n predicates.
    """
    num_out_n_tiles = cdiv(hidden, N_TILE)
    return num_row_tiles * M_TILE * num_out_n_tiles * N_TILE
