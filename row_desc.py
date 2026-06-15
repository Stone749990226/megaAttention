#!/usr/bin/env python3
"""
Host-side metadata for the fused FA + O_proj + NVLS AllReduce kernel.

Tile-size model (设计文稿.md, revised): FA and O_proj use DIFFERENT M tiles.

    FA_M_TILE    = 128   # FA tile; the two consumer warp groups each take 64 rows
    OPROJ_M_TILE = 64    # O_proj / AR tile

Each FA tile (128 Q rows) maps to up to 2 O_proj subtiles (64 rows each), indexed
by sub_m in {0, 1}. The persistent kernel recovers varlen coordinates on the hot
path with a single table read (NOT a prefix-sum search), so we build, on the host:

    cu_fa_m_blocks[b]   = sum_{i<b} ceil(seqlen_q[i] / 128)
    fa_row_desc[t]      = {batch_idx, fa_m_block}             # FA task -> coords
    cu_oproj_m_blocks[b]= sum_{i<b} ceil(seqlen_q[i] / 64)
    oproj_row_desc[t]   = {batch_idx, oproj_m_block}          # O_proj task -> coords

Mapping (FA tile -> O_proj subtile), all per 设计文稿.md:
    oproj_row_tile_id = cu_oproj_m_blocks[batch] + fa_m_block * 2 + sub_m
    oproj_m_start     = fa_m_block * 128 + sub_m * 64
    valid_m           = min(64, q_len - oproj_m_start)        # subtile exists iff > 0

Everything else (q_start/k_start/k_len) stays derivable from cu_seqlens at runtime
and is intentionally NOT cached here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FA_M_TILE = 128
OPROJ_M_TILE = 64
OPROJ_SUBTILES_PER_FA_TILE = FA_M_TILE // OPROJ_M_TILE  # 2


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def cu_seqlens_from_seqlens(seqlens: np.ndarray) -> np.ndarray:
    """[B] sequence lengths -> [B+1] exclusive prefix sum (int32), cu[0]=0."""
    seqlens = np.asarray(seqlens, dtype=np.int64)
    cu = np.zeros(seqlens.shape[0] + 1, dtype=np.int32)
    cu[1:] = np.cumsum(seqlens, dtype=np.int64).astype(np.int32)
    return cu


def _build_row_desc(seqlens_q: np.ndarray, m_tile: int):
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
class FusedMeta:
    """FA (128-row) + O_proj (64-row) schedule metadata for one varlen batch."""

    num_batch: int
    cu_seqlens_q: np.ndarray
    cu_seqlens_k: np.ndarray
    # FA tiles (128 rows)
    num_fa_row_tiles: int
    cu_fa_m_blocks: np.ndarray
    fa_batch_idx: np.ndarray          # fa_row_desc[t].batch_idx
    fa_m_block: np.ndarray            # fa_row_desc[t].fa_m_block
    # O_proj tiles (64 rows)
    num_oproj_row_tiles: int
    cu_oproj_m_blocks: np.ndarray
    oproj_batch_idx: np.ndarray       # oproj_row_desc[t].batch_idx
    oproj_m_block: np.ndarray         # oproj_row_desc[t].oproj_m_block (= fa_m_block*2+sub_m)

    # ---- FA-tile derivations (host mirror) ----
    def fa_q_len(self, t: int) -> int:
        b = int(self.fa_batch_idx[t])
        return int(self.cu_seqlens_q[b + 1] - self.cu_seqlens_q[b])

    def fa_q_tile_start(self, t: int) -> int:
        b = int(self.fa_batch_idx[t])
        return int(self.cu_seqlens_q[b]) + int(self.fa_m_block[t]) * FA_M_TILE

    def fa_valid_m(self, t: int) -> int:
        return min(FA_M_TILE, self.fa_q_len(t) - int(self.fa_m_block[t]) * FA_M_TILE)

    # ---- O_proj-subtile derivations ----
    def oproj_fa_row_tile(self, t: int) -> int:
        """Which FA tile (O_scratch row) this O_proj subtile reads from."""
        b = int(self.oproj_batch_idx[t])
        fa_m_block = int(self.oproj_m_block[t]) // OPROJ_SUBTILES_PER_FA_TILE
        return int(self.cu_fa_m_blocks[b]) + fa_m_block

    def oproj_sub_m(self, t: int) -> int:
        return int(self.oproj_m_block[t]) % OPROJ_SUBTILES_PER_FA_TILE

    def oproj_valid_m(self, t: int) -> int:
        b = int(self.oproj_batch_idx[t])
        q_len = int(self.cu_seqlens_q[b + 1] - self.cu_seqlens_q[b])
        oproj_m_start = int(self.oproj_m_block[t]) * OPROJ_M_TILE
        return min(OPROJ_M_TILE, q_len - oproj_m_start)


def build_fused_meta(seqlens_q, seqlens_k=None) -> FusedMeta:
    """Build FA (128) + O_proj (64) schedule metadata from per-sequence lengths."""
    seqlens_q = np.asarray(seqlens_q, dtype=np.int64)
    assert seqlens_q.ndim == 1 and seqlens_q.shape[0] > 0
    assert (seqlens_q >= 0).all()
    if seqlens_k is None:
        seqlens_k = seqlens_q
    seqlens_k = np.asarray(seqlens_k, dtype=np.int64)
    assert seqlens_k.shape == seqlens_q.shape

    cu_fa, fa_b, fa_mb = _build_row_desc(seqlens_q, FA_M_TILE)
    cu_op, op_b, op_mb = _build_row_desc(seqlens_q, OPROJ_M_TILE)
    return FusedMeta(
        num_batch=int(seqlens_q.shape[0]),
        cu_seqlens_q=cu_seqlens_from_seqlens(seqlens_q),
        cu_seqlens_k=cu_seqlens_from_seqlens(seqlens_k),
        num_fa_row_tiles=int(cu_fa[-1]),
        cu_fa_m_blocks=cu_fa,
        fa_batch_idx=fa_b,
        fa_m_block=fa_mb,
        num_oproj_row_tiles=int(cu_op[-1]),
        cu_oproj_m_blocks=cu_op,
        oproj_batch_idx=op_b,
        oproj_m_block=op_mb,
    )


def oproj_task_counts(num_oproj_row_tiles: int, hidden: int, N_TILE: int,
                      super_group_n_tiles: int):
    """O_proj / AR task-space sizes (设计文稿.md O_proj 规模估算).

    Returns (num_out_n_tiles, num_super_groups, total_oproj_tasks).
    """
    num_out_n_tiles = cdiv(hidden, N_TILE)
    num_super_groups = cdiv(num_out_n_tiles, super_group_n_tiles)
    total_oproj_tasks = num_oproj_row_tiles * num_super_groups
    return num_out_n_tiles, num_super_groups, total_oproj_tasks


def decode_oproj_slot(slot_id: int, num_super_groups: int,
                      super_group_n_tiles: int, hidden: int, N_TILE: int):
    """slot_id -> (oproj_row_tile_id, n_super_group, base_out_n_tile, valid_n_tiles)."""
    oproj_row_tile_id = slot_id // num_super_groups
    n_super_group = slot_id % num_super_groups
    base_out_n_tile = n_super_group * super_group_n_tiles
    num_out_n_tiles = cdiv(hidden, N_TILE)
    valid_n_tiles = min(super_group_n_tiles, num_out_n_tiles - base_out_n_tile)
    return oproj_row_tile_id, n_super_group, base_out_n_tile, valid_n_tiles
