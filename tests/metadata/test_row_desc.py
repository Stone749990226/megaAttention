#!/usr/bin/env python3
"""
CPU unit tests for row_desc.py (Phase 0, unified 128-row model). No GPU.

    python -m pytest tests/metadata/test_row_desc.py

Validates the single 128-row row-tile descriptor (FA = O_proj = AR), tail
partial tiles, the q_len == k_len precondition, the O_proj slot decode, and the
tile-padded C_sym / O_scratch workspace sizing.
"""
import numpy as np
import pytest

from mega_attention.metadata.row_desc import (
    build_row_desc,
    cu_seqlens_from_seqlens,
    cdiv,
    oproj_task_counts,
    decode_oproj_slot,
    csym_numel,
    oscratch_numel,
    ROW_M_TILE,
    FA_M_TILE,
    OPROJ_M_TILE,
    AR_M_TILE,
)


def test_tiles_are_unified_128():
    assert ROW_M_TILE == 128
    assert FA_M_TILE == OPROJ_M_TILE == AR_M_TILE == ROW_M_TILE


def test_build_row_desc_rejects_non_unified_tile_size():
    with pytest.raises(AssertionError):
        build_row_desc([128], M_TILE=64)


def test_cu_seqlens_basic():
    cu = cu_seqlens_from_seqlens([3, 5, 2])
    assert cu.tolist() == [0, 3, 8, 10]
    assert cu.dtype == np.int32


def test_row_counts():
    # 200 -> ceil(200/128)=2, 64 -> 1, 300 -> ceil(300/128)=3
    seqlens = [200, 64, 300]
    m = build_row_desc(seqlens)
    assert m.M_TILE == 128
    assert m.num_row_tiles == 2 + 1 + 3 == 6
    assert m.cu_m_blocks.tolist() == [0, 2, 3, 6]
    assert m.num_row_tiles == sum(cdiv(s, 128) for s in seqlens)


def test_row_desc_roundtrip():
    seqlens = [200, 64, 300, 1, 129]
    m = build_row_desc(seqlens)
    for t in range(m.num_row_tiles):
        b = int(m.batch_idx[t])
        mb = int(m.m_block[t])
        assert int(m.cu_m_blocks[b]) + mb == t       # flattened id round-trips
        assert 0 <= mb < cdiv(seqlens[b], ROW_M_TILE)


def test_valid_m_tail():
    # seq=200: tiles m_start 0,128 -> valid 128, 72
    m = build_row_desc([200])
    assert m.valid_m(0) == 128
    assert m.valid_m(1) == 72
    assert m.q_tile_start(0) == 0
    assert m.q_tile_start(1) == 128
    for t in range(m.num_row_tiles):
        assert 0 < m.valid_m(t) <= ROW_M_TILE


def test_valid_m_tiny_and_exact():
    # 1 -> single tile, valid_m=1 ; 256 -> two full tiles, valid_m 128,128
    m = build_row_desc([1, 256])
    assert m.num_row_tiles == 1 + 2
    assert m.valid_m(0) == 1
    assert m.valid_m(1) == 128 and m.valid_m(2) == 128


def test_q_len_k_len_precondition():
    # equal k is fine
    m = build_row_desc([200, 64], seqlens_k=[200, 64])
    assert (m.cu_seqlens_q == m.cu_seqlens_k).all()
    # unequal k violates the complete-prompt-prefill precondition
    with pytest.raises(AssertionError):
        build_row_desc([200, 64], seqlens_k=[200, 32])


def test_oproj_task_counts():
    # T=64k tokens aligned to 128 -> 512 row tiles; hidden=4096,sg=4 -> 8 sgroups
    m = build_row_desc([512 * 128])           # 512 row tiles
    n_out, n_sg, total = oproj_task_counts(m.num_row_tiles, 4096, 128, 4)
    assert m.num_row_tiles == 512
    assert n_out == 32 and n_sg == 8 and total == 512 * 8


def test_oproj_slot_decode_bijection():
    # slot_id <-> (row_tile, nsg) is a bijection over the whole task space,
    # incl. the ragged last super group (hidden=4100 -> 33 n-tiles, sg=4).
    n_out, n_sg, total = oproj_task_counts(5, hidden=4100, N_TILE=128,
                                           super_group_n_tiles=4)
    assert n_out == cdiv(4100, 128) == 33
    seen = set()
    for slot in range(total):
        rt, nsg, base, valid = decode_oproj_slot(slot, n_sg, 4, 4100, 128)
        assert slot == rt * n_sg + nsg
        assert base == nsg * 4
        assert 1 <= valid <= 4
        # last super group is ragged: 33 = 8*4 + 1 -> final sg has 1 valid tile
        if nsg == n_sg - 1:
            assert valid == 33 - (n_sg - 1) * 4 == 1
        seen.add((rt, nsg))
    assert len(seen) == total == 5 * n_sg


def test_csym_tile_padded_capacity():
    # tail rows + ragged hidden both round UP to tile padding, not logical size.
    seqlens = [200]                            # 2 row tiles (256 padded rows)
    hidden, N_TILE = 4100, 128                 # 33 n-tiles (4224 padded cols)
    m = build_row_desc(seqlens)
    assert m.num_row_tiles == 2
    got = csym_numel(m.num_row_tiles, hidden, N_TILE)
    assert got == 2 * 128 * cdiv(hidden, N_TILE) * 128
    # strictly larger than the logical valid element count
    assert got > sum(seqlens) * hidden


def test_oscratch_capacity():
    m = build_row_desc([200, 64])              # 2 + 1 = 3 row tiles
    assert oscratch_numel(m.num_row_tiles, H_local=4, D=128) == 3 * 128 * 4 * 128


def test_csym_aligned_degenerates_to_logical():
    # When seqlens align to 128 and hidden aligns to N_TILE, padded == logical.
    m = build_row_desc([4 * 128])              # 4 row tiles, exactly 512 tokens
    hidden, N_TILE = 4096, 128
    assert csym_numel(m.num_row_tiles, hidden, N_TILE) == 512 * hidden


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
