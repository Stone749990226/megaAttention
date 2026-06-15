#!/usr/bin/env python3
"""
CPU unit tests for row_desc.py (Phase 0, revised two-tile model). No GPU.

    /usr/bin/python megaAttention/test_row_desc.py

Validates FA (128) + O_proj (64) descriptors, the FA-tile <-> O_proj-subtile
mapping, tail partial tiles, and the O_proj slot decode.
"""
import numpy as np

from row_desc import (
    build_fused_meta,
    cu_seqlens_from_seqlens,
    cdiv,
    oproj_task_counts,
    decode_oproj_slot,
    FA_M_TILE,
    OPROJ_M_TILE,
)


def test_cu_seqlens_basic():
    cu = cu_seqlens_from_seqlens([3, 5, 2])
    assert cu.tolist() == [0, 3, 8, 10]
    assert cu.dtype == np.int32


def test_fa_and_oproj_counts():
    # seqlens: 200 -> fa ceil(200/128)=2, oproj ceil(200/64)=4
    #          64  -> fa 1, oproj 1
    #          300 -> fa ceil(300/128)=3, oproj ceil(300/64)=5
    seqlens = [200, 64, 300]
    m = build_fused_meta(seqlens)
    assert m.num_fa_row_tiles == 2 + 1 + 3 == 6
    assert m.num_oproj_row_tiles == 4 + 1 + 5 == 10
    assert m.cu_fa_m_blocks.tolist() == [0, 2, 3, 6]
    assert m.cu_oproj_m_blocks.tolist() == [0, 4, 5, 10]


def test_fa_row_desc_roundtrip():
    seqlens = [200, 64, 300, 1, 129]
    m = build_fused_meta(seqlens)
    for t in range(m.num_fa_row_tiles):
        b = int(m.fa_batch_idx[t])
        mb = int(m.fa_m_block[t])
        assert int(m.cu_fa_m_blocks[b]) + mb == t
        assert 0 <= mb < cdiv(seqlens[b], FA_M_TILE)


def test_oproj_row_desc_roundtrip():
    seqlens = [200, 64, 300, 1, 129]
    m = build_fused_meta(seqlens)
    for t in range(m.num_oproj_row_tiles):
        b = int(m.oproj_batch_idx[t])
        mb = int(m.oproj_m_block[t])
        assert int(m.cu_oproj_m_blocks[b]) + mb == t
        assert 0 <= mb < cdiv(seqlens[b], OPROJ_M_TILE)


def test_fa_to_oproj_subtile_mapping():
    """Each O_proj subtile maps back to the right FA tile (O_scratch row) + sub_m."""
    seqlens = [200, 300]
    m = build_fused_meta(seqlens)
    for t in range(m.num_oproj_row_tiles):
        b = int(m.oproj_batch_idx[t])
        oproj_mb = int(m.oproj_m_block[t])
        fa_mb = oproj_mb // 2
        sub_m = oproj_mb % 2
        # fa_row_tile read from O_scratch
        expect_fa = int(m.cu_fa_m_blocks[b]) + fa_mb
        assert m.oproj_fa_row_tile(t) == expect_fa
        assert m.oproj_sub_m(t) == sub_m
        # the FA tile must actually exist
        assert expect_fa < m.num_fa_row_tiles


def test_oproj_valid_m_tail():
    # seq=200: oproj subtiles at m_start 0,64,128,192 -> valid 64,64,64,8
    m = build_fused_meta([200])
    valids = [m.oproj_valid_m(t) for t in range(m.num_oproj_row_tiles)]
    assert valids == [64, 64, 64, 8]
    for t in range(m.num_oproj_row_tiles):
        assert 0 < m.oproj_valid_m(t) <= OPROJ_M_TILE


def test_fa_valid_m_tail():
    # seq=200: fa tiles m_start 0,128 -> valid 128, 72
    m = build_fused_meta([200])
    assert m.fa_valid_m(0) == 128
    assert m.fa_valid_m(1) == 72
    assert m.fa_q_tile_start(1) == 128


def test_subtile_existence_matches_q_len():
    """num_oproj_row_tiles == sum of valid subtiles per FA tile (no empty subtiles built)."""
    seqlens = [129, 64, 65, 256]
    m = build_fused_meta(seqlens)
    # Every built oproj subtile has valid_m > 0
    assert all(m.oproj_valid_m(t) > 0 for t in range(m.num_oproj_row_tiles))
    # Count expected: sum ceil(q/64)
    assert m.num_oproj_row_tiles == sum(cdiv(s, 64) for s in seqlens)


def test_oproj_task_counts_and_decode():
    m = build_fused_meta([64 * 1024])  # 1024 oproj tiles
    n_out, n_sg, total = oproj_task_counts(m.num_oproj_row_tiles, 4096, 128, 4)
    assert n_out == 32 and n_sg == 8 and total == 1024 * 8
    seen = set()
    n_out2, num_sg2, total2 = oproj_task_counts(5, hidden=4100, N_TILE=128,
                                                super_group_n_tiles=4)
    for slot in range(total2):
        rt, nsg, base, valid = decode_oproj_slot(slot, num_sg2, 4, 4100, 128)
        assert slot == rt * num_sg2 + nsg and base == nsg * 4 and 1 <= valid <= 4
        seen.add((rt, nsg))
    assert len(seen) == total2 == 5 * num_sg2


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
