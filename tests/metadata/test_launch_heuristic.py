import numpy as np
from mega_attention.metadata.row_desc import build_row_desc, cdiv
from mega_attention.metadata.launch_heuristic import (
    estimate_work_ratio, choose_launch_config, LaunchConfig)


def _bruteforce_ratio(meta, hidden, N_TILE=128):
    fa = 2 * sum(int(meta.m_block[t]) + 1 for t in range(meta.num_row_tiles))
    oproj = meta.num_row_tiles * cdiv(hidden, N_TILE)
    return fa / oproj


def test_ratio_exact_single_seq():
    meta = build_row_desc([2048])              # 16 tiles, Σ(m+1)=136
    r = estimate_work_ratio(meta, hidden=2048)
    assert abs(r - (2 * 136) / (16 * 16)) < 1e-9   # 272/256 = 1.0625


def test_ratio_matches_bruteforce_varlen():
    meta = build_row_desc([300, 1000, 128, 4096])
    assert abs(estimate_work_ratio(meta, hidden=3072)
               - _bruteforce_ratio(meta, 3072)) < 1e-9


def test_ratio_grows_with_seqlen():
    short = estimate_work_ratio(build_row_desc([2048]), hidden=2048)
    long_ = estimate_work_ratio(build_row_desc([16384]), hidden=2048)
    assert long_ > short


def test_choose_tp1_war_zero_and_sg_valid():
    meta = build_row_desc([4096])
    cfg = choose_launch_config(meta, hidden=4096, tp_size=1)
    assert isinstance(cfg, LaunchConfig)
    assert cfg.w_ar == 0
    assert cfg.sg in (1, 2, 4, 8)
    assert 1 <= cfg.sg <= cdiv(4096, 128)
    assert cfg.w_fa >= 1 and cfg.w_oproj >= 1


def test_choose_fa_heavy_biases_fa_and_coarsens_sg():
    bal = choose_launch_config(build_row_desc([4096]), hidden=4096, tp_size=1)   # r≈1
    fa = choose_launch_config(build_row_desc([32768]), hidden=2048, tp_size=1)   # r≈16
    assert fa.w_fa / fa.w_oproj >= bal.w_fa / bal.w_oproj
    assert fa.sg >= bal.sg


def test_choose_tp_gt1_allows_ar_weight():
    meta = build_row_desc([4096])
    cfg = choose_launch_config(meta, hidden=4096, tp_size=8)
    assert cfg.w_ar >= 1
