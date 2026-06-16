#!/usr/bin/env python3
"""
Tests for the standalone single-tile O_proj microkernel (oproj_tile_sm90.py).

Validates, against a torch fp32 A@W_o reference (设计稿 O_proj/AR), for one
(fa_row_tile_id, n_super_group) task:
  * GEMM numerics over the full K_local loop (K_CHUNK chunks > num_stages -> stage
    reuse, the path that exposed the Step-3 deadlock class),
  * WG1/WG2 64-row split (incl. valid_m tail: 72 -> WG2 partial, 40 -> WG2 fully masked),
  * valid_n tail predication (hidden not a multiple of N_TILE),
  * super_group out_n_tile loop incl. a ragged super_group (valid_n_tiles < sg_n_tiles),
  * tile-padded C_sym[fa_row_tile_id, m, out_n_tile, n] offset (other tiles untouched).

C_sym is pre-filled with a SENTINEL; the kernel must write ONLY valid (m<valid_m,
n<valid_n) elements of the targeted tiles, so masked elements + all other tiles stay
exactly the sentinel.

    /usr/bin/python megaAttention/test_oproj_tile.py
"""
import cuda.bindings.driver as cuda
import torch

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from oproj_tile_sm90 import OProjTile
from row_desc import build_row_desc, oproj_task_counts, decode_oproj_slot, cdiv
from reference_fused import o_scratch_reference

DT = torch.bfloat16
SENT = -7.0          # bf16-exact sentinel, far from any output magnitude


def _three(x):
    return x.unsqueeze(-1)


def _run_one(h_local, d, hidden, seqlens, fa_row_tile_id, n_super_group,
             super_group_n_tiles=4, N_TILE=128, K_CHUNK=64, num_stages=4,
             device="cuda:0", seed=0):
    """Compile+run OProjTile for one (row_tile, super_group); return everything
    needed to validate against the torch reference."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    k_local = h_local * d
    meta = build_row_desc(seqlens)
    num_row_tiles = meta.num_row_tiles
    num_out_n_tiles, num_super_groups, _ = oproj_task_counts(
        num_row_tiles, hidden, N_TILE, super_group_n_tiles)
    slot_id = fa_row_tile_id * num_super_groups + n_super_group
    _, _, base_out_n_tile, valid_n_tiles = decode_oproj_slot(
        slot_id, num_super_groups, super_group_n_tiles, hidden, N_TILE)
    valid_m = meta.valid_m(fa_row_tile_id)

    tot_q = int(sum(seqlens))
    O_full = (torch.randn(tot_q, h_local, d, device=dev, dtype=DT) * 0.1)
    # O_scratch packing [num_row_tiles, 128, h_local, d] (fp32 ref), then bf16 tile.
    O_scratch = o_scratch_reference(O_full, meta).to(DT)          # [T,128,h,d]
    A_tile = O_scratch[fa_row_tile_id].reshape(128, k_local).contiguous()

    W_o = (torch.randn(k_local, hidden, device=dev, dtype=DT) * (k_local ** -0.5))
    hidden_pad = num_out_n_tiles * N_TILE                        # pad N so TMA tiles cleanly
    W_o_pad = torch.zeros(k_local, hidden_pad, device=dev, dtype=DT)
    W_o_pad[:, :hidden] = W_o

    C_sym = torch.full((num_row_tiles, 128, num_out_n_tiles, N_TILE), SENT,
                       device=dev, dtype=DT)

    ker = OProjTile(128, N_TILE, k_local, hidden, num_out_n_tiles, num_row_tiles,
                    fa_row_tile_id, base_out_n_tile, valid_n_tiles, valid_m,
                    K_CHUNK=K_CHUNK, num_stages=num_stages)
    cA = from_dlpack(_three(A_tile), assumed_align=16)
    cW = from_dlpack(_three(W_o_pad), assumed_align=16)
    cC = from_dlpack(C_sym, assumed_align=16)
    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    compiled = cute.compile(ker, cA, cW, cC, stream)
    with torch.cuda.stream(torch_stream):
        compiled(cA, cW, cC, stream)
    torch.cuda.synchronize()

    # fp32 reference: full [128, hidden_pad] partial, sliced per out_n_tile.
    Y_ref = A_tile.float() @ W_o_pad.float()                     # [128, hidden_pad]
    return dict(C_sym=C_sym.cpu(), Y_ref=Y_ref.cpu(), meta=meta,
                num_out_n_tiles=num_out_n_tiles, num_row_tiles=num_row_tiles,
                base_out_n_tile=base_out_n_tile, valid_n_tiles=valid_n_tiles,
                valid_m=valid_m, fa_row_tile_id=fa_row_tile_id, hidden=hidden,
                N_TILE=N_TILE)


def _check(r, tol=2e-2):
    C = r["C_sym"]; Yref = r["Y_ref"]; ft = r["fa_row_tile_id"]
    N_TILE = r["N_TILE"]; valid_m = r["valid_m"]; hidden = r["hidden"]
    base = r["base_out_n_tile"]; vnt = r["valid_n_tiles"]
    written = set()
    for sg in range(vnt):
        out_n = base + sg
        valid_n = min(N_TILE, hidden - out_n * N_TILE)
        written.add((ft, out_n))
        got = C[ft, :valid_m, out_n, :valid_n].float()
        exp = Yref[:valid_m, out_n * N_TILE: out_n * N_TILE + valid_n]
        err = (got - exp).abs().max().item()
        assert err < tol, f"tile out_n={out_n}: max_abs_err={err}"
        # tail rows / cols of this tile must be UNWRITTEN (== sentinel)
        if valid_m < 128:
            assert bool((C[ft, valid_m:, out_n, :] == SENT).all()), \
                f"out_n={out_n}: rows>={valid_m} were written"
        if valid_n < N_TILE:
            assert bool((C[ft, :, out_n, valid_n:] == SENT).all()), \
                f"out_n={out_n}: cols>={valid_n} were written"
    # every other (row_tile, out_n_tile) must be entirely untouched (== sentinel)
    for t in range(r["num_row_tiles"]):
        for o in range(r["num_out_n_tiles"]):
            if (t, o) not in written:
                assert bool((C[t, :, o, :] == SENT).all()), \
                    f"untargeted tile (row={t}, out_n={o}) was modified"


# ----------------------------------------------------------------- cases ----
def test_oproj_three_warp_groups():
    ker = OProjTile(128, 128, 512, 512, 4, 1, 0, 0, 4, 128)
    assert ker.num_dma_threads == 128
    assert ker.num_mma_threads == 256
    assert ker.threads == 384
    assert ker.mma_atom_layout_mnk == (2, 1, 1)
    assert ker.num_mma_warps == 8
    assert ker.n_kchunks == 512 // 64 == 8     # > num_stages -> stage reuse


def test_oproj_full_single_supergroup():
    # k_local=512 -> 8 K-chunks > 4 stages (stage reuse); full 4-tile super group.
    r = _run_one(h_local=4, d=128, hidden=512, seqlens=[128],
                 fa_row_tile_id=0, n_super_group=0, seed=1)
    assert r["valid_m"] == 128 and r["valid_n_tiles"] == 4
    _check(r)


def test_oproj_tail_valid_m_72():
    # seq=200 -> row tile 1 has valid_m = 72 (WG2 rows 64..71 valid, 72..127 masked)
    r = _run_one(h_local=4, d=128, hidden=256, seqlens=[200],
                 fa_row_tile_id=1, n_super_group=0, seed=2)
    assert r["valid_m"] == 72
    _check(r)


def test_oproj_tail_valid_m_40_wg2_fully_masked():
    # valid_m=40 <= 64 -> WG2 (rows 64..127) writes nothing
    r = _run_one(h_local=4, d=128, hidden=256, seqlens=[40],
                 fa_row_tile_id=0, n_super_group=0, seed=3)
    assert r["valid_m"] == 40
    _check(r)


def test_oproj_tail_valid_n():
    # hidden=200 -> num_out_n_tiles=2, last out_n_tile valid_n = 72
    r = _run_one(h_local=4, d=128, hidden=200, seqlens=[128],
                 fa_row_tile_id=0, n_super_group=0, seed=4)
    assert r["num_out_n_tiles"] == 2 and r["valid_n_tiles"] == 2
    _check(r)


def test_oproj_csym_offset_nonzero():
    # 4 row tiles, 8 out_n_tiles, 2 super groups; target (row=2, sg=1) -> out_n 4..7.
    r = _run_one(h_local=4, d=128, hidden=1024, seqlens=[512],
                 fa_row_tile_id=2, n_super_group=1, seed=5)
    assert r["base_out_n_tile"] == 4 and r["valid_n_tiles"] == 4
    assert r["num_row_tiles"] == 4 and r["num_out_n_tiles"] == 8
    _check(r)


def test_oproj_ragged_supergroup():
    # hidden=640 -> 5 out_n_tiles; sg_n_tiles=4 -> 2 super groups; sg#1 -> 1 tile.
    r = _run_one(h_local=4, d=128, hidden=640, seqlens=[128],
                 fa_row_tile_id=0, n_super_group=1, seed=6)
    assert r["base_out_n_tile"] == 4 and r["valid_n_tiles"] == 1
    _check(r)


def test_oproj_large_klocal():
    # h_local=16 -> k_local=2048 -> 32 K-chunks (heavy stage reuse), realistic-ish.
    r = _run_one(h_local=16, d=128, hidden=256, seqlens=[128],
                 fa_row_tile_id=0, n_super_group=0, seed=7)
    assert r["valid_m"] == 128
    _check(r, tol=3e-2)


if __name__ == "__main__":
    import sys
    cases = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in cases:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {e}", flush=True)
            traceback.print_exc()
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    sys.exit(1 if failed else 0)
