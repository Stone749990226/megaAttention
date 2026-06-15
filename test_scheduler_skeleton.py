#!/usr/bin/env python3
"""
Phase 1 known-answer tests for the multi-mode scheduler skeleton (single GPU).

Validates the FA -> O_proj -> AR handoff PROTOCOL with stub payloads -- no
floating-point math. Asserts:
  * every FA task / O_proj task / AR owner task executes EXACTLY once,
  * O_proj only runs after all H_local heads of its row are present (order_err==0),
  * the ordered ready queue never exposes a hole (covered by exactly-once O_proj),
  * the kernel terminates (grid_sync init+exit) for several task-space shapes.

    /usr/bin/python megaAttention/test_scheduler_skeleton.py
    /usr/bin/python -m pytest megaAttention/test_scheduler_skeleton.py -x
"""
import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from row_desc import build_row_desc, oproj_task_counts
from fused_fa_oproj_ar_sm90 import FusedFaOprojArSkeleton, NUM_CTRL


def _u32(n, dev):
    return torch.zeros(n, dtype=torch.uint32, device=dev)


def run_skeleton(seqlens, M_TILE, H_local, hidden, N_TILE, super_group_n_tiles,
                 num_ctas=None, threads_per_cta=128, device="cuda:0"):
    """Compile+launch the skeleton for one task-space shape; return result dict."""
    torch.cuda.set_device(device)
    dev = torch.device(device)
    if num_ctas is None:
        num_ctas = torch.cuda.get_device_properties(dev).multi_processor_count

    meta = build_row_desc(seqlens, M_TILE)
    num_row_tiles = meta.num_row_tiles
    num_fa = num_row_tiles * H_local
    _, num_super_groups, total_oproj = oproj_task_counts(
        num_row_tiles, hidden, N_TILE, super_group_n_tiles)

    # ---- workspace (all per-kernel control state zeroed) ----
    ctrl = _u32(NUM_CTRL, dev)
    head_ready = _u32(num_row_tiles, dev)
    oproj_queue = _u32(total_oproj, dev)
    ready_count_owner = _u32(total_oproj, dev)
    ar_probe = _u32(total_oproj, dev)
    ar_done_flag = _u32(total_oproj, dev)
    head_marker = _u32(num_fa, dev)
    fa_exec = _u32(num_fa, dev)
    oproj_exec = _u32(total_oproj, dev)
    ar_exec = _u32(total_oproj, dev)
    partial_check = _u32(total_oproj, dev)

    tensors = [ctrl, head_ready, oproj_queue, ready_count_owner, ar_probe,
               ar_done_flag, head_marker, fa_exec, oproj_exec, ar_exec,
               partial_check]
    cts = [from_dlpack(t, assumed_align=4) for t in tensors]

    kernel = FusedFaOprojArSkeleton(
        num_fa=num_fa, num_row_tiles=num_row_tiles, H_local=H_local,
        num_super_groups=num_super_groups, total_oproj=total_oproj,
        num_ctas=num_ctas, tp_size=1, threads_per_cta=threads_per_cta)

    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    compiled = cute.compile(kernel, *cts, stream)
    with torch.cuda.stream(torch_stream):
        compiled(*cts, stream)
    torch.cuda.synchronize()

    return dict(
        num_fa=num_fa, num_row_tiles=num_row_tiles, total_oproj=total_oproj,
        num_super_groups=num_super_groups,
        fa_exec=fa_exec.cpu(), oproj_exec=oproj_exec.cpu(), ar_exec=ar_exec.cpu(),
        order_err=int(ctrl[8].item()),
        fa_done=int(ctrl[1].item()), oproj_done=int(ctrl[5].item()),
        ar_done=int(ctrl[6].item()), ar_done_flag=ar_done_flag.cpu(),
        partial_check=partial_check.cpu(),
    )


def _assert_shape(r):
    # exactly-once coverage
    assert bool((r["fa_exec"] == 1).all()), \
        f"FA coverage: min={int(r['fa_exec'].min())} max={int(r['fa_exec'].max())}"
    assert bool((r["oproj_exec"] == 1).all()), \
        f"O_proj coverage: min={int(r['oproj_exec'].min())} max={int(r['oproj_exec'].max())}"
    assert bool((r["ar_exec"] == 1).all()), \
        f"AR coverage: min={int(r['ar_exec'].min())} max={int(r['ar_exec'].max())}"
    # ordering: O_proj never saw a missing/incorrect head marker
    assert r["order_err"] == 0, f"order_err={r['order_err']} (head_ready violated)"
    # done counters reached targets (=> kernel terminated correctly)
    assert r["fa_done"] == r["num_fa"], (r["fa_done"], r["num_fa"])
    assert r["oproj_done"] == r["total_oproj"], (r["oproj_done"], r["total_oproj"])
    assert r["ar_done"] == r["total_oproj"], (r["ar_done"], r["total_oproj"])
    # AR terminal protection: every slot marked done exactly once
    assert bool((r["ar_done_flag"] == 1).all())
    # partials were all written (nonzero checksum)
    assert bool((r["partial_check"] != 0).all())


# ---------------------------------------------------------------- cases ----
def test_small_uniform():
    r = run_skeleton(seqlens=[256, 192, 320, 128], M_TILE=64, H_local=4,
                     hidden=1024, N_TILE=128, super_group_n_tiles=4)
    _assert_shape(r)


def test_tail_partial_tiles():
    # non-divisible seqlens -> tail partial row tiles exercised in row_desc
    r = run_skeleton(seqlens=[130, 65, 257, 1, 200], M_TILE=64, H_local=8,
                     hidden=2048, N_TILE=128, super_group_n_tiles=4)
    _assert_shape(r)


def test_super_group_8():
    r = run_skeleton(seqlens=[512, 512, 512], M_TILE=64, H_local=8,
                     hidden=4096, N_TILE=128, super_group_n_tiles=8)
    _assert_shape(r)


def test_more_contention():
    # ~64 row tiles x 8 heads = 512 FA tasks across all SMs
    r = run_skeleton(seqlens=[64 * 64], M_TILE=64, H_local=8,
                     hidden=2048, N_TILE=128, super_group_n_tiles=4)
    _assert_shape(r)


def test_single_row_tile():
    r = run_skeleton(seqlens=[40], M_TILE=64, H_local=4,
                     hidden=1024, N_TILE=128, super_group_n_tiles=4)
    _assert_shape(r)


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
