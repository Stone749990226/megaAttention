#!/usr/bin/env python3
# benchmarks/sweep_launch_config.py
"""8×H200 sweep: 12 shape × (配比, sg) 网格, 找每组最优 -> 标定粗桶表."""
import os
import sys

# Ensure the project root is on sys.path so "benchmarks" and "mega_attention" are importable
# regardless of the working directory torchrun uses.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# Also ensure src/ is on sys.path for mega_attention package.
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import torch
import torch.distributed as dist
from torch.distributed._symmetric_memory import enable_symm_mem_for_group

from mega_attention.metadata.row_desc import build_row_desc
from mega_attention.metadata.launch_heuristic import estimate_work_ratio
from benchmarks.bench_fused_fa_oproj_ar import bench_one

SHAPES = [
    ("2048,2048", 8, 2048), ("1024,1024,1024,1024", 16, 2048),
    ("4096,4096", 8, 4096), ("8192", 8, 2048),
    ("2048,2048,2048,2048,2048,2048,2048,2048", 8, 2048),
    ("8192,8192", 8, 2048), ("8192,8192", 8, 4096), ("8192,8192", 16, 7168),
    ("16384", 8, 2048), ("16384", 16, 4096),
    ("16384,16384", 8, 2048), ("32768", 8, 2048),
]
GRID = [  # (w_fa, w_oproj, w_ar, sg); w_ar 在 tp>1 生效
    (4, 1, 1, 2), (4, 1, 1, 4), (4, 1, 1, 8),
    (2, 1, 1, 4), (8, 1, 1, 4), (8, 1, 1, 8), (5, 1, 1, 8),
]


def main():
    lr = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
    dev = torch.device(f"cuda:{lr}")
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    enable_symm_mem_for_group(dist.group.WORLD.group_name)

    lines = ["| shape | r | best (w_fa,w_oproj,w_ar,sg) | best ratio |",
             "| --- | --- | --- | --- |"]
    for seqstr, h_local, hidden in SHAPES:
        seqlens = [int(x) for x in seqstr.split(",")]
        meta = build_row_desc(seqlens)
        r = estimate_work_ratio(meta, hidden)
        best = None
        for (wf, wo, wa, sg) in GRID:
            torch.cuda.empty_cache()
            res = bench_one(seqlens, h_local, hidden, wf, wo, wa, sg,
                            ws, rank, dev, iters=30, warmup=10)
            if rank == 0:
                print(f"  {seqstr} h{h_local} hid{hidden} r={r:.2f} "
                      f"w=({wf},{wo},{wa}) sg={sg} ratio={res['ratio']:.3f}", flush=True)
            if best is None or res["ratio"] > best[1]:
                best = ((wf, wo, wa, sg), res["ratio"])
        if rank == 0:
            lines.append(f"| {seqstr} h{h_local} hid{hidden} | {r:.2f} | "
                         f"{best[0]} | {best[1]:.3f}x |")
    if rank == 0:
        out = "\n".join(lines)
        print(out, flush=True)
        with open("benchmarks/sweep_results.md", "w") as f:
            f.write(out + "\n")


if __name__ == "__main__":
    main()
