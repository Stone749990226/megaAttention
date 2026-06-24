#!/usr/bin/env python3
"""生产场景模拟: 8 卡 fused FA + O_proj + real NVLS AR, 单个 workspace 编译一次、连续吞各种 shape。

torchrun --nproc_per_node=8 tests/fused/test_fused_production.py

模拟推理引擎的 prefill 主循环: 模型层 (H_local/hidden/q_per_kv/D) 固定, 每个 forward 收到一个
batch, 其 varlen seqlens / chunk 偏移 / batch 组成各不相同。一个 FusedFaOprojArWorkspace 在
bucket capacity 上编译 ONCE, 之后 NO host reset 连续 launch 一串差异很大的 layer:
kernel-start directed cleaner + phase/sign barrier 必须跨 layer 携带正确性、不串状态、不崩。

每个 layer 校验:
  * C_sym[valid] == Y_final = all_reduce_sum_rank(O @ W_o)  (跨 rank NVLS AR 真实求和);
  * exit-clean: scheduler 跑完后 exit cleaner 把 done 计数清 0 (ctrl[1]/[5]/[6])。

固定精选 layer 刻意编排:
  * 满 capacity (R=MAX_RT) 紧跟单 tile, 大 R -> 小 R, 长 k -> 短 k —— 压上一个大/长 layer
    的残留被 cleaner 漏清的路径;
  * 重复 L00/L01 —— 同 shape 在一串异形 layer 之后仍正确 (reuse 不退化);
  * q==k / q<k(对齐+不对齐 offset) / GQA / 单序列 vm44 tail / 多 super-group(hidden=640) 全覆盖。
末尾两个越界负例验证 workspace 干净拒绝 (set_layer assert), 且拒绝后 workspace 仍可用。
"""
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import cutlass  # noqa: F401  (ensures DSL runtime import)

from mega_attention.runtime import FusedFaOprojArWorkspace
from mega_attention.metadata.row_desc import build_row_desc
from mega_attention.reference.fused import fa_reference, oproj_reference

DT = torch.bfloat16

# 固定模型层参数 (一个 workspace 内不变), hidden=640 -> 5 out tiles -> 2 super groups (ragged)。
H_LOCAL, D, HIDDEN, Q_PER_KV = 4, 128, 640, 2
H_KV = H_LOCAL // Q_PER_KV
K_LOCAL = H_LOCAL * D
MAX_RT = 12                  # bucket capacity: num_row_tiles 上界
MAX_BATCH = 12
MAX_TOT_K = 2048             # KV-token capacity (chunk prefill k_len >> q_len)

# (tag, seqlens_q, seqlens_k, seed)。seqlens_k=None -> q==k。seed 相同的 layer 输入相同
# (重复 layer 用同 seed, 即真正"同一个 batch 再来一次")。所有 shape 满足 <= capacity。
LAYERS = [
    ("L00_qk_varlen",     [200, 64, 300],            None,            0),
    ("L01_chunk_mixed",   [200, 64, 300],            [512, 64, 460],  1),
    ("L02_single_vm44",   [300],                     None,            2),
    ("L03_full_cap",      [128] * 12,                None,            3),   # R=MAX_RT
    ("L04_tiny_after_big", [128],                    None,            4),   # 满->单 tile
    ("L05_chunk_aligned", [128],                     [384],           5),   # offset=256 对齐
    ("L06_chunk_unalign", [128],                     [316],           6),   # offset=188 不对齐
    ("L07_big_varlen",    [384, 300, 128, 64],       None,            7),   # R=8 varlen
    ("L08_small_after_big", [64],                    None,            8),   # 大->小 残留压力
    ("L09_long_k",        [256],                     [1024],          9),   # 长 KV 前缀
    ("L10_short_after_long", [128],                  None,            10),  # 长k->短k 残留压力
    ("L11_repeat_L00",    [200, 64, 300],            None,            0),   # 同 L00, 异形之后复跑
    ("L12_gqa_chunk",     [200, 300],                [328, 700],      12),  # GQA + q<k uneven
    ("L13_full_again",    [128] * 12,                None,            13),  # 满 cap 再来
    ("L14_repeat_L01",    [200, 64, 300],            [512, 64, 460],  1),   # 同 L01, 复跑
]

# 越界负例: set_layer 必须干净 assert (capacity 断言在任何 copy 之前, 不污染 workspace)。
NEG_LAYERS = [
    ("N0_over_num_row_tiles", [128] * 13, None),       # R=13 > MAX_RT=12
    ("N1_over_tot_k",         [128],      [4096]),      # tot_k=4096 > MAX_TOT_K=2048
]


def _gen_inputs(dev, rank, seed, tot, tot_k):
    g = torch.Generator(device=dev).manual_seed(1234 + rank + 1000 * seed)
    Q = torch.randn(tot, H_LOCAL, D, device=dev, dtype=DT, generator=g) * 0.2
    K = torch.randn(tot_k, H_KV, D, device=dev, dtype=DT, generator=g) * 0.2
    V = torch.randn(tot_k, H_KV, D, device=dev, dtype=DT, generator=g) * 0.2
    W_o = torch.randn(K_LOCAL, HIDDEN, device=dev, dtype=DT, generator=g) * (K_LOCAL ** -0.5)
    return Q, K, V, W_o


def main():
    lr = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(lr)
    dev = torch.device(f"cuda:{lr}")
    dist.init_process_group("nccl")
    rank, ws_size = dist.get_rank(), dist.get_world_size()
    gname = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(gname)

    ws = FusedFaOprojArWorkspace.create(
        gname, max_num_row_tiles=MAX_RT, hidden=HIDDEN, H_local=H_LOCAL, D=D,
        tp_size=ws_size, rank=rank, q_per_kv=Q_PER_KV, max_num_batch=MAX_BATCH,
        max_tot_k=MAX_TOT_K, dtype=DT, device=dev)
    ws.compile(num_ctas=8)
    num_out = ws.num_out

    def run_layer(tag, sq, sk, seed):
        meta = build_row_desc(sq, seqlens_k=sk)
        R = meta.num_row_tiles
        tot = int(sum(sq)); tot_k = int(sum(sq if sk is None else sk))
        Q, K, V, W_o = _gen_inputs(dev, rank, seed, tot, tot_k)

        ws.set_layer(meta, Q, K, V, W_o)
        dist.barrier()
        ws.launch()
        torch.cuda.synchronize(); dist.barrier()

        # full-chain reference: per-rank partial -> all_reduce -> Y_final (fp32)
        O_ref = fa_reference(Q, K, V, meta)
        Yp = oproj_reference(O_ref, W_o, meta)
        Yf = Yp.clone(); dist.all_reduce(Yf, op=dist.ReduceOp.SUM)
        C = ws.csym.float().cpu()
        err = 0.0
        for t in range(R):
            vm = meta.valid_m(t); qs = meta.q_tile_start(t)
            for o in range(num_out):
                vn = min(128, HIDDEN - o * 128)
                got = C[t, :vm, o, :vn]
                exp = Yf[qs:qs + vm, o * 128:o * 128 + vn].cpu()
                err = max(err, (got - exp).abs().max().item())
        fa_d, op_d, ar_d = int(ws.ctrl[1]), int(ws.ctrl[5]), int(ws.ctrl[6])
        ok = (err < 5e-2 and fa_d == 0 and op_d == 0 and ar_d == 0)
        print(f"[rank{rank}][{tag}] err={err:.4g} R={R}/{MAX_RT} "
              f"done(fa,op,ar)=({fa_d},{op_d},{ar_d})[->0] {'OK' if ok else 'FAIL'}", flush=True)
        dist.barrier()
        return ok

    allok = True
    for tag, sq, sk, seed in LAYERS:
        allok = run_layer(tag, sq, sk, seed) and allok

    # ---- 越界负例: set_layer 应抛 AssertionError, 且不污染 workspace ----
    for tag, sq, sk in NEG_LAYERS:
        meta = build_row_desc(sq, seqlens_k=sk)
        tot = int(sum(sq)); tot_k = int(sum(sq if sk is None else sk))
        Q, K, V, W_o = _gen_inputs(dev, rank, 99, tot, tot_k)
        raised = False
        try:
            ws.set_layer(meta, Q, K, V, W_o)
        except AssertionError:
            raised = True
        allok = allok and raised
        if rank == 0:
            print(f"[neg][{tag}] over-capacity rejected={raised} {'OK' if raised else 'FAIL'}",
                  flush=True)
    dist.barrier()

    # ---- 拒绝后 workspace 仍可用: 复跑 L00 ----
    allok = run_layer("L00_after_neg", [200, 64, 300], None, 0) and allok

    if rank == 0:
        print(f"\n{'ALL PASS' if allok else 'SOME FAILED'}", flush=True)
    dist.barrier(); dist.destroy_process_group()
    return allok


if __name__ == "__main__":
    main()
