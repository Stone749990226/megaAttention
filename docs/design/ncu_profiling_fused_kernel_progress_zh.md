# Fused FA + O_proj + NVLS-AR kernel 的 ncu profiling —— 进展与交接

> 状态：spike 阶段完成（已确定可行路径与一个关键阻塞问题）。下一步是把验证过的多卡路径
> 固化成 `scripts/ncu_fused_fa_oproj_ar.sh`，并解决收尾时的显存泄漏。
> 本文件是给"接手 agent"的自包含说明，不依赖任何会话上下文。

## 0. 目标

对核心 kernel `src/mega_attention/kernels/sm90/fused_fa_oproj_ar.py::FusedFaOprojAr`
（FA + O_proj + NVLS AllReduce 串在一个 persistent kernel 内）做 Nsight Compute profiling。
现有 `scripts/ncu_oproj_ar.sh` / `scripts/prof_driver.py` 只覆盖 **standalone O_proj+AR**
（`OProjARFusedKernelSM90`），**没有针对完整 fused kernel 的 ncu 脚本**——本任务就是补上它。

约束（来自用户，务必遵守）：
1. 用**专门的 minimal driver**做 ncu，**不要**用重的 `benchmarks/bench_fused_fa_oproj_ar.py`
   （它带 baseline + 正确性校验 + 计时循环，每个 application-replay pass 会把这些全重跑一遍）。
2. section 集用**纯数字编号 1,2,3,4…**，不要 2a/2b/2c。
3. section **尽量细分**（每个数字集只含一个维度），减少单次调用的 replay pass 数。
4. **不要照搬** `scripts/ncu_oproj_ar.sh` 的具体取值，更不要沿用它"~38 pass 墙"这种无依据说法。
   每个 ncu flag 要有依据（官方文档或本机实测）。
5. **只 profile 真实 `FusedFaOprojAr`**，不要 `FusedFaOprojArSkeleton`。

## 1. 环境备忘（接手机器可能不同，先自检）

- ncu：`/usr/local/NVIDIA-Nsight-Compute/ncu`，版本 **2026.2.0.0**。
  注意：**这个版本没有 `--list-kernels`**（会报 `unrecognised option`）。要确认 symbol 就直接
  profile 一次看 `==PROF== Profiling "..."` 行，或 import 后看 details 页。
- python 解释器是 **`python3`**（没有 `python`）。torch `2.11.0+cu130`，8×H200，`mega_attention` 可直接 import。
- torch 基于 CUDA 13 构建，机器驱动 570.172.08（CUDA 12.8 级）。跑 GPU 前先设 forward-compat：
  `export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH`（否则可能 "driver too old"）。

## 2. 已完成：minimal driver `scripts/prof_fused_driver.py`（已落盘，已验证）

仿 `scripts/prof_driver.py`，复用 `bench_fused_fa_oproj_ar.py::bench_one` 的 buffer 构建 +
`cute.compile` + `reset_fused`/`run_fused`，**去掉** baseline / 正确性校验 / 计时循环。流程：

```
init dist + symm_mem → 建 buffers → cute.compile(FusedFaOprojAr(...))
prime(1 launch) → warmup(W launches, 被 ncu --launch-skip 跳过)
→ nvtx.range_push("prof"); prof(1 launch); nvtx.range_pop() → destroy_process_group
```

要点（接手后改 driver 时不要破坏这些不变量）：
- **launch 顺序固定 = prime(1) + warmup(W) + prof(1)**。所以 ncu 用
  **`--launch-skip $((1+W)) --launch-count 1`** 精确命中 prof launch（W 由 `--warmup` 控制，默认 4）。
- driver **只 import / 实例化 `FusedFaOprojAr`，从不构建 skeleton** → 编译产物里只有真实 kernel，
  `-k regex:FusedFaOprojAr` 不可能误匹配 skeleton。**这就是满足约束 5 的方式**，不要再额外费力 anchor regex。
- 单卡（`WORLD_SIZE=1`）时 `tp_size==1`，kernel 内 `nvl_barrier` / multimem AR 路径被跳过，
  AR 退化为 identity；symm_mem 会打印 "Gracefully skipping multicast initialization"（非致命，正常）。
- 参数：`--seqlens/--hidden/--h_local/--w_fa/--w_oproj/--w_ar/--sg/--auto`（同 bench）+ `--warmup`。

### 已确认的 kernel symbol
profile 时实际符号形如：
```
kernel_cutlass_kernel_mega_attentionkernelssm90fused_fa_oproj_arFusedFaOprojAr_object_at__tensorptr...
```
→ **`-k regex:FusedFaOprojAr` 验证可用**。

## 3. spike 关键结论（本机 8×H200 实测）

### 3.1 单卡（kernel replay）——✅ 可用，最简单
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
ncu --target-processes all -k regex:FusedFaOprojAr --launch-skip 1 --launch-count 1 \
    --section SpeedOfLight -f -o /myworkspace/log/spike_fused_1gpu \
    torchrun --nnodes=1 --nproc_per_node=1 --master_port=29613 scripts/prof_fused_driver.py --warmup 0
```
- 成功。SpeedOfLight = **10 passes**（kernel replay，单卡无 NCCL，可随便加 section 甚至 `--set full`）。
- 默认 workload（seqlens 2048,2048, hidden 2048, H_local 8）单卡基线：
  **Duration 144.90 µs，Compute(SM) 25.40%，Memory 42.47%，DRAM 6.53%，L1 32.47%，L2 42.47%**。
- 这是看 FA + O_proj 计算热点最快的路径（AR=identity，不含真实通信）。

### 3.2 多卡 shmem + kernel replay（官方文档对"同进程树并发通信 kernel"的推荐）——❌ 不可用
依据 `ncu-docs/NsightComputeCli.md` §4.3.2：单节点、一个 ncu 包 torchrun（同进程树）应该用
`--communicator shmem --communicator-shmem-num-peers N --lockstep-kernel-launch`，但**它只支持
kernel/range replay**。实测命令：
```bash
ncu --communicator shmem --communicator-shmem-num-peers 8 --lockstep-kernel-launch \
    -k regex:FusedFaOprojAr --launch-skip 3 --launch-count 1 --section SpeedOfLight -f -o ... \
    torchrun --nnodes=1 --nproc_per_node=8 --master_port=29614 scripts/prof_fused_driver.py --warmup 2
```
**失败**，报错链：
```
==WARNING== Backing up device memory in system memory. Kernel replay might be slow...
==ERROR== Failed to save memory for replay. ... consider using --replay-mode application ...
cuda_context_state>> Failed to copy memory / Failed to save context state! → ContextSaveFailed
```
**根因**：kernel replay 要逐 pass save/restore CUDA context（含对称内存/multicast 映射），而
**跨卡 symmetric memory + NVLS multicast 的 context 无法被 save/restore**。这正是 collective kernel
不能用 kernel replay 的根本原因。**shmem 路径对本 kernel 是死路，放弃。**

### 3.3 多卡 tcp + per-rank ncu + application replay——✅ 可用（唯一可行的多卡路径）
8 个独立 ncu 实例（每 rank 一个，**手动设 dist 环境变量、用 `python3` 而非 torchrun 启动**），
靠 `--communicator tcp --communicator-tcp-num-peers 8 --lockstep-kernel-launch` 保持同步，
`--replay-mode application`（application replay 不 save/restore 显存，每 pass 重跑整个 app，
collective 在每 pass 都完整跑完，不会因 ncu 暂停单 rank 而死锁）。实测命令骨架：
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:$LD_LIBRARY_PATH
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29615 WORLD_SIZE=8 NCCL_NVLS_ENABLE=1 NCCL_ALGO=NVLS
for i in $(seq 0 7); do
  RANK=$i LOCAL_RANK=$i ncu \
    --replay-mode application --app-replay-buffer memory \
    --communicator tcp --communicator-tcp-num-peers 8 --lockstep-kernel-launch \
    -k regex:FusedFaOprojAr --launch-skip 3 --launch-count 1 \
    --section SpeedOfLight --clock-control none --kill yes -f -o "$OUT/rank$i" \
    python3 scripts/prof_fused_driver.py --warmup 2 \
    >"$OUT/ncu.rank$i.out" 2>&1 &
done
```
- **结果：8 个 `rank*.ncu-rep` 全部产出（各 ~596KB），约 180s 完成（SpeedOfLight = 10 replay passes）。**
  profiling 数据有效。→ 多卡走 **tcp + application replay** 这条路。
- 进度可从 `ncu.rank$i.out` 里 `replay pass` 行数读出（做 heartbeat / watchdog 用）。
- `--clock-control none`：本次实测用 none 正常完成（出现 "unmodified GPU clocks" 警告，可接受）。
  旧脚本声称锁频会破坏 lockstep，**此点未独立验证**，先沿用 none，接手可再实验 base。

## 4. ⚠️ 头号待解决问题：收尾时进程不退出 + 显存泄漏

3.3 里 8 个 report 在 ~180s（pass 10）就已全部写出，但**之后 8 个 ncu 进程不退出**，挂了 10+ 分钟：
- `--kill yes` 没能让 collective app 干净退出（app 卡在最后的 NCCL teardown / `dist.barrier`）。
- 这些进程最终变成 **zombie（state=Z, ppid=1）**，但 init **没有回收**它们。
- 结果：**每张 GPU 仍被占约 136GB（`mem_get_info` 显示 GPU0 只剩 3.3GB free）**，整机基本不可用，
  直到 zombie 被回收 / 显存被释放。`nvidia-smi --query-compute-apps` 此时为空（容器里看不到持有者）。
- `kill -9` 杀不掉（进程处于 D / 已 defunct）；需要更强的清理手段或排查 teardown 为什么 hang。

**这是把脚本投入使用前必须解决的阻塞问题。** 复现说明：跑完 3.3 的命令后就会出现。

可尝试的方向（按优先级，接手 agent 验证）：
1. **report 产出即收尾**：脚本里轮询 `$OUT/rank*.ncu-rep`，一旦 8 个齐了，主动
   `kill -9` 掉所有 ncu + 其子 `python3` rank 进程，再 `rm -f /dev/shm/cuda.shm.* nccl-* torch_*`。
   不要依赖 `--kill yes` 自己干净退出。
2. **改 driver 收尾**：prof launch 后**去掉最后的 `dist.barrier()`**，或给 `init_process_group`
   传 `device_id=`、设 NCCL 超时（`init_process_group(timeout=...)` / `TORCH_NCCL_*` env），
   让 teardown 不会无限等。可能 `--kill yes` 在 barrier 中途杀进程导致 NCCL D-state。
3. **排查 D-state 来源**：用 `cat /proc/<pid>/wchan`、`/proc/<pid>/status` 看卡在哪个 syscall；
   确认是 NCCL all-gather/barrier 还是 CUDA context 销毁。
4. 若显存确实被 defunct 进程 pin 住且无法回收，评估是否需要在脚本结尾做受控的
   `nvidia-smi --gpu-reset`（**有风险，必须确认无其他任务在用 GPU 才能做**）。
5. 备选：`--app-replay-buffer file`（落盘而非内存），看是否改变收尾行为。

## 5. 下一步：固化成 `scripts/ncu_fused_fa_oproj_ar.sh`

设计（多卡只保留 3.3 的 tcp+application-replay 路径，shmem 已否决）：

- **`-n 1`（单卡）**：一个 ncu 包 `torchrun --nproc_per_node=1`（或直接 `python3`），
  `--replay-mode kernel`（默认）、`--clock-control base`、默认 `-s full`。看 FA+O_proj 计算。
- **`-n 8`（默认，多卡）**：8 个 per-rank ncu + tcp + application replay（照 3.3）。
  必须实现 **§4 的收尾清理**（report 齐了就强杀 + 清 /dev/shm）+ heartbeat/STALL watchdog +
  收尾打印 per-rank `gpu__time_duration.sum`（挑 duration 最短的 "worker" rank 分析；collective 下
  早到的 rank 在 AR barrier 里 spin，duration 虚高，是 waiter）。
- 参数：`-n N` / `-s SET`（数字；单卡可 `full`）/ `-o OUTDIR` / `-p PORT` / 透传 driver 参数；
  env：`WARMUP`(默认4)、`KERNEL_REGEX`(默认 `FusedFaOprojAr`)、`CLOCK_CONTROL`。
  导出 `NCCL_NVLS_ENABLE=1 NCCL_ALGO=NVLS CUTE_DSL_LINEINFO=1`。
- launch-skip 公式：**`--launch-skip $((1+WARMUP)) --launch-count 1`**。

### 建议的数字 section 集（每集一个维度，先在本机量各自 pass 数）
```
1  SpeedOfLight                                  # 已测 = 10 passes（kernel replay；app replay 待测）
2  SchedulerStats + WarpStateStats               # 调度 + stall-reason（warp-specialized 最关键）
3  ComputeWorkloadAnalysis                       # pipe 利用率（WGMMA/FMA/ALU）
4  MemoryWorkloadAnalysis                        # 访存 pipeline
5  Occupancy + LaunchStats + InstructionStats    # 占用率限制 + 指令组成
6  MemoryWorkloadAnalysis_Chart + _Tables        # L2/DRAM/peer reduce 数据通路
7  Nvlink + Nvlink_Tables + Nvlink_Topology      # AR 的 NVLink/NVSwitch 传输
8  SourceCounters                                # SASS/源行级 stall（需 --import-source yes；
                                                 #   注意 application *range* replay 对 JIT kernel 无 SASS，
                                                 #   普通 application replay 可以）
9  RooflineChart + HierarchicalTensorRooflineChart
```
**注意**：application replay 每个 pass 重跑整个 app（NCCL init + CuTe JIT compile，每次约数十秒），
所以总耗时 ≈ pass 数 × 单次 app 时间。section 越多 pass 越多、耗时越长、§4 的泄漏累积风险越高。
**先把 §4 解决，再逐个 section 集实测 pass 数和耗时**，按需要继续拆分（如 6→6/7…）。

## 6. 验证顺序（接手后建议）
1. 解决 §4 收尾泄漏（最优先；否则连续跑会把整机显存占满）。
2. 单卡：`-n 1 -s full` 跑通，拿 FA/O_proj 计算 SOL/roofline/warp stall。
3. 多卡：`-n 8 -s 1` 跑通 + 自动收尾清理验证（跑完显存能回到接近空）。
4. 多卡逐集 `-s 2..9`，记录每集真实 replay pass 数与耗时，定出实际可行的覆盖范围（用实测数据，
   不要预设魔法上限）。
5. 分析：`ncu --import <worker-rank>.ncu-rep --page details | less`。

## 7. 参考
- 官方多进程 profiling：skill `gpu-program-reference` 下
  `references/cuda-cpp/vendored-docs/ncu-docs/NsightComputeCli.md` §4.3.2（Multi-Process Support /
  Mandatory Concurrent Kernels）、`ProfilingGuide.md`（Kernel/Application/Range Replay）。
- 旧的 standalone O_proj+AR profiling：`scripts/ncu_oproj_ar.sh`、`scripts/prof_driver.py`
  （tcp + application replay + lockstep + 8 per-rank 实例的范式来源；其 watchdog/cleanup 思路可借鉴，
  但**具体取值和"38 pass"叙事不要照抄**）。
