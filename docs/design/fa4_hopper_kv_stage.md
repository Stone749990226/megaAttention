# FA4 Hopper K/V stage 复用笔记

本文整理 FlashAttention-4 Hopper forward 中，`M=128`、两个 MMA warp group、`K/V stage=2` 时，K/V 如何通过 TMA 进入 shared memory、如何被两个 MMA warp group 消费，以及 intra-wg overlap 的寄存器数据流。

对应源码主要在：

- `flash_attn/cute/flash_fwd_sm90.py`
- `flash_attn/cute/softmax.py`

相关 launch shape 参考见：

- `docs/design/fa4_hopper_launch_shape_reference.md`

## 一句话模型

```text
K/V 只放在 shared memory stage 里。
QK 的结果 S 放在当前 WG 的寄存器 acc_S 里。
softmax 后的 P 也先在 acc_S 里，随后转成 PV 的寄存器 A operand tOrP。
PV 用的是上一轮留下来的 tOrP 和当前 V stage。
```

所以不要把 `K.stage` 或 `V.stage` 理解成临时 P buffer。K/V stage 只存 K/V。

## M=128 时两个 MMA WG 怎么分工

Hopper WGMMA 的自然 M atom 是 64 行。FA4 Hopper 对 `tile_m=128` 使用：

```text
atom_layout_mnk = (tile_m / 64, 1, 1) = (2, 1, 1)
```

因此一个 CTA 里有：

```text
WG0: producer / TMA
WG1: MMA consumer, 负责 Q rows 0..63
WG2: MMA consumer, 负责 Q rows 64..127
```

两个 MMA WG 不是分摊同一个 64-row atom，而是各自负责一个 64-row atom：

```text
Q tile [128 x D]

+---------------------+
| rows 0..63    -> WG1 |
+---------------------+
| rows 64..127  -> WG2 |
+---------------------+
```

K/V stage 是共享的：

```text
K.stage s = K block Bj [N x D]

              same K.stage s
                   |
        +----------+----------+
        |                     |
WG1: Q[0..63]   @ K(Bj)^T
WG2: Q[64..127] @ K(Bj)^T
```

V stage 也是共享的：

```text
V.stage s = V block Bj [N x Dv]

              same V.stage s
                   |
        +----------+----------+
        |                     |
WG1: P[0..63]   @ V(Bj)
WG2: P[64..127] @ V(Bj)
```

每个 WG 有自己的 accumulator、softmax state 和 `tOrP`，但读同一个 K/V shared-memory stage。

## Shared memory 布局

以 `tile_n=128`、`head_dim=128`、`head_dim_v=128`、`kv_stages=2` 为例：

```text
SharedStorage

+-------------------------------------------------------------+
| mbar_ptr_Q : Q pipeline full/empty barrier, 1 stage         |
| mbar_ptr_K : K pipeline full/empty barrier, 2 stages        |
| mbar_ptr_V : V pipeline full/empty barrier, 2 stages        |
+-------------------------------------------------------------+
| sQ : [M=128, D=128, stage=1]                                |
+-------------------------------------------------------------+
| sK : [N=128, D=128, stage=2]                                |
|                                                             |
|      +---------------------------+------------------------+ |
|      | K.stage0 [128 x 128]      | K.stage1 [128 x 128]   | |
|      +---------------------------+------------------------+ |
+-------------------------------------------------------------+
| sV : [N=128, Dv=128, stage=2]                               |
|                                                             |
|      +---------------------------+------------------------+ |
|      | V.stage0 [128 x 128]      | V.stage1 [128 x 128]   | |
|      +---------------------------+------------------------+ |
+-------------------------------------------------------------+
```

源码里 K 和 V 是两个独立 pipeline：

```text
pipeline_k: num_stages = 2
pipeline_v: num_stages = 2
```

这点很重要。K/V 没有合成一个 `pipeline_kv`。

## stage 的 full/empty 门

每个 stage 可以理解成两个门：

```text
empty barrier: producer 等它，表示 stage 可写
full  barrier: consumer 等它，表示 TMA 已经写完，stage 可读
```

producer 写一个 K stage：

```text
producer_acquire(K.stage s)
    等 empty(K.stage s)

TMA load K(Bj) -> sK[..., s]
    TMA 完成时 signal full(K.stage s)

producer_commit(K.stage s)
producer_state.advance()
```

consumer 读一个 K stage：

```text
consumer_wait(K.stage s)
    等 full(K.stage s)

WGMMA: Q @ K(Bj)^T

consumer_release(K.stage s)
    到达 empty(K.stage s)
```

`M=128` 时，consumer group 包含 WG1 和 WG2。stage 真正 empty 的条件是两个 WG 都 release：

```text
WG1 release K.stage s
WG2 release K.stage s
    => K.stage s 才能被 producer 下一轮复用
```

V stage 同理。

## 为什么 K 和 V 要分离

K 和 V 的生命周期不同：

```text
K(Bj):
    QK(Bj) 完成后就不用了
    可以尽早 release K.stage

V(Bj):
    必须等 P(Bj) 生成后做 PV(Bj)
    release 比 K 晚一个节拍
```

所以官方拆成：

```text
pipeline_k: K 用完就 release
pipeline_v: V 用完才 release
```

如果把 K/V 合到一个 stage 里，那么 K 明明已经不用了，也要等 V 被 PV 消费完才能复用，既损失 overlap，也更容易把 stage 复用协议写错。

## intra-wg overlap 的核心节拍

intra-wg overlap 的思路是：

```text
当前 block 做 QK。
上一个 block 做 PV。
```

也就是：

```text
QK(B2) 和 PV(B3) 在时间上重叠。
QK(B1) 和 PV(B2) 在时间上重叠。
QK(B0) 和 PV(B1) 在时间上重叠。
```

为了做到这一点，每个 MMA WG 需要同时持有两类寄存器：

```text
acc_S : 当前 block 的 QK 结果，之后会原地变成 P(current)
tOrP  : 上一个 block 的 P(previous)，给当前 PV(previous) 使用
acc_O : O accumulator
```

## 四个 KV block 的完整时间线

假设当前 causal row tile 要处理四个 KV block：

```text
B3, B2, B1, B0
```

这里按从右到左处理。stage=2，所以 stage 轮转：

```text
block B3 -> stage0
block B2 -> stage1
block B1 -> stage0
block B0 -> stage1
```

### Producer 侧

producer 大致按下面顺序 TMA load：

```text
time ->

1. K.stage0 <- K(B3)
2. Q stage  <- Q tile

3. K.stage1 <- K(B2)
4. V.stage0 <- V(B3)

5. K.stage0 <- K(B1)
6. V.stage1 <- V(B2)

7. K.stage1 <- K(B0)
8. V.stage0 <- V(B1)

9. V.stage1 <- V(B0)
```

注意第 5 步复用 `K.stage0` 时，必须等 WG1/WG2 都 release 之前的 `K.stage0 = K(B3)`。

第 8 步复用 `V.stage0` 时，必须等 WG1/WG2 都 release 之前的 `V.stage0 = V(B3)`。

### Consumer 侧

初始状态：

```text
smem_pipe_read = stage0
```

#### Step A: first half, 处理 B3 的 QK

```text
Shared memory:
    K.stage0 = K(B3)

WG1/WG2:
    wait K.stage0 full
    QK(B3) -> acc_S
    release K.stage0
    softmax(acc_S)
    acc_S 原地变成 P(B3)
    tOrP = P(B3)
```

Step A 结束后：

```text
tOrP = P(B3)
acc_O 还没有加 B3 的贡献
K.stage0 已 release
```

#### Step B: current=B2, previous=B3

```text
Shared memory:
    K.stage1 = K(B2)
    V.stage0 = V(B3)

WG registers at entry:
    tOrP  = P(B3)
    acc_O = current O accumulator
```

执行顺序：

```text
1. wait K.stage1 full

2. issue QK(B2)
       input : Q rows owned by this WG, K.stage1
       output: acc_S

3. wait V.stage0 full

4. issue PV(B3)
       input : tOrP = P(B3), V.stage0
       output: acc_O

5. wait_group(1)
       等较老的 QK(B2) 完成
       允许较新的 PV(B3) 继续飞

6. release K.stage1

7. mask / softmax(acc_S)
       acc_S 从 scores(B2) 原地变成 P(B2)

8. wait_group(0)
       等所有 WGMMA 完成，也就是确保 PV(B3) 完成

9. release V.stage0

10. tOrP = P(B2)
       覆盖旧的 P(B3)，给下一轮 PV(B2) 用
```

Step B 结束后：

```text
acc_O += P(B3) @ V(B3)
tOrP   = P(B2)
K.stage1 已 release
V.stage0 已 release
```

#### Step C: current=B1, previous=B2

```text
Shared memory:
    K.stage0 = K(B1)
    V.stage1 = V(B2)

WG registers at entry:
    tOrP = P(B2)
```

执行：

```text
QK(B1) uses K.stage0 -> acc_S
PV(B2) uses tOrP=P(B2) and V.stage1 -> acc_O
wait_group(1)
release K.stage0
softmax(acc_S) -> P(B1)
wait_group(0)
release V.stage1
tOrP = P(B1)
```

#### Step D: current=B0, previous=B1

```text
Shared memory:
    K.stage1 = K(B0)
    V.stage0 = V(B1)

WG registers at entry:
    tOrP = P(B1)
```

执行：

```text
QK(B0) uses K.stage1 -> acc_S
PV(B1) uses tOrP=P(B1) and V.stage0 -> acc_O
wait_group(1)
release K.stage1
softmax(acc_S) -> P(B0)
wait_group(0)
release V.stage0
tOrP = P(B0)
```

#### Step E: last half, 补最后一个 PV

```text
Shared memory:
    V.stage1 = V(B0)

WG registers at entry:
    tOrP = P(B0)
```

执行：

```text
wait V.stage1 full
PV(B0) uses tOrP=P(B0) and V.stage1 -> acc_O
release V.stage1
```

到这里，这个 row tile 对所有 KV block 的贡献都已经累积到 `acc_O`。

## P 到底在哪里

这是最容易混淆的地方。

QK 的 WGMMA 输出是 `acc_S`，也就是当前 WG 的寄存器 accumulator：

```text
acc_S = Q @ K.T
```

softmax 会原地改写 `acc_S`：

```text
acc_S = exp(scores - row_max)
```

这时 `acc_S` 逻辑上就是当前 block 的 P。

然后代码把 `acc_S` reshape 成 PV 的 A operand layout：

```text
tOrP = reshape_acc_to_frgA(acc_S)
```

默认 `mma_pv_is_rs=True`，所以 `tOrP` 是寄存器 tensor。它不会写入 K stage，也不会写入 V stage。

只有 `mma_pv_is_rs=False` 时，P 才会被写入独立的 `sP` shared-memory buffer；即便如此，也不是写入 K/V stage。

## wait_group(1) 和 wait_group(0)

在 middle step 里，代码会连续 issue 两个 WGMMA group：

```text
group old: QK(current) -> acc_S
group new: PV(previous) -> acc_O
```

`wait_group(1)` 的意思是：

```text
等待到最多只剩 1 个 WGMMA group outstanding。
```

因为此时有两个 group outstanding，所以 `wait_group(1)` 会等待较老的 QK 完成，让 `acc_S` 可以安全读取和修改。

但它允许较新的 PV 继续执行。这样 consumer 可以在 PV 还在飞的时候开始：

```text
mask current scores
online softmax current scores
```

后面的 `wait_group(0)` 表示：

```text
等待所有 outstanding WGMMA 完成。
```

这里确保 PV(previous) 完成，然后才能：

```text
release V(previous) stage
覆盖 tOrP 为 P(current)
```

如果没有这个 `wait_group(0)`，就可能在 PV 还没读完旧 `tOrP` 时，把 `tOrP` 覆盖成新的 P。

## 一张总览图

```text
time ->

Producer WG0:
    K.s0=B3    Q
    K.s1=B2    V.s0=B3
    K.s0=B1    V.s1=B2
    K.s1=B0    V.s0=B1
                V.s1=B0

Consumer WG1, rows 0..63:
    QK B3 -> P B3
    QK B2  ||  PV B3 -> P B2
    QK B1  ||  PV B2 -> P B1
    QK B0  ||  PV B1 -> P B0
                 PV B0

Consumer WG2, rows 64..127:
    QK B3 -> P B3
    QK B2  ||  PV B3 -> P B2
    QK B1  ||  PV B2 -> P B1
    QK B0  ||  PV B1 -> P B0
                 PV B0
```

其中 `||` 表示 QK(current) 和 PV(previous) 在 WGMMA pipeline 中重叠。

## 对实现的提醒

如果实现自己的 warp-specialized FA，最容易写错的是这几件事：

1. 不要把 K/V 合成一个 stage 生命周期，除非你非常清楚 release 时机。
2. K stage 在 QK 完成后可以 release；V stage 要等 PV 完成后才能 release。
3. `tOrP` 是上一轮 P，只有等上一轮 PV 完成后才能覆盖。
4. `wait_group(1)` 后只能认为较老的 QK 完成，不能认为 PV 也完成。
5. `wait_group(0)` 后才能 release V stage 和覆盖 `tOrP`。
6. `M=128` 时，WG1/WG2 都必须 release，同一个 stage 才真的 empty。

把这几个点守住，stage=2 的复用关系就清楚了。
