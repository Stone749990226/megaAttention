# exit-clean 单 barrier vs baseline 双 barrier 对比

口径: fused = kineto self device time (ms/iter); ratio = baseline(FA+GEMM+AR) / fused。
改动: 每层 2 次跨 rank nvl_barrier(init+exit) -> 1 次(exit_clean，兼作下一层 init)，
cleanup 从 kernel-start 移到 kernel-exit 并全清 capacity，藏进 straggler 等待。

| 表 | shape | fused 2bar | fused 1bar | Δ% | ratio 2bar | ratio 1bar |
|---|---|---|---|---|---|---|
| README | varlen(B=8,tot=22.6K,max=7.5K) H8/kv1 hid4096 | 1.314 | 1.284 | -2.3% | 1.314 | 1.368 |
| README | varlen(B=6,tot=13.4K,max=6.8K) H8/kv1 hid4096 | 0.809 | 0.790 | -2.3% | 1.331 | 1.371 |
| README | varlen(B=8,tot=22.6K,max=7.5K) H12/kv1 hid6144 | 2.284 | 2.270 | -0.6% | 1.203 | 1.177 |
| README | varlen(B=6,tot=13.4K,max=6.8K) H12/kv1 hid6144 | 1.367 | 1.353 | -1.0% | 1.231 | 1.274 |
| README | varlen(B=8,tot=22.6K,max=7.5K) H12/kv1 hid5120 | 2.028 | 2.021 | -0.4% | 1.178 | 1.179 |
| README | varlen(B=6,tot=13.4K,max=6.8K) H12/kv1 hid5120 | 1.213 | 1.203 | -0.9% | 1.206 | 1.221 |
| README | varlen(B=8,tot=22.6K,max=7.5K) H16/kv1 hid16384 | 5.189 | 5.175 | -0.3% | 1.184 | 1.182 |
| README | varlen(B=6,tot=13.4K,max=6.8K) H16/kv1 hid16384 | 3.071 | 3.053 | -0.6% | 1.190 | 1.195 |
| README | varlen(B=8,tot=22.6K,max=7.5K) H16/kv2 hid8192 | 2.883 | 2.870 | -0.5% | 1.304 | 1.304 |
| README | varlen(B=6,tot=13.4K,max=6.8K) H16/kv2 hid8192 | 1.710 | 1.711 | +0.0% | 1.313 | 1.312 |
| RATIO | varlen(B=6,tot=13.4K,max=6.8K) H8/kv1 hid4096 | 0.813 | 0.781 | -4.0% | 1.321 | 1.378 |
| RATIO | varlen(B=6,tot=13.4K,max=6.8K) H8/kv1 hid4096 | 1.276 | 1.267 | -0.8% | 1.417 | 1.439 |
| RATIO | varlen(B=6,tot=13.4K,max=6.8K) H8/kv1 hid4096 | 2.283 | 2.269 | -0.6% | 1.437 | 1.455 |

结论: 去掉一次跨 rank barrier 带来小而一致的提升，小/便宜 shape 相对收益最大
(hid4096 ~ -2~-4%)，大 shape 在测量噪声内 (<1%)。correctness 全部不变 (err_rel 同 baseline)。
