# 20 — 27B 容量表：从权威参数算，不把 artifact 大小当显存

> 日期 2026-11-19。数据源：`/tmp/spec_6cc1f9fa438b7094685ae417dfa40e1ca324e617.md`（85432 B，base `6cc1f9fa` 的 spec）+ `ninfer-src/README.md`。产物 `tools/capacity27.py`（13 项 selftest 全过，RED 有效）。

## 1. 为什么不能复用 capacity.py

`capacity.py` 从 `.ninfer` 内嵌 manifest 的 `objects[].bytes` 精确算权重。本地只有 4B 的模型文件（2,827,176,704 B），**没有 27B 的 manifest**。三条路都不通：

1. 下载 27B artifact（19.03 GiB）到本机：AGENTS.md 第 17 条禁止本地长跑，且 19 GiB 下载 + manifest 解析不是零成本。
2. 从 Hugging Face 拉单文件 manifest：`.ninfer` 的 manifest 内嵌在文件头之后，不是独立 sibling。
3. 用 artifact 的 `Size, GiB`（README:217-223）当权重：这是**文件字节**，不是显存 arena。本报告要证明这条路会系统性高估权重、低估容量。

改走第 4 条：从 spec §3.1/§3.2 与 README 实测表取权威参数，逐项带出处，不造 manifest。

## 2. 权威输入（每项带出处）

| 量 | 值 | 出处 |
|---|---:|---|
| full-attention 层数 | 16 | spec §3.1「27B × 4 KV heads」；README:463 佐证 |
| KV heads / 层 | 4 | spec §3.1 |
| head_dim | 256 | spec §3.1「BF16 1024 B/token/head」= 2×256×2 byte 反推 |
| 页大小 | 64 token | `src/core/paged_kv_cache.h:20` 编译期常量（报告 06） |
| StateImage | 186.8 MiB | spec §3.2（GDN recurrent FP32 144 + conv 2.81 + hidden 10 KiB + DFlash2 rings 40），README:218 的 195,897,344 B = 186.83 MiB 互证 |
| MTP | +1 层 KV | spec §3.1 末句 |
| 默认 headroom | 1024 MiB | `include/ninfer/base.h:57` |
| driver 扣减 | 400 MiB | 报告 16 沿用值 |
| graph allowance | 256 MiB（UNKNOWN） | 报告 18 的 T4 实测 `planning allowance 256.0 MiB`；27B 无本机 probe |

九档 KV 单价**直接复用** `capacity.py:167 kv_bytes_per_token_head()`——它复刻 `src/core/paged_kv_storage.h:60-125` 的 switch，并经编译 ninfer 源码实测。27B 与 4B 的 head_dim 同为 256，故九档字节数逐档一致，差异只在 KV head 数（4）和 FA 层数（16）。

## 3. 单价表与自测

`python3 tools/capacity27.py --selftest` 13 项全过：`bf16 1024 / int8_g64 528 / fp8_row256 516 / k8v4 402 / nvfp4_g16 288 / rk8v4 408 / rk4v4 280 / rk4v4_e8 280 / rk2v4_e8 216` B/token/head；27B bf16 页 = 4,194,304 B = 4.00 MiB、int8 页 = 2,162,688 B = 2.0625 MiB（与 spec §3.1 逐字节一致）；spec §3.3「100k token INT8 KV = 3.38 GB」按十进制 GB 反算精确命中。

`--prove-red` 把 bf16 页字节注入成 4,194,305，门变红并打印 `RED 证明通过`。

27B 每页组 KV（16 FA × 4 KV × 64 token）：

| dtype | B/token/head | MiB/页组 | 每 64K token |
|---|---:|---:|---:|
| bf16 | 1024 | 4.000 | 4096 MiB |
| int8_g64 | 528 | 2.0625 | 2112 MiB |
| fp8_row256 | 516 | 2.0156 | 2064 MiB |
| k8v4 | 402 | 1.5703 | 1608 MiB |
| nvfp4_g16 | 288 | 1.1250 | 1152 MiB |
| rk8v4 | 408 | 1.5938 | 1632 MiB |
| rk4v4 / rk4v4_e8 | 280 | 1.0938 | 1120 MiB |
| rk2v4_e8 | 216 | 0.8438 | 864 MiB |

## 4. 容量表（artifact 文件字节当权重上界）

`official` = 19.03 GiB groupwise-int Q4/Q5（README:219，参考表基准）。扣文件字节 + 186.8 MiB state + 1024 MiB headroom + 256 MiB graph + 400 MiB driver 后，各 dtype 的 token 数：

| 卡 | bf16 | int8 | k8v4 | nvfp4 | rk4v4 | rk2v4_e8 |
|---|---:|---:|---:|---:|---:|---:|
| 3060 Ti 8G | 0 | 0 | 0 | 0 | 0 | 0 |
| 2080 Ti 11G | 0 | 0 | 0 | 0 | 0 | 0 |
| 3060 12G | 0 | 0 | 0 | 0 | 0 | 0 |
| V100 16G（ECC） | 0 | 0 | 0 | 0 | 0 | 0 |
| 3090/4090 24G | 51.3k | 99.6k | 130.8k | 182.6k | 187.8k | 243.5k |
| 5090 32G | 180.0k | 349.2k | 458.6k | 640.2k | 658.4k | 853.6k |
| PRO 6000 48G | 444.7k | 862.6k | 1132.9k | 1581.4k | 1626.6k | 2108.5k |

## 5. 关键发现：artifact Size ≠ 显存权重

README:180 实测 RTX 3060 **12 GB** 能跑 IQ3_S（artifact 13.99 GiB）8,192 token；README:181 **16 GB** V100 能跑 IQ3_S 241,664 token。若把 13.99 GiB 文件字节当权重，12 GB 卡连权重都装不下（本报告表全 0）。README:236 给了机制：官方 27B artifact 在 16 GB 卡上用 `--embedding-q4 --lm-head-q4` 加载；`docs/cli.md:274` 说明 `--embedding-residency host` 把 token embedding 表放到 pinned host（Qwen3.8-27B 的 IQ2_S 表 388 MiB、Q8 表 1.26 GiB）。

因此**第 4 节的表是容量下界**（把整个文件塞进显存的悲观情形）。真实显存占用要扣：

- embedding 表外置：−388 MiB（IQ2_S）到 −1.26 GiB（Q8）；
- 权重 artifact 的 vision tower / MTP / DFlash2 adapter 是否全驻留，spec §3.2 的 StateImage 只算了 DFlash2 rings 40 MiB，adapter 权重另算；
- 27B 的最大上下文实测（README:278-281：3090 417,792 / 4090 405,504 / 5090 872,448 / PRO 6000 1,048,576）未标 dtype，从本报告表反推：24 GB 卡 417,792 token 落在 rk4v4（187.8k）与 rk2v4_e8（243.5k）之间仍差 1.7–2.2 倍 → 实测必然同时用了**更省权重的 artifact**（Bonsai 2 8.87 GiB 或 MXFP8 16.96 GiB）与**外置 embedding**。用 `bonsai2` 档重算 3090 24G：rk4v4 796.6k、rk2v4_e8 1032.6k，417,792 落在其间——量级自洽。

## 6. 对「24 GB 跑 27B」的直接回答

按官方 artifact + 显式容量口径，24 GB 卡跑 bf16 只有 51.3k token；换 int8_g64 99.6k，nvfp4_g16 182.6k。README 实测的 417k/405k 是**低比特 artifact（Bonsai 2 / MXFP8）+ embedding 外置 + rk4v4/rk2v4-e8 极省 KV** 共同作用，不是官方 artifact 的原样容量。所以「4090 24 GB + 27B dense nvfp4」的推荐成立，但要写清：nvfp4_g16 183k token 是官方 artifact 的保守值，实测更高需要换更省的权重档。

## 7. 六条可引用数字

1. 27B 每页 KV：bf16 4.000 MiB / int8 2.0625 MiB / nvfp4 1.125 MiB（spec §3.1 逐字节对上）。
2. 27B StateImage 186.8 MiB（spec §3.2 = README 195,897,344 B）。
3. 100k token INT8 KV = 3.38 GB 十进制（spec §3.3 反算精确命中）。
4. 24 GB 官方 artifact 容量下界：bf16 51.3k / int8 99.6k / nvfp4 182.6k token。
5. 12 GB 卡跑 13.99 GiB artifact 是实测事实（README:180）→ artifact Size 不是显存权重。
6. `tools/capacity27.py`：13 项 selftest PASS，RED 有效；输出逐卡逐 dtype 表 + 四档 artifact 切换。

## 8. 三型假阳守卫

- **语料源错**：spec 的 4.00 MiB 是权威判据，`capacity.py` 的单价函数是底物；两者独立，逐档对上才算过。
- **把测量结果当判据**：README 的 417,792 是实测，但 dtype/artifact/外置策略未标 → 只当锚点，不当容量判据；判据是 spec §3.1 的单价表（有出处、可复算）。
- **让「大」盖住「对」**：容量表全 0 的格子不是脚本 bug，是「artifact 文件字节当权重」这个假设的必然结果；报告把它解释成发现，不放宽假设去凑 README。
