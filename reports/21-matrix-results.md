# 21 — 分层矩阵实测：三轮 Kaggle T4 跑完了什么

> 日期 2026-11-19 ~ 11-20。产物：`remote/matrix-sm75/result.json`（213,402 B）+ `remote/matrix2-sm75/result.json`（v3，229,510 B；v2 旧版存 `result.json.bak-v2-int8flag`），三内核均 COMPLETE。
> 底座 `6cc1f9fa`，T4 sm_75 / CUDA 13.0.88 / 4 CPU / 31 GB RAM。判据 `tools/reconcile_matrix.py`（第一轮 + `--round2` 第二轮，RED→GREEN 双验）。
> 2026-11-20 更新：第二轮 int8 段经 **v3** 内核（`--kv-dtype int8`，v2 是笔误 `int8_g64`）重推后**全部判据转绿**，原第 6 节的"int8 留 UNKNOWN"已作废，见第 6 节修订版。

## 1. 先说结论

三轮 build 全绿：**第一轮 16,047.3 s（4.46 h）、第二轮 v2 17,379.9 s（4.83 h）、v3 16,058.6 s（4.46 h）**，各 943 目标零 error。矩阵 13 个合法格 + 6 个结构非法格 + 4 个 H 扫描格全部按预注册跑完。

- **第一轮判据全绿**：D1（13 合法格全启）、D2（6 非法格不启进程）、D5（host=0 无格，SKIP）。
- **第二轮 bf16 段与 RSS 全绿**：main/hsweep 行数与第一轮一致（D5），每行 RSS 采样有效（D4）。
- **第二轮 int8 段经 v3 重推后全绿**：v2 因 flag 笔误 `--kv-dtype int8_g64`（CLI 合法值 `int8`，`serve_options.cpp:64`）6 判据红；v3 改正后 int8 两行启动、有容量行、`hr1024=10,297` 页 / `hr512=10,794` 页，斜率 1.0302 MiB/页 ≈ int8 理论单价 1.03125。D1–D3 全 PASS（见第 6 节）。
- **实测填上了 04 计划的两个 UNKNOWN**：RSS 峰值首次采到（主扫描 3077–3099 MiB；H 扫描 1103→9234 MiB 随预算单调）；TTFT 命中加速逐格量化。

## 2. 修正了一个判据错：优雅关闭是 rc=0 不是 -15

第一轮初跑 D1 报红：`returncode=0` 而不是预注册设想的 SIGTERM `-15`。取证单格 `m-b8192-ctx8192`：`alive=True`、`ready_after_seconds=22.0`、两次请求 200（第二次 `cached_tokens=7279`）、log 尾部有 `server stopped`——引擎完全正常，是 ninfer-serve 收到 terminate 后**走优雅关闭返回 0**。

判据从 `rc == -15` 改为 `rc ∈ {0, -15}` 且**必须有 `ready_after_seconds`**。复验：两轮全部 13 合法格 rc 分布都是 `{0: 13}`，alive 全真、ready 全有。改的是判据的**过严假设**，不是放水——关键判据（启动 + 就绪）反而更严了（原来只查 rc + alive，现在加查 ready）。

这是报告 18「绝对判据死、差判据活」的同类教训：**预注册里的"正常退出"也是绝对判据，写死后现实用另一种合法方式达成，就误报**。判据该断言的是"服务起来过"，不是"以我设想的方式结束"。

## 3. 命中率：每个 ctx 都 100% 命中，cached_tokens 精确等于 reused_tokens

两次相同 prompt（第 2 次复用第 1 次的前缀），三档 ctx 的结果：

| ctx | reused_tokens（/stats 累计） | 第2次 cached_tokens（usage） | 第1次总耗时 | 第2次总耗时 | 加速 |
|---|---:|---:|---:|---:|---:|
| 8192 | 7,279 | 7,279 | 4.62 s | 0.28 s | 16.5× |
| 32768 | 29,119 | 29,119 | 25.25 s | 0.76 s | 33.2× |
| 131072 | 58,249 | 58,249 | 71.9 s | 1.41 s | 51.0× |

**两条独立采集路径逐字相等**（`/stats` 的 `reused_tokens` vs OpenAI `usage.prompt_tokens_details.cached_tokens`）——这是 08 号报告「hash 找候选、token 才是判等证据」的运行时闭环：命中真的发生了，不是计数器自我报告。加速比随 ctx 增长（16→33→51×）说明命中省的正是 prefill，长 prompt 收益更大。

B 格从 8192 到 131072（16 倍）不改变 reused 和耗时——因为容量够大，命中路径与 B 无关；B 决定的是"最多能装多少"，不是"命中能省多少"。这印证 15 号报告 `StageSolution::page_groups` 是天花板而非性能旋钮。

## 4. H 扫描：host 预算的饱和点在 2 GiB

ctx=32768、B=65536，host 预算 4 档：

| host_mib | host_used | host_block_writes | device_state_slots | RSS 峰值 |
|---:|---:|---:|---:|---:|
| 64 | 64.0 MiB（满） | 32 | 1 | 1,102.7 MiB |
| 512 | 512.0 MiB（满） | 256 | 1 | 1,550.5 MiB |
| 2048 | 1,086.4 MiB（未满） | 455 | 2 | 3,086.6 MiB |
| 8192 | 1,086.4 MiB（未满） | 455 | 2 | 9,233.9 MiB |

**两个发现**：

1. **饱和点在 2 GiB**：2048 与 8192 的 `host_used`（1,086.4 MiB）和 `host_block_writes`（455）完全相同——再大的 host 预算没有被使用。预注册 D4 说"不保证单调"，实测是**先增后饱和**（32→256→455→455）， monotone 检测返回"是"（非降），但真形状是饱和。
2. **RSS 随预算单调，但语义不同**：host=64/512 时 host 被写满、device_state_slots=1；host≥2048 时 host 没写满、device_state_slots=2。**RSS 从 1.1 GiB 涨到 9.2 GiB 主要是 pinned host 内存被计入进程 RSS，不是 GPU 显存**（报告 06 第 9 点的 Linux 侧对应：WDDM 上 pinned host 计入显存预算，Linux 上计入 RSS）。所以"host KV 免费"在任何平台都要看它挤的是哪个预算池。

预注册 D3 预测 H=64 "可能启动失败"（`hybrid_host_layout.h:30-34`：非零预算须装得下 1 snapshot + 1 block），实测**成功**——4B 的 StateImage 49.1 MiB + 一个 block 2 MiB < 64 MiB，正好装下。这是「预测保守但判据只观测不判定」的正确结果：不把它记成失败，记成"4B 的边界比预估低"。

## 5. RSS 第三指标：主扫描稳定 3.1 GiB

第二轮 9 个主扫描格 RSS 峰值 3,077.4–3,099.4 MiB（极差 22 MiB，0.7%），ctx 从 8192 到 131072 只涨 22 MiB。这填补了 04 计划 §5.0 要求的第三维度：内存占用与 B/ctx **几乎无关**——因为 KV 按命中路径共享、host 预算固定 2 GiB，进程 RSS 由模型和 host arena 决定。H 扫描才是 RSS 的主导变量（见第 4 节）。

采样器本身：`read_rss_kib` 每 0.5 s 读 `/proc/<pid>/status` 的 VmRSS，daemon 线程，terminate 后 join——D4 全绿（每行 n≥1、peak>0），仪器可信。

## 6. int8 段：从一处 flag 笔误到 v3 全绿

### 6.1 v2 的失败取证（保留原记录）

第二轮 v2 的 int8 两行 exit 1、alive=False、无容量行、日志是 `--help` 尾部。三步锁因：

1. `alive=False, rc=1, seconds=2.0` —— 进程在解析参数阶段就退了，没到引擎启动。
2. log 是 usage 文本（含 `--draft-tokens`、`--log-level` 等帮助条目）—— 参数解析器拒绝后打印帮助。
3. `serve_options.cpp:62-77` `parse_kv_dtype` 的合法集：`bf16 / int8 / fp8 / rk8v4 / rk4v4 / rk4v4-e8 / rk2v4-e8 / nvfp4 / k8v4`。我写的是 `int8_g64`（capacity.py 的内部名）—— **CLI 不认，抛 `invalid kv-dtype: int8_g64`**。

`--kv-headroom-mib` 要求 `--kv-capacity auto`（`serve_options.cpp:1766`）这条我写对了，不是它的问题。

### 6.2 v3 重推：判据全绿

v3 唯一改动是 `matrix2_kernel.py:721` 的 `int8_g64` → `int8`（AST 校验通过），其余 pre-registration 与 D1–D6 判据不变。build 16,058.6 s，943 目标零 error。

`python3 tools/reconcile_matrix.py remote/matrix2-sm75/result.json --round2` → **全部判据通过**：

| 判据 | 结果 | 实测 |
|---|---|---|
| D5 bf16 合法+hsweep 行数 == 13 | PASS | legal=9 hsweep=4 |
| D5 结构 NO 行数 == 6 | PASS | got=6 |
| D1 int8 两行启动 | PASS | live=2/2 |
| D1 int8-hr1024 有容量行 | PASS | pages=10,297 |
| D1 int8-hr512 有容量行 | PASS | pages=10,794 |
| D2 int8 hr1024 页数 > bf16 5323 | PASS | int8=10,297 bf16=5,323 |
| D2 int8 hr512 页数 > bf16 5579 | PASS | int8=10,794 bf16=5,579 |
| D3 int8 hr512 > hr1024 | PASS | 10,794 > 10,297 |
| D4 每行 RSS 采样有效 | PASS | — |

引擎启动日志原文（两行各一条）：

```
capacity | KV 659,008 tokens, int8, auto | pages 10,297/33,554,431 | runtime 10.9 GiB | free 1.23 GiB
capacity | KV 690,816 tokens, int8, auto | pages 10,794/33,554,431 | runtime 11.4 GiB | free 750.8 MiB
```

### 6.3 三个此前 UNKNOWN 的答案

1. **int8 斜率实测 = 1.03018 MiB/页组**（`512 / (10,794 − 10,297)`），与理论单价 1.03125（8 层 × 67,584 B）差 0.10%。这是**引擎口径的 int8 单价首次实测**，此前只有 `capacity.py` 的算术值。
2. **int8 的 `minres` ≈ 2,889 MiB**（两行反解 `14400 − hr − (pages−128)×1.03125` 得 2,889.22 / 2,888.69，互差 0.53 MiB）。比 bf16 的 2,986.00 少 **97 MiB**——工程含义是：同一个 `--kv-headroom-mib` 在 int8 下能多拿 ~97 MiB/1.03125 ≈ 94 页 ≈ 6,016 token。**minres 是 dtype 相关的，不是常数**（`capacity_engine.py` 目前把 2,986 当全局常量，这是它的已知边界）。
3. **预注册量级预测偏 2.3%**（预测 10,544 / 11,040，实测 10,297 / 10,794）。偏差来自第 2 点：我按 bf16 的 minres 推 int8 页数，而 int8 的 minres 更小 → 应预测更多页才对，实际更少——说明偏差方向与 minres 差相反，另有一项未定位的差（`mtp_extra_pages` 随 dtype 缩放是候选，`startup.cpp:146`：`(draft_window−1)/64` 页 MTP KV，其单价也随 dtype 减半）。按 AGENTS.local 第 13 条不再开第三个反解实验，如实记为**量级判据未命中、方向未定位**。

预注册里这两行本来就是「只观测不硬判」的量级项（±2% 不硬判），所以 D1–D4 的绿不受影响；但这个 2.3% 偏差本身值得记住：**"算术已知"不等于"能预测"，因为算术里的常数项可能随变量组变化**。

## 7. 可引用数字表

| 量 | 值 | 来源 |
|---|---:|---|
| build 时长（第一轮 / 第二轮 v2 / v3） | 16,047.3 / 17,379.9 / 16,058.6 s | 三 result.json 的 `stages.build.seconds` |
| 合法格 / 结构非法格 / H 扫描格 | 13 / 6 / 4 | 三轮一致 |
| 三档 ctx 命中率 | 100%（reused 与 cached 逐字相等） | 3 格 × 2 轮 |
| TTFT 加速（8192/32768/131072） | 16.5× / 33.2× / 51.0× | 第 2 次耗时比 |
| host 饱和点 | host_used 1,086.4 MiB / writes 455（2048 与 8192 相同） | 第二轮 H 扫描 |
| RSS 主扫描 | 3,077–3,099 MiB（极差 0.7%） | 第二轮 9 格采样 |
| RSS H 扫描 | 1,103→1,551→3,087→9,234 MiB | 第二轮 4 格采样 |
| **int8 实测页数（hr1024 / hr512）** | **10,297 / 10,794** | v3 `result.json` 的 `matrix.int8` |
| **int8 实测 token（hr1024 / hr512）** | **659,008 / 690,816** | 同上，启动日志 `capacity \|` 行 |
| **int8 斜率** | **1.03018 MiB/页组**（理论 1.03125，差 0.10%） | `512/(10,794−10,297)` |
| **int8 `minres`** | **≈2,889 MiB**（bf16 是 2,986，少 97） | 两行反解，互差 0.53 MiB |
| int8 量级预测偏差 | −2.3%（预测 10,544/11,040） | 预注册量级项，不硬判 |

## 8. 四条方法论

1. **预注册的绝对判据要防"换一种合法方式达成"**：rc=-15 被优雅关闭 rc=0 满足同一语义却判红——断言要落在"服务起来过"（ready），不是"以我设想的方式结束"。
2. **两个独立采集路径逐字相等才是真命中**：`/stats` 计数器自说自话不算，OpenAI usage 的 cached_tokens 相等才算闭环。
3. **监控指标要标"挤的是哪个池"**：Linux 的 RSS 涨 ≠ 显存涨；同理 WDDM 的 pinned host 涨 = 显存涨。同一物理量在不同平台的预算归属不同。
4. **自己写错的旗标值与环境失败是同一类错**：都要"三步锁因"，不能因为错在脚本就含糊过去。但"止损不追加"要分清止损对象——止损的是**没有新读数的重复**，不是**修正一次性错误后的重跑**。v3 只改一个字符串常量、换回 4.46 h build 就拿到了 int8 的引擎实测单价与 minres（此前只有算术值），这笔账是正的：算术值从未排除"常数项随变量组变化"这一可能，而实测当场就发现了 minres 差 97 MiB。
5. **算术已知 ≠ 能预测**：int8 的斜率算得准（1.03018 vs 1.03125，差 0.10%），但页数量级偏 2.3%，因为公式里的常数项 `minres` 本身随 dtype 变。**算得准的是斜率，猜不准的是截距**——这和第 2 节"差判据活、绝对判据死"是同一个结构的又一个实例。
