# kvmem-matrix-sm75 — 04 号分层矩阵预注册

> 写于 2026-11-19 05:50 PST。内核尚未推送。
> 复用 `<kaggle-user>/kvmem-scan-sm75` v5 的构建链（12 阶段全绿，build 4.41 h），只把扫描阶段换成 04 号矩阵。
> 底座仍是 `6cc1f9fa438b7094685ae417dfa40e1ca324e617`（kvmem `backends/versions.json` 钉的 base），T4 sm_75 / CUDA 13.0.88 / 4 CPU / 31 GB RAM。

## 一、矩阵的合法投影（跑前写死）

04 计划 §5.4 给的 B ∈ {1024, 4096, 16384, 32768} × ctx ∈ {8192, 32768, 131072} 在 ninfer 引擎下**大部分格子非法**：
`startup.cpp:1331-1332` 显式容量必须 `explicit_tokens >= max_context`，否则抛 `kv_capacity must be at least max_context`。

脚本实际用的 B 格是 `{8192, 16384, 32768, 65536, 131072}`（04 原表还有 1024/4096，但那是 `B>=128` 的下界演示值；我按 ctx 的合法上界补齐到能覆盖 131072 ctx）。

| ctx \ B | 8192 | 16384 | 32768 | 65536 | 131072 |
|---|---|---|---|---|---|
| 8192 | 跑 | 跑 | 跑 | 跑 | 跑 |
| 32768 | NO | NO | 跑 | 跑 | 跑 |
| 131072 | NO | NO | NO | NO | 跑 |

脚本按 `B >= ctx` 自动生成行，非法行标 `STRUCTURAL_NO` 不启进程。**预期合法 9 格（主扫描 5+3+1）、非法 6 格（2+4）**；H 扫描 4 格全合法（ctx=32768、B=65536），合计 **13 个真跑格**。

⚠️ **dtype 偏离 04 原文**：04 §5.0 固定面写 `int8`，本内核沿用 v5 全绿的 `bf16`（`--kv-dtype bf16`）。理由：v5 的 9 行扫描已把 bf16 路径验透，首轮矩阵要隔离的变量是 B/H/ctx，不叠加 dtype；int8 作为第二轮 or 对照另跑。这条偏离必须写进报告 20，不许当"按 04 执行"。

⚠️ **RSS 维度本轮不采**：04 §5.0/§5.3 要求 `/proc/<pid>/status` VmRSS 序列，`scan_kernel.py` 的 `run_serve` 只有 stdout/stderr 文件日志、没有侧车采样线程。内核已推（QUEUED），加采样线程要重推并重新排 4.4 h build——**不值得**。本轮只给命中率/tps/D2 合法性三维，RSS 记为 UNKNOWN 进报告 20；这不是"RSS 不重要"，是这一发的边际收益低于重排成本。

## 二、采什么

每格：`run_serve(seconds=1200, traffic=2, prompt=<长 prompt>)`，prompt 长度 `min(ctx, 65536)` token 的重复句（`make_long_prompt` 按 0.8 系数估算句子数）。两次相同请求 → 第二次应触发命中。
采集面（沿用 v5 验证过的 `read_stats`）：`/stats` 的 `context_cache.hybrid`（CUMULATIVE，`load_report.cpp:87-118`）18 个 counter + `capacity` + `occupancy`。`usage.prompt_tokens` 取真实长度。

## 三、判据（跑前写死，不许事后挪）

| # | 判据 | 预期 | 若反了说明什么 |
|---|---|---|---|
| D1 | 所有合法格 `alive_after_probe=true`、`returncode` 为正常 terminate | 全绿 | 引擎级旗标组合错，回 `serve_options.cpp` 逐行查 |
| D2 | 所有非法格标 `STRUCTURAL_NO` 且**没有**启动进程 | 6 格 | 投影逻辑写错 |
| D3 | H 扫描 `host_mib=64` 的格 | **可能启动失败**（`hybrid_host_layout.h:30-34`：非零预算须装得下 1 snapshot + 1 block，装不下启动即拒） | 若 64 也成功 → 说明 4B 的 StateImage 比预估小，是发现不是失败 |
| D4 | H 从 64→8192，`snapshot_hits`/`reused_tokens` 不保证单调 | 不单调 | fork `452acabb11` 实测 3 GiB 82.4% > 4 GiB 79.9%；单调反而是意外 |
| D5 | 每格 `host_block_writes>0` 仅当 host_mib>0 且 `snapshot_hits>0` | 成立 | v5 已证 host=0 时三项写计数全 0 |
| D6 | `unbacked_node_losses`/`host_dead_reclaims` | **不预测**（04 号的核心未知，首次实测） | — |

**唯一绝对判据**：D1/D2（结构对不对）。命中率/RSS/tps 的**形状判据**（D4/D5）优先于任何数值；04 计划 §5.5 要的曲面在首轮只给观测不给结论。

## 四、时长预算与止损

v5 build 4.41 h（943 目标）+ 模型就位 45 s + 每格 1–3 min × 13 合法格 ≈ **5.5–6 h**。timeout 28800 s（8 h）留 2 h buffer。
止损：若 build 因环境问题（非代码）失败一次，放弃本内核，以 06–19 号源码报告 + 18 号 v5 实测交付，不追加第三次环境尝试（AGENTS.local 第 13 条）。
