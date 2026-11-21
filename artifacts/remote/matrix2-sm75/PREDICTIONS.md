# kvmem-matrix2-sm75 — int8 单调性 + RSS 第三指标预注册

> 写于 2026-11-19 06:30 PST。内核代码已就绪（751 行，AST OK），**尚未推送**（第一轮 `kvmem-matrix-sm75` 仍 RUNNING，Kaggle 单 GPU 配额须串行）。
> 复用第一轮 matrix 的 12 阶段构建链（v5 全绿，build 4.41 h），底座 `6cc1f9fa438b7094685ae417dfa40e1ca324e617`，T4 sm_75 / CUDA 13.0.88。
> 本轮**唯一目的**：补第一轮主动放弃的两个维度——①int8 相对 bf16 的容量单调性；②VmRSS 峰值（04 计划 §5.0 第三指标，第一轮记 UNKNOWN）。不重跑整个矩阵。

## 一、跑什么（跑前写死）

复用第一轮 `scan_matrix` 的 `main`（合法 B×ctx 格）+ `hsweep`（host 4 档），两段完全不动，因此这两段的读数**应可与第一轮逐格对照**——这是免费的重复性检验。新增 `int8` 段：

| 段 | 行 | flag | traffic | 目的 |
|---|---|---|---|---|
| main | 9 合法 + 6 STRUCTURAL_NO | `--kv-dtype bf16`（common 继承） | 2 | 与第一轮重复性对账 |
| hsweep | host∈{64,512,2048,8192} | 同上 | 2 | 与第一轮重复性对账 |
| int8 | headroom∈{1024,512} | `--kv-dtype int8_g64 --kv-capacity auto --kv-headroom-mib N --max-context 8192` | 0 | 容量单调性探针 |

`--kv-headroom-mib` 只在 `--kv-capacity auto` 下合法（`serve_options.cpp:1758-1760`），所以 int8 段走 auto 不是显式。traffic=0 因为容量从启动日志 `engine_capacity` 行读，不依赖请求。

所有 `run_serve` 调用都经改造后的函数：Popen 后起 daemon 线程每 0.5 s 读 `/proc/<pid>/status` 的 `VmRSS` 到 list，terminate 后 set stop + join(timeout=2)，返回 `rss:{peak_kib,last_kib,n,peak_mib}`。读 /proc 无副作用，不影响服务行为。

## 二、判据（跑前写死，不许事后挪）

| # | 判据 | 预期 | 若反了说明什么 |
|---|---|---|---|
| D1 | int8 两行 `alive_after_probe=true`，启动日志有 `engine_capacity` 行 | 全绿 | flag 组合错（`--kv-dtype`/headroom 互斥），回 `serve_options.cpp` |
| D2 | int8 页数 **严格大于** 同 headroom 的 bf16（bf16: 5323@hr1024、5579@hr512，v5 实测） | int8 > bf16 | KV 单价 2 倍错又回来了，或 dtype 旗标没生效（仍是 bf16） |
| D3 | int8 的 `headroom=512` 页数 > `headroom=1024`（少扣 512 MiB 预留 → 多页） | 严格增 | auto 容量路径的 headroom 减法没生效 |
| D4 | 每行 `rss.n ≥ 1` 且 `rss.peak_kib > 0` | 全成立 | 采样线程没起来（daemon 线程异常被吞）或 /proc 路径错 |
| D5 | main/hsweep 的 bf16 行数 = 第一轮（9 合法 + 6 结构 NO + 4 hsweep） | 完全一致 | 两段被意外改动，重复性检验失效 |
| D6 | `rss.peak_mib` 跨 host_mib 的单调性 | **不预测** | 第一轮 UNKNOWN 的第三指标，首次实测，只观测 |

**绝对判据**：D1/D2/D3/D5（结构 + 单调性）。D4 是仪器自检（采样器本身工作），D6 只观测不判定。命中率/bps 数值与第一轮逐格对账，偏差 >20% 才是发现（两次独立启动的 CUMULATIVE 计数可能有预热差异）。

## 三、量级预期（只做 sanity，不当判据）

4B 模型 KV 单价（报告 16 容量口径，8 层）：bf16 2.00 MiB/page-group、int8_g64 1.03125 MiB/page-group。T4 ECC 后可用 14400 MiB、minres ≈ 2986 MiB（报告 18 `m=128` 反解）、headroom 1024/512：

- bf16 hr1024: `128 + floor((14400−1024−2986)/2.00) = 5323` ✓（v5 实测）
- int8 hr1024: `128 + floor((14400−1024−2986)/1.03125) ≈ 10544`（**量级预测，±2% 容差，不作硬判据**）
- int8 hr512: `128 + floor((14400−512−2986)/1.03125) ≈ 11040`（同上）

D2/D3 只用"大于"关系，不依赖这两个绝对值——这正是报告 18「差判据活、绝对判据死」的方法论落地。

## 四、时长与止损

build 4.41 h（943 目标）+ 模型 45 s + main 9×1–3 min + hsweep 4×1–3 min + int8 2×15 s ≈ **5.5–6 h**。timeout 28800 s（8 h）留 2 h buffer。

止损：若 build 因环境问题失败一次，放弃本内核，以报告 01–20 + 第一轮矩阵实测交付（AGENTS.local 第 13 条：同一问题最多 2 个实验）。int8/RSS 留在"未测"栏，不追加第三次环境尝试。

## 五、v2 实测结果与 v3 重推（2026-11-19 补记）

v2 两内核均 COMPLETE。第一判据结果：
- **第一轮矩阵**：全绿（D1 13 合法格、D2 6 非法格不启进程、D5 SKIP）。实测命中率 100%、TTFT 16.5/33.2/51.0×、host 饱和 2 GiB、RSS 首次采到——已写入报告 21。
- **第二轮 bf16 + RSS 段**：全绿。RSS 主扫描 3077–3099 MiB、H 扫描 1103→9234 MiB。
- **第二轮 int8 段**：6 判据全红，exit 1 + help 文本。根因 `--kv-dtype int8_g64` 非法——CLI `parse_kv_dtype`（`serve_options.cpp:62-77`）合法名是 **`int8`**。

**v3 唯一改动**：`matrix2_kernel.py:721` 的 `int8_g64` → `int8`（注释同步）。其余代码、判据、fixture 一字不动。重推 version 3，build 再花约 4.8 h。

**判据不变，仍用上面的 D1–D6**（跑前写死，不因 v2 失败而挪）。预期 int8 页数落在量级值 ~10544（hr1024）/ ~11040（hr512），D2 严格 > bf16 5323/5579、D3 hr512>hr1024。若 v3 int8 仍红且确系引擎行为（不是旗标），则记 UNKNOWN 收口，不再第四次重推。
