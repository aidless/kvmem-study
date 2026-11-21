# Ninfer + KVMem 深挖：一个生产级 KV 引擎的读、算、测

从「核验一条抖音视频说什么」开始，到读透 [NInfer](https://github.com/Neroued/ninfer) 引擎源码、从模型文件反推容量、在没有 GPU 的情况下用 Kaggle 双 T4 真实编译运行、跑完分层矩阵实验，最后把容量预测的三行全错拆成四项可复算的误差。

产出：**22 份源码报告 + 5 篇对外笔记 + 4 个可跑工具**，全部带 `path:line` 引用和实测闭环；查不到的事实一律标 `UNKNOWN`，不用记忆补。

## 这个仓库是什么

- **不是** NInfer 或 KVMem 的 fork，是**第三方视角的深度阅读 + 实测记录**。源码只引用、不重分发（见 [血缘与许可](#血缘与许可)）。
- 每一份报告回答一个具体问题，每一处数字都能从 `artifacts/` 里的原始日志/JSON 复算。
- 所有预测在开跑前写死（`artifacts/remote/*/PREDICTIONS.md`），事后对账，不对就改脚本不改判据。

## 按问题进入

| 你想问什么 | 去哪 |
|---|---|
| Ninfer 和 KVMem「合体」到底是什么、是不是真的 | [`reports/01`](reports/01-kv-offloading-landscape.md)、[`blog/01`](blog/01-agent-session-kv-checkpoint.md) |
| 每个后端加了什么补丁、为什么这么加 | [`reports/02`](reports/02-patch-taxonomy.md)（104 个 patch 分类） |
| 跨会话检查点复用怎么保证正确 | [`reports/03`](reports/03-spread-checkpoint.md) |
| KV 分页引擎的物理层设计 | [`reports/06`](reports/06-ninfer-core-source.md)、[`blog/02`](blog/02-kv-physical-layer.md) |
| 前缀缓存 / 落盘 / 磁盘层 / 准入合并 | [`08`](reports/08-prefix-layer.md) → [`09`](reports/09-disk-tier.md) → [`10`](reports/10-admission-coalescing.md) → [`14`](reports/14-persist-format.md) |
| StateImage、租约、spill、多卡分层 | [`11`](reports/11-state-image.md) / [`12`](reports/12-lease-ladder.md) / [`13`](reports/13-disk-spill.md) / [`15`](reports/15-stage-plan.md) |
| 一张卡到底能跑多少 token（4B / 27B） | [`16`](reports/16-capacity-from-manifest.md)、[`20`](reports/20-27b-capacity.md)、`tools/capacity*.py` |
| 没有 GPU 怎么办、怎么在 Kaggle 上编译运行 | [`17`](reports/17-phase-b-unblock.md) → [`18`](reports/18-phase-b-run.md)、[`blog/03`](blog/03-phase-b-run-notes.md) |
| 分层矩阵实验的真实结果 | [`21`](reports/21-matrix-results.md)、[`blog/04`](blog/04-criteria-can-be-wrong.md) |
| 三条核心主链（准入 / 缓存 / 状态）的收尾深读 | [`22`](reports/22-core-chains.md) |
| 面试怎么讲这个项目：5 分钟主线 + 追问库 | [`blog/05`](blog/05-interview-narrative.md) |
| 全部报告的摘要与 GO/NO-GO 表 | [`reports/00-SUMMARY.md`](reports/00-SUMMARY.md) |

## 目录结构

```
reports/      22 份源码报告 + 00-SUMMARY.md（摘要、GO/NO-GO 表、文件索引）
blog/         5 篇中文对外笔记（第一人称、带 path:line）
tools/        4 个可跑脚本（容量计算 / 引擎口径 / 矩阵对账）
artifacts/    实测原始数据：Kaggle 内核 result.json、启动日志、预注册判据
kernels/      5 个 Kaggle 内核脚本（可复跑的构建与扫描链）
```

## 四条贯穿全项目的结论

1. **KVMem 的真增量**不是「发明 KV 卸载」，而是三个策略决策：有界→无界 RetainAll、前缀命中→内容检索召回、独占文件→跨会话共享（host RAM 层）。NInfer 原生已有 host KV 槽、块身份、GDSF 驱逐、落盘。
2. **页大小 64 是编译期常量**（`paged_kv_cache.h:20` + `static_assert`），不是软件限制——改它等于重编全部 CUDA kernel。
3. **零拷贝的根据是一条不变量**：free 页在 append kernel 写入前不被读。它同时支撑可借性与 suspend 快照瘦身。
4. **判据比数字重要**：容量预测三行全错，但跑前写死的差判据活下来并直接指出单价错误；绝对判据全错。

另有两条容量口径的结论（stage 预算口径 ≠ 引擎容量口径，6 GB 卡上差 4.6 倍；`minres` 随 KV dtype 变 98 MiB），见 [`reports/16 §12`](reports/16-capacity-from-manifest.md) 与 [`reports/21 §6`](reports/21-matrix-results.md)。

## 关键数字

| 量 | 值 | 出处 |
|---|---|---|
| 4B 权重 / head / MTP | 2109.1 / 497.3 / 77.4 MiB | `reports/16` |
| 全常驻 | 2732.95 MiB | `reports/16` |
| KV 单价（bf16 / int8） | 2.00 / 1.03125 MiB 每 64-token 页组（8 层） | `reports/16 §5` |
| 6 GB 容量 | 59,392（bf16）/ 115,008（int8）token | `reports/16 §12.5` |
| T4 16 GB 实测容量 | 5,323 页（headroom 1024）/ 5,579（512） | `reports/18` |
| Kaggle build | 15,873.9 s，943 目标零 error | `reports/18` |
| 矩阵命中率 / TTFT | 100% / 16.5–51.0× | `reports/21` |
| Host tier 饱和点 | 2 GiB（再大无收益） | `reports/21`、`blog/04 §3` |

## 工具

| 脚本 | 作用 | 验证 |
|---|---|---|
| `tools/capacity.py` | 从 `.ninfer` manifest 算 4B 容量，两口径 + 九档 dtype | 36 项 `--selftest` PASS，`--prove-red` 有效 |
| `tools/capacity27.py` | 27B 容量表（从 spec + README 取权威参数） | 13 项 PASS |
| `tools/capacity_engine.py` | 引擎口径容量计算器，复现 `kv_capacity.cpp` 的 `m=128` | 19 项 PASS，RED 有效 |
| `tools/reconcile_matrix.py` | 两轮矩阵判据对账（`--round2`） | RED→GREEN 双验 |

```bash
python3 tools/capacity_engine.py --vram-mib 15360 --kv int8 --ecc   # T4 口径
python3 tools/capacity.py path/to/model.ninfer --selftest            # 需要模型文件
```

## 实测数据怎么来的

`artifacts/remote/` 里是 Kaggle 内核的原始输出：

- `gpu-probe/` — 逐 arch 编译**并运行**验证可跑性（`sm_75: OK`，其余 FAIL）。
- `scan-sm75-v5/` — 容量三行 + Host tier 六行 + 双卡（12 阶段全绿）。
- `matrix-sm75/`、`matrix2-sm75/` — 分层矩阵三轮（bf16 / int8 / RSS）。
- 每个目录都有 `PREDICTIONS.md`（开跑前写死的判据）与 `result.json`（实测）。

复跑需要自己的 Kaggle 账号（`kernels/` 下的脚本可直接推送；build timeout 设 28800，不是默认 14400）。

## 血缘与许可

- 引用源码 commit `6cc1f9fa438b7094685ae417dfa40e1ca324e617`（KVMem `backends/versions.json` 钉的 base，非上游主干）。
- 上游项目：[Neroued/ninfer](https://github.com/Neroued/ninfer)（Apache-2.0）、[iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all)、[kvmem/kvmem](https://github.com/kvmem/kvmem)、[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)。
- 本仓库**不含**上游源码，只有引用。本仓库自己的报告/笔记/脚本以 MIT 发布（见 [LICENSE](LICENSE)）；被引用的上游内容保留其原始许可。
- 报告里的云主机地址已脱敏为 `[T4_HOST]` 等占位符。

## 复现提示

- 所有报告带 `path:line` 引用；查不到的事实标 `UNKNOWN`。
- 实验一律上云，本机是控制台不是算力节点；长跑走 Kaggle 内核。
- `.ninfer` manifest 解析见 `tools/capacity.py`：头部 32 字节 + 126,944 B JSON。

### 路径映射

报告与笔记是研究工作区内按当时结构写的，正文里的 `kvmem-study/<子目录>` 对应本仓库的根：

| 正文写 | 本仓库 |
|---|---|
| `kvmem-study/reports/…` | `reports/…` |
| `kvmem-study/blog/…` | `blog/…` |
| `kvmem-study/tools/…` | `tools/…` |
| `kvmem-study/remote/…` | `artifacts/remote/…` |
| `notes/2026-11-17-ninfer上游与分支血缘核验.md` | `artifacts/lineage-note.md`（19 处引用，已脱敏） |
| `kvmem-study/kaggle/…` | `kernels/…` |
| `kvmem-study/models/qwen3_5_4b_mtp.ninfer` | **未随仓库发布**（2.8 GB，按 `artifacts/remote/PHASE-A-PROVENANCE.md` 自行制备） |
| `kvmem-study/ninfer-src/` | **未随仓库发布**（上游源码，按上面 commit 自行 clone） |

不改正文路径，是为了让报告与当时落盘的台账、预注册记录保持一致；对不上时以上表为准。
