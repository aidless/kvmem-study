# v1.0.0 — NInfer × KVMem 深度阅读与实测记录

**发布日期**：2026-11-21
**Tag**：`v1.0.0`（指向 commit `6dd91ad`）
**仓库**：https://github.com/aidless/kvmem-study

这是 kvmem-study 的第一个正式版本。它把一个「抖音上看到的合体新闻」追成了一套可复算、可复跑、可面试讲述的完整产物。

---

## 这个版本是什么

不是 NInfer 或 KVMem 的 fork，是**第三方视角的深度阅读 + 云端实测记录**。上游源码只引用 commit、不重分发。

- **22 份源码报告**（`reports/01`–`22` + `00-SUMMARY.md`）：从 patch 分类、分层实验计划，一路深挖到物理层、prefix 层、磁盘 L3、准入合并、租约阶梯、落盘格式、层切分求解器、容量算术、三条核心主链。
- **5 篇中文笔记**（`blog/01`–`05`）：从 agent session checkpoint 到 KV 物理层、Phase B 实跑、判据方法论，最后收束成面试叙事总纲。
- **4 个可跑工具**：`capacity.py`（36 项 selftest）、`capacity27.py`（13 项）、`capacity_engine.py`（19 项）、`reconcile_matrix.py`（RED→GREEN 双验）。
- **7 个 Kaggle 内核脚本**：convert / gpu_probe / driver_probe / build / scan / matrix / matrix2，构成一条可复跑的云端构建—扫描链。
- **实测原始数据**：11 份 `result.json` + 启动日志 + 开跑前写死的 `PREDICTIONS.md` + 脱敏的血缘核验笔记。

---

## 关键成果

### 1. 核验「合体是真的」
KVMem（`backends/ninfer` submodule）钉住 NInfer fork `iamwavecut/ninfer-all @ 6cc1f9fa`，v0.19.1 双后端统一版可查。血缘笔记进一步查清：这个 base 不在上游主干上，是分支线 2026-11-04 的快照，上游 master 已分叉并用另一套设计做了跨请求共享。

### 2. 无 GPU 也能算容量
`stage_fits` 是纯整数运算、零 CUDA 依赖，所有输入都在已落盘的模型 manifest 里。从 `.ninfer` 的 126,944 B manifest 反推出 32 层权重 2109.1 MiB、head 497.3、MTP 77.4、GDN state 49.1，全常驻 2732.95 MiB。6 GB 卡容量：bf16 59,392 token / int8 115,008 token。

### 3. 把「没有 GPU 主机」改写成「购买执行」
本地 RTX 3060 Laptop 是 sm_86，不能当算力节点；四条 ssh 云通道全 DEAD。逐 arch 编译**运行**实测确认 T4 的 sm_75 可用（`nvcc -arch=sm_86` 在 T4 上能编译、运行才报 `no kernel image`）。Kaggle 双 T4 免费，v5 构建 4.41 h / 943 目标零 error，把问题从「拥有资源」变成「购买执行」。

### 4. 判据比数字重要
容量预测三行全错（+23.9%），但跑前写死的**差判据**活了下来并直接指出单价错误：预测两档差 455 页、实测 256 页，比值 0.5626 ≈ 2.00/2.25。四项误差精确闭合（ECC −960 / stage +1536 / resident 漏 −623.8 / headroom 双扣 +1024），闭合差 0.05 MiB。差判据只做减法、绝对值误差在相减时抵消——不依赖绝对值的判据，对绝对值全错的预测仍给出可用结论。

### 5. 矩阵实测
命中率 100%（`reused_tokens` 与 `usage.cached_tokens` 逐字相等）、TTFT 加速 16.5–51.0×、Host tier 2 GiB 饱和、`minres` 随 dtype 变（int8 比 bf16 少 98 MiB）。三轮内核（build → scan → matrix → matrix2 v3）全部全绿。

---

## 这次收尾（相对未打 tag 的上一版）

- 补 `artifacts/lineage-note.md`——19 处报告引用此前是死链
- 修 `reconcile_matrix.py` 的 GREEN fixture bug（`ready_after_seconds` 缺失导致 D1 误红）
- 加 GitHub topics（8 个检索关键词）
- 加 CI（`.github/workflows/selftest.yml`），4 条命令在 push/PR 自动跑，云端实跑全绿
- README 补路径映射与血缘笔记条目

---

## 已知边界（诚实标注，不追）

- 真 `headroom=0` 的引擎容量未实测（`reports/18`）
- int8 页数量级偏 2.3% 的方向、`minres` 98 MiB 差的成因未定位（`reports/21 §6`）
- ROCm/HIP 侧是否存在对应 WDDM 的 pinned host 约束（`reports/06`）
- 上游 master 那套 prefix 设计的运行时表现、LMCache 的本项目可比收益（`artifacts/lineage-note.md`）

这些按 `AGENTS.local` 第 13 条停止，已写进 `CONTRIBUTING.md`。

---

## 复现

四个不依赖模型的 selftest 可直接跑，CI 每次 push 也在跑：

```bash
python3 tools/reconcile_matrix.py --selftest
python3 tools/reconcile_matrix.py --selftest --round2
python3 tools/capacity_engine.py --selftest
python3 tools/capacity27.py --selftest
```

`capacity.py` 需要模型文件路径（`python3 tools/capacity.py <model.ninfer> --selftest`）。模型与上游源码未随仓库发布，制备方法见 `artifacts/remote/PHASE-A-PROVENANCE.md` 与 README「血缘与许可」。

许可：MIT（见 `LICENSE`）。上游项目保留各自原始许可。
