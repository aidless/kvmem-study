# 把 22 份报告讲成 5 分钟：一个 KV 引擎项目的面试叙事总纲

这篇不是新研究，是把已经落盘的东西整理成能开口讲、能被追问的结构。素材 22 份源码报告 + 4 篇笔记 + 4 个工具，都带 `path:line` 和实测闭环；这里只做取舍和排序——讲什么、用什么数字、被追问时翻哪一份。

底座：`kvmem-study/`，commit `6cc1f9fa438b7094685ae417dfa40e1ca324e617`（kvmem `backends/versions.json` 钉的 base，血缘见 `notes/2026-11-17-ninfer上游与分支血缘核验.md`）。全部结论索引在 `reports/00-SUMMARY.md`。

## 一、5 分钟主线（先背下这段）

> 我核验了一条「Ninfer + KVMem 合体」的视频，确认它是真的，然后花了两周把它从「一条抖音文案」读成一个完整的生产级 KV 引擎案例。KVMem 的价值不是发明了 KV 卸载——NInfer 原生就有 host KV 槽、块身份、GDSF 驱逐和落盘——它的真增量是三个策略决策：有界改无界、前缀命中改内容检索召回、独占文件改跨会话共享。为了验证这套设计，我读完了引擎源码，从模型文件反推容量，在没有 GPU 的困境下用 Kaggle 双 T4 真实编译运行（build 4.41 小时，943 目标零 error），跑完分层矩阵，最后把容量预测的三行全错拆成四项可复算的误差。**最大的收获不是那几个数字，而是一条方法：跑前写死的差判据活了下来、绝对判据全错——判据比数字重要。**

五步展开，每步背后都有一个「我错在哪」：

| 步骤 | 讲了什么 | 我错在哪 → 怎么发现 |
|---|---|---|
| 1. 核验 | 合体是真的；KVMem 三个策略增量 | 起初以为「合体」是新引擎，实际是 fork + patch 链（`reports/01`、血缘笔记） |
| 2. 读源码 | 页大小 64 是编译期常量、零拷贝靠一条不变量 | 起初把 block 64 当软件限制，读到 `paged_kv_cache.h:20` + `static_assert` 才发现是硬件/ABI 约束（`reports/06`） |
| 3. 算容量 | 从 manifest 反推，两套口径差 4.6 倍 | KV 单价只算了 K 一个向量，**全部数字偏小整 2 倍**；外部 spec 的 27B 页大小当裁判才抓出（`reports/16 §11`） |
| 4. 上云实测 | Kaggle 双 T4，编译成功 ≠ 可运行 | 四连败：树根认错 / 挂载深一层 / 缺 meson-nasm / `CUDA::cuda_driver`；逐 arch **运行**才知 sm_75 是唯一可跑档（`reports/17`） |
| 5. 对账 | 预测三行全错，但差判据活了 | headroom 标签 bug（`if 0:` 为假）、漏 resident、ECC、单价口径四项叠加；`minres` 随 dtype 变 98 MiB（`reports/18 §3.1`、`reports/21 §6`） |

## 二、五分钟之后的追问库

**Q：KVMem 到底改了什么，值得一个项目？**
A：三个策略决策，不是新机制。NInfer 原生已有 host KV 槽/块身份/GDSF 驱逐/`--prefix-cache-file` 落盘；磁盘层甚至已经是内容寻址 + 跨会话 dedupe。KVMem 换的是：有界→无界 RetainAll、前缀命中→内容检索召回、Host RAM 层跨进程共享（把 L3 的能力上移到 L2）。补丁 +16,651 行的原因是**改一个核心假设就得拆掉围绕它的全部补偿机制**（settle 降级、restore 贪心、几何/预算校验在无界世界全作废）。见 `reports/08 §15`、`blog/01`。

**Q：为什么 block 固定 64？能改吗？**
A：`src/core/paged_kv_cache.h:20` `kPagedKVPageSize = 64` 是编译期常量，`paged_kv_address.cuh:8-13` 用 `position >> 6` 取页号、`& 63` 取偏移，还有 `static_assert`。改它等于重编全部 CUDA kernel——不是软件开关。见 `reports/06 §1`。

**Q：零拷贝为什么安全？**
A：一条不变量——free 页在 append kernel 写入前不被读（`evictable_kv_pool.h:16-30`）。它同时支撑两件事：页可借给别的池；suspend 快照能用 `pool.free_extents()` 把空闲页当洞挖掉而瘦身。见 `reports/06 §7`、`blog/02`。

**Q：一张卡能跑多少 token？**
A：必须先问口径。stage 预算口径（喂 `stage_fits`，`fixed=3134.73 MiB` + 9 层单价）和容量口径（引擎 `--kv-capacity auto`，`minres` + `m=128` + 8 层斜率）回答两个不同问题，6 GB 卡上差约 4.6 倍。容量口径：6 GB bf16 **928 页 = 59,392 token**、int8 **1,797 页 = 115,008 token**；T4 16 GB bf16 实测 5,323 页（默认 headroom 1024）/ 34 万 token。上限还要和 `max_position_embeddings 262144` 取 min。见 `reports/16 §12`、`tools/capacity_engine.py`。

**Q：为什么两套口径差这么多？**
A：8 层 vs 9 层——MTP 那份 KV 属于 backend pool，不在主链斜率里（`startup.cpp:1465-1471`）。我一开始混用，把 stage 的数当容量报，低估 4.6 倍。见 `reports/16 §12.4`。

**Q：没有 GPU 你怎么实测的？**
A：本机 3060 Laptop 是 sm_86，有权重也跑不了构建；四条 ssh 云通道全 DEAD。改走 Kaggle 双 T4（sm_75），逐 arch 编译**并运行**验证可跑性——`nvcc -arch=sm_86` 在 T4 上编译会成功，运行才报 `no kernel image is available`，所以编译成功不是证据。构建内核四连败后，v5 唯一改动是 timeout 从 14400 提到 28800，跑通 16,047 s、943 目标零 error。见 `reports/17`、`blog/03`。

**Q：容量预测错在哪，为什么还说成功？**
A：三行绝对数全错（+23.9% / +23.3% / 真 headroom=0 那行没跑），根因四项：脚本 `if headroom:` 对 0 为假导致标签与旗标不符、漏算 resident 623.8 MiB、T4 开 ECC 打了 0.9375 折、单价用 2.25 而非 2.00。**但跑前写死的差判据**（两行应差 455 页）实测 256 页，比值 0.5626 ≈ 2.00/2.25，直接指出了单价错误。差判据只做减法，前三项误差在减法里抵消或变成常数。见 `reports/18 §3.1`、`blog/04`。

**Q：`minres` 98 MiB 的差怎么回事？**
A：诚实答案——**未定位**。bf16 2986 MiB、int8 ≈2888.44 MiB（用固定斜率后两行各给半开区间、取交集中点）。`mtp_extra_pages` 随 dtype 缩放是候选，但按「同一问题最多 2 个实验」的纪律不再开第三个反解实验。这个「不知道就写不知道」本身就是答案的一部分。见 `reports/21 §6`、`README §6`。

**Q：这套方法能迁移吗？**
A：四条已跨项目验证：①判据比数字重要（差判据活、绝对判据死）；②可复算 ≠ 起点对（KV 单价 2 倍错）；③算得准斜率、算不准截距（minres 随 dtype 变）；④NO-GO 有保质期（计划文档里的不可行判定被实测推翻后必须回头改）。共同防线是「跑前写死 + 事后只定位不改判据 + 独立路径逐字对表」。见 `blog/03`、`blog/04`、台账 `L-ops-746/758/797/832`。

## 三、数字速查（被追问时直接报）

| 量 | 值 | 出处 |
|---|---|---|
| Ninfer / KVMem star | 3.3k / 1.3k | `reports/01` |
| 页大小 | 64 token（编译期常量） | `paged_kv_cache.h:20` |
| 4B 权重 / head / MTP | 2109.1 / 497.3 / 77.4 MiB | `reports/16` |
| 全常驻 | 2732.95 MiB | `reports/16` |
| KV 单价（bf16 / int8） | 2.00 / 1.03125 MiB 每 64-token 页组（8 层） | `reports/16 §5` |
| 6 GB 容量 | 59,392（bf16）/ 115,008（int8）token | `reports/16 §12.5` |
| T4 实测 | 5,323 页（hr=1024）/ 5,579（hr=512） | `reports/18` |
| build | 15,873.9 s / 943 目标零 error | `reports/18` |
| 命中率 / TTFT | 100% / 16.5–51.0× | `reports/21` |
| Host 饱和点 | 2 GiB（再大无收益） | `reports/21`、`blog/04 §3` |

## 四、诚实边界（面试时主动说，别等被戳）

- **实测**：容量三行、Host tier 六行、双卡 Legacy、矩阵 13 格、int8 页斜率。数据落盘在 `remote/*/result.json`，预注册在 `remote/*/PREDICTIONS.md`。
- **算术**：4B/27B 容量表、KV 单价、九档 dtype、`m=128` —— 从 manifest/spec 复算，非运行读数。
- **UNKNOWN**：真 headroom=0 从未实测；int8 2.3% 偏差方向、98 MiB 成因未定位；ROCm/HIP 侧是否有对应 WDDM 约束未知。查不到就写 UNKNOWN，不用记忆补（`README §复现提示`）。

这四块讲完，正好 5 分钟主线 + 10 分钟追问。再多就是逐份报告的细节，翻 `reports/00-SUMMARY.md` 即可。

---

笔记索引：`01-agent-session-kv-checkpoint.md`（会话 KV 检查点）、`02-kv-physical-layer.md`（KV 物理层）、`03-phase-b-run-notes.md`（容量实测四件事）、`04-criteria-can-be-wrong.md`（判据也会写错）、`05-interview-narrative.md`（本篇：面试叙事总纲）。完整源码报告见 `reports/00-SUMMARY.md`（01–22）。
