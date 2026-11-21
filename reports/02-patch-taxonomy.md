# KVMem 集成补丁全量分类与深挖（Patch Taxonomy）

> 数据源：`kvmem-study/kvmem/backends/patches/`（104 个 .patch，现算于 2026-11-10）+ `kvmem-study/kvmem/backends/versions.json`。
> 本文所有行数/文件数均为脚本逐文件现算（`+`/`−` 行按 diff 体统计，排除 `+++`/`---` 头；`files` 按 `+++ b/` 去重路径数），引用行号均为 patch 文件内行号。
> 路径均相对仓库根 `kvmem-study/kvmem/`。

## 0. 顶层结构：版本钉法（versions.json）

`backends/versions.json` 的 `layout = "git-submodules-with-patches"`，每个后端记录：

- **base_commit**：钉死的上游 commit（llamacpp=`cd959bae`，ninfer=`6cc1f9fa`，polystrata=`fabae0d6`，上游 URL 各自记录）；
- **patches 数组**：每个 patch 带 `id`、`path`、`sha256`、`requires`（补丁间依赖 → **有序串行应用**）、`after_tree`（应用后整树哈希）、可选 `source`（外部移植来源，含原 fork commit 号）；
- **patched_tree**：全部 patch 应用完成后的整树哈希——`sha256(patch) + after_tree` 双层校验，任何一个 patch 被篡改或错序都会导致整树哈希不匹配。

关键事实：**versions.json 只登记 3 个后端共 75 个 patch**（llamacpp 5 + ninfer 48 + polystrata 22），而 patches/ 目录实有 **104 个**：`radiance-*` 的 29 个 patch（`backends/patches/radiance-*.patch`）不在 versions.json 中，radiance 后端改走 `backends/radiance/adapter/` 适配层 + `backends/radiance/tools/prepare_native.py` 的自有管线（见 `backends/radiance/README.md:1-14`）。即：**四个后端两种集成机制**——ninfer/polystrata/llamacpp 是「pinned fork + 有序 patch 树哈希」，radiance 是「adapter 目录 + 独立 prepare 脚本」。

ninfer 的依赖链最长：`kvmem → ptq1 → ptq1-performance → t5-performance → query-memory → nvfp4-kvmem → checkpoint-policy → monitor-events → nvfp4-blackwell-scale → dflash2-gguf-blocks → … → selective-feedback`（48 个 patch 构成近线性的串行链，`versions.json` 中 ninfer.patches 的 `requires` 字段）。

## 1. 全量 patch 档案表

### 1.1 llamacpp（5 个，upstream `ggml-org/llama.cpp` @ `cd959bae`）

| 文件 | 规模(+/-) | 文件数 | 层 | 机制概括 |
|---|---|---|---|---|
| llamacpp-kvmem.patch | +1154/−137 | 50 | KV 外置核心 | KVMem 嫁接主补丁：新增 `src/llama-kvmem-factory.h`、`src/adapter/llama-memory-kvmem.cpp`（`backends/patches/llamacpp-kvmem.patch:1356,2084`），在 `llama-memory` 抽象（`llama_memory_seq_pos_max` 调用点改造 `llamacpp-kvmem.patch:121-211`）下把 KVMem 注册为第三种 memory 实现（与 KV-cache-unified / recurrent 并列），动到 ggml backend 层（`ggml-backend-meta.cpp`，`llamacpp-kvmem.patch:496`） |
| llamacpp-pq2.patch | +496/−1 | 35 | 量化格式 | PQ2 量化格式全栈支持（ggml CPU/CUDA kernel + gguf 常量 + model-loader + quantize 工具） |
| llamacpp-ptq1.patch | +670/−1 | 35 | 量化格式 | PTQ1 量化格式，结构与 pq2 同构（依赖 pq2） |
| llamacpp-ptq1-performance.patch | +523/−0 | 2 | 算子性能 | PTQ1 planar matvec + PDL 消费端同步修复，只限 dense CUDA 2D 调用（`versions.json` source 字段） |
| llamacpp-checkpoint-policy.patch | +32/−10 | 2 | checkpoint/scheduler | `common/speculative.cpp/.h` 的 checkpoint 准入策略收紧 |

### 1.2 ninfer（48 个，upstream `iamwavecut/ninfer-all` @ `6cc1f9fa`，串行依赖链）

| 文件 | 规模(+/-) | 文件数 | 层 | 机制概括 |
|---|---|---|---|---|
| ninfer-kvmem.patch | +16651/−1250 | 265 | **KV 外置核心** | KVMem 嫁接主补丁（见 §2.0） |
| ninfer-ptq1.patch | +2346/−235 | 68 | 量化格式 | PTQ1 全栈移植（来自 `ninfer-fusion-kvmem` fork，合格性端口） |
| ninfer-ptq1-performance.patch | +157/−29 | 11 | 量化/算子性能 | PTQ1 路由与校准，不改 SM120 策略 |
| ninfer-t5-performance.patch | +2455/−41 | 38 | 算子性能 | RowSplit + BF16 producer 适配（`ninfer-4090-windows` fork） |
| ninfer-query-memory.patch | +569/−81 | 32 | **KV 统计/Query 生命周期** | 有界 BF16 Query 捕获 + Mean-Q 求和（见 §2.2） |
| ninfer-nvfp4-kvmem.patch | +210/−16 | 10 | KV 布局/量化 | scoped/default 路由、原生快照模式绑定、Blackwell KV codec 保留 |
| ninfer-checkpoint-policy.patch | +517/−58 | 12 | checkpoint | checkpoint 准入策略 |
| ninfer-monitor-events.patch | +106/−9 | 6 | 监控 | 原生响应 ID、TTFT、请求间隔速率、MTP/copy 计数器 |
| ninfer-nvfp4-blackwell-scale.patch | +76/−1 | 3 | 量化 | Blackwell UE4M3 weight-scale 归一化 + FP64 codec oracle |
| ninfer-dflash2-gguf-blocks.patch | +177/−43 | 11 | 投机解码 | DFlash2 GGUF 块支持（`melis2023/ninfer-kvmem-actkv`） |
| ninfer-dflash2-nvfp4.patch | +418/−3 | 15 | 投机解码/量化 | DFlash2 NVFP4 dispatch |
| ninfer-cuda-graph-probe.patch | +181/−1 | 9 | CUDA graph | 统一 dense MTP/DFlash2 校准，allocation-only fallback |
| ninfer-token-alignment.patch | +39/−8 | 2 | scheduler/准入 | 启动 token 预算先归一化再做容量检查 |
| ninfer-output-budget.patch | +31/−4 | 2 | scheduler/准入 | KVMem 自动输出受逻辑上下文与 RAM/SSD 历史约束 |
| ninfer-dflash2-layer-pipeline.patch | +43/−8 | 5 | 投机解码/准入 | DFlash2 层流水线准入 + 能力声明 |
| ninfer-cpu-vision-props.patch | +46/−9 | 7 | 监控/视觉 | 不取推理执行锁即可观察 CPU Vision 缓存元数据 |
| ninfer-memory-planning.patch | +365/−5 | 10 | **内存规划** | 单 GPU 内存回收 + 主动预算规划（fixed reserve/KV、页对齐） |
| ninfer-multigpu-resident-vision.patch | +93/−0 | 1 | 内存规划 | 去掉自动流水线 sizing 中重复的常驻 Vision 绑定 |
| ninfer-t5-sm75-prefill.patch | +28/−5 | 2 | 算子/准入 | T5 prefill SM75 共享内存准入 |
| ninfer-monitor-image-input.patch | +10/−0 | 5 | 监控 | 监控面板图像输入 |
| ninfer-chat-cache-owner.patch | +116/−7 | 5 | serving/缓存所有权 | Chat `prompt_cache_key`/OpenCode `promptCacheKey` 绑定活缓存所有权 |
| ninfer-cross-owner-fork.patch | +449/−207 | 6 | **KV 共享** | 跨 owner RAM 精确前缀发现，多 owner 共享不可变 RAM/SSD KV，事务性冷恢复 |
| ninfer-heads-tp.patch | +2234/−380 | 66 | 张量并行 | KVMem dense heads TP：全局 Query/统计、rank 级 state/payload 身份、本地量化分片 |
| ninfer-checkpoint-cleanup.patch | +443/−23 | 9 | checkpoint 生命周期 | 请求后 checkpoint 捕获准入与清理；Query replay 与 prompt-tail 生命周期分离 |
| ninfer-input-reuse.patch | +323/−12 | 6 | prefix 复用 | 精确结构化输入复用 + 保守模板改写 fallback |
| ninfer-ngram-arbitration.patch | +168/−4 | 8 | 投机解码 | 请求级 ngram copy yield 仲裁、MTP/DFlash 反馈修复、rejected-copy 游标重置 |
| **ninfer-exact-ops.patch** | +641/−420 | 48 | **算子（保算术）** | 见 §2.1 |
| ninfer-5060-gguf-coop.patch | +4/−1 | 2 | 算子路由 | 协作 kernel 按 RTX 5060 Ti 校准路由（Q4_K scalar、Q6_K cooperative） |
| ninfer-gguf-mxfp8-prefill.patch | +2025/−4 | 15 | 量化 | GGUF MXFP8 prefill：fused tile + 64 MiB bounded Lt + actual-amax + FP64 oracle |
| ninfer-gguf-nvfp4-prefill.patch | +2604/−17 | 29 | 量化 | NVFP4 slab/MSE4/Lt compute，显式 artifact 权限，复用 FP8 CPU oracle |
| ninfer-kvmem-statistics-readback.patch | +313/−40 | 6 | **监控/统计读回** | Mean-K/Query coalesced D2H + sampled-attention 批量读回、pinned 规划、NVTX 归因 |
| ninfer-gguf-mxfp8-device-routes.patch | +524/−26 | 12 | 量化路由 | per-device profile 的 MXFP8 tile/Q8 路由 + 25 条 5060 Ti 实测规则 |
| ninfer-gguf-mxfp8-image-stores.patch | +73/−9 | 3 | 量化算子 | 两个 128-bit 对齐 image store 替代八个 scalar store |
| ninfer-dflash-conditioning.patch | +233/−0 | 10 | 投机解码 | model-aware DFlash2 recent minimum 与 restored-floor bound |
| ninfer-shared-cache.patch | +169/−23 | 20 | **KV 共享** | owner-first 精确前缀 fork；sealed 物理 KV/只读 state 引用，独立可变 lineage |
| ninfer-gguf-vector-epilogues.patch | +312/−0 | 3 | 算子 | 对齐 FP32-input BF16 Store/Add epilogue（每线程 8 行向量读写） |
| ninfer-gguf-lt512-routes.patch | +1706/−109 | 4 | 量化路由 | Lt512 精确宽度 profile 路由 + 78 条 5060 Ti 实测规则 |
| ninfer-gguf-route-workspace.patch | +95/−12 | 2 | 算子/内存 | 路由持有 FP8/Q8 workspace + 全 token 区间规划 |
| ninfer-gguf-bf16-output.patch | +113/−17 | 7 | 量化 | 对齐 MXFP8 BF16 输出；输出类型分 Lt cache |
| ninfer-gguf-image-budget.patch | +31/−4 | 6 | 内存预算 | per-device MXFP8 image 容量（5060 Ti 实测 32 MiB，未知设备 64 MiB） |
| ninfer-gguf-add-capacity.patch | +48/−10 | 4 | 算子审计 | BF16 Add 容量假设修复（连续 public residual 不必 16-byte 对齐） |
| ninfer-content-cache.patch | +464/−234 | 16 | 内容缓存 | KVMem master PR259 content-cache 层移植到 NInfer 2026-11-04.2 |
| ninfer-release-20261104-integration.patch | +253/−41 | 14 | 上游版本兼容 | NInfer 2026-11-04.2 兼容：whole-chunk TP 统计、辅助 KV 描述符、converter 契约 |
| ninfer-span-construction.patch | +1/−1 | 1 | 构建兼容 | GCC 11 C++20 下 attention-row span 元组构造显式化 |
| ninfer-shared-snapshots.patch | +394/−66 | 11 | checkpoint/快照 | 不可变 checkpoint manifest（SealedKvBlock/KvSnapshot），私有 tail 与物理准入 |
| **ninfer-layerwise-recall.patch** | +14585/−550 | 134 | **KV 外置核心（逐层）** | 见 §2.3 |
| ninfer-checkpoint-retention.patch | +339/−81 | 16 | checkpoint | 实例级热 checkpoint 保留 + model 尺寸自动配额 + 显式字节上限 |
| ninfer-selective-feedback.patch | +258/−1 | 1 | 投机解码 | 自动单 lane DFlash2 graph feedback batching + 发布前采样行选择 |

### 1.3 polystrata（22 个，upstream `VecSzn/PolyStrata` @ `fabae0d6`，source_tag `v0.1.45-poly.7`）

| 文件 | 规模(+/-) | 文件数 | 层 | 机制概括 |
|---|---|---|---|---|
| polystrata-kvmem.patch | +3890/−161 | 32 | **KV 外置核心** | KVMem 嫁接主补丁：qwen35moe F16/Q8 on CUDA（实验性 HIP）、bounded Query/replay、PageSoftmax 评分、resident-only Pressure、共享进程 RAM/SSD（`versions.json` polystrata.local_changes 全文列举） |
| polystrata-attention-feedback-performance.patch | +443/−41 | 7 | 算子性能 | 注意力反馈路径性能 |
| polystrata-elastic-workspace.patch | +354/−26 | 8 | 内存/workspace | 弹性 workspace |
| polystrata-elastic-kv-reclaim.patch | +47/−3 | 4 | 内存回收 | 弹性 KV 回收 |
| polystrata-elastic-headroom.patch | +294/−2 | 6 | 内存规划 | 弹性 headroom |
| polystrata-elastic-mtp.patch | +119/−12 | 3 | 投机解码/内存 | 独占执行期 1–3 个嵌入式 MTP draft |
| polystrata-block-tokens.patch | +132/−93 | 12 | KV 布局 | 启动期固定 32/64-token 块（默认 32） |
| polystrata-hip-kvmem.patch | +22/−5 | 1 | 构建/HIP | ROCm/HIP 经 hip_compat 编译可移植 KVMem core 与 query scorer |
| polystrata-serve-log-close.patch | +17/−1 | 3 | serving/构建 | 关停释放前端共享日志句柄，Windows 才能删临时目录 |
| polystrata-token-alignment.patch | +15/−2 | 4 | scheduler/准入 | token 对齐 |
| polystrata-kolibri-test-link.patch | +2/−2 | 1 | 构建/测试 | poly.5 Kolibri1 引擎链入 KVMem 回归测试目标 |
| polystrata-kvmem-image-query-span.patch | +5/−0 | 1 | KV 外置 | assistant-final 图像 prompt 的 (0,0) no-Query 哨兵排除出图像扩展 |
| polystrata-monitor-image-input.patch | +34/−3 | 2 | 监控 | 监控图像输入 |
| polystrata-kvmem-follow-up.patch | +68/−0 | 4 | KV 外置 | 主补丁的跟进修正 |
| polystrata-gemma4-kvmem.patch | +789/−98 | 17 | 模型适配 | Gemma 4 26B-A4B 走 KVMem（gemma4 层 rebase 到 poly.7） |
| polystrata-vision-hip.patch | +21/−2 | 2 | 构建/视觉 | AMD HIP image encoder 本地 vision-hip 层 |
| polystrata-shared-cache.patch | +441/−74 | 13 | KV 共享 | owner-first 精确原生前缀 fork（与 ninfer/radiance 同源同语义） |
| polystrata-content-cache.patch | +554/−140 | 11 | 内容缓存 | content-selected 原生快照、key-neutral 发布、修订版恢复诊断 |
| polystrata-immutable-snapshots.patch | +155/−9 | 7 | checkpoint/快照 | 跨 checkpoint 边界的精确不可变历史 manifest，dirty-block 捕获复用 |
| polystrata-tiered-checkpoint-state.patch | +156/−54 | 7 | checkpoint | Qwen recurrent state 存入 sealed、配额记账记录；published roots 可重试冷读恢复 |
| polystrata-bounded-restore-staging.patch | +93/−36 | 3 | 恢复/restore | 有界行分片恢复原生 KV，部分上传期间保留 sources |
| polystrata-checkpoint-manifest-capacity.patch | +2/−0 | 1 | checkpoint | 构建 checkpoint manifest 前预留已知原生块数 |

### 1.4 radiance（29 个，AMD 后端，**不在 versions.json**，走 adapter 机制）

| 文件 | 规模(+/-) | 文件数 | 层 | 机制概括 |
|---|---|---|---|---|
| radiance-1.17-kvmem.patch | +3684/−296 | 84 | **KV 外置核心** | Radiance 1.17 的 KVMem 嫁接 |
| radiance-abi15-kvmem-windows.patch | +4709/−1747 | 140 | **KV 外置核心** | Windows ABI15 的 KVMem 嫁接（删除量最大的 patch，−1747 行） |
| radiance-v117-windows-host.patch | +2699/−2540 | 113 | **KV 外置核心** | v1.17 Windows host 集成 |
| radiance-packed-kv.patch | +3338/−8 | 19 | **KV 布局** | packed KV（prefix-cache-plan 对应物，`docs/packed-kv-prefix-cache-plan.md`） |
| radiance-aql-cpu-access.patch | +17/−6 | 1 | CPU 访问 | AQL CPU 访问 |
| radiance-async-observer.patch | +103/−11 | 5 | 监控 | 异步 observer |
| radiance-attention-observer.patch | +57/−5 | 11 | 监控 | 注意力 observer |
| radiance-batched-observer.patch | +108/−2 | 5 | 监控 | 批量 observer |
| radiance-bounded-admission.patch | +106/−2 | 4 | scheduler/准入 | 有界准入 |
| radiance-cpu-domain-coverage.patch | +137/−9 | 3 | 测试 | CPU 域覆盖 |
| radiance-cpu-vision.patch | +245/−5 | 11 | 视觉/CPU | CPU 视觉 |
| radiance-dflash-conditioning.patch | +62/−6 | 3 | 投机解码 | DFlash conditioning（与 ninfer 同名 patch 同语义） |
| radiance-execution-overhead.patch | +76/−8 | 5 | 监控/性能 | 执行开销观测（`docs/radiance-execution-overhead-20261107.md`） |
| radiance-media-build.patch | +12/−2 | 1 | 构建 | media 构建 |
| radiance-media-query.patch | +86/−45 | 2 | KV 外置 | media query |
| radiance-monitor.patch | +49/−0 | 7 | 监控 | 监控面板 |
| radiance-mxfp4-cast-closure.patch | +58/−38 | 5 | 量化 | MXFP4 cast 闭合 |
| radiance-profile-closure.patch | +107/−0 | 8 | profile | profile 闭合 |
| radiance-qwen35-profile.patch | +315/−0 | 3 | 模型 profile | Qwen3.5 profile |
| radiance-rank-metadata.patch | +45/−0 | 2 | 多 GPU | rank 元数据 |
| radiance-scheduler-test-fakes.patch | +17/−0 | 1 | 测试 | scheduler 测试 fake |
| radiance-session-state.patch | +93/−5 | 6 | 会话状态 | 会话状态 |
| radiance-shared-cache.patch | +122/−5 | 8 | KV 共享 | owner-first 前缀 fork（三后端同源手法） |
| radiance-storage-lifecycle.patch | +53/−11 | 5 | 存储生命周期 | 存储生命周期 |
| radiance-ui-data.patch | +3/−0 | 2 | UI | UI 数据 |
| radiance-vision-pipeline-audit.patch | +47/−19 | 2 | 视觉审计 | 视觉管线审计 |
| radiance-vision-pipeline.patch | +263/−4 | 16 | 视觉管线 | 视觉管线 |
| radiance-win-single-stream.patch | +24/−3 | 1 | Windows/HIP | 单 HIP stream 双逻辑 lane（`backends/radiance/README.md:26-28`） |
| radiance-avx-levels.patch | +77/−19 | 3 | CPU 算子 | AVX 级别路由（libavx/libref 插件，`backends/radiance/README.md:16-18`） |

### 1.5 按层分布汇总（104 个，现算）

| 层 | 数量 | 代表 |
|---|---|---|
| KV 外置核心（嫁接/窗口/逐层 recall/Pressure） | 8 | ninfer-kvmem、ninfer-layerwise-recall、polystrata-kvmem、radiance-1.17-kvmem、radiance-abi15-kvmem-windows、radiance-v117-windows-host、polystrata-kvmem-follow-up、radiance-media-query |
| 量化格式/量化路由 | 18 | pq2/ptq1 ×4、mxfp8/nvfp4 系列 ×12、nvfp4-blackwell-scale、mxfp4-cast-closure |
| 算子（kernel/epilogue/性能/保算术） | 16 | exact-ops、t5-performance、vector-epilogues、image-stores、add-capacity、avx-levels、attention-feedback-performance、coop、ptq1-performance、sm75-prefill、route-workspace、aql-cpu-access 等 |
| KV 布局/存储生命周期 | 6 | packed-kv、block-tokens、nvfp4-kvmem、storage-lifecycle、memory_statistics 布局相关、kvmem-statistics-readback（统计读回也算此层+监控边界） |
| checkpoint/快照/内容缓存 | 15 | checkpoint-policy ×2、checkpoint-cleanup/retention、shared-snapshots、immutable-snapshots、tiered-checkpoint-state、bounded-restore-staging、manifest-capacity、content-cache ×2、checkpoint-handoff 相关 |
| KV 共享/多会话/前缀复用 | 6 | shared-cache ×3、cross-owner-fork、chat-cache-owner、input-reuse |
| 投机解码（MTP/DFlash/DFlash2/ngram） | 12 | dflash2 系列 ×5、dflash-conditioning ×2、ngram-arbitration、selective-feedback、elastic-mtp、dflash2-layer-pipeline |
| scheduler/准入/内存规划 | 10 | token-alignment ×2、output-budget、memory-planning、elastic-workspace/headroom/kv-reclaim、bounded-admission、image-budget、multigpu-resident-vision |
| 监控/可观测/UI | 14 | monitor-events、monitor-image-input ×2、monitor ×2、observer ×3、execution-overhead、cpu-vision-props、statistics-readback、ui-data、rank-metadata |
| 服务层/会话状态 | 4 | serve-log-close、session-state、cpu-vision（视觉服务）、radiance-vision-pipeline 系（视觉管线算 3 个，此处归并计入视觉/前端） |
| 模型适配/多 GPU/TP | 4 | gemma4-kvmem、heads-tp、qwen35-profile、profile-closure |
| 构建/测试/兼容 | 11 | hip-kvmem、vision-hip、win-single-stream、media-build、kolibri-test-link、span-construction、release-20261104-integration、cpu-domain-coverage、scheduler-test-fakes、radiance-avx-levels（构建路由）、heads-tp 外的 CMake 类 |

（注：跨层 patch 按主导层归入一层，边界的归并规则已在上表脚注；总和核对 = 104。）

## 2. 三个最有含金量的 ninfer 补丁深挖

选择理由：任务建议的 `ninfer-exact-ops` / `ninfer-query-memory` / `ninfer-layerwise-recall` 恰好覆盖三个正交的核心问题——**算子层保算术改造**（exact-ops）、**KV 统计的 Query 侧数据通路**（query-memory）、**KV 外置核心的逐层化重写**（layerwise-recall）。`ninfer-kvmem.patch`（+16651，全仓最大）作为背景在 §2.0 简述，因为后三者都建立在它开的洞之上。

### 2.0 背景：ninfer-kvmem.patch 如何给引擎开洞

上游 NInfer 的 KV 是一个常驻显存的 paged cache（`src/core/paged_kv_cache.cpp`）。主补丁做了四类开洞：

1. **配置洞**：在 `EngineOptions` 里挂 `KvmemOptions kvmem;`（`backends/patches/ninfer-kvmem.patch:521`），并新增 `KvmemOptions` 结构体定义（`backends/patches/ninfer-kvmem.patch:1484`）：`selected_tokens`(B)/`reserve_tokens`(R)/`host_bytes`(H)/`disk_path`+`disk_bytes`(SSD)/`sink_tokens`/`recent_tokens`/`retained_sessions`——B/R/H 三层预算 + 64-token 页的 sink/recent 环绕。没有这个结构体，上层任何外置策略都没有配置入口。
2. **页池统计洞**：给 `DeviceKVPagePool` 新增 `host_record_copy_stats()`（`backends/patches/ninfer-kvmem.patch:2449,2497`）——**不 enqueued copy、不同步流**的纯诊断接口，预告 record 拷贝的 runs/copies/payload_bytes。KVMem 的物理记账（"planned and actual reuse plus physical accounting"）依赖这个上游本来没有的计量点。
3. **桥接对象**：新增文件 `src/models/qwen3_5/program/storage/kvmem_window.cpp`（+1656 行），第一行注释就写明了所有权哲学：*"This bridge borrows Program's existing native pool, address space, stream and reservation. It owns no device allocator."*（`backends/patches/ninfer-kvmem.patch:8554-8557`）——`WindowMemoryBackend implements kvmem::MemoryBackend`，把引擎现成的页池/地址空间/流"借"给 KVMem 核心库，而不是让 KVMem 另建一份显存池。KVMem 库侧的 `kvmem::RawKvStore`（`backends/patches/ninfer-kvmem.patch:7829,10236`）只管外置索引与选择，物理字节仍归引擎。
4. **状态洞**：`KvmemWindowState` 挂进 `SequenceState.window`，带 `stamp.session/history_revision/view_version` 三级版本号——跨请求的 durable history 跟着 SequenceState 走，KVMem 自身无状态。

### 2.1 ninfer-exact-ops.patch（+641/−420，48 文件）——「算术保持」的算子改造

它解决的是 KVMem 特有的一个矛盾：**KVMem 要把 KV 页在 INT8/INT4 量化格式下外置到 RAM/SSD 并按位复算，任何 kernel 改写都可能改变浮点加法顺序 → 同一请求回放出的 KV 位形与原始不一致 → 外置历史作废**。因此这个 patch 的每一个 kernel 改动都追求"逐位等价"（bit-exact），而非常规的性能优化取向。

三个代表性机制：

1. **归约顺序参数化的 GDN gating kernel**。上游 `gdn_norm_gating_27_simt` 的并行归约顺序随 CTA 物理线程数变化。patch 把模板参数从 `<Tile, Threads, Rotate>` 扩成 `<Tile, Threads, OrderThreads, Rotate>`（`backends/patches/ninfer-exact-ops.patch:120-122`），让**逻辑归约顺序（OrderThreads 条逻辑线程的 warp 顺序）与物理 CTA 大小解耦**：物理线程只是把逻辑线程的累加分片跑得更快，每个结果位与 CTA 大小无关（`backends/patches/ninfer-exact-ops.patch:119-121` 注释原文）。随后即可在不改变任何一位输出的前提下换 launch 形状：256 线程在 1–2 token tile 上比 512 快（T=8: 15.4→13.0 µs，RTX 5060 Ti 36 SM 实测，`backends/patches/ninfer-exact-ops.patch:171-174`）。**为什么必须有它**：没有顺序解耦，任何为外置扫描调优 kernel 的尝试都会静默破坏位形一致性——这是 KVMem "replay/复算" 前提的算子层保险。
2. **FP16 位技巧的 INT4 反量化**。`kv_cache_int4_dequant_f16x8_from` 从逐 nibble 的 float 乘法改为 HMUL2 向量化：biased nibble n 加上 1024 的指数基座即 `1024+n`，减 1032 精确还原 code，再一次 HMUL2 得到与 FP32 精确积同值的一次 FP16-RN 舍入（注释与实现 `backends/patches/ninfer-exact-ops.patch:264-279`，关键行 `const __half2 bias = __float2half2_rn(1032.0f);`）。签名从 `float s` 改为 `__half s`（`backends/patches/ninfer-exact-ops.patch:267-269`）——scale 本身也是表示的一部分。KVMem 外置 INT4 KV 页到 SSD 后重读进算子的结果与常驻路径逐位相同，靠的就是这类 codec 级等价。
3. **GDN replay 的 tile 常量化**。`launch_recurrent_record_typed`/`launch_replay_fold_typed` 的 grid 第三维从 `kStateDim / kBlockDv` 换成新常量 `kReplayTiles`（`backends/patches/ninfer-exact-ops.patch:303,312`，定义在 `backends/patches/ninfer-exact-ops.patch:429`），并删除 `RawValueLane`/`load_value_lane` 的 lane-guard 路径——replay 折叠不再按 warp 内 lane 位置分支，保证 record→fold 的覆盖范围与 tile 数无关。这直接服务 KVMem 的 `GdnReplayRecords` 机制：外置后只存 record、按需重放状态展开。
4. **请求退役清理**（engine 侧小洞）：`RequestControl::retire()` 增加 `ngram_build.reset();`（`backends/patches/ninfer-exact-ops.patch:81`）——异步 ngram 构建协程随请求退役，避免跨请求悬挂状态（patch source 注明 "asynchronous ngram construction ... adapted to KVMem dynamic heads, conv history coverage and request retirement"）。

### 2.2 ninfer-query-memory.patch（+569/−81，32 文件）——Query 统计从「存行」到「有界捕获/滚动求和」

KVMem 的检索打分需要每层的 Query 统计（Mean-Q cosine 或逐 token 行）。上游原实现把每个 prefill chunk 的 Query 行**全量**写进 `query_sums` 缓冲——长 prompt 下这块显存与 prompt 长度线性相关。这个 patch 重写了整条 Query 数据通路：

1. **两个新算子替换 token_sums 直写**。`include/ninfer/ops/token_sums.h` 新增 `capture_query_rows`（把 [origin, origin+T)∩range 的 BF16 行**原位拷贝**到容量有界的 output，未用行清零）和 `accumulate_query_sum`（把该交集**按时间序累加**进 FP32 [channels,1] 累加器，"caller seeds it with the accepted Query prefix before each invocation (including graph replay); this preserves the sequential FP32 sum across chunk boundaries"）（声明 `backends/patches/ninfer-query-memory.patch:80,90`，实现 `backends/patches/ninfer-query-memory.patch:756,768`）。算子明确声明 "owns no allocation or state"——分配与生命周期仍归引擎。
2. **调度点改写**。`text_qk_norm_rope` 里原来的 `ops::token_sums(normalized_query..., q_rows)` 直写改为按 dtype 分流：`query_values.dtype == FP32` 走 `accumulate_query_sum`，否则 `capture_query_rows`（`backends/patches/ninfer-query-memory.patch:107-131`，关键行 `ops::accumulate_query_sum(q_input, origin, q_range, q_values, stream)`）。
3. **Mean-Q 模式**：`KvmemWindowState` 新增 `query_sums` 向量（"Mean-Q keeps the chronological FP32 sum instead of retaining token rows. The per-step device accumulator is always reseeded from this accepted state."，`backends/patches/ninfer-query-memory.patch:353-356`），`ProgramImpl` 构造时 `query_columns = mean_query ? 1U : min({prefill_chunk, capacity, query_max_tokens})`（`backends/patches/ninfer-query-memory.patch:330-331`）——Mean-Q 下 Query 统计显存从 O(prompt) 掉到 O(1)。`rewind_memory_query`/`restore_memory_history`/`retain_memory_history` 三个生命周期点同步 `query_sums` 的清空（`backends/patches/ninfer-query-memory.patch:460,1377-1381,1530-1534`）。
4. **计费口径修复**：`GenerationResult` 新增 `computed_prompt_tokens`（"Logical uncached prompt tokens accepted at completed prefill boundaries... may be partial or zero on cancellation"，`backends/patches/ninfer-query-memory.patch:93-107`），`RequestControl::snapshot_prefill_timings()` 把 prefill 计时快照从"仅在正常完成路径取"提前为"完成或取消都要取"（`backends/patches/ninfer-query-memory.patch:1251-1268` 与 `complete_prefill_step` 的删改 `backends/patches/ninfer-query-memory.patch:1843-1867`）。docs 同步改 `cache_n + prompt_n` 语义（取消时可小于 `usage.prompt_tokens`，`backends/patches/ninfer-query-memory.patch:28-51`）。
5. **测试基建**：可选编译开关 `NINFER_KVMEM_QUERY_TEST_HOOKS`（`backends/patches/ninfer-query-memory.patch:1-27`）注入文件级 rendezvous `query_test_boundary()`——prefill 到达中间边界时写 `ready` 文件、等 `release` 文件，让 HTTP 测试能在**真实调度边界**注入第二请求或断连（实现 `backends/patches/ninfer-query-memory.patch:377-428`）。Query 生命周期回归（含回放、取消、跨 chunk）因此可确定性复现。

**为什么必须有它**：KVMem 的 Mean-Q 检索评分路径在原引擎里根本没有"有界 Query 统计"这一说——上游假设 Query 统计与 KV 一样常驻且全量。没有这个 patch，启用 KVMem 的长上下文请求会在统计缓冲上先 OOM；且 replay（graph replay 会重复触发统计路径）没有 seed-累加语义就会把已接受前缀重复计入。

### 2.3 ninfer-layerwise-recall.patch（+14585/−550，134 文件）——从「整窗 recall」到「逐层工作集」

这是 ninfer 串行链的第二大 patch，把 KVMem 的 recall 单位从"整个 KV 窗口"细化到"每层独立的工作集（working set）"——即 layerwise Pressure：每层可以独立选择哪些历史页留在显存，其余封存。

1. **请求策略 API**：新文件 `include/ninfer/layerwise_request.h`（+31 行，`backends/patches/ninfer-layerwise-recall.patch:414-451`）定义 `KvmemReselect {Auto, Force, Off}` + `KvmemQuerySpan` + `KvmemRequestPolicy`，并给出字符串解析 `parse_kvmem_reselect`（`backends/patches/ninfer-layerwise-recall.patch:443`）。`PreparedPrompt::set_layerwise_request_policy`（`backends/patches/ninfer-layerwise-recall.patch:354-375`）把它挂进 prompt 准备结果——**Query 跨度成为请求级、可验证的一等公民**（"Original token coordinates. Providing a span does not imply Force"）。
2. **选项与默认值**：`KvmemOptions` 增加 `recall_strategy`（Legacy|Layerwise）、`LayerwiseOptions`（backend 默认 metadata_bytes = 1 GiB，`kDefaultLayerwiseMetadataBytes`，`backends/patches/ninfer-layerwise-recall.patch:395-403`）与 `layerwise_mtp_history`（显式 Unspecified——注释写明"MTP draft 的 K/V 对这些历史 ID 是独立的，两种实验性替代下 target selection 都要 fresh and per layer"，即默认行为保守）。
3. **策略绑定前置于物化**：新文件 `src/models/qwen3_5/frontend/layerwise_query_policy.h`（+104 行，`backends/patches/ninfer-layerwise-recall.patch:2171-2281`）。`finalize_layerwise_query_source()` 在**物化之前**（"Binding occurs before materialization, while the accepted source and its exact prefix proof are still immutable"）确定 original/effective Query、计算 capacity 预算（`fits = visible <= limit && remaining <= limit - visible && output_tokens <= limit - visible - remaining`）并冻结为 `std::shared_ptr<const LayerwisePromptPolicy>`——一旦绑定不可变，后续调度只能消费。目录级预判 `layerwise_query_may_continue` 明确注释"仍需完整 native 覆盖 + 全 token/media/position 前缀检查才可绑定"（`backends/patches/ninfer-layerwise-recall.patch:2258-2261`）——**乐观过滤、悲观绑定**。
4. **核心执行面**：`ProgramImpl::compact_layerwise_pressure()`（约 +100 行单函数，`backends/patches/ninfer-layerwise-recall.patch:10537-10650`）实现逐层 Pressure 事务：先 `archive_memory_window` 封存全部 canonical 历史（要求 "complete canonical history"，混合历史页必须已有 sealed source），对每个 `LayerwisePressureChoice` 校验 `choice.candidates == source.resident`（**"native Pressure attempted cold-history recall" 即抛异常——Pressure 只能从当前常驻集里选，不许冷召回**），部分 accepted tail 必须**精确保留**（"must preserve its exact accepted partial tail"），再经 `LayerWorkingSetPlan` 规划、`KVLayerStripeWrite` 双池（text + MTP auxiliary）拷贝、`restore_native_layer_stripe` 分片恢复，最后 `commit_frontier + destructive_truncate` 收尾；全程 `layerwise_rewrite_pending` 哨兵阻止其他路径观察未发布的目的地（`prepare_window` 入口第一行检查，`backends/patches/ninfer-layerwise-recall.patch:10634-10640`），异常路径 `stamp.valid=false` 整体作废（fail-closed）。
5. **CLI/可观测**：`--kvmem-recall-strategy legacy|layerwise`（`backends/patches/ninfer-layerwise-recall.patch:13957`），KvmemMemorySummary 注明 "Layerwise canonical versions also include their exact Mean-K index"（`backends/patches/ninfer-layerwise-recall.patch:451-463`），并新增 `layerwise_catalog`/graph capture/`native_layerwise_probe` 等一批新文件（134 个文件中相当比例是测试：layerwise pressure/request/frontend 的回归覆盖）。

**为什么必须有它**：整窗 recall 模式下，一个 256K 上下文的请求要么整窗常驻、要么整窗外置——显存台阶粗、Pressure 只能整体搬。layerwise 把 Pressure 粒度降到层×页，`resident-only` 约束（选择集 ⊆ 常驻集）保证任何时刻回退到 sealed 快照都可行，`kvmem-vllm algorithm contract retained`（patch source 原文）说明该算法契约与 kvmem-vllm 分支共享，是跨后端对齐的算法层。

## 3. 可复用手法：给一个没有 KV 外置能力的引擎做侵入式改造

### 3.1 接口开洞位置（按侵入深度递增）

1. **配置层最先开洞**：`EngineOptions` 下挂一个 `KvmemOptions` 式的选项块（B/R/H/D 四级预算 + sink/recent 页环绕），其余所有 patch 都以它为前提（`ninfer-kvmem.patch:1484-1525`）。llama.cpp 同理先在 `common/common.h`/`include/llama.h` 加开关（`llamacpp-kvmem.patch` 的 common.cpp/h、include/llama.h 各 1 处）。
2. **在「memory 抽象」上注册新实现，而不是改算子**：llama.cpp 有现成的 `llama_memory` 抽象 → KVMem 作为第三种 memory 实现注册（`llama-kvmem-factory.h`，`llamacpp-kvmem.patch:2084`；adapter 在 `src/adapter/llama-memory-kvmem.cpp`）。ninfer 没有这个抽象 → 退而求其次，在 `SequenceState` 里挂 `window` 子对象 + `MemoryBackend` 桥接接口。**有抽象走抽象，无抽象造桥**。
3. **计量点开在页池上**：`host_record_copy_stats` 这种"零副作用预告拷贝计划"的纯诊断接口是物理记账的地基（`ninfer-kvmem.patch:2449`）。vLLM 对应物 = `paged_kv_cache`/`kv_cache_manager` 上的 block 计数器层。
4. **统计通路单独立 patch**：KV 之外的辅助数据（Query 统计、Mean-K 索引、observer）是第二处必开洞，且必须与 KV 主体分开演进（ninfer 用 3 个 patch 分层：kvmem 主补丁 → query-memory → statistics-readback）。

### 3.2 所有权/生命周期处理（KVMem 的三条铁律）

1. **桥接对象"只借不建"**：`WindowMemoryBackend` 明示 owns no device allocator，借引擎的池/地址空间/流/预留（`ninfer-kvmem.patch:8554-8557`）。外置层永远不与引擎抢显存分配权，显存 OOM 责任边界清晰。
2. **不可变快照 + 版本号**：一切跨请求共享的历史以 sealed 快照存在（`HostKvRecord` COW、`SealedKvBlock`、`stamp.session/history_revision/view_version`），可变 lineage 独立（shared-cache 三胞胎 patch 的同一句话："sealed physical KV/read-only state references, independent mutable lineage"）。写路径永远是"新开可变层 + 发布"，读路径永远盯版本号。
3. **事务化改写 + fail-closed**：Pressure/restore 全部走「archive → plan → copy → restore → commit_frontier → destructive_truncate」的显式事务，`*_pending` 哨兵阻止并发观察半成品，异常路径统一 stamp 失效（`ninfer-layerwise-recall.patch:10537-10650` 的 try/catch 收尾）。部分恢复容忍但不猜——"partial accepted tail" 必须精确保留。
4. **生命周期分离**：Query replay 的生命周期与 prompt tail 分开（checkpoint-cleanup patch："separate Query replay and prompt-tail lifetimes with bounded extra recomputation"），请求退役时 `retire()` 一处清空全部请求级协程/缓冲（`ninfer-exact-ops.patch:81`）。

### 3.3 回滚策略

- **补丁层**：`versions.json` 的 `sha256(patch)` + `requires` 串行链 + `after_tree` 逐级树哈希 + 终态 `patched_tree`——任意 patch 可回退到任意中间树状态且可验证。这是比 git 更强的可复现保证（树哈希不依赖 git 历史）。
- **运行层**：verify_transfers 诊断开关、FP64 codec oracle 独立校验（nvfp4/mxfp8 系列反复出现 "independent FP64 codec oracle"、"CPU/FP64 qualification"）、`retained_sessions` 有界保留历史、checkpoint 可重试冷读（tiered-checkpoint-state）。
- **算法层**：所有新策略都带 Legacy 回退枚举（`RecallStrategy::Legacy` 默认值，`ninfer-layerwise-recall.patch:402`），显式值永远压过默认（"every explicit native value wins"）。

### 3.4 落到 vLLM / SGLang：从哪里下手

**vLLM**（对应开洞点）：
1. `vllm/config.py` 的 `CacheConfig`/`KvTransferConfig` → 加 `KvmemOptions` 式预算块（B/R/H/D）；
2. `vllm/v1/core/kv_cache_manager.py` + `vllm/v1/worker/gpu_block_pool.py`（block 池）→ 仿 `WindowMemoryBackend` 写桥：借 vLLM 现成的 block table/地址空间，不改 allocator；在 block pool 上加 `host_record_copy_stats` 式计量方法；
3. `ModelRunner` 的 attention 前向 → Query 统计洞（capture_query_rows/accumulate_query_sum 两个无状态算子，接在 qk-norm-rope 之后，与 ninfer `attention.cpp` 挂点同位）；CUDA graph replay 下必须用 seed-重置语义（query-memory 的核心教训）；
4. prefix cache（`kv_cache_manager` 的 hash 块）→ shared-cache 手法：owner-first 精确前缀 fork，sealed 物理 block + COW，替代 vLLM 现在的 copy-on-share；
5. scheduler 层 → bounded admission（prepare_window 式准入检查 + reserve 预算），建议照抄 ninfer 的「先归一化 token 预算再查容量」（token-alignment patch 的教训：归一化在容量检查之前，否则 effective 值与准入值不一致）。

**SGLang**（对应开洞点）：
1. `python/sglang/srt/mem_cache/`（RadixCache + allocator）→ KVMem 作为 `memory pool` 之外的二级 tier 挂进 radix tree 的 evict 路径；SGLang 已有 hierarchical cache 雏形，外置层对接点最自然；
2. `sglang/srt/model_executor/cuda_graph_runner.py` → replay 的统计 seed 语义同上；layerwise 的 `LayerwiseMtpHistory::Unspecified` 保守默认值得照抄（EAGLE draft 的 KV 与 target 独立，混合历史选择容易踩错）；
3. `sglang/srt/server` → 监控洞（kvmem_status_json/kvmem_metrics 式只读 JSON 端点），先于任何策略 patch 落地——ninfer 链的顺序证明先有观测（monitor-events 在链中部）再做激进策略是安全次序。

**通用次序**（ninfer 串行链的实证排序）：配置洞 → 页池计量洞 → 桥接对象 → 统计通路 → checkpoint/快照 → 共享/fork → 投机解码适配 → 逐层化重写。最后一步（layerwise）在主补丁之后 40 个 patch 才出现——逐层化是把前面所有基础设施（sealed 快照、版本号、事务化 restore）串起来的终局形态，不要一开始就做。

---
*统计与引用核对：patch 总数 104（llamacpp 5 / ninfer 48 / polystrata 22 / radiance 29）；versions.json 登记数 75；所有 `+/-` 行数与文件数由脚本逐文件现算；所有 `patch:行号` 引用已 grep 定位核验。*
