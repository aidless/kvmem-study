# KVMem Radiance 后端深挖：AMD RDNA4/ROCm 路线与「跨 ISA 后端接入需要什么」

> 调查日期：2026-11-10。只读分析 KVMem 仓库（`kvmem-study/kvmem`），所有论断带相对仓库根 `path:line` 或 URL；查不到的写 UNKNOWN。

## 1. Radiance 是什么

**上游仓库**：[StillDeadcode/radiance（Codeberg）](https://codeberg.org/StillDeadcode/radiance)，官网 [radiance.stilldeadco.de](https://radiance.stilldeadco.de/)。定位：**面向 AMD GPU 的独立开源推理引擎**，C++/HIP 编写，服务语言模型、图像/视频/语音/音乐生成，HTTP API 兼容 OpenAI/Anthropic/llama-server 端点（来源：Codeberg 仓库 README 摘要，web 搜索 2026-11-10）。

**它不是 fork，也不从属于 KVMem 生态**：KVMem 通过固定上游 archive + 补丁链的方式接入，而非 fork：

- 固定版本：v1.17.0 / commit `19d4e8fe0e18071b9b6300f15fc4ccf38dc4fa3d`，archive SHA-256 `40062b03…f8abbd1`（`backends/radiance/upstream.json`）
- ABI 门：上游 base ABI 35，KVMem ABI `1263927331`（`backends/radiance/upstream.json`；历史 ABI15 代为 `1263927297`）
- 许可：Apache-2.0，保留上游归属（`backends/radiance/UPSTREAM_NOTICE`、`docs/radiance-integration.md:197`）
- 构建方式：`python backends/radiance/tools/prepare_native.py` 拉取固定上游 → 打补丁 → 构建 `radiance-kvmem-server` worker（`docs/radiance-integration.md:130-134`）

**集成形态**：三层结构——
1. Radiance 引擎原生侧（补丁注入 hooks + `core/kvmem/` 新目录，如 `radiance-shared-cache.patch` 新增 `core/kvmem/cache_owner.hpp`）；
2. KVMem 公共 C++17 契约核心（`kvmem/include/kvmem/memory_contract.hpp`，`docs/memory-backend-contract.md:5`）；
3. KVMem 侧适配器 `backends/radiance/adapter/`（45+ 文件），实现 `MemoryBackend` 接口并对接 Radiance 引擎的 `rad::KVManager` / `core/mem/kv.h`。

## 2. Radiance 侧 KV/state 模型，与 NInfer 的差异

| 维度 | Radiance（AMD/ROCm） | NInfer（NVIDIA/CUDA） |
|---|---|---|
| 模型容器 | 自有 `.rad` 格式（`qwen3.8-27b-iq3r-s-v4.rad`，13.20 GiB，含视觉塔与 DFlash2 载荷，`docs/radiance-iqr-v4-integration-20261105.md:16-25`） | GGUF 制品（`docs/multi-backend-differences.md:8` 上下文） |
| KV 编码 | 引擎原生 FP8 KV（默认 fp8/f32；packed backport 增 `fp8_v4`/`k6_v4`），几何 head_dim 256、GQA 6、**原生 block 16 tokens**（`backends/radiance/upstream.json` `packed_kv_backport`） | BF16/INT8 native payload + 独立 scale planes；**页 64 tokens**、ninfer 检索块固定 64（`docs/multi-backend-differences.md` 粒度约定表） |
| Recurrent state | F32（或 f16）状态，GDN materialize 走引擎导出函数，核对 producer ABI/layout（`backends/radiance/adapter/state_materialization.hpp:10-31`） | GDN StateImage 精确 checkpoint + query 重放（`docs/memory-backend-contract.md:210,272`） |
| 草稿 | 内嵌 MTP（共享 packed 主 KV 组）与内嵌 DFlash2（独立 FP8 draft cache，训练深度 7），启动前核对容器草稿类型/深度（`docs/radiance-iqr-v4-integration-20261105.md:19,27`） | MTP 独立原生页池 + ngram（`docs/multi-backend-differences.md`） |
| 多卡并行 | **原生张量并行**：每卡持连续 head shard，`--tp-wire exact`，Query/KV 头数必须能均分；无 P2P 时集合通信走主机暂存（`docs/radiance-integration.md:29-37`） | **按层切分流水线**：一逻辑历史、一选页结果、一份 H 预算，各卡持自己层的 KV，MTP 池在 rank 0，不引入 TP/NCCL（`docs/ninfer-multigpu-validation-20261006.md` 实现来源节） |
| 统计捕获 | 每 rank 独立捕获 mean-K/Q，按头拼接、头加权 attention 合并；单 rank 可直接消费（`docs/radiance-adapter-staging-20261107.md:20-23`） | 按层 owner 的 mean-K/Q、candidate keys，Host 按全局 Attention 层顺序收集（`docs/ninfer-multigpu-validation-20261006.md` 适配段） |

两侧共用同一公共策略：Query scorer（page-softmax / meanq-cosine）、图/LRFU、Pressure 只从驻留历史选块、工具续接不做冷召回——这是公共契约层的收益（`docs/radiance-integration.md:11-12` 与 `docs/multi-backend-differences.md` 工作集表逐行一致）。

## 3. Radiance 接入改了什么：补丁清点（按层分类）

补丁目录 `backends/patches/radiance-*.patch` 共 33 个，分三代。当前 1.17 资格链 7 项 + 后续追加，历史 ABI15 链 17 项 + 主集成，全部 SHA-256 记录在 `backends/radiance/upstream.json`。

**A. 主集成（引擎级，每代一个）**
- `radiance-1.17-kvmem.patch`（6780 行）：1.17 主集成——CMake、ABI、调度器/观察者 hooks、KVMem 接入点（upstream.json `native_patch`）
- `radiance-abi15-kvmem-windows.patch`（14110 行）：ABI15 代主集成，含 Windows 构建路径
- `radiance-v117-windows-host.patch`（13885 行）：1.17 的 Windows host 构建补齐

**B. KV/state 与调度语义（核心接入逻辑）**
- `radiance-packed-kv.patch`（3555 行，改 `abi/rad_abi.h`）：packed KV 格式 backport（fp8_v4/k6_v4），MTP 共享 packed 主 KV 组（upstream.json `packed_kv_backport.scope`）
- `radiance-session-state.patch`：会话/recurrent 状态在 step_observer 的挂钩
- `radiance-storage-lifecycle.patch`：存储生命周期（capture/park/restore）挂钩
- `radiance-rank-metadata.patch`（`core/mem/kv.h`）：rank 元数据，支撑 TP 每 rank KV 组
- `radiance-bounded-admission.patch`（`core/sched/scheduler.h`）：物理 KV 池之外引入独立逻辑上下文 admission（12K 验证中修的 sampling 容量缺陷即在此层，`docs/radiance-iqr-v4-integration-20261105.md:97`）
- `radiance-dflash-conditioning.patch`（`core/mem/kv.h`）：DFlash 窗口保留——保存/恢复接受位置的实际草稿 KV，替代「回放后 epoch 失效重累计」旧策略（`docs/radiance-iqr-v4-integration-20261105.md:29`）
- `radiance-attention-observer.patch`（`core/runtime/ctx.h`）：attention 捕获挂钩（accepted attention 统计来源）

**C. 观察/监控/共享缓存（KVMem 侧数据通路）**
- `radiance-async-observer.patch`、`radiance-batched-observer.patch`、`radiance-monitor.patch`（均在 `core/runtime/step_observer.h`）：异步/批量统计观察与监控面
- `radiance-shared-cache.patch`：新增 `core/kvmem/cache_owner.hpp`，同进程不可变 host/SSD 前缀共享
- `radiance-ui-data.patch`（`core/server/sink.h`）：UI/监控数据导出
- `radiance-media-query.patch`（`core/server/query_span.h`）：媒体请求的 Query span 边界

**D. 视觉/媒体**
- `radiance-cpu-vision.patch`、`radiance-vision-pipeline.patch`、`radiance-vision-pipeline-audit.patch`、`radiance-cpu-image-*`：CPU 视觉编码器与异步编码管线（`docs/radiance-integration.md:43-57`）
- `radiance-media-build.patch`：FFmpeg 媒体依赖构建开关

**E. 构建/内核 profile/平台**
- `radiance-qwen35-profile.patch`、`radiance-profile-closure.patch`（`libr4d/CMakeLists.txt`）：TP1 dense 内核闭包（46 HIP units / 159 rows，`backends/radiance/upstream.json` `focused_kernel_profile`）
- `radiance-avx-levels.patch`（`libavx/`）：CPU AVX 部署层级（level 2 = AVX2/FMA/F16C）
- `radiance-mxfp4-cast-closure.patch`：MXFP4 cast 内核闭包
- `radiance-cpu-domain-coverage.patch`（`core/build/rad_builder.cpp`）：CPU domain 用例覆盖
- `radiance-aql-cpu-access.patch`（33 行，`core/device/aql.cpp`）：**HSA 不能确认 CPU 可访问显存池时，AQL 内核参数环/录制参数改走一致性主机内存**（`docs/radiance-integration.md:40-41`）——ROCm 特有的内存可访问性问题
- `radiance-win-single-stream.patch`（`core/runtime/ctx.cpp`）：Windows 单 stream 约束
- `radiance-scheduler-test-fakes.patch`（`tests/`）：调度器测试替身
- `radiance-execution-overhead.patch`（`core/engine.cpp`）：tape 重放/完成事件/borrowed spill 执行开销优化

## 4. 跨 ISA 视角：同一契约在 CUDA 与 ROCm 两侧的实现差异

### 4.1 硬件无关的部分（公共层已经解决）

`memory-backend-contract.md` 的核心刻意做成 **无引擎、无 GPU runtime、无 Tensor 类型**（`docs/memory-backend-contract.md:5`）：`MemoryBackend`/`MemoryTransfer` 全部必需操作为纯虚函数（`:199`），且明确「公共核心不持有 CUDA stream/event」（`:319`）。因此以下全部硬件无关，两侧零改动共用：

- 三坐标模型（原始 token 位置 / compact 地址 / 物理页，`:77-89`）
- `HostKvStorage`/`SpillFile` RAM/SSD 分层、配额 H/D、读租约、`seal()`、fork 语义（`:9-56`）
- prepare → transfer → publish / abort 事务状态机（`:212-237`）
- 前缀 checkpoint、LCP、内容快照目录、LRFU（`:93-163, 327-335`）
- 不透明字节 payload + `PayloadLayout` 平面描述——公共层不解释量化编码、不重新量化（`:193, 245`）

「新推理后端的最小适配步骤」六步（`docs/active-kv-tiering.md:63-76`）全部落在这一侧：描述原生 KV 布局 → 建 `HostKvStorage` → lease/source 规则 → 引擎安全边界接 prepare/wait/publish → 闲置历史同记录移动 + 错误映射 → RAM 先行、SSD/取消/满盘/热冷并发验证。**这六步对第三种 GPU 生态照抄即可，一行 GPU 代码都不用写。**

### 4.2 AMD 侧必须重做的部分（Radiance 实测经验）

1. **设备 API 桥**：CUDA→HIP。适配器只在两处接触设备 API：公共页 scorer 经私有 shim（`backends/radiance/adapter/hip_compat/cuda_runtime.h:1-27`，`cuda*` → `hip*` 宏映射 + `__shfl_down_sync` 重定义），页搬运直接用引擎自己的 `RadStream`——**明确不用 cast 的 HIP stream**（`backends/radiance/adapter/page_transfer.hpp:15-22`：「Uses Radiance streams, NOT casted HIP streams」）。契约要求引擎是唯一的设备分配器/stream/event/graph 所有者（`docs/memory-backend-contract.md:194`），所以桥只需透传引擎已有的 stream 句柄。
2. **FP8 KV / F32 state**：编码留在引擎，公共层只见字节平面。Radiance 的 FP8 是引擎原生格式（fp8_v4/k6_v4 packed backport），GDN 状态 materialize 通过引擎导出函数 `GdnMaterialize` 完成，并在恢复前核对 producer 的 ABI、layout 和导出函数（`backends/radiance/adapter/state_materialization.hpp:10,29`）。NInfer 侧对应物是 BF16/INT8 scale planes + StateImage。**跨 ISA 结论：量化/状态编码永远不进公共层，接新卡只需让引擎提供「导出 accepted state / 导出 KV 平面」的稳定 ABI。**
3. **Stream/执行生命周期**：ROCm 无 CUDA graph，KVMem 侧改用引擎 tape 重放 + 完成事件（`radiance-execution-overhead.patch`，`KVMEM_RAD_TAPE_MODE`），并修 Windows 单 stream 约束（`radiance-win-single-stream.patch`）。D2H 借出流必须属于设备且活过 drain（`page_transfer.hpp:19-22`）。
4. **内存可访问性**：RDNA4 特有坑——显存池 CPU 可访问性需 HSA 确认，不可确认时内核参数缓冲退回一致性主机内存（`radiance-aql-cpu-access.patch`，`docs/radiance-integration.md:40-41`）；Windows 还有 GPU 地址位宽修复（`docs/radiance-integration.md:91`）。NVIDIA 侧没有这一类问题。
5. **TP/并发模型**：Radiance 每 rank 一个 `MemoryBackend`，`RankMemory` 聚合多 rank（`backends/radiance/adapter/rank_memory.hpp:7-13`：「One logical view backed by consecutive head shards」），接受统计按头拼接、attention 头加权合并（`docs/radiance-adapter-staging-20261107.md:20-23`）；集合通信 `--tp-wire exact`，无 P2P 时走主机暂存。NInfer 则是按层 owner。**两种模式都在公共契约上成立，第三生态二选一即可，`RankMemory` 聚合模式是硬件无关的纯 host 代码。**
6. **异步 staging 纪律**：checkpoint KV H2D staging 封顶 16 MiB、复用一对 allocation/event（`docs/radiance-adapter-staging-20261107.md:33-45`）；spill 借出 pinned 源给 CPU 回调，CPU worker 零设备操作（`:27-31`）。这些是性能层，非契约硬性要求，但直接复用了「公共层持有引用、引擎持设备」的边界。

### 4.3 接第三种 GPU 生态（国产卡/其它 ROCm 卡）的最小适配面清单

对照 `docs/active-kv-tiering.md:63-76` 六步 + Radiance 实测增量：

| # | 工作 | 层 | 硬件相关？ |
|---|---|---|---|
| 1 | 描述引擎原生 KV 层/页/plane/stride/有效 token + 量化附属 | 适配器（纯 host） | 否 |
| 2 | 建 `HostKvStorage`，捕获后 `seal()` | 公共层现成 | 否 |
| 3 | lease/source/写新记录规则 | 公共层现成 | 否 |
| 4 | 引擎安全边界接 prepare/wait/publish，绑 generation+内容版本 | 适配器 | 否 |
| 5 | 闲置历史同记录移动；`BudgetExceeded`/`TransferFailed` 映射 | 适配器 | 否 |
| 6 | RAM 回归 → 强制 SSD/闲置恢复/取消/满盘/热冷并发 | 验证流程 | 否 |
| 7 | **设备 API 桥**：stream/event/async copy 透传引擎自有句柄（HIP/厂商 SDK） | 适配器 | **是** |
| 8 | **页搬运**：D2H/H2D + pinned staging 上限（Radiance 用 16 MiB） | 适配器 | **是** |
| 9 | **统计捕获**：accepted 边界的 mean-K/Q，GQA 组 max；TP 时按 rank 拼接 | 适配器+引擎 hooks | **是** |
| 10 | **状态导出 ABI**：GDN/recurrent state materialize/restore，核 producer ABI+layout | 引擎侧 | **是** |
| 11 | **调度 hooks**：step observer、逻辑 admission 与物理池分离（bounded-admission 模式） | 引擎补丁 | 部分 |
| 12 | **平台坑位**：CPU 可访问显存确认/回退、地址位宽、单 stream 约束、编译器差异 | 引擎/构建 | **是** |

**结论**：契约面（六步）是纯 host、零 GPU 代码；真正的跨 ISA 增量只有 #7–#12 六项，其中最重的是 #9（统计捕获 hooks）与 #10（状态导出 ABI）——它们要求引擎有可挂钩的执行边界，这正是 Radiance 补丁链里 `step_observer` 系列和 `core/mem/kv.h` 系列占据的层。公共层 `memory_contract.hpp` / `native_kv_payload.hpp` 本身零改动。

## 5. 未验收 / 明确不做的范围（诚实边界）

以下全部有明确记录，引用处为原始声明：

- **Windows 原生未资格验证**：`backends/radiance/upstream.json` 两代均 `"native_windows_qualified": false`；2026-10-28 有 256K/33 请求完整运行记录（`docs/radiance-integration.md:137-147`，RSS 峰值 10925.13 MiB）但只是运行记录而非资格验收。
- **TP2 未资格验证**：本轮 worker 二进制 capability 收窄为最多 1 device、TP1，不能用其跑 TP2；通用 profile 保留但未验证（`docs/radiance-iq3v4-runbook-20261105.md:125`、`backends/radiance/upstream.json` `qualification_limits`）。双卡（2×R9700）有功能验证但明确声明不据此推断多卡加速比（`docs/radiance-integration.md:35-38`）。
- **general/all 内核 profile 未验证**：只验证 `qwen35-dense-tp1`（46 HIP units/159 rows），profile 内未合格路径 fp8_v4/k6_v4 也在其中（upstream.json `focused_kernel_profile`）。
- **JPEG 模型请求未验证**：JPEG decoder 已编进 FFmpeg bundle 但未测（upstream.json `qualification_limits`）。
- **完整模型 fp8_v4/k6_v4 KV 与 f16 全模型 state 未资格验证**（upstream.json `qualification_limits`）。
- **视频/音频不支持**（`docs/radiance-integration.md:47`）。
- **RDNA3 未资格**（upstream.json `historical_qualification`）。
- **低分辨率 OCR 质量限制**：原始 PNG suite 21/22，OCR `5829→5029`，普通 native 无 observer 同样复现；2 倍分辨率 probe 两路径均正确（`docs/radiance-iqr-v4-integration-20261105.md:101`）。
- **模型质量未通过**：256K Task2 中间 31 个批次确认仅 11/31 同时含正确 ID 与校验和（execution-overhead 复测为 11/31，`docs/radiance-execution-overhead-20261107.md:230-231`）；「运行合同通过 ≠ 模型质量通过」（`docs/radiance-iqr-v4-integration-20261105.md:5`）。
- **性能边界**：MTP7 vs MTP3 只有功能正确性证据，不声称更快（`docs/radiance-integration.md:22-23`）；执行开销优化的 5% decode 目标未达成、无 p95 streaming-gap 结论（`docs/radiance-execution-overhead-20261107.md:169-190`）；adapter staging 的端到端提速只在单对 256K 运行上观察（`docs/radiance-adapter-staging-20261107.md:109-118`）。
- **持久化边界**：SSD 缓存文件仅当前进程有效，不支持重启恢复、跨进程共享（`docs/radiance-integration.md:15-16`、`docs/memory-backend-contract.md:52-53`）；无跨进程 KV 共享、无 GPU 页共享（upstream.json `shared_cache_qualification.scope`）。
- **环境差异记录**：WSL DrvFS 临时目录 unlink 后 ftruncate 出 errno 2，同二进制 Linux ext4 通过（`docs/radiance-iq3v4-runbook-20261105.md:151`）；WSL ThreadSanitizer 因地址映射失败退出、无竞态检测结论（`docs/memory-backend-contract.md:71-72`）。
- **llama 侧对照**：公共存储层 Linux 原生 CUDA 推理验收也尚未完成（`docs/memory-backend-contract.md:73-74`）——即 Linux 原生验收证据目前集中在 Radiance 侧。

## 6. 关键验证数字存档（摘要）

| 项 | 值 | 出处 |
|---|---|---|
| 上游 pin | v1.17.0 / `19d4e8fe…`，archive SHA `40062b03…` | `backends/radiance/upstream.json` |
| 资格设备 | RX 9060 XT 16 GiB，gfx1200，TP1/C1 | `docs/radiance-iqr-v4-integration-20261105.md:7,20` |
| KV/state | FP8 KV / F32 state（默认），prefill 256 | 同上 `:22-23,62` |
| 12K 两阶段 | 155 checks 通过；问题轮 cached 10,240 tokens | 同上 `:97` |
| 四配置短探针 | DFlash2-7 端到端 40.09 tok/s；原生 decode 140.41 tok/s | 同上 `:68-73` |
| 256K Task2（staging 优化后） | prefill 650.45 tok/s，RSS 峰值 8219 MiB，pinned 峰值 16 MiB | `docs/radiance-adapter-staging-20261107.md:119-127` |
| DFlash recent 下限 | W2048/block128 → effective recent 2176 | `docs/radiance-v117-merge-audit-20261106.md:7` |
