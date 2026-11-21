# KV 分层 trade-off 实验：可行性判定与执行计划（04）

> 生成：2026-11-10（可行性研究轮，未下载模型、未编译、未跑实验）。
> 对象：固定 Qwen3.5-4B dense，KVMem-NInfer 后端扫 `--kvmem-budget × --kvmem-host-mib × 上下文长度`，量 decode 吞吐 / KV 检索命中率 / 进程 RSS。
> 证据来源：本地仓库 本地 kvmem 克隆（VERSION=0.19.0）+ 上游 `iamwavecut/ninfer-all@6cc1f9fa438b7094685ae417dfa40e1ca324e617`（tarball 解压于 解压目录，与 `backends/versions.json` 中 ninfer `base_commit` 一致）+ GitHub Releases API 实测。

---

## 0. 结论一览（GO / NO-GO）

| 环境 | GPU / 架构 | 判定 | 一句话依据 |
|---|---|---|---|
| 本机 Ubuntu 24.04 + RTX 3060 Laptop 6GB | sm_86 | **GO（smoke 级）** | sm_86 是上游默认构建目标；4B 模型实测 GPU 峰值 ~3.0 GiB，6GB 放得下；须源码构建（本机无 nvcc，需先装 CUDA toolkit） |
| 云端 Windows GPU 实例（RTX 20/30/40/50） | sm_75/86/89/120a | **GO（零构建，最强路径）** | v0.19.1 官方 Windows 包覆盖 nvidia-20/30/40/50 unified（GitHub API 实测 15 assets）；全部 KVMem-NInfer 验收都在 Windows 上；但手头云通道无现成 Windows GPU，需新租 |
| 云端 Linux GPU 实例（T4 [T4_HOST] / 阿里 ECS GPU / SeetaCloud） | T4=sm_75 | **GO-条件（先 smoke 后扫描）** | sm_75 在官方 Linux CUDA 12.9.2 构建矩阵内；kvmem 侧 SM75 T5 共享内存溢出已修（2026-10-29）。⚠️ 原文「该修复只做过 PTX 验证、kvmem patch 链从未在任何 Linux 原生 GPU 上验收」**已被 Kaggle T4 推翻**（2026-11-18，见 `18-phase-b-run.md`）：Linux 原生 GPU 上的 patch 链已实机验收，但**走的是 Kaggle 而非 ssh 主机**，本机 ssh 通道（T4/SeetaCloud/ecs-us）到 2026-11-14 仍全 DEAD，故云 ssh 路径本身仍未验
| Kaggle（P100） | sm_60 | **NO-GO（硬）** | 上游明确 "Pascal and older 6.x → no build"（gpus.md L26），任何构建都到不了 P100 |
| Kaggle（T4） | sm_75 | **已跑通（2026-11-18）** | ⚠️ **本行判定已被实测推翻**：原判「NO-GO（工程性）」的理由是「4 vCPU 内完成 350 分钟级构建，会话时长不可控」。实测把 build 阶段 timeout 从 14400 s 调到 **28800 s**（v5）后，`build` 用 **15,873.9 s（4.41 h）** 编完 943/943 目标零 error，12 阶段全绿，T4 上 9 行扫描正常退出（见 `17-phase-b-unblock.md` 与 `18-phase-b-run.md`）。**结论：会话时长不是墙，timeout 才是。** 唯一仍成立的约束是 sm_75 必须在白名单内（见 1.4 节）
| ModelScope Notebook（V100/A10） | sm_70/sm_86 | GO-条件（同云 Linux，备选） | V100 走 cuda12.9 构建（sm_70）、A10 走 sm_86；同样卡在源码构建 |
| 任何 AMD GPU 通道（AMD Radeon Cloud MI300X 等） | 非 CUDA | **NO-GO（硬）** | NInfer 是 CUDA 引擎，无 ROCm/HIP worker 路径（kvmem 的 HIP 工作在 llama 后端侧，ninfer 后端无） |

**最大阻塞点**：Linux 平台不存在任何预编译 KVMem worker —— 所有 Linux 环境（本机/云端）都必须源码构建 kvmem-patched ninfer，而仓库从未记录过 Linux 原生 GPU 验收（`docs/multi-backend.md:179` 明言"不能将 WSL 主机测试当作 Linux CUDA 验收"；`docs/multi-backend-repository.md:184-186` 只说机制上"按 ninfer 自己的 docs/vcpkg.json 构建"）。最短可行路径 = **租一台 Windows RTX 30/40 GPU 实例直接用官方 zip**；最省钱的路径 = 本机/云 Linux 源码构建（一次构建，多环境复用）。

> **2026-11-19 追加**：第二条路径已经验证成立。Kaggle T4（4 vCPU、16 GB 显存）一次会话内完成源码构建 + patch 链 + 9 行扫描，`build` 用 4.41 h（`timeout 28800`，见 `17-phase-b-unblock.md`）。**本报告的实验矩阵（见下文）因此可以在 T4 上跑，不需要再租 Windows 实例或等 ssh 主机。** 但 T4 是 sm_75，与 40 系（sm_86）不同档，本报告里「RTX 30/40 行为差异」那格在 T4 上只能验一半。

---

## 1. 问题一：Linux/云端能否跑 NInfer 后端？

### 1.1 上游 ninfer-all 的 Linux 支持是完整的（官方级）

- **CI/发布矩阵**：官方 native Linux 发布矩阵 = CUDA 12.9.2 → archs `70;75;80;86;89`；CUDA 13.4.3 → archs `75;80;86;89;120a`，linux job 跑在 `ubuntu-24.04`（容器 debian:13-slim），产出 universal Linux 包 + SHA256SUMS 并推送 HF bucket：`/tmp/ninfer-all-*/.github/workflows/native-release.yml:112-116,140-144,164-167`。
- **原生构建文档**：Ubuntu 24.04 需要 `gcc-13, cmake, ninja, meson, nasm, zlib1g-dev, libcurl4-openssl-dev` + NVIDIA CUDA Toolkit 12.8+，`scripts/fetch-media-sources.sh` 拉 FFmpeg/dav1d 源码包，然后 `cmake -S . -B build-sm86 -G Ninja -DCMAKE_CUDA_ARCHITECTURES=86 -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF && cmake --build build-sm86 --parallel 2`：`/tmp/ninfer-all-*/docs/build-linux.md`（Native Ubuntu 24.04 build 节）。
- **容器镜像**：`ghcr.io/iamwavecut/ninfer-all:cuda12.9`（含 70;75;80;86;89）与 `:cuda13.4`（含 75;80;86;89;120a），单执行文件含全部架构：`/tmp/ninfer-all-*/docs/build-linux.md`（Container image 节）。
- **架构边界**：`86`=RTX 30（默认，也跑 RTX 40）；`75`=Turing（RTX 20/TITAN RTX/**即 T4 所属代际**），CUDA 12.8+，上游标注 "measured; built-in device profiles"（gpus.md 卡表 L23、L147-153）；`70`=V100 仅 CUDA 12.8/12.9；**Pascal 6.x 及更老 = "no build"（gpus.md L26）→ P100 无支持**；Hopper/Blackwell datacenter 当前无构建（gpus.md L25）。
- 注意：上述上游 Linux 支持**不含 kvmem patch**——上游镜像/发布包是未打 kvmem 补丁的 `ninfer-serve`。

### 1.2 KVMem patch 链的 Linux 构建：机制通、验收缺

- 补丁机制：`backends/versions.json` 中 ninfer 条目锁定上游 `6cc1f9fa438b...`，48 个有序 patch（`backends/versions.json` ninfer 段），经 `python scripts/prepare-backends.py` 制备后校验 patched tree（`docs/multi-backend-repository.md`，patched_tree `cd29ac101266cc6e0df181b82c76c0d140d365d2`）。
- kvmem 核心注入：patch `ninfer-kvmem.patch` 在 ninfer 的 CMake 里加 `set(NINFER_KVMEM_SOURCE_DIR .../kvmem)` + `kvmem_enable_cuda_recall(...)`：`backends/patches/ninfer-kvmem.patch:55-61`。
- Windows 有完整 wrapper（`scripts/windows/build-backends.ps1 -Backend ninfer`，MSVC 14.44 + CUDA 13.2 + vcpkg manifest：`docs/multi-backend-repository.md`）；**Linux 没有对应 wrapper**，文档只给出公共核心（`cmake -DKVMEM_BUILD_LLAMA=OFF -DKVMEM_BUILD_SERVER_TESTS=OFF`）的 Linux 命令，ninfer 部分指回上游构建文档（`docs/multi-backend-repository.md:179-186`）。
- 验收状态：全部 ninfer+kvmem 真实模型验收在 Windows/MSVC2022/CUDA13.2、RTX 5060 Ti (sm_120a) 与 5050 Laptop 上（`docs/ninfer-release-20261104.md`：4B/MTP2 B=384/128 H=2048MiB、4B/MTP2 InputReuse B=8192/256 H=1024MiB 通过）；KV 活跃分层（RAM/SSD）的公共层 17/17 通过、WSL/GCC 地址与泄漏检查通过，但原生 GPU 性能验收只在 Windows（`docs/active-kv-tiering-validation-20261010.md` 开头 + 性能方法节）；`docs/active-kv-tiering.md:81-82` 承认 "Windows 和 Linux 使用同一契约及不同的偏移文件 I/O 实现，具体平台是否完成原生 GPU 验收以验收记录为准"——目前无 Linux 记录。
- 结论：**Linux 源码构建是"应该能编"级（机制完整：上游官方 Linux 构建 + CMake 注入），不是"已验收"级**。这正是要 smoke 的原因。

### 1.3 官方 release 只有 Windows

GitHub Releases API 实测（2026-11-10，`https://api.github.com/repos/kvmem/kvmem/releases`）：`v0.19.1`（2026-10-31 发布，15 assets）仅含 `kvmem-v0.19.1-windows-x64-nvidia-{20,30,40,50}-unified.zip` 及 sha256/validation.json；另有 `v0.19.poly1`（nvidia-50 polystrata）与 `v0.19.0-rc1`（含 `windows-rtx20-sm75 / rtx30-sm86 / rtx40-sm89 / rtx50-sm120a / v100-sm70-cuda12.9` 五包）。**无任何 Linux asset。** Windows 一键包定位见 `docs/windows-one-click.md`（含 UI + worker 发现，识别 `ninfer-kvmem-server.exe`）。

### 1.4 sm_75 边界（决定 T4/Kaggle-T4 判定）

- kvmem 侧：SM75 T5 tall prefill kernel 曾需 68,608B shared memory > SM75 上限 65,536B，已修（t5_project.cu 改 64×64 GEMM，25,600B）；补丁链末端 `ninfer-t5-sm75-prefill.patch`；**验证只在 RTX 5060 Ti/5050 上以 PTX 方式执行 SM75 代码，明确声明 "No physical SM75 test, complete SM75 worker link, package replacement or release upload is claimed"**：`docs/ninfer-sm75-t5-validation-20261029.md:69`。
- 上游侧：Turing `75` 是 "measured" 档（TITAN RTX 24GB/RTX 2080 Ti 11GB 实测过 Bonsai 2、Qwen3.8-27B 等），`cuda12.9` 容器含 75：`/tmp/ninfer-all-*/docs/gpus.md:23,147-153`。
- 判定：**T4 (sm_75) 架构在支持范围内**，但 kvmem-patched 全链路在物理 SM75 上零实机记录 → T4 实验=首次实机验收，必须 smoke 先行。

---

## 2. 问题二：模型从哪来？

### 2.1 没有现成的 Qwen3.5-4B `.ninfer` 分发 → 必须自行转换

- Web/ModelScope 检索（2026-11-10）：ModelScope 上 `.ninfer` 社区分发只有 Qwen3.8-27B 系（`docs/radiance-integration.md:97` 的 Zeshion/Qwen3.8-27B-Radiance-KVMem-MXFP4；`docs/ninfer-all-migration-validation.md:109` 的 Merkyor GGUF-NInfer Q2LynnStyle 制品）。**UNKNOWN（查过：web_search bing/exa、ModelScope API）**：Qwen3.5-4B 的 `.ninfer` 公开分发。
- 官方验证过的转换源（`docs/qwen35-dense-family.md`）：`unsloth/Qwen3.5-4B-MTP-GGUF` 的 `Qwen3.5-4B-Q4_K_M.gguf`，rev `86835bf9949e4d14d6860f7910b1340ad4f271a9`，2,834,975,040 bytes，SHA256 `3874209241c9a397e2f62cd3f70f80fd2dfbf0dfccb6838416bdb48a714e8630`；转换为完整 32+1 层 `.ninfer` 2,827,176,704 bytes；转换契约 52 项回归通过。
- ModelScope 渠道实测（2026-11-10）：`https://modelscope.cn/api/v1/models/unsloth/Qwen3.5-4B-MTP-GGUF` 返回 200；`Qwen/Qwen3.5-4B`（config/tokenizer 源）返回 200。HF 直连超时（宿主 fake-ip 代理下慢），**ModelScope 是国内首选下载通道**。
- UNKNOWN（查过 ModelScope API 树接口）：pin revision 的文件树查询返回 404（`/repo/files?Revision=86835...`），实际下载时需以 master 分支文件 + SHA256 校验落地（哈希在 2.1 有权威值，下载后逐字节核对）。

### 2.2 转换工具存在且 Linux 可用（CPU 即可）

- 转换器：`python3 -m tools.convert --model <config/tokenizer 目录> --recipe qwen3_5_dense_gguf --source gguf=<GGUF 路径> --device cpu --out <输出>.ninfer`。
- **recipe `qwen3_5_dense_gguf` 不在上游**，由 kvmem patch 注入：`backends/patches/ninfer-kvmem.patch:1323`（recipe 表条目："Every text, embedding, head and MTP tensor in its GGUF block format, byte for byte; an untied BF16 head as exact BF16 words"）、`:1359`（说明）、`:1384` 附近给出完整命令模板。上游 `tools/convert/gguf_blocks.py` 只有 `qwen3_8_27b_gguf`（`/tmp/ninfer-all-*/tools/convert/gguf_blocks.py:607,657`），18 种 `gguf_*` 块格式全支持（Q4_K_M 属 Q4_K 系）。
- 运行条件：Python 3.11 + PyTorch + NumPy，`--device cpu` 显式可用（patch 内命令示例即 `--device cpu`，`backends/patches/ninfer-kvmem.patch` bonsai 命令节）→ **转换不需要 GPU，阿里 CPU ECS / 本机均可做**。
- 顺序约束：必须先 `python scripts/prepare-backends.py` 制备 patched tree，转换器才能看到 recipe；`prepare-backends.py --check` 可验证补丁链完整性（`docs/multi-backend-repository.md`）。

### 2.3 模型资源量级（容量判定依据）

- `.ninfer` 约 2.83 GB（磁盘），4B+MTP 在 RTX 5050 Laptop 8GB 上实测 GPU 峰值 2922–3022 MiB、启动 3.1–3.3s（`docs/benchmarks/cuda-graph-probe-20261027.json`，model=Qwen3.5-4B-MTP-Q4_K_M 记录）→ 本机 6GB 卡在 4B + MTP + 中等 KV budget 下成立；SSD/RAM 分层参数另行加压（见 §3.3）。

---

## 3. 问题三：指标从哪取？

三个指标全部有官方采集面，无需自造探针（采集器脚本本身要在执行轮新建，属计划内）。

### 3.1 decode 吞吐

- **权威口径**（照抄 `docs/long-context-benchmark-2026-09-14.md:34` + `docs/radiance-integration.md:187`）：decode = 总生成 token（含思考）/ 生成总耗时；prefill 有效 = 新增输入 token / prefill 总耗时。
- **采集面 A（per-request）**：OpenAI 流式响应逐 chunk 时间戳 + `usage`（`scripts/summarize_canary.py:39-52` 解析 `prompt_tokens` / trace 字段的模式可复用）。
- **采集面 B（服务端权威）**：worker JSONL 活动日志（`--request-log-jsonl`，`docs/native-cli-configuration-spec.md:548-550`）每 5 秒输出 `prefill`（扣除前缀缓存命中）与 `decode`（最终提交 token，MTP 计接受提交量）区间计数（`/tmp/ninfer-all-*/docs/serving.md:2635-2641`），以及 `/stats` = `/v1/load` + uptime 平均 `throughput_tokens_per_second`（`serving.md:665-673`）。
- **采集面 C（Prometheus）**：`GET /metrics` 文本格式（`serving.md:782`），便于连续采样。

### 3.2 KV 检索命中率

- 服务端 `context_cache.hybrid` 对象：`snapshot_hits`（从快照恢复的准入数）、`reused_tokens`（恢复时复用的 prompt token 数）、`device_blocks`/`evictable_blocks`/`tree_blocks`、Host 层 `host_block_writes`/`host_block_restores`/`host_write_bytes`/`host_restore_bytes`、`evicted_blocks`、`unbacked_node_losses` 等（`/tmp/ninfer-all-*/docs/serving.md:2646-2659`）——**分层实验的直接证据面：下沉/回迁字节量、驱逐量、复用量都在这里**。
- 请求级复用：per-request trace 的 `replayed_rows` / `new_text_rows` 字段（`scripts/summarize_canary.py:48,61` 已有解析先例）。
- 命中率定义（实验内操作化）：`hit_rate = reused_tokens / prompt_tokens`（per-request），聚合层用 `snapshot_hits/reused_tokens` 区间增量。**VERIFY**：kvmem patch 是否在 `/stats` 增加专属 counter（`backends/patches/ninfer-kvmem-statistics-readback.patch` 存在但未逐字段核对）——smoke 轮抓一次 `/stats` 全文确认字段名后再固化采集脚本。
- 检索算法背景（不计分依据）：`docs/graph-recall.md`（Graph recall C 为当前源码默认， Mean-Q cosine / normalized GQA max 为可选）。

### 3.3 进程 RSS

- **口径**：运行期峰值 VmRSS，不含模型加载（`docs/long-context-benchmark-2026-09-14.md:36`）；Windows 用 working set、WSL 用 VmRSS、显存用 Windows 专用计数（`docs/radiance-integration.md:187`）。
- **采集器先例**：`scripts/multimodal_canary.py:383`（`re.search(r'^VmRSS:\s+(\d+)', status, re.M)` 读 /proc）+ NVML 显存采样（:273,318-330）——该脚本目前是 llama 侧（无 ninfer 分支），**执行轮需写一个 ninfer 版 driver**（SSH 驱动云端时同样适用，读 /proc/\<pid\>/status + `/stats` 轮询）。
- **实验关键点**：`--kvmem-host-mib H` 与 `--kvmem-disk-mib D` 都**不是**进程 RSS 上限——mean-K、状态、传输暂存、OS page cache 另计（`docs/active-kv-tiering.md:25`）；RSS 对 H 的实际响应曲线正是本实验要测的 trade-off 主轴之一。
- 崩溃保护先例：swap 阈值自停逻辑（`long-context-benchmark-2026-09-14.md:65`，`SWAP_STOP.json`）。

---

## 4. 分环境判定细节与缺口清单

### 4.1 本机 6GB（GO，smoke 级）

- 架构：3060 Laptop = Ampere sm_86，上游默认构建目标（build-linux.md 表）。
- 缺口（执行前要补）：本机无 `nvcc/cmake/ninja`（2026-11-10 探测：`which nvcc` 空、`gcc-13` 已装、RAM 14GB、16 核）→ 需装 CUDA toolkit 12.8+（约 3-4 GB 下载）+ cmake/ninja；磁盘余量需 ≥15GB（构建产物 sm_86 kernel bundle 约 450MB：`/tmp/ninfer-all-*/cmake/NinferBundles.cmake:3`，加源码/模型/中间物）。
- 职责定位（呼应 2026-10-28 上云纪律）：本机只做 ≤1 小时级 smoke（构建 + 起 worker + 单请求正确性 + 指标面验证），正式扫描上云。

### 4.2 云端 Windows GPU（GO，零构建）

- 判定依据：§1.3 官方包 + §1.2 全部验收在 Windows。RTX 30/40 卡租到即可跑。
- 缺口：手头云通道（AMD T4 ssh 实例、阿里 ECS、SeetaCloud、Kaggle、ModelScope）**均无 Windows GPU 现货**，需要新租通道（费用与可用性执行时再探）；这是唯一 GO 但"差一步"的环境。
- 包内容澄清：一键 zip 是控制台/UI+Python runtime（`docs/windows-one-click.md`），worker/模型单独提供；worker 即包内 `ninfer-kvmem-server.exe`（发布资产名 nvidia-XX-unified 对应 sm_75/86/89/120a）。

### 4.3 云端 Linux GPU（GO-条件）

- T4 实例（[T4_HOST]:[T4_PORT]，16GB HBM）：sm_75 官方构建覆盖（§1.1/§1.4）；16GB 显存给 KV budget 与 H 压测留足空间；**风险**：sm_75 全链路首跑（§1.4）+ kvmem patch 链 g++-13 可移植性未验证（UNKNOWN，查过 patch 列表无 Windows-only 标记但未试编译）。
- 其余 Linux GPU（A10=sm_86、V100=sm_70@cuda12.9、A100=sm_80）：同路径，架构全支持。
- 阻塞解除路径：上游容器镜像 `ghcr.io/iamwavecut/ninfer-all:cuda12.9` 可先行验证"未打 patch 的 ninfer 在该卡上能跑 Qwen3.8"（正向基线），再上 kvmem patched 源码构建。

### 4.4 Kaggle（NO-GO）

- P100：`gpus.md:26` "Pascal and older | 6.x | none | no build" → 硬性无支持。
- T4：架构支持但 (a) 只有 Linux → 需源码构建；(b) 上游 CI 单架构构建 `timeout-minutes: 350`（`/tmp/ninfer-all-*/.github/workflows/native-linux.yml:38`），Kaggle 4 vCPU 只会更慢且会话时限内未必完成；(c) kvmem patch 链 Linux 构建本身未验证，一环失败全链废 → **投入产出不成比例**。
- 结论：Kaggle 不进入本实验；若未来上游发布 Linux 官方包（native-release.yml 已有发布机制）可重评。

### 4.5 AMD GPU 通道（NO-GO）

- NInfer = CUDA 引擎；kvmem 的 ROCm/HIP 工作（`docs/building-hip-on-windows.md`、`docs/rocm10-beta2-validation.md`）都在 llama 后端/公共核心侧，ninfer 后端无 HIP 路径 → MI300X 等 AMD 卡对本实验无用。

---

## 5. 可执行计划

### 5.0 总原则

- 每个数字落盘可溯源：沿用 `docs/benchmarks/*.json` 的 schema 惯例（`gpu_model`、`gpu_uuid`、`worker_sha256`、`sampled_gpu_peak_mib`、`startup_seconds` 等字段先例见 `docs/benchmarks/cuda-graph-probe-20261027.json`）；每格记录：worker SHA256、`.ninfer` SHA256、完整命令行、JSONL 活动日志、`/stats` 终态快照、RSS 采样序列、`metrics.json`。
- 固定参数面（两后端契约，`docs/multi-backend-differences.md`）：block=64 tokens 固定；`--kvmem-budget` B≥128 且 64 倍数；`--kvmem-gen-reserve` R 64 倍数；`--kvmem-host-mib` MiB 硬预算；KV dtype 用 `int8`（`--kv-dtype` 默认 bf16，int8 是 ninfer 快速预填充允许集成员：`docs/native-cli-configuration-spec.md:406,599`）；MTP 与并发固定（`--max-concurrency 1`）。
- 变量矩阵（三层主轴）：B ∈ {1024, 4096, 16384, 32768}；H ∈ {64, 512, 2048, 8192}（H=64 视作近全下沉、8192 视作近全驻留）；ctx ∈ {8192, 32768, 131072}。

### 5.1 阶段 A：模型制备（云端 CPU 或本机，~1-2h 墙钟）

```bash
# 1. 制备 patched tree（含 qwen3_5_dense_gguf recipe）
cd kvmem-study/kvmem && python3 scripts/prepare-backends.py --check

# 2. 下载 GGUF（ModelScope 渠道）+ config/tokenizer，记录 SHA256
# modelscope download unsloth/Qwen3.5-4B-MTP-GGUF Qwen3.5-4B-Q4_K_M.gguf
# 校验: sha256sum == 3874209241c9a397e2f62cd3f70f80fd2dfbf0dfccb6838416bdb48a714e8630
#       （docs/qwen35-dense-family.md 权威值）

# 3. 转换（CPU，约 3GB 产物）
python3 -m tools.convert \
  --model /path/to/Qwen3.5-4B-config-tokenizer \
  --recipe qwen3_5_dense_gguf \
  --source gguf=/path/to/Qwen3.5-4B-Q4_K_M.gguf \
  --device cpu --out models/qwen35-4b-mtp.ninfer
# 溯源: 命令模板 backends/patches/ninfer-kvmem.patch:1384 附近
# 产物登记: 大小+SHA256 写入 ledger（对齐 qwen35-dense-family.md 的 2,827,176,704 bytes 参考）
```

### 5.2 阶段 B：构建 worker（按环境二选一）

**B1 云 Linux（T4 实例，CUDA 12.9.2，archs 75）**：
```bash
# 上游 native build（docs/build-linux.md 原文步骤）+ kvmem 注入
git -C kvmem submodule update --init backends/ninfer        # 或按 versions.json checkout base_commit
python3 scripts/prepare-backends.py                          # 打 48 个 kvmem patch
export CC=gcc-13 CXX=g++-13 CUDACXX=/usr/local/cuda-12.9/bin/nvcc CUDAHOSTCXX=g++-13
cmake -S backends/ninfer -B build-sm75 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF
cmake --build build-sm75 --parallel <nproc>
# 产物: build-sm75/apps/ninfer-serve（kvmem.patch 自动注入 KVMEM_SOURCE_DIR，见 ninfer-kvmem.patch:55-61）
```
估算：上游 CI 上限 350 min（native-linux.yml:38）；16 核实机估计 1.5-3h（UNKNOWN，首跑即得真实值）。**smoke 断言**：`ninfer-serve --help` 正常、起 worker 加载 .ninfer、单请求 8K ctx 输出非空、`/stats` 返回 JSON。

**B2 云 Windows（RTX 30/40，零构建）**：下载 `kvmem-v0.19.1-windows-x64-nvidia-30-unified.zip`（+ .sha256 核对）→ 解压 → UI 或 `scripts/windows/start-ninfer.ps1` 起 worker（默认 B=36864/R=16384/Host=12288MiB/ctx=204800/int8：`docs/multi-backend-differences.md`）。

### 5.3 阶段 C：指标采集器（执行轮新建 1 个脚本，云端跑）

- 输入：worker 地址 + 一格参数 + 请求序列定义。
- 行为：起参重启 worker → 按 `docs/long-context-benchmark-2026-09-14.md` 的多轮工具回放法构造递增上下文（每轮 +8K token 固定内容、断言 prompt 落在 [ctx-1024, ctx-512]、后续请求有缓存命中）→ 流式计时 + `usage` 记录 → 每 1s 采样 `/proc/<pid>/status` VmRSS + `/stats`（或 5s JSONL 落盘）→ 结束写 `metrics.json` + `metrics.csv`（schema 对齐 `scripts/summarize_canary.py:69-114`）。
- 命中率字段首格人工核对 `/stats` 全文（§3.2 VERIFY 项），固化后批量跑。

### 5.4 阶段 D：扫描矩阵与时长估算

- **主扫描（12 格）**：H 固定 2048，B×ctx 全组合 4×3。
- **H 扫描（4 格）**：ctx 固定 32768、B 固定 4096，H ∈ {64, 512, 2048, 8192}。
- **可选 SSD 轴（2 格）**：`--kvmem-disk-path` + `--kvmem-disk-mib`（`docs/active-kv-tiering.md` 三参数契约），仅 GO 环境加测。
- 单格工作量（33 请求/长格、短格 8 请求）：128K 格累计 prefill ~261K token；按 27B 在 5060Ti 有效 prefill 793-941 tps 折算（`docs/radiance-integration.md:141,165`），4B 估 >1500 tps → 128K 格 ≈ 3-4 min prefill + decode ≈ 1-2 min + 加载/清理 ≈ **5-8 min/格**；32K 格 ≈ 2-3 min；8K 格 ≈ 1-2 min。
- 总时长（主扫描 12 格 + H 扫描 4 格）：**约 50-90 min 纯测 + 开销 ≈ 2-3h**；一次性构建另计。每格独立输出目录、每格记录启停。
- **时长均为估算（标 est）**，smoke 后用真实首格数值修正（这是"数字来源"纪律的一部分：估算不进结论）。

### 5.5 阶段 E：报告

- 产出：`kvmem-study/reports/05-tiering-results.md` + `docs/benchmarks/` 惯例 JSON 落盘；三指标（decode tps / hit_rate / RSS 峰值）对 B、H、ctx 的曲面 + 每 格数字→采集文件路径映射表。
- 结论判据：H 增大→decode tps 提升/命中率变化/RSS 增长的实际斜率；找出"RSS 换吞吐"的拐点格；记录 `unbacked_node_losses`/`evicted_blocks` 异常格。

### 5.6 环境优先级（执行顺序建议）

1. **A（模型制备）** 在阿里 CPU ECS 或本机：无 GPU 依赖。
2. **B1+C+D 在 T4 实例**：一次构建解决 sm_75 云 Linux 全部扫描；同 tarball 可在本机 3060 重编 sm_86 做交叉验证（可选）。
3. **B2（Windows 包）** 作为"官方已验收路径"的对照组：若能低成本租到 RTX 30/40 Windows 实例，优先用它跑主扫描、把 T4 结果当交叉验证；租不到则以 T4 为主战场并在报告中标注"Linux 首验收"属性。
4. Kaggle/AMD：排除。

---

## 6. UNKNOWN 与风险登记

| # | 事项 | 查过哪里 | 影响 | 处置 |
|---|---|---|---|---|
| U1 | ~~kvmem 48-patch 链在 Linux/g++-13 的可编译性~~ **已解（2026-11-18）** | v5 实测：`prepare_ninfer` 51.0 s 打完 48 patch（head `6cc1f9fa438b`），`build` 15,873.9 s 编完 **943/943 目标零 error**，g++ 13.3.0 + CUDA 13.0.88 | — | 结论：Linux/g++-13 可编；**但需两处环境补齐**（meson+ninja+nasm；CUDA 13 的 `CUDA::cuda_driver` 要用 `CMAKE_PROJECT_INCLUDE` shim 补 IMPORTED 目标） |
| U2 | ~~`/stats` 中 kvmem 专属 counter 精确字段名~~ **已解（2026-11-18）** | v5 六行 Host tier 全部拿到 `context_cache.hybrid` 的 18 个计数器，字段名与 `prefix/hybrid_cache.h:56-74` 逐字一致；`make_stats_report`（`src/serve/load_report.cpp:87-118`）用 `previous={}` 重建 → 块是 **CUMULATIVE 非区间** | — | 结论：字段名已实测；**采样要注意它是累计值**，两轮之间要自己减，不能直接当区间读数 |
| U3 | Qwen3.5-4B `.ninfer` 公开分发 | web_search（bing/exa）、ModelScope API | 无 → 必须自行转换 | 按 §5.1 转换，SHA256 对齐权威值 |
| U4 | ModelScope pin revision 文件树 404 | `/api/v1/models/.../repo/files?Revision=86835...` 返回 404（repo 本体 200） | 下载路径 | 下载时用默认分支 + 哈希校验兜底 |
| U5 | Linux 实机构建时长 | 上游 CI timeout 350min（native-linux.yml:38） | 排期 | 首跑实测后回填 |
| U6 | 本机 CUDA toolkit 未装（探测：`which nvcc` 空） | 本机 2026-11-10 | 本机 smoke 前置 | 装 12.9 toolkit 或改容器镜像跑 smoke |
| U7 | Windows GPU 云通道现货与价格 | 未逐一询价（仅确认手头 5 通道无 Windows GPU） | 对照组可行性 | 执行轮询价；租不到则 T4 单主战场 |

## 7. 证据文件索引

- 本地：`docs/multi-backend.md`、`docs/multi-backend-differences.md`、`docs/multi-backend-repository.md`、`docs/linux-launcher.md`、`docs/active-kv-tiering.md`、`docs/active-kv-tiering-validation-20261010.md`、`docs/qwen35-dense-family.md`、`docs/ninfer-release-20261104.md`、`docs/ninfer-sm75-t5-validation-20261029.md`、`docs/graph-recall.md`、`docs/long-context-benchmark-2026-09-14.md`、`docs/radiance-integration.md`、`docs/native-cli-configuration-spec.md`、`docs/windows-one-click.md`、`docs/benchmarks/cuda-graph-probe-20261027.json`、`backends/patches/ninfer-kvmem.patch`、`backends/versions.json`、`scripts/multimodal_canary.py`、`scripts/summarize_canary.py`、`scripts/native_stats.py`。
- 上游（/tmp/ninfer-all-6cc1f9fa438b.../）：`docs/build-linux.md`、`docs/gpus.md`、`docs/serving.md`、`docs/weight-conversion.md`、`docs/recipes.md`、`.github/workflows/native-release.yml`、`.github/workflows/native-linux.yml`、`tools/convert/gguf_blocks.py`。
- 网络：GitHub Releases API `repos/kvmem/kvmem/releases`（2026-11-10 实测）；ModelScope API `models/unsloth/Qwen3.5-4B-MTP-GGUF`、`models/Qwen/Qwen3.5-4B`（2026-11-10 实测 200）。
