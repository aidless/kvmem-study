# 报告 17：Phase B 解卡——Kaggle 双 T4 与 sm_75 的合法性

**结论先行**：Phase B（构建 + 扫描）从 2026-11-10 卡到今天，原因是"没有 Linux GPU 主机"。今天这条堵点解开了，而且不是勉强凑合，是**官方支持范围内的目标**：

| 项 | 值 | 出处 |
|---|---|---|
| GPU | Tesla T4 ×2，各 15,360 MiB | `remote/gpu-probe/gpu_probe.json` |
| compute capability | **7.5**（Turing） | 同上 `compute_cap` |
| 驱动 | 580.178.04 | 同上 `driver_version` |
| CUDA | 13.0.88 | 同上 `nvcc` |
| CPU / RAM / 磁盘 | 4 核 / 31 GB / 20 GB | 同上 `cpu`、`mem_gb`、`disk` |
| NInfer 支持 | **sm_75 是官方六档之一** | `CMakeLists.txt:32`、`docs/build-linux.md:13` |
| 运行期准入 | capability 75 需 Turing 镜像，否则抛 | `src/models/qwen3_5/program/planning/startup.cpp:1394-1398` |
| 功能完整性 | **"Every model, KV format and drafter runs."** | `docs/build-linux.md:128` |

## 一、探测：从"猜"到"只有一档能跑"

上一轮推的探测内核 `<kaggle-user>/kvmem-gpu-probe`（`kvmem-study/kaggle/gpu_probe/probe.py`）不是只打印 `nvidia-smi` 就算完。它的核心是一段**逐架构编译运行**：

```python
for arch in ("sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_120a"):
    nvcc -arch=$arch -o /tmp/t_$arch probe.cu
    /tmp/t_$arch        # 打印 gpu ok 才算这一档能在这张卡上真跑
```

结果（`remote/gpu-probe/kvmem-gpu-probe.log:32-41`）：

```
compile_run[sm_75]: OK
compile_run[sm_80]: FAIL
compile_run[sm_86]: FAIL
compile_run[sm_89]: FAIL
compile_run[sm_90]: FAIL
compile_run[sm_120a]: FAIL
```

**为什么这个设计是对的**：`nvcc -arch=sm_86` 在一台只有 T4 的机器上**编译会成功**（nvcc 不检查本机显卡），失败发生在运行时——`no kernel image is available for execution on the device`。如果只测"能不能编译"，五档都会报 OK，结论就是错的。**编译成功不是可运行性证据，加载执行才是。**

一档 OK、五档 FAIL，这个形状本身就证明了探测是有效的：它不是"全部失败"（那会是环境坏了），也不是"全部成功"（那会是探测没真跑）。

## 二、合法性：sm_75 不是兼容层凑数

真正让我确认这条路能走的，不是 T4 有 15 GB 显存，而是三处代码/文档把 sm_75 写成**一等目标**。

### 2.1 构建期白名单

```cmake
# CMakeLists.txt:32
if(NOT _ninfer_architecture MATCHES "^(70|75|80|86|89|120a)$")
  message(FATAL_ERROR "NInfer supports CMAKE_CUDA_ARCHITECTURES of 70, 75, 80, 86, 89 and 120a ...")
```

```cmake
# CMakeLists.txt:45-47
if("75" IN_LIST _ninfer_architectures)
  set(NINFER_ARCH_TURING ON)
endif()
```

白名单只有六个值。sm_75 在其中，且有自己的 tier 开关 `NINFER_ARCH_TURING`。CMakeLists.txt:1-8 的头部注释把每一档的定位写清楚了：

> The current implementation supports Ampere sm_86 and early Ada sm_89, plus sm_80 (GA100, e.g. CMP 170HX) as an unmeasured compatibility target ... sm_70 (Tesla V100) builds the same compatibility path with the Ampere warp primitives it lacks (cp.async, ldmatrix, the m16n8 mma.sync shapes) emulated ... **Turing sm_75 (TITAN RTX, RTX 20 series) runs the same mma.sync path through the Turing tier below.**

注意它给 sm_80 打的标签是 "unmeasured compatibility target"，给 sm_75 的却是"有 Turing tier"——**有 tier 的档比"兼容目标"档地位更高**。

### 2.2 运行期准入（最关键的一行）

```cpp
// src/models/qwen3_5/program/planning/startup.cpp:1394-1398
if (capability == 70 || capability == 80 || capability == 86 || capability == 89 ||
    capability == 120 ||
    (capability == 75 && ops::device_tier(context.device) == ops::DeviceTier::Turing)) {
    continue;
}
throw std::invalid_argument(
    "Qwen3.5 family runtime requires compute capability 7.0, 7.5, 8.0, 8.6, 8.9 or 12.0 "
    "(7.5 needs a build with 75 among CMAKE_CUDA_ARCHITECTURES)");
```

这是"能不能跑"的权威判据，它比任何文档都硬：

- capability 7.0/8.0/8.6/8.9/12.0 无条件放行。
- **7.5 需要额外条件**：`device_tier(device) == DeviceTier::Turing`。也就是说**光有 T4 不够，二进制里必须有 Turing 那一档的 device code**，否则 CUDA 加载的是别的档的镜像，tier 探测对不上，直接抛。
- 错误信息自己把修法写出来了：`7.5 needs a build with 75 among CMAKE_CUDA_ARCHITECTURES`。

`DeviceTier` 枚举（`src/ops/common/device_tier.h:22-33`）把四档的代价写得很直白：

```cpp
// sm_75: Turing forms of the sm_80 instructions (sm75_mma.cuh) and kernels sized for 64 KiB of
// shared memory per block.
Turing,
```

"Turing forms of the sm_80 instructions" —— 也就是说 sm_75 不是"少几个 kernel"，是**同一批 kernel 的 Turing 形态**。这是移植，不是阉割。

### 2.3 能力边界（唯一的真实约束）

`docs/build-linux.md:119-128` 是我找到的最诚实的一段硬件约束描述：

> Turing's tensor cores multiply FP16 and INT8 but not BF16 or TF32, it has no `cp.async`, and a block has 64 KiB of shared memory. A `75` build runs the kernels of the other targets with Turing forms of those instructions and Turing shapes of the kernels that do not fit:
>
> - Integer routes (the T2 ternary projections, A8 activations, INT8-family attention, GGUF `dp4a` and MMQ kernels) and FP16 routes run on the native tensor cores.
> - **BF16 products run on FP16 tensor cores for every output whose operands are FP16 values, and in FP32 for the others; no value is narrowed.** BF16-cache attention multiplies Q, K and V as their FP16 roundings, and every attention PV product with BF16 values rounds the probabilities to FP16.
> - **Every model, KV format and drafter runs.** The Linear routes come from the legacy table only, and `NINFER_LINEAR_ROUTES=unified` is refused.

三条各自重要：

1. **没有 bf16 张量核 ≠ 不能跑 bf16 模型**。做法是"在 FP16 张量核上算以 FP16 为操作数的乘，其余在 FP32 里算，**不窄化任何值**"。代价是速度，不是正确性。这是把"硬件缺一条指令"翻译成"算法等价变换"的标准做法，值得记。
2. **"Every model, KV format and drafter runs"** 直接回答了我最担心的问题：Phase A 制备的那个 `.ninfer`（184 个 bf16 对象 + 96 个 q4_k + 35 个 q6_k）不需要重新量化。
3. **唯一限制是 `NINFER_LINEAR_ROUTES=unified` 被拒**，Linear 路线只能走 legacy 表。这是 route 选择问题，不是模型支持问题。

附件 `docs/build-linux.md:103-117` 甚至给了 Turing 的原生构建命令，与我推的内核参数一致：

```bash
cmake -S . -B build-sm75 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DNINFER_BUILD_APPS=ON
cmake --build build-sm75 --parallel
```

## 三、为什么这条通道比等 ssh 主机好

回头看，从 11-10 到今天，我在等的东西是"一台 Linux GPU 主机"。三次探测（`[T4_HOST]:[T4_PORT]`、`[ECS_US_HOST]:22`、`[SEETACLOUD_HOST]:[PORT]`）全 DEAD，`ecs-hz` 活着但是 2 核 / 1 GB RAM / 无 GPU / 无 nvcc —— 实测过，`free -g` 的 `available` 是 0。

Kaggle 这条通道的账：

| | ssh 主机 | Kaggle 内核 |
|---|---|---|
| 可用性 | 每次要探测，随时 DEAD | CLI 推送即跑，状态可查 |
| 成本 | 按小时计费 | 免费（配额内） |
| 单次上限 | 自定 | ~12 h |
| 磁盘 | 自定 | 20 GB working + 更大 /tmp |
| 复现性 | 手工环境 | 脚本即环境，版本记录在 result.json |
| 缺点 | — | 会话结束不留进程；产物必须落 `/kaggle/working` 才能拉回 |

**真正的转折点是认识上的**：我一直在找"一台常驻的 GPU 机器"，而这件事需要的其实是"一次能跑两小时的 GPU 执行"。前者稀缺，后者随处可得。这不是妥协，是把问题从"拥有资源"改写成"购买执行"。

## 四、这一轮推了什么

内核 `<kaggle-user>/kvmem-build-sm75`（`kvmem-study/kaggle/build_sm75/`），版本 1，`enable_gpu: true`、`enable_internet: true`、`dataset_sources: ["<kaggle-user>/kvmem-src"]`。

六个阶段，每个阶段失败都立即落 `result.json`（不是攒到最后）：

1. **extract_kvmem** —— 在 `/kaggle/input/**` 全树找 `*.tar.gz`（新版 Kaggle 的挂载路径仍会漂），解到 `/tmp/work`。
2. **prepare_ninfer** —— 把 `backends/ninfer` 钉到 `6cc1f9fa`（`git fetch --depth 1 origin <sha>`，失败回退全量 fetch），断言 `HEAD == base_commit`，然后给 kvmem 根建本地 commit，让 `prepare-backends.py` 走 patch 分支逐个 verify，而不是走 release 分支要 `SOURCE-MANIFEST.json`。
3. **probe_toolchain** —— 打印 `nvidia-smi -L`、`nvcc --version`、`gcc`、`cmake`、`ninja`、`df /tmp`、`free -g`；**cmake < 3.28 就用 pip 装**（`cmake_minimum_required(VERSION 3.28)` 是硬门），ninja 缺失同样 pip 装。
4. **fetch_media** —— 跑 `scripts/fetch-media-sources.sh`，拉 FFmpeg 9.0.2 + dav1d 1.5.4（sha256 pin，构建期不再联网），记录目录与每个文件大小。
5. **configure + build** —— `-DCMAKE_CUDA_ARCHITECTURES=75`、`-DNINFER_BUILD_APPS=ON`、Ninja、`--parallel 4`。configure 后回读 `CMakeCache.txt` 里 `CUDA_ARCHITECTURES / CUDA_COMPILER_VERSION / CXX_COMPILER / MEDIA_SOURCES_DIR` 存档——**构建参数必须从缓存里回读，不能只信自己传了什么**。
6. **smoke** —— 找到所有 `ninfer*` 可执行文件，各跑 `--help`；再跑 `--list-devices`，看 NInfer 自己的 tier 探测能不能认出 T4。这一步是"能构建"与"能跑"之间的分界。

## 五、还没解决的问题

**构建能不能成功，还不知道。** 已知风险按可能性排序：

1. **FFmpeg 静态编译**（最大风险）。`fetch-media-sources.sh` 只负责下载，CMake 里要把 FFmpeg 编成静态库。4 核上 FFmpeg + dav1d + 整个引擎，可能 60–120 分钟。若超时或依赖缺失（nasm/yasm 之类汇编器），要在内核里补装。
2. **Turing shapes 的 kernel 集合是否完整**。`src/ops/common/rowsplit_a8_mma.cuh:58-60` 提到 Turing 的 staging ring 只有 2 阶段、64 KiB 共享内存装不下 512-token tile 的两阶段。这类"装不下"是**编译期常量选择**，理论上已处理好，但没跑过就是没跑过。
3. **磁盘**：20 GB `/kaggle/working`，但中间产物全在 `/tmp`。模型 2.8 GB + 源码 94 MB + 构建产物可能 5–10 GB。`/tmp` 在 Kaggle 上通常更大，已让内核打印 `df -h /tmp` 备案。
4. **`--list-devices` 这个参数可能不存在**。我是从 CLI 风格推测的，`smoke` 阶段用了 `check=False`，不存在也只会记一个非零退出码，不会中断。

## 六、对前面报告的影响

- **报告 16 的容量算术不用改**。`stage_fits` 是纯整数运算（`src/core/stage_plan.h:202-213`），与 CUDA 架构无关。T4 的 15,360 MiB ×2 只是把 `--vram-mib` 换成 15360 再跑一遍 `tools/capacity.py`：

  ```
  python3 tools/capacity.py models/qwen3_5_4b_mtp.ninfer --vram-mib 15360 --kv int8_g64
  ```

  这是 Phase B 的第一件要做的对账：**算术说能装多少，实测跑起来能装多少**。报告 16 已经明确指出两个口径（budget 阶 vs measured 阶）差 6.4 倍，这个差异必须在真机上量出来才算数。

- **报告 04 的分层实验计划现在可执行**。它的核心变量是 Host tier 预算（**`ninfer-serve --host-kv-mib`**，注意不是 KVMem 包装层的 `--kvmem-host-mib`，两者是不同的二进制）与上下文长度的交互。T4 有 **31 GB 主机 RAM**，这刚好是 Host tier 的主战场——比 6 GB 的 3060 Laptop 更适合测 L2 分层，因为主机内存足够大到能观察"Host 够用 → 不再走 L3"的拐点。

- **双卡值得单独测**。`--devices` + `--stage-layers` 的层切分在 Turing 上是否成立，报告 15 的 `solve_stage_plan` 给了求解器，但"两卡都是同构 sm_75"是求解器没测过的情形（它的样例是 V100 + TITAN RTX + 30 系混合）。

## 七、下一步（构建成功后立即执行）

1. **容量对账**：`capacity.py --vram-mib 15360` 的预测 vs 真机 `--max-context` 能起到的实际值，两口径都量。
2. **Host tier 分层扫描**：`--host-kv-mib` 取 `{0, 2048, 8192}` × 上下文 `{8K, 32K, 128K}`，采 `HybridCacheCounters` 的 18 个计数器（`prefix/hybrid_cache.h:56-74`）。
3. **跨会话复用实证**：两个会话喂同一前缀，看 `blocks_reattached` 与 `reused_tokens` 是否如报告 08 预测的那样在 Host 层命中（微秒级、零拷贝），而不是掉到 L3（毫秒级 + 一次拷贝，`kRestoreBudget = 8s`）。
4. **两卡层切分**：`--devices 0,1 --stage-layers 16,16`，量 `crosses_after`（`src/core/stage_plan.h:120`）承诺的"每 stage 边界恰好一次跨卡"。

---

**一句话**：堵了四天的不是算力，是一个错误的假设——以为需要一台自己的 GPU 机器。实测只需要一个能跑两小时的 GPU 执行，而 sm_75 在源码里是有名有姓的一档。

## 八、附录：旗标语义（读 `src/serve/serve_options.cpp` 得来，写报告 17 时踩过）

写"下一步"时我一度把 KVMem 包装层的旗标名写成了 NInfer 原生的旗标名。这是两个不同二进制的两套 CLI，混用会直接报"未知参数"。实读源码后确认：

| 目的 | 原生旗标 | 出处 | 备注 |
|---|---|---|---|
| KV 容量 | `--kv-capacity <tokens\|auto>` | `serve_options.cpp:706-710`、`parse_kv_capacity:80-85` | **单位是 token 不是页组**；默认 `explicit_capacity(8192)` = 8192 token（`serve_options.h:47`） |
| 请求上下文上限 | `--max-context <int>` | `:701-705` | 与 KV 容量是两个独立的闸 |
| Host tier | `--host-kv-mib <mib>` | `:902-912` → `context_cache.host_kv_capacity_bytes` | 代码把它标成 `legacy_cache_flag`；首选名是 `--host-cache-mib`（`:913`） |
| 显存留白 | `--kv-headroom-mib <mib>` | `:711-716` | 别名 `--vram-headroom-mib` 是另一个 fork 的叫法；`0` = 关掉自动留白 |
| KV 格式 | `--kv-dtype <...>` | `:1200-1203` | |
| 磁盘 tier | `--disk-kv-path` / `--disk-kv-gib` | `:953-958` | 同样带 legacy 标记 |
| 卡选择 | `--device <id>` / `--devices <ids>` | | `--devices` 走 `device.model_parallel()` |
| 层切分 | `--stage-layers <n,n,...>` | | 与 `StagePlan::from_layer_counts` 对应（`stage_plan.h:1-120`） |

两条容易踩的硬约束，都写在 `serve_options.cpp` 里：

1. **显式容量与自动容量互斥且各自不许带对方的参数**：`validate_options` 明确抛 `Engine explicit kv_capacity must not carry automatic headroom`、`Engine automatic kv_capacity must not carry explicit tokens`（`model_instance.cpp:58-72`）。也就是说 `--kv-capacity 65536` 配 `--kv-headroom-mib 512` 会直接启动失败——留白只属于自动模式。
2. **legacy 缓存旗标与 hybrid 缓存旗标是两套模式，同给即错**：`legacy_cache_flag` 与 `hybrid_option_flag` 两个变量专门用来在跨模式冲突时报出"最后见到的是哪个"（`serve_options.cpp:639-643`）。

这两条对 Phase B 的扫描设计是硬约束：**扫描矩阵不能把 `--kv-capacity N` 和 `--kv-headroom-mib` 放同一行配置里**，那会让整行实验在启动阶段就死掉，而错误信息看起来像"参数拼错了"，不像"实验设计错了"。
