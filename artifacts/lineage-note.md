# NInfer 上游与分支血缘核验（他山之石，2026-11-17）
> 发布说明：本笔记是 kvmem-study 研究过程中对上游血缘的一次性核验记录。
> 正文引用的 `reports/NN-*` 指本仓库 `reports/` 目录；个人绝对路径与云主机地址已脱敏。
> 这里的所有 GitHub API 读数都可以按表格里的命令重跑（需网络）。

> 触发：用户「他山之石可以攻玉 看看github」。
> 目的：在引用外部实现前，先量出「我们读的那个 base 到底站在哪条线上」——这一步不成立，后面抄什么都是错的。
> 结论先行：**kvmem 钉的 `6cc1f9fa` 不在上游主干上，是 iamwavecut 那条线的快照；上游 master 与它分叉（+1895 / −523），并且上游已经用另一套设计做了「跨请求共享保留前缀」。**

## 一、实测表（全部可复算）

| # | 命令（要点） | 输出 |
|---|---|---|
| 1 | `GET /repos/Neroued/ninfer` | `stars=3372 fork=false default_branch=master pushed_at=2026-11-17T05:37:01Z` |
| 2 | `GET /repos/iamwavecut/ninfer-all` | `fork=true parent=Don-Chad/ninfer-3090 master pushed_at=2026-11-17T00:15:16Z` |
| 3 | `GET /repos/Neroued/ninfer/compare/6cc1f9fa...master` | **`status=diverged ahead_by=523 behind_by=1895 merge_base=f76e19c0fbd0`** |
| 4 | `GET /repos/iamwavecut/ninfer-all/compare/6cc1f9fa...master` | `status=ahead ahead_by=325 behind_by=0 merge_base=6cc1f9fa` |
| 5 | `raw.githubusercontent.com/{repo}/6cc1f9fa/src/core/disk_kv_bridge.h` | `200`（两库都有） |
| 6 | 同上，`.../program/prefix/hybrid_cache.h`、`/src/core/stage_plan.h` | `200 / 200`（两库都有） |
| 7 | `raw.githubusercontent.com/Neroued/ninfer/HEAD/src/core/disk_kv_bridge.h` | **`404`** |
| 8 | `GET /contents/src/models/qwen3_5/program/prefix?ref=HEAD`（Neroued） | **`Not Found`**（整个目录没了） |
| 9 | `GET /contents/src/core?ref=HEAD`（Neroued） | 有 `paged_kv_cache.*`、`expert_cache.*`、`state_image.*`…**无** `disk_kv_bridge/disk_kv_store/host_kv_store/host_kv_clamp/kv_loan/evictable_kv_pool/stage_plan` |
| 10 | `GET /contents/src/core/io?ref=HEAD` | 只有 `file.* read_queue.*`（未迁到 io） |
| 11 | `GET /repos/Neroued/ninfer/commits?path=src/models/common/context/...` → `ebdc67e584bc`（2026-11-16） | `fix(cache): share retained prefixes across requests`，**83 文件**：`+prefix.cpp 981`、`+prefix.h 212`、`+request_ledger.{h,cpp} 66/81`、`−identity.h 73`、`−qwen/prefix_identity.{h,cpp} 102/424`、`recovery.h +68/−245`、`recovery_store.h` 从 qwen 上移 |
| 12 | `raw .../HEAD/src/models/common/context/prefix.h` | `:16-17` `using PrefixId = runtime::PrefixId; using PrefixKey = runtime::PrefixKey;`（契约层在 `runtime/contract/recovery.h`）；`:41-43` 注释「**Sparse, request-owned demand evidence. Computing a later key resumes at the nearest already requested frontier; no digest array grows with every token.**」 |
| 13 | `raw .../HEAD/src/models/common/context/prefix.cpp` | `:13-16` `kInitial{1469598103934665603, 7809847782465536322}`、**`kPrime{1099511628211, 14029467366897019727}`**；`:18-21` `mix = rotl(d^v,17)*p0 ; rotl(d+(v^0x9e3779b97f4a7c15),29)*p1`；`:34-45` `mix_media` 仍混 `0x6e696e6665722d76`（"ninfer-v"）+ modality/grid/patch/timestamps/spans，时间戳改 `stamp==0 ? 0.0 : stamp` |
| 14 | `raw iamwavecut/ninfer-all/HEAD/src/core/CMakeLists.txt` | `:21` 与 `:75` **仍 `CUDA::cuda_driver`**（上游无任何修复） |
| 15 | `GET /commits?path=cmake`（Neroued，近 15 条） | 最新 `58733a86f169 2026-10-22 fix(media): align JPEG decoding with Pillow`；**无 cuda_driver / CUDAToolkit 相关提交** |
| 16 | LMCache `dev/README.md` | 自述「KV cache management layer」，卖点是 persistent store / 跨引擎复用 / MP 架构（2026-04）/ 多节点 P2P CPU 内存共享 GA（2026-01）/ 5000+★ |

## 二、逐条判断

**（A）底座是分支线，不是主干。** `6cc1f9fa`（2026-11-04）在 `iamwavecut/ninfer-all` 上是 master 的祖先（+325 且 behind 0），在 `Neroued/ninfer` 上与 master **已分叉**（merge_base 早于它 1895 个提交）。我们 06–14 号报告读的 `src/core/{disk_kv_bridge,stage_plan,host_kv_store,evictable_kv_pool,kv_loan,host_kv_clamp}`、`src/models/qwen3_5/program/{prefix,storage}` 这套系统，**只存在于这条分支线**。报告里写「NInfer 的物理层」时，准确说法是「iamwavecut 线 2026-11-04 快照的物理层」。

**（B）上游主线用另一套设计做了同一件事。** `ebdc67e5` 把「每请求拷贝一份 identity」换成「精确共享前缀存储 + 请求自有 output ledger」，并删除 `identity.h` / `qwen/prefix_identity.*`。这正是我 09/10 号报告里归给 KVMem 的「跨请求/跨会话共享」增量——**上游自己在 6cc1f9fa 之后 12 天做了**，只是换了实现路线（`runtime::PrefixKey` 契约下沉 + request ledger，而不是 hybrid cache + slab 池）。

**（C）我 09 号报告的摘要机制已被上游改写，且改写理由正是我报告里点出的成本。** base 上是「双乘子 xorshift + 逐字节累积 + 每 frontier 一个 128 位摘要」，我在 09 号引的注释自己承认 156K-token prompt 上「**10 ms of every admission**（RTX 5060 Ti, 2026-10-28）」；上游新代码 `prefix.h:41-43` 明写「**no digest array grows with every token**」，`kPrime` 换成 `{1099511628211, 14029467366897019727}`，`mix` 改为 `rotl` 双路。→ 引用 10 ms 这个数必须带 sha 与日期，否则等于引用一个已被作者修掉的缺陷。

**（D）`CUDA::cuda_driver` 没有上游先例。** fork master `src/core/CMakeLists.txt:21`/`:75` 仍硬链 `CUDA::cuda_driver`；上游 `cmake/Dependencies.cmake:1` 仍只 `find_package(CUDAToolkit REQUIRED)`，`cmake/` 下自 2026-10-22 起无相关提交。→ 我在 v5 里自造的「符号链接 + `CMAKE_PROJECT_INCLUDE` shim」不是重复劳动，但也**没有社区验证**，这一条只能靠自己的 RED（原命令报 `Target "ninfer_core" links to: CUDA::cuda_driver but the target was not found`）与 GREEN（configure 通过）判。

**（E）LMCache 只作假设源，不作结论。** 它是 Python 层 + 跨引擎 connector，与 ninfer 的进程内 C++ 分页引擎不同量纲；它的 P2P 共享、persistent store 可作为「我们还没做的机制」清单，但**未测量**其在本项目前提下的收益（无 GPU、无 multi-node），按 `AGENTS.local.md` 第十五条不进结论。

## 三、对既有产物的影响

| 产物 | 影响 | 处置 |
|---|---|---|
| 06/07/08/09/10/12/13/14 号报告 | 描述对象需加「iamwavecut 线 @6cc1f9fa」限定 | 各报告开头已写 base sha；**待补一句线名**（不改结论） |
| 09 号报告的 10 ms/admission、双乘子常数 | 只对 base 成立，上游已改 | 引用处带 sha + 「上游 2026-11-16 已换成稀疏 frontier 指纹」 |
| 「KVMem 增量 = 跨请求共享」 | 缩水为「在该分支线的该时点」 | 改成按线/按 sha 表述 |
| 15/16 号（stage 数学、容量算术） | 不受影响（纯整数运算 + manifest 实数） | 不动 |
| Phase B 扫描内核 | 不受影响（编的是 pin 住的 base） | 不动 |

## 四、证据等级与未测量

- **等级 A**（可直接复算）：第 1–15 行全部来自 GitHub API / raw 响应，任何人有网络即可重跑。
- **未测量**：① 上游 master 那套 prefix 设计的运行时表现（无 GPU 构建，Phase B 编的是 base）；② `iamwavecut/ninfer-all` 的 +325 提交里是否也有同类 prefix 重构（未逐条读）；③ LMCache 与本项目的可比收益（不同语言、不同后端、无多节点）；④ 上游删除该子系统是否有公开的 issue/PR 说明（`Fixes #448` 是本次 cache 重构的关联号，未展开读）。

## 五、续查（2026-11-17，用户批准后：issue 正文 + 分支线 +325 + 报告加注）

### 5.1 issue #448：那次 83 文件重写的动机是缺陷，不是新特性

- 标题：*Prefix cache reuse stalls at a stale checkpoint for growing prompts (a754fcf3 → d9ad9dea)*，state `closed`，1 条评论。
- 报告者环境（原文）：RTX 5090 D 32 GB / driver 617.14 / CUDA 13.4 / Qwen3.8-27B NVFP4（artifact sha256 `74d2c57145e6…`）/ 旗标 `--kv-dtype nvfp4 --max-context 262144 --context-capacity auto --max-concurrency 2 --prefill-chunk 4096` / `dflash2` draft 4 / `checkpoint_interval 8192` / KV ≈ **417,488 tokens** / VRAM 31.5 GiB / 启动回归 16/16。
- 现象：长对话从 143k 长到 158k token 时，引擎**反复报同一个过期检查点命中**（`cache 44,684`），每轮重 prefill **70–100k token**；受控三请求探针（冷 / 加尾 / 完全相同）在 `a754fcf3`(v6) 上复用 ≈100%，在 `d9ad9dea`(v7, master) 上复用 **0 token**。**短 benchmark 看不出来，只在长上下文 + 长活进程上出现。**
- 处置：报告者 2026-11-16 04:49 UTC 回滚到 `a754fcf3`；`Neroued` 2026-11-17 评论「`ebdc67e584bc` fixed here」。
- **对我上午判断（B）的修正**：上游不是「顺手把 KVMem 的增量做了」，而是**修一个真实的长上下文复用失效**——复用停在过期 frontier/anchor 上；换掉 per-request identity 拷贝是修法，不是目的。

### 5.2 分支线 +325 提交里，我报告描述的"设计"有 3 处已被当成 bug 修掉

| commit | 日期 | 它承认的缺陷（原文要点） | 被我写成的"设计" |
|---|---|---|---|
| `452acabb11` | 11-08 | 持 Device slot 的快照其 **tail 必须留在 Device**，Host 驱逐不能删它的 Device 副本；车道 image 的 Host 写要保住**最新被取代的 tap**；块写穿不得挤掉**比本车道最有价值保留快照更值钱**的保留项；快照命中距上次命中/发布**满半窗**时留一半命中到下一窗，**突发命中一个不留**；Host-tier 落盘**升到 format 5**（带 periodic 标记） | 08 号「统一 slab 池、不分固定配额」与 14 号「version 3」都是**该时点的缺陷形态** |
| `8ba2753efd` | 11-10 | **整体落在 tail block 内的媒体项**（Qwen3.8 上 4 token 的短视频）**不在任何 vision key 里**，只差这一段 clip 的 prompt 也能从快照往后复用；修法 `media_exclusion` 把 tap 排除延到该项**起始块的末尾**（`tap_planner.h +23/−1`，`hybrid_program.cpp +22/−25`，附两条真测试） | 10 号把 `inside_vision`（`range.begin < frontier < range.end`）写成"严格双侧判据"——**它有洞** |
| `af60c489cd` | 11-10 | 带 hybrid 缓存的多卡 Engine **启动即失败**：层检查只读一个 shard 的 StateImage 池，state/KV-record/fork 拷贝只传一个 stream；改为跨 shard 检查 + 每 rank 各自的 compute stream | 我推的「层切分与 Host tier 永远不可能在同一进程里测」是 **base 的启动缺陷**，不是引擎边界；`38a10aa49a feat(serve): keep the hybrid cache over several devices with --use-alt-prefix-caching` 已解 |

**直接后果**：Phase B 扫描内核 docstring 里写的「双卡行是 Legacy 模式，hybrid 块缺席是结论不是缺口」，**只对钉住的 base 成立**，不能作为引擎性质写进报告 18。

### 5.3 上游给出的可引用实测数（Host tier 收益，RTX 4090 / Qwen3.8-27B）

- 266 例重放：modelled prefill **79,045 s → 74,855 s**；42 例更快，8 例更慢（最多 **+12.1%**）。
- issue #222 的轮转场景：**3 GiB Host tier 复用 82.4 %** prompt token（修前 **5.8 %**）；**4 GiB 复用 79.9 %**（修前 17.5 %）。
- **3 GiB 优于 4 GiB，非单调** —— 与 08 号报告对「统一 slab 池给 B×ctx 扫描引入交互项、曲线不单调」的警告一致，现在有第三方实测支撑。

### 5.4 分支线其余未读但相关的提交（250 条中命中 42 条）

`cc65dbb398 feat(kv): bf16p, BF16 KV rows packed losslessly`（**新的 KV dtype**，16 号容量表只有 bf16/int8）；`1189298679 feat(experts): read disk experts from a drive mirror or straight into device memory`；`cb2419f95a fix(runtime): default device snapshot slots yield to the KV capacity`；`43fd83d838 fix(qwen4_exp): run staged GGUF host misses eagerly under WDDM`（呼应 06 号 WDDM 那条）；`7414bfae27 perf(qwen4_exp): pin context-cache pages while the Engine is idle`；`452acabb11`/`af339d1fd5` 的 `docs/maintainer/hybrid-prefix-cache-spec.md`（**上游维护者规格文档，我们没读过**）。

### 5.5 处置记录

06–14、15、16、00-SUMMARY 共 **12 份报告**已在 H1 下插入"底座口径"注（每份 1 处，插入前逐份留 `.bak-<UTC>-before-lineage-note`）；15/16 注为"不受影响"。`pw` 12 份全部 ALLOW。

### 5.6 顺带量到的一条环境事实：Kaggle T4 镜像的 libcuda 在哪

`kvmem-scan-sm75` v1 在 `cuda_driver` 阶段就失败（`no libcuda.so in the image`），九个候选目录全空。用探针内核 `<kaggle-user>/kvmem-driver-probe` 一次问清（结果 `kvmem-study/remote/driver-probe/driver_probe.json`）：

| 判据 | 实测 |
|---|---|
| `ctypes.CDLL("libcuda.so.1")` | **成功**，`cuInit(0)` 返回 0，`cuDeviceGetName` → `Tesla T4` |
| `/proc/self/maps` 里的真实路径 | **`/usr/local/nvidia/lib64/libcuda.so.580.178.04`** |
| 该目录内容 | `libcuda.so` → `libcuda.so.1` → `libcuda.so.580.178.04`，外加 `libnvidia-ml.so*`、`libcudadebugger.so*` |
| 其余目录 | `/usr/local/cuda/{lib64,lib64/stubs,targets/...}` 与 `/usr/lib/x86_64-linux-gnu`、`/usr/lib64`、`/usr/lib` **全部无匹配** |
| 链接后能否运行 | `nvcc -arch=sm_75 ... -lcuda` 编译通过，运行输出 `driver ok: 2 device(s), first=Tesla T4` |

→ **判据顺序应当是「先问 loader，再扫目录」**：`find / -name 'libcuda*'` 在这台机器上只命中 Julia 源码里的同名 `.jl` 文件，扫目录会得出"没有驱动库"的错误结论，而 `ctypes` 一行就给出真路径。
同一修法已进 `scan_kernel.py`：`CUDA_LIB_DIRS` 前置 `/usr/local/nvidia/{lib64,lib}`，并新增 `loader_path()`（`ctypes` + `/proc/self/maps`）作为兜底；本地实跑该函数返回本机 `/usr/lib/x86_64-linux-gnu/libcuda.so.595.99.02`，同目录 unversioned 符号链接存在。
