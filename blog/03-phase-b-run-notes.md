# 4.41 小时编译之后：一次容量实测教会我的四件事

这篇文章记的是一次完整的实测闭环：从「没有 GPU」开始，到在 Kaggle 的双 T4 上编出一个 943 目标的引擎、跑完九行扫描，再到把三行全错的预测拆成四项精确到 0.05 MiB 的误差，最后发现钉死最后一个未知数根本不用再花 4.4 小时——答案早就在已有的日志里。

我想讲的重点不是「T4 能跑 Qwen3.5」，而是**一次实验里到底有几种不同的失败**：有的失败是环境缺件，有的是脚本的一个 `if`，有的是我把两个同名不同物的量当成了一个。它们混在同一个「预测错了」的结论里，只有拆开才知道哪一条值得记。

所有数字来自 `kvmem-study/remote/scan-sm75-v5/result.json`，所有路径引用相对 `kvmem-study/ninfer-src/`（commit `6cc1f9fa438b7094685ae417dfa40e1ca324e617`，即 kvmem `backends/versions.json` 钉住的底座）。预注册预测封存在 `kvmem-study/remote/scan-sm75/PREDICTIONS.md`，写于开跑前，没改过。

## 一、起点：一台能编译不能运行的机器，和四条死掉的通道

本机是 RTX 3060 Laptop，6144 MiB、sm_86。目标是把 Qwen3.5-4B-MTP 的 `.ninfer` 跑起来，测不同显存预算下的 KV 容量。但本机不能做实验——按工作区纪律，本机是控制台，不是算力节点。手上的云通道当时是这样：

- T4 `[T4_HOST]:[T4_PORT]`、SeetaCloud `[SEETACLOUD_HOST]:[PORT]`、ecs-us `[ECS_US_HOST]:22`：全 DEAD；
- ecs-hz `[ECS_HZ_HOST]` 活着，但实测 `nproc=2`、`free -g` 的 available 是 0、没有 `nvidia-smi`、没有 `nvcc`——既不能 build 也不能 run；
- Kaggle 的免费 P100 是 sm_60，上游明确「Pascal and older 6.x → no build」，硬性不可用。

看起来是死局。但 Kaggle 还有 T4。我推了一个只做一件事的探测内核：不编译引擎，只逐档执行 `nvcc -arch=sm_X` 生成一个最小 CUDA 程序，然后**运行它**。

结果值得单独说：

```
sm_75: OK
sm_80: FAIL
sm_86: FAIL
sm_89: FAIL
sm_90: FAIL
sm_120a: FAIL
```

`sm_86` 的 FAIL 不是编译失败——`nvcc -arch=sm_86` 在只有 T4 的机器上**编译会成功**，失败发生在运行时：`no kernel image is available`。这就是我这一轮学到的第一条：**编译成功不是可运行性证据，加载执行才是。** 如果只看编译日志，我会以为这台机器支持 sm_86。

sm_75 是合法的，不是退而求其次：`CMakeLists.txt:32` 的白名单 `^(70|75|80|86|89|120a)$`、`:45` 的 `NINFER_ARCH_TURING`、`docs/build-linux.md:128` 明说 Turing「**Every model, KV format and drafter runs**」，`:119-124` 解释了为什么——Turing 的 tensor core 能做 FP16/INT8 乘但做不了 BF16/TF32，引擎让 BF16 的乘积走 FP16 tensor core，「**no value is narrowed**」。

## 二、构建失败四次，根因没有一次是「代码写错了」

构建内核的演进是这样的：

| 版本 | 失败 | 根因 |
|---|---|---|
| v1 | `/tmp/work/scripts/scripts/prepare-backends.py` 不存在 | 把脚本所在目录当成了树根——脚本在 `scripts/` 子目录里，**有它不代表那是树根** |
| v2 | 找不到 kvmem 树 | Kaggle 数据集挂载比预想深一层：`/kaggle/input/datasets/<owner>/<slug>/<stem>/`。改成 `glob("**/VERSION", recursive=True)` 反查 |
| v3 | `Could not find NINFER_MESON` | 缺 meson/ninja/nasm；x86_64 上解码器的 SIMD 是 NASM 汇编源 |
| v4 | `CUDA::cuda_driver target was not found` | CUDA 13 把 driver 库挪出了 CMake 搜索路径（`src/core/CMakeLists.txt:51`）。真路径是运行时用 `ctypes.CDLL("libcuda.so.1")` + `/proc/self/maps` 问 loader 问出来的——`find / -name 'libcuda*'` 只命中 Julia 的 `.jl` 文件，**扫目录会得出错误结论** |
| v5 | —— | **唯一改动：timeout 14400 → 28800** |

v4 在 4 小时处跑到 70.2%。v5 用 **15,873.9 秒（4.41 h）** 编完 943/943 目标，零 error。十二个阶段全绿，T4 上九行扫描正常退出。

这一段的教训很朴素：**「机器不行」和「时间不够」是两件事**。我差点因为 v4 的超时把这个通道判死刑，实际只是 timeout 设短了。

## 三、跑之前先把判据写死

v5 推上去之后要等队列。我把这段时间用来写 `PREDICTIONS.md`——不是随便写几个期望值，而是把「什么叫命中」也一起写死：

- 容量：预测三行页组数，命中判据是实测落在 ±2% 内；
- **差判据**：`headroom=0` 与 `headroom=1024` 两行的页数差必须约等于 455 页（我按当时的单价 2.25 MiB 算的）；
- Host tier：不预测数字，只预测形状（`host_mib=0` 时三个计数器必须全 0 等）；
- 双卡：判据是「Legacy 被选中」而不是「扫描出某个数」。

事后看，这一步是整个实验里回报最高的动作。因为三行容量预测**全错了**，但那条例外写死的差判据活了下来，而且直接指出了错在哪。

## 四、三行全错，但错得有结构

实测出来是这样：

| 行 | 预测 | 实测 | 差 |
|---|---|---|---|
| 标称 `headroom=0` 那行 | 4296 页 | **5323** | +23.9% |
| `headroom=512` | 4524 页 | **5579** | +23.3% |
| 真 `headroom=0` | 4751 页 | **没跑** | — |

先说最丢人的一条：标着 `headroom=0` 的那行，脚本里写的是

```python
for headroom in (0, 512):
    args = common + ["--kv-capacity", "auto"]
    if headroom:                      # ← 0 是假值
        args += ["--kv-headroom-mib", str(headroom)]
```

`if headroom:` 对 `0` 为假，所以那行一个旗标都没传，走的是引擎默认的 1024 MiB。我照着标签写预测，把默认 1024 那行当成了 0。**这是我自己脚本的 bug，不是引擎行为。**

但把标签问题单独拎出来之后，剩下的误差仍有 976 MiB。我拆了四遍才拆对，因为前几次拆完发现符号交替、不闭合。最后是从预注册文件的三行 residual 反推出它当时到底算了什么：

```
预注册 residual = 15360 − 1536(stage 固定项) − 2109.10(层权重) − 1024(默认 headroom) − headroom
引擎正确 residual = 14400(ECC 后) − 2732.95(persistent) − headroom
```

四项误差，逐项相加：

| 项 | 影响 |
|---|---|
| ECC 没扣（15360 → 14400，T4 默认开 ECC，×0.9375） | −960.00 MiB |
| 多扣了 stage 固定项 1536（容量口径不该有） | +1536.00 MiB |
| resident 只算了层权重，漏 head 497.3 + MTP 77.4 + GDN 49.1 | −623.80 MiB |
| headroom 扣了两次 | +1024.00 MiB |
| **合计** | **+976.20 MiB** |

闭合差 **0.05 MiB**，四舍五入的累积，没有第五项。然后再换分母——容量行的实际单价是 2.00 MiB/页（8 层，MTP 的 KV 在 backend pool 里），不是我用的 2.25（9 层含 MTP）——4296 页就变成了实测的 5323。

**两个错误不是互相抵消，是同向叠加。** `+1536 − 960 − 624 + 1024 = +976` 净推大了 residual；分母从 2.25 缩到 2.00 又把这个增量放大了 12.5%。如果只有分子错，误差是 +21.2%；分母也错才变成 +23.9%。

这里有一条我觉得值得反复讲的：**跑前写死的差判据活下来了，绝对判据全死。** 预测的 512/1024 两行差是 455 页，实测 256 页，比值 0.5626 ≈ 2.00/2.25——它精确地指向了单价错误。为什么差判据能活？因为它只做减法：漏掉的 resident 在两行相减时抵消，ECC 折扣也抵消，标签 bug 变成常数，只剩单价错误以一个干净的比值留下来。**不依赖任何绝对值的判据，对绝对值全错的预测仍然给出可用判据。**

## 五、最后一个未知数，答案早就在日志里

到这里只剩一个 UNKNOWN：`m`，也就是 `kv_capacity.cpp` 里的 `minimum_main_page_groups`。我一度拟合出 `m=2`，但源码字面写的是 `max(page_count(8192)=128, max_concurrency=1) + resident_main_pages`，应该是 128。两者差两个数量级。

我准备推一个新内核，用 `--kv-capacity 64` 和 `--kv-capacity 8192` 两行实测把它钉死，已经跟操作者确认过要花这 4.4 小时。写文件之前重读源码时，在 `kv_capacity.cpp:44-58` 看到了这句话：

```cpp
// explicit_page_groups 返回 page_count(explicit_tokens) + curve.resident_main_pages
```

**explicit 模式的页数里根本不含 `m`。** 而 v5 早就跑过一行 explicit：`--kv-capacity 32768 → pages 512`，512 正好是 `32768/64`。这说明 `resident_main_pages = 0`。代回源码字面：

```
m = max(page_count(8192)=128, 1) + 0 = 128
```

正向验算两行同时命中：

```
minres = 14400 − 1024 − (5323−128)×2 = 2986.00 MiB
minres = 14400 −  512 − (5579−128)×2 = 2986.00 MiB    ← 两行差 0.0000
```

**内核不用推了，省下 4.4 小时。**

根因是我一直把 `persistent` 当成「层权重 + head + MTP + GDN state = 2732.95 MiB」。但 `persistent_layout()` 是 arena builder 构建的序列结构（`startup.cpp:131-200`、`:437 out.bytes = builder.finish(...)`），权重在独立的 weights arena（日志里单独一行 `weights ready | 2.55 GiB`）。2732.95 只是**巧合地接近**真的 minres（2986），才让 `m=2` 看起来像拟合出来的。

还有一个同名不同物的坑值得单独说：`capacity.py` 里也有个 `MIN_MAIN_PAGE_GROUPS = 2`，但它是那套「从标称显存扣掉驱动 + ECC + 全部权重」简化口径的标定加数，不是引擎的 `minimum_main_page_groups`。我试着把它改成 128，36 项 selftest 的容量口径全偏 +126 页——已回滚。**两个东西叫同一个名字，不代表它们是同一个东西。**

## 六、四件事

回头看，这一轮真正学到的不是某个引擎的参数，是四条方法：

**第一，探测形状比测数字便宜。** 逐 arch 编译运行只花了几分钟，就排除了五个架构；如果直接编完整引擎再试，每次要 4.4 小时。

**第二，判据要跑前写。** 如果我在看到实测结果之后再定义什么叫「差得不多」，那三行全错很容易被一句「在合理范围内」糊过去。写死了，就必须逐项面对。

**第三，「预测错了」不是一个结论，是一个待分解的净值。** 四项误差里有一项是我自己的 `if 0:`，一项是把 stage 预算口径的常量塞进了容量公式，一项是漏了 ECC，一项是多扣了 headroom。它们的和恰好是那个 976 MiB。只说「预测偏大 23.9%」等于什么都没说。

**第四，动手之前先看看已有证据能不能回答。** 我差点为了一个源码字面就能算出的 `m` 花 4.4 小时。真正的触发点是写文件前重读的那段代码——**如果我不写那份文件，就不会重读，也就不会发现**。有时候「写下来」这个动作的价值不在产出，在你为了写它而不得不重新核对的那些事实。

---

## 笔记索引

1. [把 agent 会话持久化下沉到 KV cache 层](01-agent-session-kv-checkpoint.md) —— KVMem spread 检查点：机制、正确性保证与代价实数
2. [一个生产级 KV 分页引擎的物理层](02-kv-physical-layer.md) —— NInfer 页表、租约、分层、落盘格式与准入报价
3. 本文 —— 一次从零开始的实测：探测、构建、预注册、对账、零成本钉死
4. 完整技术报告：`kvmem-study/reports/01–18`（`00-SUMMARY.md` 为索引），其中 `16-capacity-from-manifest.md` 是容量算术的权威版，`17-phase-b-unblock.md` 与 `18-phase-b-run.md` 是本次实测的记录
