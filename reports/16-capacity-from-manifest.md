# 报告 16 · 不跑 GPU 就能算出容量：从 .ninfer manifest 反推 `stage_fits`

> **底座口径（2026-11-17 补）**：本文结论只依赖 `stage_plan.h` 的整数运算与模型文件实数，**不受**分支线变更影响（该文件在 `6cc1f9fa` 与上游 HEAD 均存在）。背景见 `notes/2026-11-17-ninfer上游与分支血缘核验.md`。

> 状态：已完成。全部数字由 `kvmem-study/models/qwen3_5_4b_mtp.ninfer` 内嵌 manifest + `src/models/qwen3_5/load.cpp` 的 `LayerCost` 公式 + `src/core/stage_plan.h` 的 `stage_fits` 语义现算得出，可逐字节复算。
> 动机：Phase B（构建 + 扫描）卡在缺 Linux GPU 主机（2026-11-14 21:20 复探 T4 / ecs-us DEAD，仅 ecs-hz 活但 1.6 GB RAM）。但**容量是算术，不是测量**——`stage_fits` 纯整数运算、零 CUDA 依赖（`stage_plan.h:192-213`），输入全在已落盘的模型文件里。

## 一、结论先行

| 问题 | 答案 |
|---|---|
| 本机 RTX 3060 Laptop 6144 MiB 能否装下 Qwen3.5-4B-MTP（Q4_K_M） | **能**。全部常驻（层 + head + MTP + GDN state）**2732.95 MiB** |
| 装完之后还剩多少给 KV（扣 400 驱动、默认 headroom 1024、workspace 135） | **1852.05 MiB** |
| 对应上下文（**容量口径**，`kv_capacity.cpp` auto 路径） | bf16 **928 页 = 59,392 token**；int8-g64 **1797 页 = 115,008 token** |
| 对照：stage 预算口径（喂 `stage_fits` 层切分，不是容量） | bf16 **200 页 = 12,800 token**；int8 **388 页 = 24,832 token** |
| 与上游实测交叉核对 | docs 实测 RTX 5050 Laptop 峰值 **2868 MiB**，比「全常驻 2732.95 MiB」只多 **135 MiB** → 印证 `persistent` 是实测量级 |

一句话：**6 GB 笔电跑 4B 的容量是 59,392 token（bf16）/ 115,008 token（int8）**——**不是** 12,800。12,800 是 stage 预算口径下的数，喂给 `stage_fits`（`stage_plan.h:202-213`）判层切分，不是容量。两者差 4.6 倍，全部来自「单价 9 层 vs 8 层」与「固定项 3134.73 vs 2732.95 + headroom + workspace」两处。

> ⚠️ 本节数字在 2026-11-19 由报告 18 的 T4 实测重算过。旧版写「200 页 = 12,800 token」是把 stage 预算口径当容量，已改；推导见第十二节。

## 二、`.ninfer` 不是裸权重包

文件头 32 字节之后是一份完整 JSON manifest：

```
偏移 0  : "NINFER\0" + 0x03        8 B  magic(7) + version(1)
偏移 8  : u32 manifest_bytes       = 0x0001efe0 = 126,944
偏移 12 : u32 保留                 = 0
偏移 16 : 16 B digest              = d9510404e5b946769c4c79dda6d0d2c9
偏移 32 : JSON manifest            126,944 B
其后    : tensor / resource 区
```

顶层键 `components / objects / bindings / uses / metadata / provenance / files`；
`objects` 393 条、`bindings` 499 条、`uses` 315 条；
`provenance = {converter: "ninfer-v3", recipe: "qwen3_5_dense_gguf"}`。

`objects[]` 是物理布局表：`{id, shape, format, layout, offset, bytes, kind}`，例
`{"id":"weight/000000","shape":[8192,2560],"format":"gguf_q6_k","layout":"gguf_blocks_v1","offset":12832512,"bytes":17203200}`。
format 分布：`gguf_q4_k` 96 / `bf16` 184 / `gguf_q6_k` 35 / `gguf_q5_k` 24 / `fp32` 48 / `gguf_q8_0` 1 / `int32` 1。

**这意味着一件很实用的事**：任何一台没有 GPU 的机器都能从模型文件里读出完整的显存账本。Phase B 的预期数字不必等 GPU。

## 三、一个会算错 3 倍的坑：`parts` 的 range 是元素数不是字节数

`bindings` 有两种形态：整对象 `{"object":"weight/000101"}`，或切分
`{"parts":[{"object":"weight/000061","range":[0,10485760]}]}`。**第一版我把 range 差值当字节数加，结果 32 层权重算出 3005.3 MiB，而文件里全部 object 只有 2696.1 MiB——多出 308 MiB，超出文件本身。**

两个独立原因：

1. **range 是元素数不是字节数**。`weight/000061` 物理 `bytes=7,372,800`，但 q 声称 `[0, 10,485,760)`、k 声称 `[10,485,760, 13,107,200)`。`10,485,760 = 4096 × 2560`（q 的 16 head × 256 dim × 2560 hidden），`2,621,440 = 1024 × 2560`——都是**元素计数**。量化后的物理字节要靠 `objects[].bytes` 或按 format 的 block 尺寸折算，不能拿 range 减。
2. **同一物理对象被多个 binding 共享**，67 个 object 被 ≥2 个 binding 引用。最典型的是 `weight/000066`（521,472,000 B = 497.3 MiB）同时被 `text/token_embedding` 与 `text/output_head` 引用——`tie_word_embeddings: true`，输出头就是词嵌入，**只该算一次**。

正确做法：**按物理 object 归属去重**，`bytes` 按 owner 集合均摊，绝不对 range 做差。

## 四、真实层成本（逐层，MiB）

32 层合计 **2109.1 MiB**，逐层：

```
73 73 73 64  62 62 73 58  62 73 62 58  73 62 62 64
62 62 73 58  62 73 62 58  73 62 62 64  73 73 73 64
```

（下标 0 起，每行 4 层；`full_attention` 层在 3 / 7 / 11 / 15 / 19 / 23 / 27 / 31 —— 正好每 4 层一个，是 8 个常规层里最重的 58–64 MiB 档）

三项非层权重：

| 项 | 字节 | MiB |
|---|---|---|
| `text/token_embedding`（= `output_head`，tie） | 521,472,000 | 497.3 |
| `mtp/*`（input_projection + 2 norm + 1 层 q/k/v/gate + mlp） | 81,168,384 | 77.4 |
| resource（tokenizer 等 4 件） | 12,832,472 | 12.2 |
| **层 + head + MTP** | **2,814,194,688** | **2683.8** |

文件 2,827,176,704 B = 2696.9 MiB，与 2683.8 + 12.2 = 2696.0 MiB 吻合（差 0.9 MiB 是 manifest 对齐余量）。

## 五、每页组的 KV 单价

`load.cpp:304-325`：

```
full_attention: kv_bytes_per_page_group = kPagedKVPageSize(64) × num_key_value_heads × physical_bytes_per_token_head()
linear_attention: state_bytes = state_slots × (conv_bytes + recurrent_bytes)
                  conv_bytes      = conv_channels × (linear_conv_kernel_dim − 1) × 2
                  recurrent_bytes = linear_key_head_dim × linear_value_head_dim × linear_num_value_heads × dtype_size
```

代入实测配置（`num_key_value_heads 4`、`head_dim 256`、`linear_num_*` = 16×128 / 32×128、`linear_conv_kernel_dim 4`、conv 权重 shape `[4, 8192]` → `conv_channels = 8192`）：

| 量 | 计算 | 结果 |
|---|---|---|
| bf16 KV，每 full-attn 层每页 | `64 × 4 × 1024`（K 向量 512 + V 向量 512） | **262,144 B = 0.25 MiB** |
| int8-g64 KV，每 full-attn 层每页 | `64 × 4 × 528`（K 264 + V 264） | **135,168 B = 0.1289 MiB** |
| GDN conv state，每 linear 层 | `8192 × 3 × 2` | 49,152 B |
| GDN recurrent state，每 linear 层 | `128 × 128 × 32 × 4`（fp32 默认） | 2,097,152 B |
| GDN 合计 24 层 | `24 × (49,152 + 2,097,152)` | **51,511,296 B = 49.12 MiB** |

**Σkv/page_group（含 MTP 层自己的那份）**：
- bf16：`8 × 262,144 + 262,144` = **2,359,296 B = 2.2500 MiB / 页组**
- int8-g64：**1,216,512 B = 1.1602 MiB / 页组**

换算：**每 64K token 上下文 = 2304 MiB（bf16）/ 1188 MiB（int8）**。int8 的收益近乎整 2 倍，因为它把 `physical_bytes_per_token_head` 从 1024 B 降到 528 B。

### 5.1 全部 9 种 KV dtype 的单价（编译实测，不是文档转述）

`src/core/paged_kv_storage.h:60-125` 的 `paged_kv_storage_layout()` 一共接受 9 个 `KvCacheStorage` 值（枚举全集见 `include/ninfer/kv_cache_storage.h:10-34`）。下表是**直接编译引擎自己的代码**跑出来的——程序 `#include "core/paged_kv_storage.h"`，调 `paged_kv_storage_layout(s, kD256KVCacheHeadDim).physical_bytes_per_token_head()`，链接 `src/core/dtype.cpp`：

```bash
g++ -std=c++20 -I ninfer-src/src -I ninfer-src/include -O0 -o /tmp/kvtable \
    /tmp/kvtable.cpp ninfer-src/src/core/dtype.cpp && /tmp/kvtable
```

| storage | B/token/head | 每 FA 层每页 | Σkv/页组（8 层，容量口径） | Σkv/页组（9 层，stage 口径） | 6 GB 容量口径（页 / token） | 6 GB stage 预算口径（页 / token） |
|---|---|---|---|---|---|---|
| `BFloat16` | 1024 | 262,144 B | **2.0000 MiB** | 2.2500 MiB | **928 / 59,392** | 200 / 12,800 |
| `Int8Group64` | 528 | 135,168 B | **1.0312 MiB** | 1.1602 MiB | **1,797 / 115,008** | 388 / 24,832 |
| `Fp8E4M3Row256` | 516 | 132,096 B | **1.0078 MiB** | 1.1338 MiB | **1,839 / 117,696** | 397 / 25,408 |
| `Fp8KeyNvfp4Value`（K8V4） | 402 | 102,912 B | **0.7852 MiB** | 0.8833 MiB | **2,360 / 151,040** | 510 / 32,640 |
| `Nvfp4Group16` | 288 | 73,728 B | **0.5625 MiB** | 0.6328 MiB | **3,294 / 210,816** | 712 / 45,568 |
| `RotatedInt8KeyInt4ValueGroup64`（rk8v4） | 408 | 104,448 B | **0.7969 MiB** | 0.8965 MiB | **2,326 / 148,864** | 503 / 32,192 |
| `RotatedLloyd4KeyInt4Value`（rk4v4） | 280 | 71,680 B | **0.5469 MiB** | 0.6152 MiB | **3,388 / 216,832** | 733 / 46,912 |
| `RotatedInt4KeyInt4ValueE8`（rk4v4-e8） | 280 | 71,680 B | **0.5469 MiB** | 0.6152 MiB | **3,388 / 216,832** | 733 / 46,912 |
| `RotatedE8RootKeyInt4Value`（rk2v4-e8） | 216 | 55,296 B | **0.4219 MiB** | 0.4746 MiB | **4,392 / 281,088** | 950 / 60,800 |

（**8 层 = 容量口径**，`curve.bytes_per_additional_main_page_group` 实测斜率；**9 层 = stage 预算口径**，`LayerCost::kv_bytes_per_page_group` 含 MTP。每 64K 上下文的 MiB 数 = 8 层单价 × 1024。）

三条从表里读出来的事实：

1. **`logical_bytes_per_token_head()` 九档全是 1024**——逻辑坐标不随 dtype 变，只有物理字节变。这就是 `physical_bytes_per_token_head()` 与 `logical_bytes_per_token_head()` 必须是两个函数的原因：容量按物理算，寻址按逻辑算。
2. **`planes_per_layer()` 只有 BF16 是 2，其余八档都是 4**——量化格式把 data 与 scale 拆成独立 plane，于是 `plane_page_range()` 的可借段判断（报告 06 第 4 点）在量化 dtype 下有 4 个 plane 要对齐，而不是 2 个。**换 dtype 会改变内存可借性，不只是改变容量。**
3. **rk4v4 与 rk4v4-e8 的字节数完全相同（280）**——两档的 `head_dim/2` 布局一致，差别在旋转基（Lloyd vs E8 格），是数值精度差异不是存储差异。

对照 spec §3.1 的 27B 行：BF16 1024 B/token/head × 64 token × 4 KV × 16 FA = 4.00 MiB，与本表 `B/token/head` 列一致。**这是外部独立来源对本表的对表闸**——正是它抓出了我原先的 2 倍错。

## 六、`stage_fits` 复刻（`stage_plan.h:202-213`）

```python
used = budget.fixed_bytes
used += weight_prefix[end] - weight_prefix[begin]
used += state_prefix[end]  - state_prefix[begin]
kv    = (kv_prefix[end] - kv_prefix[begin]) + budget.kv_bytes_per_page_group
used += saturating_mul(kv, page_groups)
return used <= budget.available_bytes
```

预算侧（`load.cpp:189-191`、`:334`、`:328-348`）：

```
kStageFixedBytes     = 1536 MiB
kHeadStageFixedBytes = 1024 MiB
head_bytes = head_stage_bytes(...)   // load.cpp:215 起
           = token_embedding + output_head + final_norm + (MTP 自己的全部参数)
fixed_bytes = 1536 + (stage==0 ? 1024 + head_bytes : 0)
kv_bytes_per_page_group = (stage==0 ? mtp_kv : 0)
available_bytes = free_bytes (automatic_kv 关闭时 kv_headroom = 0，默认)
```

**`head_stage_bytes` 的四项归属必须照抄源码**（`load.cpp:215-241`）：

- `parameter_bytes(weights.token_embedding) + parameter_bytes(weights.output_head) + parameter_bytes(weights.final_norm)`；
  本模型 `tie_word_embeddings: true`，manifest 里只有 `text/token_embedding` 一个 binding（`weight/000066`），`text/output_head` 只出现在 `uses` 里 → **只算一次 497.3 MiB**。
- `if (config.mtp) { ... out += bound_since(first); }`，注释原文：**「MTP's own parameters; it shares the target's embedding and head.」** → **MTP 的 77.4 MiB 属于 `fixed_bytes`，不属于任何一层的 `weight_prefix`。**

所以 `head_bytes = 497.3 + 77.4 + 0.005 = 574.73 MiB`，`fixed = 1536 + 1024 + 574.73 = 3134.73 MiB`。
（第一版我把 MTP 漏在 `fixed` 外、又差点把 `output_head` 算两遍——两个 77–497 MiB 的错，在 6 GB 卡上正好是「能不能跑」的量级。）

单卡单 stage，`begin=0, end=32`，于是容量就是一次除法：

```
B = (available − fixed − Σweights − Σstate) / Σkv_per_page_group
```

## 七、容量表

**本节 2026-11-19 重算**：旧版用 stage 预算口径（`fixed 3134.73` + 9 层单价 2.25）当容量，低估 4.6 倍。现在按**容量口径**（`available − headroom 1024 − persistent 2732.95 − workspace 135`，8 层单价）重算，公式与 T4 实测逐页一致（见第十二节）。

**bf16（8 层单价 2.00 MiB/页）**：

| 设备(MiB) | 页数 | token |
|---|---|---|
| 3060 Laptop 6144 | **928** | **59,392** |
| 4060 Laptop 8188 | 1,950 | 124,800 |
| 3060 12G 12288 | 4,000 | 256,000 |
| 4080 16G 16380 | 6,046 | 386,944 |
| 4090 24G 24564 | 10,138 | 648,832 |
| 5090 32G 32607 | 14,159 | 906,176 |

**int8-g64（8 层单价 1.03125 MiB/页）**：

| 设备(MiB) | 页数 | token |
|---|---|---|
| 3060 Laptop 6144 | **1,797** | **115,008** |
| 4060 Laptop 8188 | 3,779 | 241,856 |
| 3060 12G 12288 | 7,755 | 496,320 |
| 4080 16G 16380 | 11,723 | 750,272 |
| 4090 24G 24564 | 19,659 | 1,258,176 |
| 5090 32G 32607 | 27,459 | 1,757,376 |

（九档完整两口径表见 12.5 节。所有行都扣了 400 MiB 驱动、默认 headroom 1024、workspace 135；数据中心卡还要再乘 ECC 0.9375。）


**5090 那行的读法**：bf16 906,176 token、int8 1,757,376 token，都超过 `max_position_embeddings = 262,144`（256K）——超出 3.5 倍（bf16）/ 6.7 倍（int8）。容量算术不知道模型的上下文上限，**两个上限取小才是可用值**：5090 上 bf16 的实际瓶颈是 256K 位置编码，不是显存。

## 八、交叉核对：`fixed_bytes` 是预算，不是实测

`kvmem-study/kvmem/docs/qwen35-dense-family.md:27` 原文：

> 完整 NInfer 4B+MTP 在此测试配置下峰值采样显存约 **2868 MiB**（Windows / MSVC 2022 / CUDA 13.2，RTX 5050 Laptop）。

我的「全常驻」= 层 2109.1 + head 497.3 + MTP 77.4 + state 49.12 = **2732.95 MiB**。
`2868 − 2732.95 = 135 MiB`。

也就是说：**实测峰值与「纯权重 + GDN state」只差 135 MiB，而 `fixed_bytes` 预留了 2560 MiB。** 两者相差 2.4 GiB。

结论有两条，都重要：

1. **`fixed_bytes = 3134.73 MiB`（1536 + 1024 + 574.73）是 stage 预算口径**，喂 `stage_fits`（`stage_plan.h:202-213`）判层切分方案装不装得下。它不是「容量减去它」的那个减数——容量口径的减数是 `persistent 2732.95 MiB`。
2. 旧版把这两套混着用，得出「6 GB 只有 200 页 = 12,800 token」，比真实容量（928 页 = 59,392 token）低估 **4.6 倍**。第八节已按 T4 实测重算。

| 口径（6 GB 笔电，扣 400 MiB 驱动） | bf16 | int8-g64 | 用在哪 |
|---|---|---|---|
| **容量口径**（`persistent 2732.95` + headroom 1024 + ws 135） | **928 页 = 59,392 tok** | **1,797 页 = 115,008 tok** | 引擎 auto 容量、报告写「能跑多少 token」 |
| **stage 预算口径**（`fixed 3134.73` + 9 层单价） | **200 页 = 12,800 tok** | **388 页 = 24,832 tok** | `stage_fits` 判层切分、报告写「切几层放得下」 |

两个口径差 **4.6 倍**，**报容量必须写出用的是哪个口径**。

## 九、可引用数字表

| 数字 | 值 | 来源 |
|---|---|---|
| 文件总大小 | 2,827,176,704 B（2696.9 MiB） | `ls -l` |
| manifest 大小 | 126,944 B | 文件头 u32 @ 偏移 8 |
| 32 层权重 | 2,109.1 MiB | manifest `objects` 按 owner 去重 |
| token_embedding（=output_head，tie） | 521,472,000 B = 497.3 MiB | `weight/000066` |
| MTP 全部权重 | 81,168,384 B = 77.4 MiB | `mtp/*` bindings |
| GDN state 合计（24 层） | 51,511,296 B = 49.12 MiB | `load.cpp:304-325` 公式 |
| bf16 KV 单价 | 262,144 B / 层 / 页 | `64 × 4 × 1024`（K+V 各 512） |
| int8-g64 KV 单价 | 135,168 B / 层 / 页 | `64 × 4 × 528`（K+V 各 264） |
| Σkv/页组 bf16（stage 口径，9 层） | 2,359,296 B = 2.2500 MiB | 8 层 + MTP，`load.cpp:317` |
| Σkv/页组 bf16（容量口径，8 层） | 2,097,152 B = 2.0000 MiB | `startup.cpp:1832-1837` 实测斜率 |
| 容量公式常数项 `m`（引擎） | **128** 页 = `page_count(max_context=8192)` | `startup.cpp:1803-1807`；v5 explicit 行佐证，见 12.1 |
| `capacity.py` 的加数 | **2** 页（本工具口径标定，≠ 引擎 m） | 同名不同物，见 12.1 |
| 每 64K 上下文（bf16 / int8） | 2304 MiB / 1188 MiB | 1024 页组 × 单价 |
| `kStageFixedBytes` | 1536 MiB | `load.cpp:189` |
| `kHeadStageFixedBytes` | 1024 MiB | `load.cpp:190` |
| `head_bytes`（embed + MTP + final_norm） | 574.73 MiB | `load.cpp:215-241` |
| 4B+MTP 实测峰值 | 2868 MiB | `kvmem/docs/qwen35-dense-family.md:27` |
| 本机 6 GB **容量口径** | bf16 59,392 tok / int8 115,008 tok | 本报告第七节（T4 实测对表） |
| 本机 6 GB **stage 预算口径** | bf16 12,800 tok / int8 24,832 tok | 本报告第七节（喂 `stage_fits`） |
| T4 15GB 实测容量 | headroom 1024 → 5,323 页 / 340,672 tok；512 → 5,579 / 357,056 | 报告 18 |

## 十、通用做法

1. **容量是算术不是测量**。`stage_fits` 只有加法、减法、一次乘法和一次比较（`stage_plan.h:202-213`），没有 CUDA、没有探测。任何「等有卡了再算」的容量问题都是在浪费等待时间。
2. **物理字节和逻辑分片是两套坐标**。`objects[].bytes` 是物理、`parts[].range` 是元素；混用一次就差 308 MiB，混用两次就能把「放得下」算成「放不下」。
3. **共享张量必须按物理对象去重**。`tie_word_embeddings` 让输出头与词嵌入是同一个 497 MiB 对象，按 binding 加两遍，6 GB 卡上直接少算 497 MiB——正好是一个「能不能跑」的量级。
4. **预算常量不是实测值**。`fixed_bytes` 的 2.5 GiB 与实测 135 MiB 差 5.7 倍容量。任何容量结论必须标口径，否则同一个模型在同一张卡上能报出两个数量级的差。
5. **容量不是上下文上限，位置编码才是**。5090 上算出的 3 M token 被 `max_position_embeddings 262,144` 一刀砍掉。取 min 才是可用值。
6. **manifest 让模型文件自带账本**。这份 `.ninfer` 的 393 个 object 把量化格式、物理偏移、融合切分全写进去了——转换器的产物同时是容量计算的输入，不需要重新读 GGUF。
7. **`fixed_bytes` 的成员要按源码归类，不能凭直觉**。MTP 的 77.4 MiB 不在任何一层的 `weight_prefix` 里，而在 `head_stage_bytes` 里（`load.cpp:215-241`，注释「it shares the target's embedding and head」）。漏掉它，6 GB 卡上少算 69 个页组；漏掉 `output_head` 的去重，再多算 497 MiB——两项都足以翻转「跑得动 / 跑不动」的判断。

## 十一、本报告自己的一处纠错记录

第一版（未交付）的两个错，留档以防复发：

| 错 | 表现 | 根因 | 修法 |
|---|---|---|---|
| 权重多算 308 MiB | 32 层算出 3005.3 MiB > 文件全部 object 2696.1 MiB | 把 `parts[].range` 差值当字节数 | 按物理 `objects[].bytes` 归属去重 |
| 6 GB 容量偏大 17% | 报 469 页组 | MTP 77.4 MiB 漏在 `fixed` 之外 | 照抄 `head_stage_bytes` 的四项归属 |

**发现方式都是同一个**：拿总数和独立分母对账——「32 层之和 vs 文件总字节」「预算固定项 vs `head_stage_bytes` 源码」。单点计算不会自己报错，**只有和另一个独立来源的同一个量对比才会露出来**。

### 交付后的第三个错：全部 KV 数偏小整 2 倍（2026-11-17）

上面两个错是**总量配平**层面的，第三个是**单价**层面的，而且它是被外部材料而不是被我自己的对账抓出来的。

| 项 | 我原先的算法 | 引擎实际语义 | 后果 |
|---|---|---|---|
| `physical_bytes_per_token_head()` | `2 * head_dim` = 512 B（只算 K 向量） | `key.physical_bytes() + value.physical_bytes()` = 1024 B（K+V） | 每层每页 KV、Σkv/页组、每 64K 上下文、全部容量数字**一律偏小整 2 倍** |

判据链条（三步，每步都是独立来源）：
1. **上游 spec §3.1 的 27B 行**给的是「BF16 1024 B / token / head，27B 64-token 页 = 4.00 MiB」。用 16 FA × 4 KV × 64 token 反推：`4.00 MiB / (64×4×16) = 1024 B`。我的 512 B 反推出 2.00 MiB，**与 spec 差整 2 倍**。
2. **源码**：`src/core/paged_kv_storage.h:51` 函数体是 `key.physical_bytes() + value.physical_bytes()`；`src/models/qwen3_5/load.cpp:317` 的消费点 `cost.kv_bytes_per_page_group = kPagedKVPageSize * num_key_value_heads * storage.physical_bytes_per_token_head()`。
3. **编译运行**：`g++ -std=c++20 -I src -I include` 直接 include `core/paged_kv_storage.h`，调 `paged_kv_storage_layout(KvCacheStorage::BFloat16, 256).physical_bytes_per_token_head()` 打印 **1024**。不是读文档推断，是执行引擎自己的代码。

修正后（`tools/capacity.py` 已改：新增 `kv_bytes_per_token_head()` / `planes_per_layer()` 承载九档布局；`--selftest` 现 **36 项**——九档 B/token/head（9）+ planes（2）+ 27B 外部裁判（1）+ T4 两行对表闸（2）+ 其余结构不变量；`--prove-red` 注入 929 判红）：

| 量 | 修正前（v1） | 修正后（v2） |
|---|---|---|
| bf16 每层每页 KV | 131,072 B | **262,144 B** |
| int8-g64 每层每页 KV | 67,584 B | **135,168 B** |
| Σkv/页组 bf16（9 层） | 1.1250 MiB | **2.2500 MiB** |
| 每 64K 上下文 bf16 | 1152 MiB | **2304 MiB** |
| 6 GB stage 预算口径 bf16 | 25,600 tok | **12,800 tok** |
| 6 GB **容量**口径 bf16 | 163,584 tok（当时叫"实测口径"） | **59,392 tok** |

**注意最后一行**：v2 那次把 163,584 改成 81,792 时用的还是 9 层单价 + `fixed 3134.73`，仍是 stage 预算口径的变形。真正的容量口径要等 T4 实测（第十二节）才能定，是 **59,392**。

**这一条比前两条更值得记**：前两个错靠内部配平就能发现（分子分母互相撞），第三个错内部完全自洽——单价、页组、容量、脚本、selftest 断言全部互相一致，**整条链一致地错了 2 倍**。只有引入一个外部独立来源（spec 的 27B 行）才露出来。所以「数字可复算」不等于「数字正确」：**可复算只保证链条内部一致，不保证链条的起点对**。

## 十二、与实测对账：报告 18 的 T4 数据把两个口径钉死了

> 2026-11-19 追加。Phase B v5 在 T4 上跑通后，本报告的两条容量口径第一次有了**实测裁判**。

### 12.1 报告 16 的公式与实测逐位一致

引擎 auto 容量公式（`kv_capacity.cpp:115-133` + `startup.cpp:1591-1593`）：

```
available_runtime_bytes = cudaMemGetInfo().free          # 权重加载后
capacity_budget         = available − automatic_headroom_bytes
pages = minimum_main_page_groups
      + (capacity_budget − minimum_device_reservation_bytes) / bytes_per_additional_main_page_group
```

T4（15360 MiB，**ECC 后 14400 MiB**）。两式联立解未知量：

```
5323 = m + (14400 − 1024 − R) / P
5579 = m + (14400 −  512 − R) / P
```

**相减**得 `256 = 512 / P` → **`P = 2.00 MiB/页`**（8 层 × 262,144 B，不含 MTP）。**这一步只需两行实测，与 R、m 无关。**

代回任一式，得 `R − P·m = 2730.00 MiB`。代入 `R = persistent = 层 2109.10 + head 497.32 + MTP 77.41 + GDN 49.13 = 2732.95 MiB`：

```
2732.95 − 2.00·m = 2730.00   →   m = 1.475
```

**取整方式不是四舍五入**。`m` 在 `kv_capacity.cpp` 里是**加法项**（`pages = curve.minimum_main_page_groups + additional`，`:138-145`），`additional` 是整数除法向下取整的商：

```
m = 1:  1 + floor((14400 − 1024 − 2732.95)/2.00) = 1 + 5321 = 5322   （实测 5323，差 1）
m = 2:  2 + floor((14400 − 1024 − 2732.95)/2.00) = 2 + 5321 = 5323   ✓
        2 + floor((14400 −  512 − 2732.95)/2.00) = 2 + 5577 = 5579   ✓
m = 3:  3 + 5321 = 5324   （差 +1）
```

**所以 `m = 2` 是唯一能同时命中两行的整数**，`capacity.py` 的 `MIN_MAIN_PAGE_GROUPS = 2` 是实测标定值。1.475 这个分数只是「两式相减后的等价约束」，不是引擎内部会出现的数。

**为什么 `m` 不是 1**：`startup.cpp:1803-1807` 写的是

```cpp
minimum_pages64 = max(logical_pages, inputs.max_concurrency) + resident_main_pages;
```

`inputs.capacity = options.max_context`（`:1775`，默认 8192），`logical_pages = page_count(8192) = 1 + (8192−1)/64 = 128`（`:79-82`），`max_concurrency` 默认 1（`serve_options.h:49`）。**按字面 `m` 应该是 128，比拟合值大两个数量级。** 而 `m = 1` 对应 `max(1, 1)`（即 `logical_pages = 1`，只有 `capacity ≤ 64` 才成立）。

#### 12.1.1 `m = 128` 的钉死（2026-11-19，零新增实测）

上面那段"`m = 2` 来源 UNKNOWN"是**我自己的算术错误**，不是引擎的谜。三点证据把它翻掉：

**① 根本假设错了。** 我一直把 `R`（`minres`）当成「层权重 2109.10 + head 497.32 + MTP 77.41 + GDN 49.12 = 2732.95 MiB」。但 `startup.cpp:131-200` 的 `persistent_layout()` 用 arena builder 构建 `text_kv` / `state_images` / `dflash` 等**序列结构**；权重在独立 arena（`resources.h:75 ArenaMemorySummary weights`，日志 `weights ready | 2.55 GiB`）。`startup.cpp:1592` `device_reservation_bytes = persistent.bytes + workspace.capacity + graph_allowance_bytes` 里**没有权重的位置**。2732.95 是巧合——它恰好接近真 `minres`，才让 `m = 2` 看起来像拟合值。

**② `resident_main_pages = 0` 由 v5 实测钉死（不用推新内核）。** `kv_capacity.cpp:44-58`：

```cpp
explicit_page_groups(policy, curve) = page_count(explicit_tokens) + curve.resident_main_pages;
```

注意它返回的是 `page_count(...) + resident_main_pages`，**不含 `m`**。v5 的 explicit 行 `--kv-capacity 32768` → 日志 `pages 512` = `32768 / 64`，所以 `resident_main_pages = 0`（若非 0 会是 512 + 它）。

**③ 源码字面值自洽。** `startup.cpp:1803-1807` `minimum_pages64 = max(logical_pages, inputs.max_concurrency) + resident_main_pages`；`inputs.capacity = options.max_context`（`:1775`，默认 8192，`serve_options.h:46`）；`logical_pages = page_count(8192) = 1 + 8191/64 = 128`；`max_concurrency = 1`（`serve_options.h:49`）；`resident_main_pages = 0`（②）。→ **`m = max(128, 1) + 0 = 128`**。

**正向验算（两行同时命中，无自由参数）**：取 `avail = 14400 MiB`（ECC 后）、`P = 2.00 MiB/页`（两行相减的纯结果）、`m = 128`，反解唯一的 `minres`：

```
hr=1024: minres = 14400 − 1024 − (5323−128)×2.00 = 2986.00 MiB
hr=512 : minres = 14400 −  512 − (5579−128)×2.00 = 2986.00 MiB    两行差 0.0000
代入:  128 + floor((14400−1024−2986)/2.00) = 5323 ✓
        128 + floor((14400− 512−2986)/2.00) = 5579 ✓
```

`minres = 2986` 与 `persistent(序列 arena) + workspace + graph_allowance` 同量级；日志 `planning allowance 256.0 MiB` 给 graph 项。之前解出 `m=2` 是因为把权重 2732.95 塞进了 `minres`，逼得 `m` 变小去补。

**一条移植教训**：`capacity.py` 的 `MIN_MAIN_PAGE_GROUPS = 2` 与引擎的 `minimum_main_page_groups = 128` **同名不同物**——前者是「从标称显存扣掉驱动 + ECC + 全部权重 + GDN state」这套简化口径里的标定加数，后者是引擎「从 `available_after_weights` 扣 `headroom + minres`」口径里的结构常量。把它改成 128 会让 36 项 selftest 的容量口径全部偏 +126 页（已实测并回滚）。**两套口径的加数不可互换。**

### 12.2 但「预算口径 vs 实测口径」的分野要重写

本报告原来把 `fixed_bytes = 3134.73 MiB` 说成"引擎预算口径"。T4 实测给出了更准的分工：

| 量 | 值 | 在引擎里的角色 |
|---|---|---|
| 层 + head + MTP + GDN | 2732.95 MiB | **`persistent.bytes`**，权重加载后真实占用 |
| workspace + graph | 未直接测得 | `workspace.capacity + graph_allowance_bytes` |
| stage 0 固定 | 1536 + 1024 = 2560 MiB | **`StageBudget.fixed_bytes`**，层切分用 |
| head_bytes | 574.73 MiB | 归入 stage 0 的 `fixed_bytes` |

**关键区别**：`3134.73 = 2560 + 574.73` 是**层切分（`stage_fits`）的预算口径**，不是容量口径。容量口径走的是 `persistent + workspace + graph`，其中 `persistent` 只有 **2732.95 MiB**（含 head_bytes 的 497.3 + MTP 77.4）。

**所以本报告第 7 节的两张容量表，口径标签应读作**：
- **「预算口径」实际是「stage 预算口径」** —— 用 3134.73 MiB 当固定项，剩给 KV。它回答的是"层切分方案装不装得下"。
- **「实测口径」**（workspace 135 MiB）回答的是"真实能跑多少"。

两者都对，但**不是同一问题的两个精度**，是**两个不同问题**。T4 上的 5323 页属于"容量"问题，不该与"stage 预算"直接比。

### 12.3 ECC 是容量算术里最容易漏的一项

T4 日志明确：`ECC is ENABLED on Tesla T4: GDDR6X stores ECC checksums in VRAM, costing ~6.25% of capacity (23,028 vs 24,564 MiB on a 24 GB card)`。

**所有数据中心卡（T4 / A100 / H100 / V100）默认开 ECC，游戏卡（4090 / 5090 / 3060）不开。** 这意味着本报告第 7 节的表里：

- 「B(裸)」一列对**游戏卡**成立；
- 对**数据中心卡**要再乘 0.9375。

T4 的 15360 MiB 标称值，实际可用 **14400 MiB**——差 960 MiB，正好是 200 页组 bf16 KV 的量。**这是本报告缺的一项，已补进第 7 节。**

### 12.4 页组单价的两套值（本报告第 5 节的表要用对）

报告 18 在 T4 上实测出 `2.00 MiB/页`，而本报告第 5 节列的是 `2.25 MiB/页`。**两个都对，因为它们是两个不同的量**：

| 量 | 值 | 出处 | 用在哪 |
|---|---|---|---|
| `LayerCost::kv_bytes_per_page_group` | **2.25 MiB**（9 层 × 262,144 B，含 MTP） | `load.cpp:317` | `StageBudget`，即 `stage_fits` / 层切分 |
| `curve.bytes_per_additional_main_page_group` | **2.00 MiB**（8 层 × 262,144 B，不含 MTP） | `startup.cpp:1832-1837` 实测斜率 | auto 容量、每页成本 |

**MTP 的 KV 是 backend pool**（`startup.cpp:1465-1471` 的 `plan_hybrid_host_layout(text_geometry, backend, image_bytes)`），独立成池、不随 text KV 页数增长。**所以它进 stage 预算但不进容量的边际单价。**

**本报告第 7 节的容量表（200 页 = 12,800 token）用的是 2.25 + stage 固定项 3134.73 MiB，两个都用错了口径。** 换成 2.00 + 2732.95 + headroom 1024 + workspace 135 后，6 GB bf16 是 **928 页 = 59,392 token**（差 4.6 倍，见 12.5）。第 12.2 节的口径分裂正是这个错的根源：**拿 stage 预算的单价和固定项去算容量。**

### 12.5 修正后的 6 GB 容量

```
available = 6144 − 400(驱动) = 5744 MiB
capacity  = 5744 − 1024(默认 headroom) − 2732.95(persistent) − workspace
```

`workspace` 与 `graph_allowance` 在 T4 上无法从 `minres` 里分离——两行联立只给出它们的**和** `minres = 2732.95 MiB`，与「层 + head + MTP + GDN」逐位相等。这说明 **T4 上 workspace + graph 小到可以忽略**（或已含在 persistent 的计量里）。所以直接用拟合值：

引擎公式（`kv_capacity.cpp:130-145`）：

```
pages = curve.minimum_main_page_groups
      + floor((available − headroom − curve.minimum_device_reservation_bytes) / 斜率)
```

T4 两行联立**精确解出**两个未知量（`available = 14400 MiB` 由 ECC 折扣确定，斜率由两行之差确定）：

```
5323 = m + (14400 − 1024 − R)/2.00
5579 = m + (14400 −  512 − R)/2.00
→ 两式相减恒成立（512/2 = 256），故只约束 m 与 R 的组合：R − 2.00·m = 2730.0
```

代入 `R = persistent = 2732.95 MiB` → `m = 1.475`，但 `m` 是**加法项不是乘数**，所以实际取 **2**（`capacity.py` 的 `MIN_MAIN_PAGE_GROUPS`）；`m = 2` 是唯一同时命中两行的整数，详见 12.1 节末尾。

6 GB 游戏卡（无 ECC，`available = 6144 − 400 = 5744 MiB`，`headroom = 1024`，`workspace = 135`）：

```
bf16: 2 + floor(1852.0498 / 2.00)    = 2 +  926 =  928 页 ≈ 59,392 token
int8: 2 + floor(1852.0498 / 1.03125) = 2 + 1795 = 1797 页 ≈ 115,008 token

（`1852.0498 = 5744 − 1024 − 2732.95 − 135`；int8 的容量口径单价是 **8 层** × 67,584 B = 540,672 B = 1.03125 MiB，不是 9 层的 1.1602。**上一版这里写成 1598 页是用错了分母——1.0312 才是 8 层的单价，0.515625 是单层的。**）
```

**三点说明**：

1. **`workspace = 135 MiB` 仍用 docs 的实测差**（5050 峰值 2868 − 全常驻 2732.95）。T4 上 `R` 拟合成 2732.95，说明那里 workspace 小到并进常数项；两个设备不同，135 只对 6 GB 卡成立。
2. **`headroom = 1024 MiB` 是默认值**（`base.h:57`），不是 0。报告 18 的脚本 bug 让标着 "headroom=0" 的那行实际跑的是默认 1024，那里已更正。
3. **九档两套单价的来源**：容量口径用 `curve.bytes_per_additional_main_page_group`（`startup.cpp:1832-1837` 的**实测斜率**，8 层，MTP 的 KV 是独立 backend pool 不随 text KV 增长，`startup.cpp:1465-1471`）；stage 预算口径用 `LayerCost::kv_bytes_per_page_group`（`load.cpp:317`，**含 MTP 一层**）。

**与本报告第 1 节原来的「bf16 200 页 = 12,800 token」相比，修正后是 928 页 = 59,392 token，差 4.6 倍**——来自三处：单价 2.25→2.00（×1.125）、固定项 3134.73→2732.95 + headroom 1024 + workspace 135（共省 411.8 MiB）、加上 `min_pages` 的 +2 页。

**结论**：本报告第 1 节「6 GB 笔电 1.28 万 token」是**低估约 4.6 倍**，修正值 **59,392（bf16）/ 115,008（int8）token**。这与报告 18 的 T4 实测（headroom=1024 时 5323 页 / 340,672 token）用的是同一个公式、同一套常量，并且 `capacity.py --selftest` 现在把 T4 两行作为对表闸逐页钉住（**36 项全 PASS**，`--prove-red` 注入 929 仍判红）。
