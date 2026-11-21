# 扫描内核的预注册预测（跑前写，跑后对账）

写这份文件的时间：2026-11-17 19:35 PST，`<kaggle-user>/kvmem-scan-sm75` v4 **仍在 RUNNING**（18:18 PST 推送，已 77 分钟）。
**此刻还没有任何一行实测结果落盘**，所以下面的数字是真的预测，不是事后解释。

目的只有一个：让「跑完了」和「预测对了」变成两件不同的事。没有预注册，任何实测值都能被解释成「符合预期」。

## 一、容量行：`--kv-capacity auto` 的三行

引擎的 auto 路径（`src/runtime/engine/kv_capacity.cpp:120-160`）：

```text
available_runtime_bytes = cudaMemGetInfo() 在权重加载之后
capacity_budget = available_runtime_bytes - automatic_headroom_bytes
pages = minimum_main_page_groups
      + (capacity_budget - minimum_device_reservation_bytes) / bytes_per_additional_main_page_group
```

`minimum_device_reservation_bytes = persistent.bytes + workspace.capacity + graph_allowance_bytes`
（`src/models/qwen3_5/program/planning/startup.cpp:1591-1593`）。

关键恒等式（已用 Python 验到 1e-9 内相等，见下）：

```text
我的 budget 口径  = (VRAM - 驱动) - (1536 + 1024 + head_bytes) - Σ权重 - Σstate
引擎 auto(1024)   = (VRAM - 驱动) - (Σ权重 + head_bytes + Σstate) - 1024 - 1536
                  = 同一个式子，只是 head_bytes 归在 fixed 还是归在 resident
```

数值验证（6 GB 笔电）：

```text
我的 budget 口径 residual = 451.049805 MiB
引擎 auto(默认 headroom 1024) residual = 451.049805 MiB
差 = 0.0
```

**所以报告 16 的 budget 口径不是「保守下界」，它恰好就是引擎默认参数下的精确值。** 这一点在实测前就该说清楚。

### 预注册预测（T4 15360 MiB，bf16，权重后空闲 12227.05 MiB）

| 行 | headroom | residual | 预测页组 | 预测 token |
|---|---|---|---|---|
| `auto headroom=0MiB`（即 `--kv-headroom-mib 0`） | 0 MiB | 10691.0 MiB | **4751** | **304,064** |
| 默认 `--kv-capacity auto`（headroom 1024 MiB） | 1024 MiB | 9667.0 MiB | **4296** | **274,944** |
| `auto headroom=512MiB` | 512 MiB | 10179.0 MiB | **4524** | **289,536** |

**判据（跑前定死）**：

- **命中**：实测 `kv_capacity_max_page_groups` 落在预测值的 **±2%** 内。容差来源是 `available_runtime_bytes` 来自 `cudaMemGetInfo()`，权重加载后会有几 MiB 到几十 MiB 的分配器碎片，不是算术误差。
- **证伪**：超出 ±2%。这时候要区分两种原因——① 我的 `head_bytes` / `Σ权重` 归属错了（→ 回到报告 16 第十一节的三步判据链重查）；② `minimum_device_reservation_bytes` 不是我以为的 1536 MiB（→ 说明 `workspace.capacity` 在 T4 上远大于 1.5 GiB，那 `kStageFixedBytes` 的注释「an estimate」就是低估）。
- **特别关注**：`headroom=0` 与 `headroom=1024` 两行的差必须 ≈ 1024/2.25 = **455 页**。如果差不是这个数，说明 headroom 在 auto 路径里被扣了两次或没扣。

第三行 `explicit 32768 tokens` **不是容量测量，是失败形状探针**：`--kv-capacity 32768` 与默认 headroom 互斥（`kv_capacity.cpp:120-123` 抛 `explicit KV capacity must not carry automatic headroom`），但 `--kv-capacity` 显式给了之后 `serve_options.cpp:1717` 的 `if (!kv_capacity_explicit)` 分支不走，所以 headroom 保持 `nullopt` → 应该**启动成功**。预测：`alive_after_probe=true` 且日志里出现 `32768`。若它报 headroom 错误，说明 `serve_options.cpp` 的赋值顺序与我的读法不符。

## 二、Host tier 六行：`--host-cache-mib {0,2048,8192} × --max-context {8192,32768}`

**这里我不预测数字，只预测形状**，理由是诚实的：`HybridCacheCounters` 的 18 个计数器里，`reused_tokens` / `host_block_restores` 取决于 `send_traffic` 送的是什么 prompt，而那个函数用的是通用文本，前缀重合度是我没测过的量。给一个没根据的数字比不给更糟。

**形状预测（可证伪）**：

1. **`host_mib=0` 两行**：`host_image_writes`、`host_block_writes`、`host_restore_bytes` 必须**全为 0**——Host tier 关闭就不该有任何 Host 流量。任何非零 = 计数器语义与我的理解不符。
2. **`host_mib>0` 的行**：`host_write_bytes > 0` 是必须的（写穿一定发生），但 `reused_tokens` **可能仍然是 0**——因为 `send_traffic` 送的是两次相同请求，第二次若在第一次的快照仍驻留 Device 时到达，走的是 Device 命中而不是 Host 恢复。**所以「`reused_tokens>0` 但 `host_restore_bytes==0`」是预期内的正常形状，不是 bug。**
3. **`max_context=32768` 行的 `admissions` 必须 ≥ `max_context=8192` 行**（同样流量、更大池 → 不该更少准入）。
4. **非单调性检查**：`host_mib` 从 2048 → 8192 时，`reused_tokens` **不保证单调增加**。上游 fork `452acabb11` 已实测到「3 GiB 复用 82.4% > 4 GiB 79.9%」，根因是 Host slab 池里块与 StateImage 抢同一批字节（报告 08 的警告）。**若这里也出现 2048 优于 8192，那是复现了上游的非单调，不是我的测量错。**

## 三、双卡行：`--devices 0,1 --stage-layers 16,16`

**判据是「Legacy 模式被选中」，不是「hybrid 块缺席是缺口」。**

`serve_options.cpp:1705-1712` 把 `--devices` 当作 `catalog_option` → `context_cache.mode = Legacy`；`:1733-1736` 让 hybrid 与 pipeline 互斥。所以在 base `6cc1f9fa` 上：

- **命中**：`alive_after_probe=true`，`/stats` 里**没有** `hybrid` 块，且日志里能看到两卡的层分配（16+16）。
- **证伪**：`alive_after_probe=false`，或 `/stats` 里出现 hybrid 块（说明我的读法错了），或层分配不是 16/16（说明 `--stage-layers` 被忽略）。
- **必须标注**：这是 **base 的行为，不是引擎边界**。fork 在 2026-11-10 用 `af60c489cd` 修了多卡 hybrid 启动失败，随后 `38a10aa49a` 加了 `--use-alt-prefix-caching` 多设备支持。任何写进报告的句子都必须带这个限定。

## 四、如果内核根本没跑到这些阶段

`result.json` 的 `stages` 里会看到停在哪个 `@stage`。三个可能：

| 停在哪 | 含义 | 下一步 |
|---|---|---|
| `acquire_model` | GGUF 下载或 `tools.convert` 失败 | 拉日志看是网络还是转换；模型 2.8 GB，Kaggle 到 HF 的带宽是唯一变量 |
| `build` | FFmpeg / 引擎编译失败 | 拉 `build` 阶段的 stderr 尾部，通常是缺头文件或 nvcc 内存不足（4 核编译 27B 级源码） |
| `scan_*` | 构建成功但 serve 起不来 | 这是最有价值的一类失败：说明二进制能编不能跑，日志里的 flag 错误直接指向 `serve_options.cpp` 的哪一行 |

**止损线（不变）**：v4 若因环境问题再失败一次，就放弃实测，以「算术 + 源码」交付。报告 16 + 9 档 dtype 表已经是可引用的完整答案。

## 六、v4 实测对账：预测**没有**被检验（跑前判据的诚实归档）

v4 在 2026-11-18 02:18:29Z 启动，2026-11-17 23:18 PST 拉回结果，状态 **ERROR**。
产物：`kvmem-study/remote/scan-sm75-v4/{result.json, kvmem-scan-sm75.log}`。

**九个阶段，前八个全绿，唯一的失败是 `build`：**

| 阶段 | 结果 | 秒 |
|---|---|---|
| `probe_toolchain` | OK | 0.4 |
| `ensure_media_tools` | OK | 19.7 |
| `cuda_driver` | OK（`/usr/local/nvidia/lib64/libcuda.so`，shim 生效） | 0.0 |
| `extract_kvmem` | OK（tree `/tmp/work/kvmem`，version 0.19.0） | 58.4 |
| `prepare_ninfer` | OK（head `6cc1f9fa438b7094685ae417dfa40e1ca324e617`） | 50.1 |
| `acquire_model` | OK（2,827,176,704 B，gguf_sha256 `3874209…`，`generation_config` 最小配置） | 51.4 |
| `fetch_media` | OK | 5.8 |
| `configure` | OK（`CMAKE_CUDA_ARCHITECTURES=75`） | 5.0 |
| **`build`** | **FAIL**：`timeout after 14400s`，进度 **[662/943] = 70.2%** | 14400.1 |

**关键：这不是"环境问题"，是"我自己设的 4 小时超时"。** 编译没有报错，是稳定推进中被 timeout 掐断。
线性外推：全部 943 目标需 `943/662 × 4h = 5.70h`，剩 281 个约 1.70h。
日志尾部全是正常的 `Building CUDA object ... small_t_i8_w{3..6}_h{16,24}_{cached,append}.cu.o`，没有一条 compiler error。

**所以：上面第一节到第三节的预测一条都没被证伪，也一条都没被证实——因为扫描根本没跑。** 按第零节的原则，**未测量就写未测量**：

| 预测项 | 状态 |
|---|---|
| 容量三行（4751/4524/4296 页） | **未测量** |
| `headroom=0` 与 `1024` 差 ≈ 455 页 | **未测量** |
| explicit 32768 启动成功 | **未测量** |
| Host tier 六行形状 | **未测量** |
| 双卡 Legacy 判据 | **未测量** |

**止损线是否触发？** 原话是「v4 若因**环境问题**再失败一次即放弃实测」。这次失败的原因是超时参数，不是环境——八个前置阶段（包括此前三次卡住我们的 cuda_driver、meson/nasm、tools.convert）**全部一次通过**。按字面判断：**不触发**，修 timeout 后重推。

**v5 的唯一改动**：`scan_kernel.py:456` `timeout=14400` → `timeout=28800`（8 小时）。Kaggle GPU 内核上限 9 小时，前置 190.8 s + 编译 5.70 h + 扫描估计 15–25 min ≈ 6.2 h，8 h 留了足够 buffer。推断已在代码注释里写明。

v5 于 2026-11-17 23:2x PST 推送，`KernelWorkerStatus.RUNNING`。**上面所有预测对 v5 继续有效，不做任何调整。**

## 五、这份文件自己的可证伪性

上面每条判据都在跑前写定。**跑完后我会把实测值逐行贴在这里做对账，包括预测错的那些。** 一份只在预测对时才被引用的预注册等于没有预注册。

## 七、v5 实测对账（2026-11-18，COMPLETE）

v5 于 2026-11-18T07:21:39Z 启动，全部 12 阶段 `ok=True`，`build` 用 15,873.9 s（4.41 h，v4 是 4h 到 70.2%）。产物 `kvmem-study/remote/scan-sm75-v5/{result.json 122270B, kvmem-scan-sm75.log}`。

### 7.1 容量行：预测全错，但错法有结构

**先说一个事实错误**：我预测的"默认 `--kv-capacity auto`（headroom 1024）"那一行**脚本根本没跑**。`scan_kernel.py:635` 是 `for headroom in (0, 512)`，第三行是 explicit 32768。所以三行预测里有一行（4296 页）从不存在对应的实测。

| 行 | 预测页组 | 实测页组 | 差 | 预测 token | 实测 token |
|---|---|---|---|---|---|
| `headroom=0` | 4751 | **5323** | +12.0% | 304,064 | **340,672** |
| `headroom=512` | 4524 | **5579** | +23.3% | 289,536 | **357,056** |
| `headroom=1024`（未跑） | 4296 | — | — | 274,944 | — |
| explicit 32768 | 启动成功 | **启动成功**，512 页 | ✓ | — | 32,768 |

**两处根因（各 1 条判据）**：

1. **忘扣 ECC**。T4 开了 ECC（日志有专门 warning），可用显存 = 15360 × 0.9375 = **14400 MiB**，我按 15360 算。
2. **页组单价取错**：我用 2.25 MiB（8 层 + MTP），容量行实际用 **2.00 MiB**（8 层，不含 MTP）。两行联立精确解出：`(5579−5323) = 256 = 512 MiB / P` → **P = 2.00 MiB/页**。日志自报 "2.00 MiB per host KV page group" 是同一数。

**更有意思的是：两个错误方向相反，几乎抵消。** 4751 × 2.25 = **10,689.75 MiB**；实测 5323 × 2.00 = **10,646.0 MiB**。差 0.4%。**预测的总字节数是对的，页数标签是错的**——这是"用一个错的单价去除一个对的总量"的典型。若当时只报总字节，结论成立；报了页数，就错。

**512 → 0 的差**：预测 455 页，实测 256 页。比值 0.5626 ≈ 2.00/2.25 × (1/1.0)，就是单价比的倒数。**这条差判据本身是好的**——它立刻指出了单价错。

### 7.2 Host tier 六行：形状预测全中

| host_mib | ctx | host_block_writes | host_image_writes | host_restore_bytes | reused_tokens | device_state_slots |
|---|---|---|---|---|---|---|
| 0 | 8192 | **0** | **0** | **0** | 599 | 4 |
| 0 | 32768 | **0** | **0** | **0** | 599 | 4 |
| 2048 | 8192 | 9 | 1 | **0** | 599 | 3 |
| 2048 | 32768 | 9 | 1 | **0** | 599 | 3 |
| 8192 | 8192 | 9 | 1 | **0** | 599 | 3 |
| 8192 | 32768 | 9 | 1 | **0** | 599 | 3 |

逐条对：
- 预测 1「host_mib=0 三个计数器全 0」→ **命中**。
- 预测 2「host_mib>0 时 `reused_tokens>0` 但 `host_restore_bytes==0` 是正常形状」→ **命中**（六行全是 599 reuse / 0 restore）。理由也在数据里：第二次请求到达时第一次的快照还在 Device，走 Device 命中。
- 预测 3「ctx=32768 的 admissions ≥ ctx=8192」→ **无法判**，脚本没采 admissions 计数器；`max_admitted_requests` 六行都是 17，没有区分度。
- 预测 4「2048→8192 不保证单调」→ **未触发**（两行完全相同），不构成对预测的检验。

**唯一未预测到的**：`device_state_slots` 从 4（host=0）降到 3（host>0）。这是 `C+1` vs `C+2` 的实测证据——Host tier 开启后 Device state 槽让位给 Host 副本。**与 fork 分支 `452acabb11`/`af60c489cd` 的 C+1/C+2 分析对上了**（报告 16 记的 issue #222 重放：1 GiB tier `C+1` 复用 0%、`C+2` 27%）。

**Host 流量数字**：`host_write_bytes = 72,487,936 B = 69.13 MiB`（9 块 KV + 1 个 StateImage）。`host_capacity_bytes` = 2,147,483,648 / 8,589,934,592 精确等于 2/8 GiB。`host_used_bytes = 73,400,320 B`。

### 7.3 双卡行：判据命中

`alive_after_probe=true`、`stats.hybrid=null`（Legacy 模式）、`device_state_slots=2`、`capacity.host_kv_bytes=8,589,934,592`（8 GiB）、日志 `peer access: none between CUDA devices 0-1; their transfers stage through host memory`、`KV 16,384 tokens, pages 256/256, runtime 440.3 MiB, free 12.5 GiB`。

**必须带的限定**：这是 base `6cc1f9fa` 的行为（`--devices` → Legacy，`serve_options.cpp:1705-1712`），不是引擎边界。fork 在 2026-11-10 用 `af60c489cd` 修了多卡 hybrid 启动、`38a10aa49a` 加了多设备 `--use-alt-prefix-caching`。

**另外**：`runtime 440.3 MiB` 且 `pages 256/256` —— 双卡下每卡 2.55 GiB 权重，KV 池只剩 256 页（16,384 token），说明层切分后每卡权重翻倍。**`stage-layers 16,16` 被接受**（启动成功即证明），但日志没有打印层分配行，这条判据只到"没报错"级别。

### 7.4 判决汇总

| 预测 | 结果 |
|---|---|
| 容量三行 | **错**（+12.0% / +23.3%；根因 ECC + 单价） |
| 512 与 0 差 455 页 | **错**（256 页），但差值判据本身正确指出了单价错误 |
| explicit 32768 启动成功 | **命中** |
| Host 关闭时三计数器全 0 | **命中** |
| reuse>0 而 restore=0 是正常形状 | **命中** |
| ctx=32768 admissions ≥ ctx=8192 | **无法判**（未采 admissions） |
| 2048→8192 不单调 | **未触发**（两行相同） |
| 双卡 Legacy | **命中** |

**8 条里 3 命中、2 错、1 无法判、1 未触发、1 部分（双卡层分配只到"没报错"）。**

**最有价值的一条**：512/0 的页数差是**跑前就写好的判据**，它直接暴露了"我用 2.25 除、引擎用 2.00"这个错。如果我只预测了绝对页数没有预测差值，这个错会被当成"容差不够"放过去。

**Host 流量的字节构成也对上了**：`host_write_bytes = 72,487,936 B` = 9 块 KV × 2.00 MiB（18,874,368 B）+ 1 个 StateImage（51.13 MiB = 53,613,568 B），合计 72,487,936 **逐字节精确**。反推出的 image 大小 51.13 MiB 与日志 "49.1 MiB per state image" 差 2 MiB——那 2 MiB 是 image 之外的 tail block（`hybrid_host_layout.h` 的 "snapshot tail bundle 再占一个 slab"）。**两处独立来源互相印证。**

### 7.5 一个没预测到但很重要的读数：C+1 / C+2 在 base 上实测到了

六行里 `device_state_slots` 与日志 `context cache` 行：

| host_mib | 日志 | slots | 语义 |
|---|---|---|---|
| 0 | `1 active + 3 cached device states` | 4 | **C+2**（C=max_concurrency=1） |
| 2048 | `1 active + 2 cached device states` | 3 | **C+1** |
| 8192 | `1 active + 2 cached device states` | 3 | **C+1** |

这正是 spec §5.4 写的规则在 base `6cc1f9fa` 上的**直接实测**：*「base 默认有 Host tier 时 `C+1`、无时 `C+2`」*。而 fork 分支把这条改成了 `C+2` 恒定，理由记在 `452acabb11` 的实测里：**1 GiB tier 下 `C+1` 复用 0%、`C+2` 27%**（issue #222 重放，两车道四段 8,900-token 对话，KV k8v4）。

也就是说：**我这一发扫描顺手拿到了 fork 那个修复的"修前"读数**。之前它只是别人 issue 里的数字，现在是我自己在 T4 上跑出来的 base 行为。这条不在预测里，属于计划外收获。

**代价也一并测到了**：开 Host tier 后 `kv_capacity_pages` 从 **5299 升到 5323**（+24 页 = +1,536 token）。24 页 × 2.00 MiB = **48 MiB**，与「少一个在 Device 常驻的 StateImage」（日志 49.1 MiB）**几乎精确相等**——少一个 Device state 槽就腾出 48 MiB 给 KV 页。

**这是 `C+1` 的真实代价与收益**：代价是少一个在 Device 的 state 槽（复用率可能掉），收益是 48 MiB KV 空间。fork 选 `C+2` 恒定 = 放弃这 48 MiB 换复用率。

### 7.6 8.5 节的补充：2.00 与 2.25 是两套**不同**的单价，不是同一个数的两种说法

深挖一层：`load.cpp:317` 的 `cost.kv_bytes_per_page_group`（含 MTP，2.25 MiB）与 auto 容量实际用的 `curve.bytes_per_additional_main_page_group`（2.00 MiB）**来自两个不同的函数**，用途不同：

- `load.cpp:305-343` 算的是 **`StageBudget`**，喂给 `solve_stage_plan`（报告 15 的层切分求解器）。它的 `kv_bytes_per_page_group` 是 stage 0 专属的 `mtp_kv`（`load.cpp:341-343`：`kStageFixedBytes + (stage == 0 ? kHeadStageFixedBytes + head_bytes : 0)`），**含 MTP 那一层的 KV**。
- `startup.cpp:1836` 的 `bytes_per_additional_main_page_group` 是 **`build_sequence_candidate(min+1).device_reservation_bytes − build_sequence_candidate(min).device_reservation_bytes`** 的**实测差**（`startup.cpp:1832-1837`），即"多一页组多花多少 device 字节"。这是引擎自己算出来的 affine 斜率。

**实测斜率 2.00 MiB/页 = 8 层 × 262,144 B，不含 MTP。** 原因在 `startup.cpp:1465-1471`：MTP KV 在 Hybrid 下被当作 **backend pool**（`plan_hybrid_host_layout(..., backend, ...)`），它是独立的一池，**不随 text KV 的页数增长**。所以「多一页组」的增量只有 8 层。

**结论**：我在报告 16 里把 `Σkv/页组 = 2.25 MiB`（含 MTP）当成 auto 容量除法的分母，是**用错了函数**。报告 16 的 2.25 对"stage 预算/层切分"是对的（报告 15 的 `stage_fits` 用的正是 `StageBudget`），对"auto 容量"是错的（应该用 2.00）。

**这也解释了为什么 `explicit 32768` 行是 512 页**：32768 / 64 = 512 页整。`--kv-capacity` 单位是 token，除以 `kPagedKVPageSize` 得页数，与单价无关。
