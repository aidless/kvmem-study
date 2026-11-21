# 18 · Phase B 实跑：v5 全绿，三行容量预测全错，Host tier 形状全中

> 底座：全部路径与行号来自 `kvmem-study/ninfer-src`，即 kvmem `backends/versions.json` 钉住的 `6cc1f9fa438b7094685ae417dfa40e1ca324e617`（fork `iamwavecut/ninfer-all`）。上游 HEAD 已 `diverged, ahead 523 / behind 1895`，本报告只对底座成立。
> 判据：`kvmem-study/remote/scan-sm75/PREDICTIONS.md`（写于 2026-11-17 19:35 PST，跑前预注册）。实测值与预测值逐行对账在第七节，预测错的照贴。

## 一、结果一句话

v5 **全绿**：12 个阶段 `ok=True`，`build` 用 4.41 h 编完 943 个目标，T4 上 `ninfer-serve` 起来跑了 9 行扫描，全部正常退出。容量预测**全错**（+23.9% / +23.3%），根因**四项**（含我自己的脚本标签 bug；另有 `m` 常量一度误标 UNKNOWN，已由 v5 explicit 行钉死）；Host tier 六条形状预测**全中**，双卡 Legacy 判据**命中**；顺手拿到 spec §5.4 的 `C+1`/`C+2` 在 base 上的**首次实测**。

## 二、v5 的十二个阶段

产物 `kvmem-study/remote/scan-sm75-v5/{result.json 122,270 B, kvmem-scan-sm75.log}`，`t0=2026-11-18T07:21:39Z`，`ok=true`，`arch=75`。

| 阶段 | 秒 | 说明 |
|---|---|---|
| `probe_toolchain` | 0.5 | Tesla T4 / gcc 13.3.0 / cmake 3.31.10 / ninja 1.13.2 / RAM 31 GB |
| `ensure_media_tools` | 30.6 | meson + ninja + make + nasm |
| `cuda_driver` | 0.0 | `/usr/local/nvidia/lib64/libcuda.so` + `CMAKE_PROJECT_INCLUDE` shim |
| `extract_kvmem` | 77.7 | tree `/tmp/work/kvmem`，version 0.19.0 |
| `prepare_ninfer` | 51.0 | head `6cc1f9fa438b7094685ae417dfa40e1ca324e617` |
| `acquire_model` | 45.4 | 2,827,176,704 B，`tools.convert` 在本内核内从 GGUF 转 |
| `fetch_media` | 3.5 | FFmpeg/dav1d 已静态编入 |
| `configure` | 5.8 | `CMAKE_CUDA_ARCHITECTURES=75` |
| **`build`** | **15,873.9（4.41 h）** | 943/943 全过，零 error |
| `scan_capacity` | 80.9 | 3 行 |
| `scan_host_tier` | 82.9 | 6 行 |
| `scan_two_cards` | 17.1 | 1 行 |

**v4 → v5 的唯一差别就是 timeout 14400 → 28800。** v4 在 4 h 到 70.2%，v5 用 4.41 h 走完，剩 3.6 h buffer 未用。

## 三、容量三行：预测全错，错法有结构

T4 的 ECC 是开着的（日志有专门 warning：`ECC is ENABLED on Tesla T4: GDDR6X stores ECC checksums in VRAM, costing ~6.25% of capacity`）。

| 行 | 预测页组 | 实测页组 | 差 | 预测 token | 实测 token |
|---|---|---|---|---|---|
| `headroom=1024`（脚本标 "0"，旗标未传） | 4296 | **5323** | **+23.9%** | 274,944 | **340,672** |
| `headroom=512` | 4524 | **5579** | **+23.3%** | 289,536 | **357,056** |
| `headroom=0`（真 0） | 4751 | **该行未跑** | — | 304,064 | — |
| `explicit 32768` | 启动成功 | **启动成功**，512 页，free 10.6 GiB | ✓ | — | 32,768 |

**根因四项 + 一项已澄清的误判**：第四项单价错之外，`m` 曾被视为第五个 UNKNOWN，后来发现它根本不是独立未知量（见下）。

**① 脚本没传 `--kv-headroom-mib 0`，标签与实际不符。** `scan_kernel.py:636-638`：

```python
for headroom in (0, 512):
    args = common + ["--kv-capacity", "auto"]
    if headroom:                                   # ← 0 是假值
        args += ["--kv-headroom-mib", str(headroom)]
```

`if headroom:` 对 `0` 为假 → 那行**一个旗标都没传**，走 `KvCapacityPolicy::automatic()` 的默认值 `kDefaultKvCapacityHeadroomBytes = 1024 MiB`（`include/ninfer/base.h:57`、`engine_options.h:34`）。所以标签 `auto headroom=0MiB` 实际是 **headroom=1024**。我在写预测时把标签当真值了——**预测 4296 页那一行确实跑过，只是它的旗标是默认的 1024**。

**② 预测公式漏了 623.8 MiB 的 resident。** 预测时用的 residual 是 `15360 − 2560(fixed) − 2109.1(层) − 1024(headroom)`，漏掉 `head_bytes 497.3 + MTP 77.4 + GDN 49.1 = 623.8 MiB`。引擎的 `minimum_device_reservation_bytes = persistent.bytes + workspace.capacity + graph_allowance_bytes`（`startup.cpp:1591-1593`）把这三项都算进去了。

**③ 忘扣 ECC。** T4 开了 ECC（日志 warning：`ECC is ENABLED on Tesla T4: GDDR6X stores ECC checksums in VRAM, costing ~6.25% of capacity (23,028 vs 24,564 MiB on a 24 GB card)`）。可用应为 `15360 × 0.9375 = 14,400 MiB`，我按 15,360 算。

**④ 页组单价取错。** 我用 2.25 MiB（8 层 + MTP），容量行实际用 **2.00 MiB**（8 层，不含 MTP）。两行联立精确解出：`(5579−5323) = 256 = 512 MiB / P` → **`P = 2.00 MiB/页`**。日志自报 `2.00 MiB per host KV page group` 是同一个数。

**这四项的净效果是：绝对页数判据错（+23.9%），而「两档之差」判据成立且正好 256 页。** 差的判据不含 ECC、resident 漏项与单价的绝对量，所以它对；绝对页数的判据含了这些错，所以它错。逐项复算见 3.1，四项全部闭合到 0.05 MiB。

**最有意思的一点：错误的方向是交替的。** 四项里 b（多扣 stage 1536）和 d（headroom 扣两次）把 residual 推大，a（ECC）和 c（漏 resident）把它推小；`+1536 − 960 − 624 + 1024 = +976 MiB`。而页组从 4296 变到 5323 是 **+1027 页 = +23.9%**——因为分母同时从 2.25 缩到 2.00，把分子的增量又放大了 12.5%。**如果只有分子错、分母对，误差是 +21.2%；分母也错，就变成 +23.9%。** 两个错误同向叠加，没有出现"互相抵消"的假象。

**验算（用核实过的引擎公式复现实测）**：`pages = m + (14400 − headroom − 2732.95) / 2.0`，两行联立解出 `m ≈ 1.5`、`resident = 2732.95 MiB`（与报告 16 的「全常驻 2732.95 MiB」逐位一致）。m 是加法项不是四舍五入，只有整数 **2** 同时命中两行（`m=1` 差 1 页、`m=3` 差 +1 页）：

| headroom | 复算 | 实测 |
|---|---|---|
| 1024 | 5323.0 | **5323** ✓ |
| 512 | 5579.0 | **5579** ✓ |
| 0（未跑） | 5835.0 | — |

**跑前写死的差判据是本轮最成功的一条**：预测 512/1024 两行的差为 455 页，实测 **256 页**，比值 0.5626 ≈ 2.00/2.25。这条差**直接指出了单价错误**。如果只预测绝对页数没有预测差值，这个错会被归进"容差不够"放过去。

**修正后的读法**：`5323 页 / 340,672 token` 是 **headroom=1024（默认）** 的读数；`5579 页 / 357,056 token` 是 **headroom=512** 的读数。真 headroom=0 从未实测。

### 3.1 四项误差的逐项复算（闭合到 0.05 MiB）

**先把预注册的公式反推出来。** 三行 residual 与 headroom 的差值精确等于 1024 / 512，说明它把 headroom 减了两次；再对 15360 反推扣减总量，得到唯一解：

```text
预注册 residual = 15360 − 1536(stage) − 2109.10(层) − 1024(默认 headroom) − headroom
```

三行逐字复现（差 0.10 MiB 是四舍五入）：

| 行 | 预注册算式 | 结果 | 文件里写的 |
|---|---|---|---|
| hr=0 | 15360 − 1536 − 2109.10 − 1024 − 0 | 10690.90 | 10691.0 ✓ |
| hr=1024 | 15360 − 1536 − 2109.10 − 1024 − 1024 | 9666.90 | 9667.0 ✓ |
| hr=512 | 15360 − 1536 − 2109.10 − 1024 − 512 | 10178.90 | 10179.0 ✓ |

**再把引擎的正确公式并排：**

```text
正确 residual = 14400(ECC 后) − 2732.95(persistent) − headroom
```

hr=1024 那行：预注册 `9666.90`，正确 `10643.05`，**差 +976.15 MiB**。

**四项分解（精确闭合）：**

| 项 | 动作 | 对 residual 的影响 |
|---|---|---|
| a | ECC 未扣（15360 → 14400） | **−960.00** |
| b | 多扣了 stage 固定项 1536（容量口径不背 stage） | **+1536.00** |
| c | resident 漏算 623.8（head 497.3 + MTP 77.4 + GDN 49.1） | **−623.80** |
| d | headroom 被扣两次（默认 1024 已含在 `1536+1024+head_bytes` 里，又减了一遍） | **+1024.00** |
| **合计** | | **+976.20** |

实测差 **+976.15**，闭合差 **0.05 MiB**（四舍五入累积）。**四项全部复现，没有第五项。**

**页组层面：**

| 口径 | 算式 | 页组 |
|---|---|---|
| 预注册（hr=1024） | `9666.90 ÷ 2.25` | **4296** |
| 逐项修正 residual | `10643.05 ÷ 2.25` | 4730 |
| 再换单价（2.25→2.00） | `10643.05 ÷ 2.00` | 5321 |
| 加常数项 m | `5321 + 2` | **5323** ✓ |

**为什么"逐项加减"曾经对不上**：a/b/c/d 里 b、d 是**加**项、a、c 是**减**项，四项同量级（1536 / 1024 / 960 / 624）且符号交替，任何一项漏掉或顺序算错都会让末位偏出几百页。**只有把它们全部还原成一个算式（而不是一串操作）才能闭合。** 我第一版写的"①+512 ②+311 ③+480 ④−11%"是错的，因为它把不同轴上的量当成可加项。

## 四、Host tier 六行：形状预测全中

| host_mib | ctx | kv_pages | device_state_slots | host_block_writes | host_image_writes | host_restore_bytes | reused_tokens |
|---|---|---|---|---|---|---|---|
| 0 | 8192 | 5299 | **4** | **0** | **0** | **0** | 599 |
| 0 | 32768 | 5299 | **4** | **0** | **0** | **0** | 599 |
| 2048 | 8192 | 5323 | **3** | 9 | 1 | **0** | 599 |
| 2048 | 32768 | 5323 | **3** | 9 | 1 | **0** | 599 |
| 8192 | 8192 | 5323 | **3** | 9 | 1 | **0** | 599 |
| 8192 | 32768 | 5323 | **3** | 9 | 1 | **0** | 599 |

- 预测 1「host=0 时三个计数器全 0」→ **命中**。
- 预测 2「host>0 时 `reused_tokens>0` 而 `host_restore_bytes==0` 是正常形状」→ **命中**。数据给出理由：两次请求都是 600 token 同文本，第二次到达时快照还在 Device，走 Device 命中（`cached_tokens: 599`），根本不需要 Host 恢复。
- 预测 3「ctx=32768 的 admissions ≥ ctx=8192」→ **无法判**。脚本采的是 `max_admitted_requests`，六行全 17，无区分度；`admissions` 计数器根本没在输出里。
- 预测 4「2048→8192 不保证单调」→ **未触发**，两行完全相同。上游 fork `452acabb11` 的非单调（3 GiB 82.4% > 4 GiB 79.9%）在这套流量下不出现，因为总量只有 69 MiB，2 GiB 池根本装得下全部写入。

**Host 流量的字节构成逐字节对上了**：`host_write_bytes = 72,487,936 B` = 9 块 KV × 2.00 MiB（18,874,368 B）+ **一个 StateImage 53,613,568 B（51.13 MiB）**。日志的 `49.1 MiB per state image` 与反推值差 2 MiB，那 2 MiB 是 `hybrid_host_layout.h` 说的 snapshot tail bundle 单独占的一个 slab——**两处独立来源互相印证**。`host_capacity_bytes` 精确等于 2 GiB / 8 GiB，`host_used_bytes` 73,400,320 B。

## 五、计划外收获：base 上的 C+1 / C+2

六行里唯一没被预测但很关键的读数：

| host_mib | 日志 | slots | 语义 |
|---|---|---|---|
| 0 | `1 active + 3 cached device states` | 4 | **C+2**（C = max_concurrency = 1） |
| 2048 / 8192 | `1 active + 2 cached device states` | 3 | **C+1** |

这是 spec §5.4 那条规则在 base `6cc1f9fa` 上的**直接实测**：*「base 默认有 Host tier 时 `C+1`、无时 `C+2`」*。fork 分支把它改成 `C+2` 恒定，理由写在 `452acabb11` 的实测里：**1 GiB tier 下 `C+1` 复用 0%、`C+2` 27%**（issue #222 重放，两车道四段 8,900-token 对话，KV k8v4）。

**我这一发顺手拿到了 fork 那个修复的"修前"读数。** 之前它只是别人 issue 里的数字，现在是我自己在 T4 上跑出来的 base 行为。

**代价也一并量出来了**：开 Host tier 后 `kv_capacity_pages` 从 5299 **升到** 5323（+24 页 = +1,536 token）。24 × 2.00 MiB = **48 MiB**，与"少一个 Device 常驻 StateImage"（日志 49.1 MiB）几乎相等——少一个 Device state 槽就腾出 48 MiB 给 KV 页。fork 选 `C+2` 恒定 = 放弃这 48 MiB 换复用率。

## 六、双卡行：判据命中，限定必带

`--devices 0,1 --stage-layers 16,16 --kv-capacity auto --max-context 16384`：

- `alive_after_probe=true`、`/stats` 里 **`hybrid` 字段为 `null`**（Legacy 模式被选中，`serve_options.cpp:1705-1712`）。
- 日志 `engine | peer access: none between CUDA devices 0-1; their transfers stage through host memory`（T4 无 P2P）。
- `capacity | KV 16,384 tokens, bf16, auto | pages 256/256 | runtime 440.3 MiB | free 12.5 GiB`——**每卡权重翻倍**（2×2.55 GiB），KV 池只剩 256 页。
- `host_kv_bytes=8,589,934,592`（8 GiB 默认 Host tier）、`device_state_slots=2`。

**必须带的限定**：这是 base 的行为，不是引擎边界。fork 在 2026-11-10 用 `af60c489cd` 修了多卡 hybrid 启动失败，随后 `38a10aa49a` 加了多设备 `--use-alt-prefix-caching`。任何写进报告的句子都要带这句。

**层分配只到"没报错"级别**：启动成功说明 `--stage-layers 16,16` 被接受，但日志没有打印层分配行，这一格没有独立证据。

## 七、判决汇总

| 预测 | 结果 |
|---|---|
| 容量（headroom 1024 / 512） | **2 错**（+23.9% / +23.3%）；另真 0 那行未跑、标签与旗标不符 |
| `headroom=1024` 与 `512` 差 ≈ 455 页 | **错**（256 页），但**这条差判据正确指出了单价错误** |
| explicit 32768 启动成功 | **命中** |
| host=0 时三计数器全 0 | **命中** |
| `reuse>0` 而 `restore=0` 是正常形状 | **命中** |
| ctx=32768 的 admissions ≥ ctx=8192 | **无法判**（未采 admissions） |
| 2048→8192 不保证单调 | **未触发**（两行相同） |
| 双卡 Legacy + 层分配 16/16 | **部分命中**（Legacy ✓，层分配无独立证据） |

**8 条里 3 命中、2 错、1 无法判、1 未触发、1 部分。**

**错的两条都是绝对值判据，活下来的三条都是形状判据。** 这不是巧合：形状判据（"全 0"、"reuse>0 而 restore=0"、"Legacy 被选中"）只需要知道机制的定性行为，不需要知道任何精确数值；绝对值判据要一次做对四处口径。**代价是形状判据的信息量低**——它无法区分"容量 5323 页"和"容量 8000 页"。要拿数字，就得承担算错的风险，并把每一处口径写下来供事后对账。

## 八、可引用数字表（全部来自 v5 result.json）

| 数 | 值 | 来源 |
|---|---|---|
| T4 可用显存（ECC 后） | 14,400 MiB | `15360 × 0.9375`，日志 ECC warning |
| KV 页组单价（8 层 bf16） | **2.00 MiB** | `8 × 262,144 B`，与日志 `2.00 MiB per host KV page group` 一致 |
| `headroom=1024`（默认）容量 | 5,323 页 / 340,672 token | `capacity \|` 行；**脚本标签写成 "0"** |
| `headroom=512` 容量 | 5,579 页 / 357,056 token | 同上 |
| 两档差 | 256 页 = 512 MiB ÷ 2.00 MiB/页 | 两行联立，与日志单价一致 |
| 容量公式常数项 `m`（引擎） | **128** 页 = `page_count(8192)` | `startup.cpp:1803-1807`；v5 explicit 512=`32768/64` 钉死 `resident_main_pages=0`，见报告 16 §12.1.1 |
| `minres`（两行反解） | **2986.00 MiB**（两行差 0.0000） | 报告 16 §12.1.1；`persistent`(序列 arena)+workspace+graph，**不含权重** |
| 预注册 → 实测的 residual 差 | **+976.15 MiB**，四项闭合到 0.05 | 见 3.1：ECC −960、stage +1536、resident −624、headroom 双扣 +1024 |
| explicit 32768 | 512 页 / free 10.6 GiB | 同上 |
| Host 写穿字节（host=2048/8192） | 72,487,936 B（69.13 MiB） | `host_write_bytes` |
| Host 写穿构成 | 9 块 KV 18,874,368 B + StateImage 53,613,568 B | 反推，与日志 49.1 MiB + 2 MiB tail 一致 |
| `device_state_slots` | host=0 → 4（C+2）；host>0 → 3（C+1） | `capacity.device_state_slots` + 日志 |
| 开 Host tier 的 KV 余量 | 5299 → 5323（+24 页 = 48 MiB） | 六行对比 |
| 双卡 KV 池 | 256 页 / 16,384 token，runtime 440.3 MiB | 双卡 `capacity \|` 行 |
| 双卡 P2P | `peer access: none`，经 host memory 中转 | 双卡日志 |
| v5 编译耗时 | 15,873.9 s（4.41 h） | `build.seconds` |