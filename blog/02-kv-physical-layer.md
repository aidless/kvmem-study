# 一个生产级 KV 分页引擎的物理层

大多数人讲 KV cache 优化，讲的是算法：PagedAttention 怎么消碎片、GQA 怎么省显存、量化怎么换精度、前缀缓存怎么提命中率。这些都没错，但它们停在一个假设上——KV 是"一块被管理的显存"，管理它的手段是分配与释放。

真把 KV 做成一个能被多会话共享、能被挂起恢复、能跨机器切分、能落到 SSD 再捞回来的东西，你会发现难点根本不在算法，在**物理层的记账**：一页现在被谁引用、它有没有一份 Host 副本、那份副本的内容版本对不对得上、这块空闲内存能不能借给别人、借出去之后回来还认不认得出自己。

这篇文章解剖 NInfer 的 KV 物理层——一个单机 C++/CUDA 推理引擎（commit `6cc1f9fa`），它的 KV 子系统有页表、有租约、有分层、有落盘格式、有准入报价，本质上是一个微型内存管理器。所有引用都是 `path:line`，读者可对照源码。我读完 32 个源文件后写下的十一份笔记在文末列出，这里是提炼。

## 一、页大小 64 是编译期常量，不是配置

`--block-size 32` 这种旗标在别处很常见。NInfer 里没有：

```cpp
// src/core/paged_kv_cache.h:20
static constexpr std::uint32_t kPagedKVPageSize = 64;

// src/ops/kernel/paged_kv_address.cuh:8-13
constexpr std::uint32_t kPagedKVPageShift = 6;
constexpr std::uint32_t kPagedKVPageMask  = 63;
static_assert(kPagedKVPageSize == (1 << kPagedKVPageShift));
```

decode kernel 取页靠 `position >> shift`，取页内偏移靠 `position & mask`。**位运算写进了 kernel**，所以改页大小不是改配置，是重编全部 CUDA kernel。这条一旦看清，很多"为什么它不支持 block size 32"的疑问就变成"它为什么要把页大小钉死"——因为钉死换来的是 decode 路径上一次移位代替一次除法。

这也是读源码比读文档值钱的地方：文档说"block 固定 64"，源码告诉你固定在哪一层。固定在配置层意味着改一个数字，固定在编译期意味着改一个架构决策。

## 二、两级分配：占位与物化分开

`reserve()` 只占容量，不绑物理页；`materialize()` 才绑（`src/core/paged_kv_cache.h:305`、`:316-319`）。这听上去像常见的 lazy allocation，但它解决的是一个具体问题：**预填充的容量必须能被承诺，而承诺的时候不该动显存**。

预算的唯一强制点在 `cover()`（`src/core/paged_kv_cache.cpp:463-482`）：

```cpp
need = reserved_ + extra;
free = capacity - lent - loaned - allocated;
if (need > free) { if (!recall_() || ...) return std::nullopt; }
```

注意 `free` 减了三个东西：`lent`（整段借出去、循环看不见的）、`loaned`（借给 expert cache 蹭的）、`allocated`（已物化的）。三个词各是一种不同的借用强度，下面第六节讲。

失败不是返回错误码，是抛 `ContextCacheExhausted`（`paged_kv_cache.cpp:513-520`）。这决定了上层必须有一个降级阶梯——第十节。

## 三、句柄防悬挂是结构性的，不是约定

CUDA 内存最容易出的错是 use-after-free：一个指针指向的页已经被别人回收。NInfer 的答案是让句柄**不可能**指向已回收的页：

```cpp
// src/core/paged_kv_cache.h:160-179
struct DeviceKVPageHandle {
  DeviceKVPagePool* owner_;  std::uint32_t index_;  std::uint32_t generation_;
};
```

每页有一个 `page_generations_`（`:399`），回收时递增。句柄构造函数全是 private，只 friend 池和 Lease。于是"拿着过期句柄用内存"在编译期就构造不出来，而不是靠 code review 抓。

## 四、分配策略直接决定"多少内存可以借"

这条是整份源码阅读里最反直觉的发现。

`materialize()` 优先取 `preferred_predecessor + 1` 或 `destination.back().index_ + 1`（`paged_kv_cache.cpp:549-566`）——优先取物理上紧邻的页。看着像 cache locality 优化，其实是**可借性的前提**：只有 page-major 连续的一段才能被 lend，因为 `plane_page_range`（`h:271-274`）只表达连续 run。

推论：**一个碎片化的池，即使空闲页够多，也借不出去。** 分配策略在这里不是性能选项，是功能前提。很多"为什么我的池明明有 2 GB 空闲却租不到显存"的问题，答案在分配器的连续性偏好里。

## 五、零拷贝的根据是一条不变量，不是一项技术

分层最贵的是拷贝。NInfer 的分层窗口（`src/core/evictable_kv_pool.h:16-30`）能把"打开窗口"做到不拷贝、不暂停 decode，全部理由写在一处：

1. store 绑在**今天就在用的偏移**上——plane 指针、block table、CUDA Graph 全部有效，所以重新映射之后没人需要改指针。
2. 只借**空闲页**，并且保证「**no page is read before the append kernel writes it**」——于是关闭窗口只需重映射，不需回拷。
3. `lease()` / `close()` **不 quiesce stream**，窗口开着 decode 照跑。

第 2 条这一句不变量支撑了两个特性：可借性，以及挂起快照的瘦身（`src/models/qwen3_5/program/residency.cpp:49-52` 用 `pool.free_extents()` 把空闲页当洞挖掉，注释说「on an idle server they are most of the persistent arena」）。

**"写前不读"这种句式值得记住。** 它是那种一句话就能撑起一整套优化、但不说出来就会被后人无声破坏的前提。写这类代码的人把它写成注释，读这类代码的人应该把它抄下来。

## 六、借有两种强度，而它们的上限不同

`lend` 与 `loan`（`src/core/paged_kv_cache.h:286-299`）不是同义词：

- **lend**：整段移出循环，admission 看不见了，给窗口租显存。
- **loan**：页仍算 available，给 MoE expert cache 蹭。上限是 `unpromised_free_pages()`——空闲减去"未来必定物化的预留"。

`loan_pages` 第一行就是断言：

```cpp
// src/core/paged_kv_cache.cpp:436-438
throw std::invalid_argument("Paged KV loan takes pages a reservation needs");
```

也就是说：**借出去的额度必须是"我现在确定用不上"的额度**。把还没承诺的空闲借出去，等于把未来的失败提前到今天，而且是别人替你还。

借还计划本体在 `src/core/kv_loan.h`：`kMaxLoanUnitPages = 256`（`:19-21`，每个 granule 需要超过 256 页的极小 scale plane 不借——注释算过账「cost the loan nothing but a few per cent of yield」）；只收完全落在借用页内的 granule（`:44-46`，共享 granule 永不 unmap）；**从最高空闲页开始借**（`:88-92`「the highest pages go first, which is where the allocator leaves the pool free the longest」）。最后这条是把分配器的连续性偏好反过来用：既然分配器从低位往上要，高位就是"最晚被要回去"的，先借它们。

## 七、跨驱动模型：Windows 上 host KV 不免费

这一条会让人踩坑，因为它的现象和原因隔得很远。

`src/core/host_kv_clamp.h:11-23`：Windows/WDDM 下，pinned host memory 被映射进 GPU 地址空间时**计入显存预算**。于是引擎把 host KV 夹到 `(free_device − 1 GiB) / 2` 再向下取整到整页。

后果：「host KV 是免费的，显存只放热数据」这个在 Linux 上成立的心智模型，搬到 Windows 直接 OOM，而现象是"显存莫名其妙被吃掉"——因为吃它的不是 CUDA allocation，是驱动。这段代码是纯算术、无 CUDA 依赖，单测在 `tests/host/test_host_kv_clamp.cpp`，可以在一台没有显卡的机器上验。

## 八、故障注入是设计的一部分

`src/core/evictable_kv_pool.h:117-122`：

```cpp
enum class LeaseFault { None, Unmap, Overlay, Access };
```

三个故障精确落在单个 granule 搬移的三个阶段之间。而这套机制能被测试的前提是明确的契约：**中途失败时把每颗送回原映射再抛，arena 复原如初、可重试。**

大多数代码的失败契约是"抛异常，希望调用方重来"。这里多了一句话——"抛之前先复原"。代价是异常路径更复杂，收益是调用方不必区分"失败了一次"和"失败到不能再用"。

## 九、把逐层恢复落在拷贝层

"分层恢复"经常被讲成调度问题：先恢复哪几层、和计算怎么交错。NInfer 把它做成**参数顺序问题**：

`copy_from_host_records(..., plane_begin, plane_end, ...)`（`paged_kv_cache.h:369-390`）让调用方把消费方马上要用的 per-layer planes 排在前面。上游 `submit_restore`（`hybrid_cache.cpp:551-617`）先发一个 prelude（tail + backend KV + continuation hidden + DFlash 局部状态）记一个事件，然后**按 `HybridRestoreLayer` 正序逐层发拷贝、每层一个事件**；消费侧 `land_restore` 只等 prelude：

```cpp
cudaStreamWaitEvent(consumer, restore_.prelude);
```

于是第 1 层的计算与第 20 层的拷贝重叠。**层序不是调度决策，是 memcpy 参数**——这个认识把"怎么交错"从运行时逻辑降级成数据准备，省掉一整套状态机。

配套的期望管理也在这层：`settle_prefetch()`（`hybrid_cache.cpp:651-661`）只同步**最后一个预取批的最后一层事件**，因为每层一个事件 + 单流 FIFO，最后一个事件完成就意味着之前所有批、所有层都完成了。一条流把依赖图换掉了（注释在 `poll()`：「every write shares the transfer stream ... so each list completes in submission order」）。

## 十、租约是四级降级链，任何一级都不许 launch 失败

Device KV 的容量不是"申请多少给多少"，是四级：

1. **扩租约**（`ensure_sequence_kv_lease`）：三档 `{margin_tokens, 2*page, page}`，末级必须等于分配器最小粒度 64。
2. **settle**：`lease_settled / lease_space_limited / lease_minimum_target`，保证"至少能跑完这一 step"。
3. **抢保留缓存**（`reclaim_device_kv_for_lease`）：一次一个 victim，排序键 `(recency, id)`，跳过有活跃边的和非 `shared_automatic_reclaimable` 的。
4. **缩答案预算**（`cap_remaining`）：`finish_reason = length`。

四级走完仍然失败 = 这是一条 bug，不是一种降级。设计意图写在两处注释里：

- `target() = min(cap, max(pages, min(reach, wanted)))` 三重 clamp 的目的是「**so the reservation can only fail on space**」——失败归因设计。任何一次失败都能明确回答"是空间不够，还是我算错了"。
- `only a decode step may extend the lease`——准入期不许偷偷扩租约，否则准入算出来的容量 B 不可信。

还有一个更细的公平性判据（`src/serve/resource_manager.h:2298-2299`）：**被扩展者不能成为扩展者的受害者**。当"牺牲 A 来扩 B"时，只有当 B 的能力严格包含 A 时才是净赚，否则窄的会拖垮宽的。

顺带修一个我先前读错的判断：算术预判（`can_resize_entitlement`，常态路径，池饱和时每步重进）与 `catch(bad_alloc)`（兜"账面与实际无原子性"）**两层都在**，不是二选一。好的容量管理从来不是"预判代替兜底"，是"预判处理绝大部分，兜底处理预判不可能知道的那部分"。

## 十一、内容寻址落到磁盘：每 frontier 一个 128 位摘要

`src/core/disk_kv_bridge.h:6-11` 的注释把契约写得比多数设计文档都清楚：

> A KV page covering `[p * 64, F)` is keyed by `digest(F)`… the same content in two sessions dedupes to one page. The tag separates engine configurations (speculative backend, proposal head, KV dtype)… A miss of any kind (never spilled, evicted, corrupt) makes the engine recompute.

三步都重要：**按内容去重**（两个会话的相同内容落成同一页）、**tag 区分引擎配置**（投机后端、proposal head、KV dtype 不同就不能混用）、**任何 miss 都整体重算**（不做部分恢复的猜测）。

摘要本体（`src/models/qwen3_5/program/prefix_identity.cpp:60-79`）是双射混合：

```cpp
kDigestOffset     = {1469598103934665603, 7809847782465536322};
kDigestMultiplier = {0x9e3779b97f4a7c15, 0xd6e8feb86659fd93};
kTokenDigestTag   = 0x74ULL << 40;   // token 首字高 16 位清零，防 [a][b] ≡ [ab] 歧义
```

xor → 奇数乘 → xorshift，三步都是双射。于是"两个 prompt 差一个词必然差摘要"是**代数事实**，不是密码学假设——碰撞只可能来自摘要空间的有限性，不可能来自这三步本身。

这里有一个我见过最好的性能实测注释：

> word-wise mixing costs a few cycles per word where a byte-wise FNV-1a step cost eight dependent multiplies, which on a 156K-token prompt was **10 ms of every admission** (RTX 5060 Ti host, 2026-10-28)

带硬件型号、带日期、带被替换掉的方案、带绝对量。这种注释比任何 benchmark 表都耐用。

以及一句限定范围的自我声明（`prefix_identity.h:64-66`）：「This is only a content shortlist: exact token and ResidentPrefixIdentity comparison remains authoritative for reuse.」——**摘要只是短名单，逐 token 比对才是判据**。非加密哈希会碰撞，所以必须能当场否掉候选。

## 十二、落盘格式：让 `path` 在任何时刻都是一份完整旧快照

`--prefix-cache-file` 的格式（`hybrid_persist.cpp:157-241` / `:248-415`）有几个能直接抄的决定：

**表前置、slab 后置。** load 先算恢复计划再 `seek`，200 GB 文件也只读被选中的 8 块。`reader.read()` 不计 seek 越过部分，所以进度条是诚实的。

**尺寸恒等式不读 footer 就能判截断**：`file_bytes == payload + slabs * slab_bytes + kFooterBytes`。

**三段校验各给不同处置**：身份（magic + version）→ 指纹（模型 / KV 格式 / 二进制）→ 几何（`slab_bytes / image_bytes / block_payload_bytes / text_page_stride / ...`）。判据是「改它之后旧字节是被**误读**（→指纹）还是只是**位置错位**（→几何）」——前者要拒，后者要换计划。

**三段全过才 `phase.emplace(...)`**：不属于我的文件不该在启动进度条上占一格。

**临时文件 + rename**，取消点长在 slab 之间**并且在 rename 之前再一次**（`kSaveAbandoned = "abandoned; the previous file is kept"`）。于是 `path` 在任何时刻要么是旧快照、要么是新快照，永远不是半份。

**"元数据对、数据未初始化"必须全清**：块已经登记进 index 但 slab 没读入时抛异常，hash 和 token 都对——命中会返回垃圾字节。这类失败最难在线上定位，所以直接 `clear()` 全清，连 `required_host_bytes` 也清掉（否则调用方会据此建议"加大 tier"）。

## 十三、容量是算术，不是测量

最后讲一个方法上的事。想知道"这张卡能跑多大上下文"，常见做法是上机扫。但容量算术其实只有加减乘除：

```
used = fixed_bytes                       // 1.5 GiB + 1 GiB + head_bytes
     + Σ 权重                             // 逐层
     + Σ GDN state                       // 每层固定，不随 token 增长
     + (Σ 每层 KV/页组 + 固定KV/页组) × page_groups
fits ⟺ used ≤ available_bytes
```

（`src/core/stage_plan.h:202-213`；纯整数运算，零 CUDA 依赖）

所有输入都在模型文件里。`.ninfer` 头部 32 字节后内嵌一份 126,944 B 的 JSON manifest，393 个 `objects[]` 各带 `shape / format / bytes / offset`——**模型文件自带完整账本**。任何一台没有 GPU 的机器都能算出完整容量表。

我按这个方法算了 Qwen3.5-4B（32 层、8 个 full-attention、KV heads 4、head_dim 256）：

| 量 | 值 |
|---|---|
| 32 层权重 | 2109.1 MiB |
| head（tied embedding + final_norm） | 497.3 MiB |
| MTP 参数 | 77.4 MiB |
| GDN state（24 层） | 49.12 MiB |
| bf16 KV 单价 | 262,144 B / 层 / 页（64 token，K 与 V 各 512 B/token/head） |
| Σkv / 页组（8 层 + MTP） | 2.2500 MiB |

于是 **每 64K token 上下文 = 2304 MiB（bf16）/ 1188 MiB（int8-g64）**。在 RTX 3060 Laptop 6 GB 上：权重 + state 常驻 2733 MiB，剩 451 MiB，得 200 页组 = **12,800 token**。

但这里有个必须标注口径的坑：`fixed_bytes` 预留了 2560 MiB，而上游实测（`docs/qwen35-dense-family.md:27`，RTX 5050 Laptop 峰值 2868 MiB）只比"全常驻"多用了 135 MiB。也就是说 **`fixed_bytes` 是给大卡留的保守 workspace 预算**，在 6 GB 卡上系统性低估。按实测口径重算，同一张卡能到 **81,792 token**——两个口径差 6.4 倍。

**报容量必须写口径**，否则同一个模型在同一张卡上能报出两个数量级的差。

算这套东西的时候我自己踩了两个坑，值得写下来：`bindings[].parts[].range` 是**元素数不是字节数**（一个 object 物理 7,372,800 B，range 却声称到 10,485,760——那是 4096×2560 个元素），拿它当字节累加会让 32 层之和超过整个文件 308 MiB；`tie_word_embeddings: true` 让 `token_embedding` 与 `output_head` 指向同一个 497.3 MiB 对象，按 binding 加就算两遍。两次都是靠**拿总数撞另一个独立分母**（层之和 vs 文件总字节）才发现，单点计算不会自己报错。

## 十四、这份代码值得学的六个习惯

不是设计模式，是习惯：

1. **把不变量的名字写进注释**。"no page is read before the append kernel writes it"、"the highest pages go first, which is where the allocator leaves the pool free the longest"——一句话撑起一整套优化，后人想改的时候知道自己在破坏什么。
2. **失败契约要写全**。不只写"抛异常"，还要写"抛之前复原"（lease 中途失败把每颗送回原映射）。
3. **把设计盲点写在决策点旁边**。根节点不携带 block extra、LoRA 身份对第 0 块内快照不可见，所以带 adapter 的请求禁止复用该快照——限制和它的理由在同一处（`hybrid_program.cpp:55-61`）。
4. **不变量在读写两侧各验一次**。写入侧靠 `file_index.at()` 保证 parents-first，读取侧再靠范围检查验一遍：读的不是自己的写操作。
5. **性能数字带硬件和日期**。10 ms/admission、RTX 5060 Ti、2026-10-28。
6. **预算常量与实测值分开记**。`fixed_bytes` 2.5 GiB 是预算，2868 MiB 是实测。两个数字都存在、都有出处，才谈得上判断余量。

## 十五、一句话总结

**KV cache 不是一块显存缓冲，是一个有页表、有租约、有分层、有落盘格式、有准入报价的微型内存管理系统。** 算法决定它能省多少，物理层决定它能不能在多会话、多卡、断电重启之后依然说得出"我知道这份字节是谁的"。

后者不性感，但它是"能用"和"看起来能用"的全部区别。

---

### 附：十一份源码笔记

| 笔记 | 覆盖 |
|---|---|
| 01 | KVMem / LMCache / Mooncake / vLLM / SGLang 五方案横评 |
| 02 | 104 个后端 patch 分类 |
| 03 | spread 检查点策略 |
| 04 | 分层实验计划 |
| 05 | Radiance AMD 后端 |
| 06 | 物理层：页表、两级分配、借还、WDDM 夹取 |
| 07 | 上层：一页两副本、epoch、suspend 七门、快照完整性 |
| 08 | prefix 层：块身份五元组、统一 slab 池、GDSF 恢复 |
| 09 | 磁盘层：内容寻址摘要、4 KiB 槽、65/25/10 配额、8 s 死线 |
| 10 | 准入：报价零拒绝、`fits()` 自洽会计、跨请求合并经济学 |
| 11–16 | StateImage 四态 / 租约阶梯 / spill 时机 / 落盘格式 / 分层求解器 / 容量算术 |
