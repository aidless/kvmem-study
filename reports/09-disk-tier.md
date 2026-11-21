# 09 · 磁盘（L3）层：内容寻址、可重建索引与"恢复不如重算"的判据

> **底座口径（2026-11-17 补）**：本文读的是 `iamwavecut/ninfer-all` 分支线 @`6cc1f9fa`（2026-11-04）的源码树。该线**不是上游主干**——`Neroued/ninfer` master 与它已分叉（`compare` 实测 `diverged, ahead_by=523, behind_by=1895, merge_base=f76e19c0fbd0`），且 2026-11-04 之后本线与上游都改过本文描述的部分机制（已核到 4 处，见 `notes/2026-11-17-ninfer上游与分支血缘核验.md`）。**引用本文任何断言请连带 sha 与日期。**

> 承接 06（设备页）→ 07（逻辑页 / Host arena）→ 08（前缀块 / Host slab）。本篇是存储栈最底层：**磁盘 KV 库**。
> 源码：`src/core/disk_kv_bridge.{h 175行, cpp 338行}`、`src/core/disk_kv_store.h 200行`、`src/models/qwen3_5/program/storage/disk_tier.cpp 471行`、`src/models/qwen3_5/program/prefix_identity.{h 99行, cpp 134行}`，commit `6cc1f9fa…`。
> 一句话结论：**NInfer 的磁盘层已经是内容寻址 + 跨会话去重**——这一条推翻了我 08 号报告里「跨会话共享是 KVMem 独有」的表述，本篇第十一节给修正后的定位。

---

## 一、三层存储的实际分工

| 层 | 介质 | 单位 | 淘汰 | 身份 |
|---|---|---|---|---|
| L1 | 显存 | 64-token 页（`kPagedKVPageSize`） | `cover()` 预算 + lease/loan | page index + generation |
| L2 | pinned host RAM | slab（块 bundle / StateImage） | GDSF 密度 | hash + extra + 64 token 原文 |
| **L3** | **磁盘文件** | **4 KiB 对齐的 slot** | **LRU** | **128 位滚动摘要 + tag + frontier** |

三层各自的"淘汰判据"完全不同：L1 是容量算术、L2 是价值/字节、L3 是时间。**这不是没统一，是三层该优化的量本来就不同。**

## 二、身份模型：每 token frontier 一个 128 位滚动摘要

```cpp
// src/core/disk_kv_bridge.h:6-11（文件头注释，全文照抄）
// Identity model: the engine keeps a rolling 128-bit digest per token frontier, a pure function of
// the first F tokens. A KV page covering [p * 64, F) is keyed by digest(F), which the live sequence
// can compute when it spills and an incoming request computes from its own prompt, with no other
// persistent metadata; the same content in two sessions dedupes to one page. The tag separates
// engine configurations (speculative backend, proposal head, KV dtype) whose bytes differ for the
// same tokens. A miss of any kind (never spilled, evicted, corrupt) makes the engine recompute.
```

三件事一次说清：

1. **摘要按 frontier 计，不按页计**：页 `[p*64, F)` 的 key 是 `digest(F)`。所以**落盘与恢复两侧都能独立算出同一个 key**，不需要任何持久元数据。
2. **同一段内容在两个会话里 dedupe 成一页**——磁盘层是内容寻址的。
3. `tag` 区分**字节不同但 token 相同**的引擎配置（投机后端、propose head、KV dtype）。

**「A miss of any kind makes the engine recompute」** 是整层的失败语义：磁盘缓存失效只需重算，不需要正确性兜底。这条决定了后面所有的宽松设计（无 fsync、可重建索引、恢复超时就放弃）。

## 三、摘要本体：词级混合，附一个带实测的性能理由

```cpp
// prefix_identity.cpp:60-79
constexpr DigestPair kDigestOffset{1469598103934665603ULL, 7809847782465536322ULL};
constexpr DigestPair kDigestMultiplier{0x9e3779b97f4a7c15ULL, 0xd6e8feb86659fd93ULL};
constexpr std::uint64_t kRewriteDigestDomain = 0x6e696e6665722d72ULL;  // "ninfer-r"
constexpr std::uint64_t kVisionDigestDomain  = 0x6e696e6665722d76ULL;  // "ninfer-v"
constexpr std::uint64_t kLoraDigestDomain    = 0x6e696e6665722d6cULL;  // "ninfer-l"
// Tags the first word of a token record. That word has its top 16 bits clear and the domain words
// above do not, so the word sequence of a prompt is a unique encoding of its records.
constexpr std::uint64_t kTokenDigestTag = 0x74ULL << 40U;             // 't'

// Mixes one 64-bit word into each lane: an xor, an odd multiply and a xorshift, each a bijection
// of the lane, so two prompts that differ in one word always differ afterwards. Word-wise mixing
// costs a few cycles per word where a byte-wise FNV-1a step cost eight dependent multiplies, which
// on a 156K-token prompt was 10 ms of every admission (RTX 5060 Ti host, 2026-10-28).
```

四个可搬走的点：

1. **「双射」是抗碰撞论证的全部**：xor（双射）× 奇数乘（模 2⁶⁴ 双射）× xorshift（双射）→ 两个 prompt 只要有一个词不同，摘要必然不同。**这不是密码学假设，是代数事实**——比"用了个好哈希"严谨得多。
2. **词级而非字节级，附实测数**：字节级 FNV-1a 每字节 8 次依赖乘法 → 156K token 的 prompt 每次 admission 花 **10 ms**（RTX 5060 Ti host，2026-10-28）。改词级混合降到"每词几周期"。**这是我读到的第一个带具体硬件和日期的性能数字**，可直接引用。
3. **域分隔符也是 ASCII 模型名**：`0x6e696e6665722d72` = "ninfer-r"（rewrite）、`-v`（vision）、`-l`（lora），与 08 号报告的 `kVisionExtraSeed` 同一套约定。
4. **`kTokenDigestTag = 0x74 << 40` 是自定界编码**：token 记录首字的高 16 位为零，域字的高 16 位非零 → **扫描时能唯一解析出记录序列**，不会把 `[a][b]` 和 `[ab]` 混成一个。这是防长度扩展歧义的标准手法，比"记录间加分隔符"省且更硬。

> ⚠️ 实操：`kVisionDigestDomain` 那行在 `cat`/`grep` 里会被渲染成 `0x6e696e6665722d76ULL` 之类的乱码。**我这次用 `od -c` 复核了第 62/63 行的原始字节**，确认真值是 `0x6e696e6665722d72` / `0x6e696e6665722d76`。凡是 `0x6e...` 开头的可打印魔数，终端都可能吃掉——`od -c` 是唯一可信通道。

## 四、摘要是「候选表」，精确身份才权威

```cpp
// prefix_identity.h:64-66（类注释）
// One rolling digest per token frontier. This is only a content shortlist: exact token and
// ResidentPrefixIdentity comparison remains authoritative for reuse. Keeping it separate from the
// exact identity avoids retaining hash-only state in immutable capture backings.
```

精确身份 `ResidentPrefixIdentity` 的字段（`prefix_identity.h:16-60`）：

```cpp
std::vector<std::uint8_t> token_types_;
std::array<std::vector<std::int32_t>, 3> positions_;      // 三个位置轴
std::vector<VisionItem> vision_items_;
std::vector<std::uint32_t> rewrite_execution_frontiers_;
std::array<std::uint64_t, 2> lora_key_;
```

**注意 `positions_` 是三个独立的 `int32` 轴**，不是单个位置——这对 Qwen3.5 这种多模态/多轴位置编码是必须的。

**分离的理由写得很精确**：「avoids retaining hash-only state in immutable capture backings」。捕获后的 backing 是不可变共享的，往里塞只用于查找的哈希状态会污染它。**查找结构与权威身份分开存，不是洁癖，是共享不可变对象的约束。**

> 这条与 08 号报告的 `SavedBlock{hash, tokens}` 是**同一纪律的两个实现**：hash 负责找，原文负责判。KVMem 03 号报告的「content proof 链」也是这个结构。**三份独立实现给出同一个答案**，说明这不是某个工程师的偏好。

## 五、磁盘存储：自描述槽 + 可重建索引 + 不做 fsync

```cpp
// disk_kv_store.h:3-12（文件头注释，全文照抄）
// A file-backed, self-describing, LRU page store: the disk (L3) tier under the Host KV arena.
//
// The data file is an array of 4 KiB-aligned slots. Each slot starts with a 48-byte header (magic,
// content identity, CRC-32 of the payload, LRU stamp), so the identity-to-slot index can always be
// rebuilt by scanning the slots. The index file (<path>.idx) is replaced atomically (temporary
// file and rename) whenever the live set changes; there is no fsync, so a crash can lose the last
// unpublished pages but never makes a stale row trusted: every index row is checked against its
// slot header on open, and every read verifies the payload CRC.
```

四层防线，一层比一层省：

| 层次 | 机制 | 崩了会怎样 |
|---|---|---|
| 槽头 | 48 字节 magic + 身份 + CRC + LRU 戳 | 槽自描述，索引可全量重建 |
| 索引 | `.idx` 原子替换（tmp + rename） | 丢最后一批未发布的页 |
| 打开时 | 每行索引**逐条**对槽头 | 陈旧行立即可疑 → 丢弃 |
| 每次读 | 校验 payload CRC | 损坏页 → miss → 重算 |

**「no fsync」这个决定很硬**：文件是纯缓存，崩溃丢失只等于 miss。而 fsync 会让每次落盘变成毫秒级同步等待。**判据是"失去它有没有正确性后果"，没有就不付这个代价。**

槽头结构（`disk_kv_store.h:123-136`，`static_assert(sizeof(Header) == 48)`）：

```cpp
struct Header {
    std::uint32_t magic = 0;       // 0x4E44564B = "NDVK"
    std::uint32_t reserved = 0;
    std::uint64_t last_used = 0;   // LRU 戳，可持久化
    std::uint64_t lo = 0, hi = 0;  // 128 位摘要
    std::uint32_t tag = 0;
    std::uint32_t frontier = 0;
    std::uint32_t crc = 0;
    std::uint32_t pad = 0;
};
```

`last_used` **落在磁盘上**，所以重启后 LRU 顺序能恢复（`disk_kv_store.cpp:323-324` 打开时 `clock_ = max(clock_, slot_header.last_used)` 并重建 `lru_` 树）。**热页不会因为重启被当成冷页踢掉**——很多实现的磁盘 LRU 一重启就全乱。

## 六、为什么用 positioned I/O 而不是 mmap

```cpp
// disk_kv_store.h:170-177（注释全文）
// Payloads move through positioned reads and writes, not the mapping: a store larger than
// memory makes mapped copies fault page by page, and a mapped write into a sparse slot on a
// full disk raises SIGBUS where a positioned write returns an error, which a pure cache turns
// into "not stored". False on any I/O error or short transfer.
```

两个理由，第二个是真正的坑：

1. 库比内存大 → mmap 拷贝会逐页缺页；
2. **满盘时 mmap 写稀疏槽会 `SIGBUS`（进程死），而 `pwrite` 返回错误**——纯缓存把它翻译成"没存上"就完事了。

> **「错误应该表现为返回值还是信号」决定了一个缓存能不能在满盘时活着。** 这条对我以后写任何文件缓存都适用：mmap 只用于读、且文件小于内存。

索引本身仍走 mmap（`base_`），因为是元数据级小结构。

## 七、预算切分：65 / 25 / 10

```cpp
// disk_kv_bridge.cpp:37-52
const bool backend              = options_.backend_page_stride != 0;
const std::size_t main_share    = backend ? 65 : 85;
const std::size_t backend_share = backend ? 25 : 0;
const std::size_t state_share   = 100 - main_share - backend_share;
const auto open = [&](DiskKVKind kind, std::size_t stride, std::size_t share) {
    Family& target = family(kind);
    target.stride  = stride;
    const std::size_t budget = options_.capacity_bytes / 100 * share;
    if (stride == 0 || budget < stride) { return; }      // 装不下一页 → 这个 family 直接不开
    ...
};
open(DiskKVKind::MainKV,    options_.main_page_stride,    main_share);
open(DiskKVKind::BackendKV, options_.backend_page_stride, backend_share);
open(DiskKVKind::StateImage,options_.state_page_stride,   state_share);
if (family(DiskKVKind::MainKV).store == nullptr) {
    throw std::invalid_argument("disk KV budget does not hold one main KV page");
}
```

**固定配额而不是统一池**——这与 08 号报告的 Host tier（统一池、不分配额）**正好相反**。

为什么？我的解读：Host 是**同一进程内**、淘汰靠价值/字节密度，统一池能让高价值对象抢到字节；磁盘是**三个独立文件**、淘汰靠 LRU，LRU 跨类比较没有意义（StateImage 的一次命中省下的时间和一个 KV 页的一次命中差几个数量级，但 LRU 不知道）。**固定配额 = 给"没法和别人比价值的类别"预设比例。**

> 这是个真实的架构矛盾，值得记：**同一个系统在 L2 用统一池、在 L3 用固定配额**，原因是两层能拿到的价值信号不同。L2 有 `prefill_seconds` 可以算字节价值，L3 只有时间戳。

三个"拒绝无效配置"的点：① `capacity_bytes == 0` 或 `main_page_stride == 0` → throw；② 某 family 预算装不下一页 → **静默不启用该 family**（但不报错）；③ Main 都开不出来 → throw。**分级拒绝**：核心能力缺 → 抛；可选能力缺 → 静默降级。

## 八、写路径：不许留洞

```cpp
// disk_kv_bridge.h:80-86（注释全文）
// Copies the page into the queue, waiting up to `timeout` for room: a caller writing a prefix
// chain must not leave a hole, since a restore stops at the first missing page. True when
// queued or already stored.
bool spill_page_wait(const DiskKVIdentity& id, DiskKVKind kind,
                     std::span<const std::byte> bytes, std::chrono::milliseconds timeout);
```

**「restore stops at the first missing page」是整个设计的枢纽**：前缀链是有序的，中间缺一页，后面全废。所以写链必须**阻塞等位**（`spill_page_wait`），不能像普通缓存那样"满了就丢"。

配套数字（`disk_kv_bridge.h` 私有段）：

| 常量 | 值 | 理由（注释原文/推导） |
|---|---|---|
| `kQueueCapacity` | 512 | 队满时 owner spill 等待或重试，不丢 |
| `kWorkerCount` | **8** | 「Several streams are needed to reach NVMe bandwidth; the store serializes only bookkeeping」 |
| `kFlushInterval` | 64 | 索引发布批量：「slots are self-describing and a lost index is rebuilt」 |

**8 个写线程**这个数很具体：NVMe 需要多队列并发才吃满带宽，而 store 只串行化簿记（不是串行化 I/O）。`kQueueCapacity = 512` 配 8 worker 意味着队列深度 64/worker，合理。

三种写接口对应三种语义：
- `spill_page_wait` — 阻塞等位，写前缀链用；
- `spill_page_sync` — 调用线程直接写，给大 blob（StateImage）用，不排队；
- `try_probe` / `try_submit` — 非阻塞，队忙返回 null ticket 让调用方稍后重试。**「Submitted bytes are borrowed and must stay alive and unchanged until the ticket leaves Pending, cancellation included」**——借出去的字节在 ticket 落定前不能动，包括取消时。

## 九、恢复路径：32 页一批、8 秒死线、按批检查

```cpp
// storage/disk_tier.cpp:392-427
const auto deadline = std::chrono::steady_clock::now() + kRestoreBudget;   // 8s
synchronize(compute_streams);          // 已入队的 page activation / state reset 先完成
const auto restore_chain = [&](...){
    const std::uint32_t page_count = kv_pages_for_frontier(chain_frontier);
    if (page_count > addresses.mapped_pages(address)) { return false; }
    const HostKVAllocationView staging = disk_staging_arena->writable_view(allocation);
    for (std::uint32_t begin = 0; begin < page_count; begin += kDiskStagingPages) {   // 32
        if (std::chrono::steady_clock::now() > deadline) { return false; }            // 按批查死线
        ...
        ids.push_back(disk_identity(sequence.prefix_digests,
                                    std::min((page + 1U) * kPageTokens, chain_frontier)));
        ...
        if (!disk_kv->restore_pages(ids, kind, staging_span)) { return false; }
        pages.physical_pool().copy_from_host(staging.subview(0, count), destinations, transfer_streams);
        synchronize(transfer_streams);
    }
    return true;
};
```

五个要点：

1. **最后一批的 key 用 frontier 而非页边界**：`min((page+1)*64, chain_frontier)`。所以 frontier=1000 的链，最后一页 key 在 1000 而不是 1024。这解释了为什么 spill 侧也要按 frontier 算。
2. **`kRestoreBudget = 8s` 的判据是"不如重算"**（文件头注释 `:24-25`：「A restore that takes longer than recomputing would is abandoned for the recompute」）。**恢复有硬死线，超了就重算**——这是缓存该有的姿态：缓存永远不该让请求比不缓存更慢。
3. **每批 32 页**（`kDiskStagingPages`），批间 `synchronize`。为什么同步而不是开流水线：因为 staging 只有一份，下一批要覆盖它。**用简单换确定性。**
4. **`restore_pages` 一次多页，任一页缺/坏即全 false**：「False as soon as any page is missing or corrupt」。与"链不许有洞"一致。
5. **`page_count > mapped_pages` 先判**：不先要页就不去读盘，避免读回来放不下。

## 十、`disk_tier.cpp` 的其余五个可引用常量与设计

| 项 | 值/内容 | 位置 |
|---|---|---|
| `kDiskStagingPages` | 32 页/批 | `:20` |
| `kSpilledAnchors` | 3（rewrite seam + 最多 3 个最早的长 anchor） | `:22-25` |
| `kRestoreBudget` | 8 s | `:26` |
| `kReleaseSpillBudget` | 20 s（materialization 之外释放的 owner 在此预算内入队） | `:28` |
| `kShutdownSpillBudget` | 60 s（关停时写常驻续写） | `:30` |

`kSpilledAnchors = 3` 的理由写得很实在：

> 「The rewrite seam and at most this many of the earliest long anchors are written with an owner: a follow-up turn re-renders the conversation, so boundaries inside generated text rarely match and the early, stable boundaries are the ones a later prompt shares.」

**按"后续 turn 会重新渲染对话"这个使用模式，决定只存"稳定的早边界"**——生成文本内部的边界几乎不会在下一轮匹配上。**存少而准，胜过多而全。**

profile 目录命名（`:40-55`）：

```cpp
std::string name = "kv" + storage + "_main" + main_stride + "_backend" + backend_stride + "_state" + state_bytes;
if (yarn.factor > 1.0F) name += "_yarn" + lround(factor*1000) + "_" + native_context;   // YaRN 旋转每个 key
if (yarn.interpolation_factor > 1.0F) name += "_pi" + lround(interpolation_factor*1000) + "_" + threshold;
```

**「YaRN changes every rotated key, so its KV must never meet an unscaled run's」**——用目录名物理隔离，比在文件里存参数再校验更彻底：**跨配置的字节根本不在同一个目录里，误读在打开文件之前就不可能发生。**

> 这是"用文件系统布局代替运行时校验"的范例。08 号报告的 `Geometry` 校验是运行时防线，这里是**布局防线**——能前移到越早的层越好。

## 十一、修正 08 号报告的定位：跨会话共享已经在磁盘层了

我 08 号报告末尾那张表写「跨进程共享 ❌」——**不准确**。磁盘层是内容寻址的，`disk_kv_bridge.h:9-11` 明确写了「the same content in two sessions dedupes to one page」。

修正后的精确表述：

| 能力 | NInfer 原生 | KVMem 换的 |
|---|---|---|
| 内容寻址的磁盘页库 + 跨会话 dedupe | ✅ **已有**（128 位滚动摘要 + tag + frontier） | 不改 |
| 索引可重建、原子替换、CRC 校验 | ✅ 已有 | 不改 |
| LRU 淘汰 + 持久化 LRU 戳 | ✅ 已有 | 不改 |
| 前缀块身份 + Host slab + GDSF | ✅ 已有（08 号） | 不改 |
| **Host RAM 层跨进程共享** | ❌ 每进程独占 arena | ✅ 换 |
| **无界保留（RetainAll）** | ❌ 三层全有界 | ✅ 换 |
| **内容检索召回**（非前缀命中） | ❌ 只有前缀 | ✅ 换 |

**所以 KVMem 的三个真增量里，"共享"不是"从无到有"，而是"从磁盘层上移到 RAM 层"。**

这个差别对性能是决定性的：磁盘命中是**毫秒级 + 一次内存拷贝**（8s 死线、32 页/批），RAM 层命中是**微秒级 + 零拷贝**（06 号报告的可借性）。**KVMem 真正在卖的是"把内容寻址从 L3 升到 L2"，而不是"引入内容寻址"。**

> 这也是为什么 KVMem 的补丁要动 Host arena 那么深（+16651）：磁盘层的 dedupe 早就成立，但要让它发生在 RAM 里，就必须把 L2 的"有界 + GDSF + 独占"三个假设全拆掉。**同一套语义在不同层，代价不是一个数量级。**

## 十二、可直接引用的数字（本报告）

| 项 | 值 | 来源 |
|---|---|---|
| 槽对齐 / 槽头 | 4 KiB / 48 字节（`static_assert`） | `disk_kv_store.h:5,138` |
| 槽头 magic | `0x4E44564B`（"NDVK"） | `disk_kv_store.h:134` |
| 索引 magic / 版本 | `0x4944564B`（"IDVK"） | `disk_kv_store.h:157` |
| CRC | CRC-32 zlib 多项式 `0xEDB88320`，slicing-by-8 | `disk_kv_store.h:120-121` |
| 预算切分 | 65/25/10（有 backend）/ 85/15 | `disk_kv_bridge.cpp:38-40` |
| 写队列 / 写线程 / 索引批量 | 512 / 8 / 64 | `disk_kv_bridge.h` 私有段 |
| 恢复批大小 | 32 页 | `disk_tier.cpp:20` |
| 恢复死线 | 8 s | `disk_tier.cpp:26` |
| 释放 / 关停预算 | 20 s / 60 s | `disk_tier.cpp:28,30` |
| 落盘锚点 | rewrite seam + ≤3 最早长 anchor | `disk_tier.cpp:22-25` |
| 摘要 | 2×64 位 = 128 位，词级混合 | `prefix_identity.cpp:60-79` |
| 字节级 vs 词级摘要代价 | 156K token prompt 每次 admission **10 ms**（RTX 5060 Ti host，2026-10-28） | `prefix_identity.cpp:69-72` |
| 摘要 lane 混合常数 | offset `{1469598103934665603, 7809847782465536322}`，multiplier `{0x9e3779b97f4a7c15, 0xd6e8feb86659fd93}` | `prefix_identity.cpp:60-61` |
| token 记录标记 | `kTokenDigestTag = 0x74 << 40`（高 16 位清零 → 自定界） | `prefix_identity.cpp:66` |

---

## 十三、这一层教给我的四条通用做法

1. **缓存的身份用"能两边独立算出的纯函数"**，不存元数据。`digest(F)` 是前 F 个 token 的纯函数 → spill 侧和 restore 侧各算一遍就对上，不需要任何一侧写索引给对方读。
2. **哈希永远只是候选表，原文才是判据。** 三层独立实现（磁盘摘要+CRC、前缀块 token 原文、KVMem 内容 proof 链）给同一答案。
3. **不 fsync 的前提是"损坏 = miss"**。先确定失效能被降级成什么，再决定付不付持久化代价。反过来做（先上 fsync 再想失败语义）会把缓存写成数据库。
4. **"恢复不如重算"要落成硬死线**，而且按批检查而不是只在开头检查一次。8 秒这个数是经验值，但**机制**（死线 + 放弃 + 重算）是结构性的。
