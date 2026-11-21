# 14 · 前缀缓存的落盘格式：分段、取消点与三段式校验

> **底座口径（2026-11-17 补）**：本文读的是 `iamwavecut/ninfer-all` 分支线 @`6cc1f9fa`（2026-11-04）的源码树。该线**不是上游主干**——`Neroued/ninfer` master 与它已分叉（`compare` 实测 `diverged, ahead_by=523, behind_by=1895, merge_base=f76e19c0fbd0`），且 2026-11-04 之后本线与上游都改过本文描述的部分机制（已核到 4 处，见 `notes/2026-11-17-ninfer上游与分支血缘核验.md`）。**引用本文任何断言请连带 sha 与日期。**

> 08 号报告读了 `hybrid_persist.cpp` 的**格式骨架**（magic、块记录、指纹与几何、取消点位置）。这篇把 `save()` 与 `load()` 的**完整流程**读完——包括为什么表必须在前面、为什么 load 的校验分三段、以及"文件损坏"与"文件属于别人"为什么用不同的处置。
> 源码：`src/models/qwen3_5/program/prefix/hybrid_persist.cpp`（`save :157-241`、`load :248-415`；常量与结构 `:32-99`）。
> 一句话结论：**这个格式把"能不能读"拆成三段独立校验（身份→指纹→几何），每段失败给不同的处置；把"读哪几块"交给一个预先算好的计划，让 slab 区可以随机 seek——于是小 tier 只读它要的那几块。**

---

## 一、文件布局

```
offset 0     magic      8   "NINFHPC1"
         +   version    4   = 3
         +   fp_len     4
         +   fingerprint fp_len
         +   Geometry   7 字段
         +   n_nodes    4
         +   n_snaps    4
         +   块表        n_nodes × 284
         +   快照表      n_snaps × 变长
         ┌─ payload 起点（reader.position() 在这里取值）
         +   slab 区     slabs × slab_bytes      ← 先所有 block slab，再所有 snapshot slab
         +   footer      "NINFEND1" + 4 + 4
```

**块记录 284 字节，其中 256 是 token 原文**（08 号报告的结论：hash 只找候选，token 才判等）：

```cpp
// hybrid_persist.cpp:64-70
struct SavedBlock {
    std::int32_t parent = kRootIndex;
    std::uint64_t hash  = 0;
    std::uint64_t extra = 0;
    std::array<TokenId, pc::kBlockTokens> tokens{};   // 64 × 4 = 256
};
```

**快照记录变长**：`tail_len` 决定读多少 token，`slabs` 数由几何算出来（`image_slabs + (tail_len != 0 ? 1 : 0)`），`first_slab` 是 slab 区内的序号——**写入时算，读取时重算并核对**。

## 二、为什么表在 slab 前面

这是整个格式最重要的一个决定。`load()` 里：

```cpp
// hybrid_persist.cpp:315-318
out.required_host_bytes = slabs * slab_bytes;
out.host_bytes          = static_cast<std::uint64_t>(config_.host_slabs) * slab_bytes;

const pc::HostRestorePlan plan =
    pc::plan_host_restore(parents, shapes, config_.host_slabs, config_);
```

**先读完全部表（284×n 字节，很小），算出 `plan`（哪些块要、哪些快照要），然后才去读 slab。**

如果反过来（每块后面紧跟自己的 slab），要读到第 100 块的元数据就得先 seek 过前 99 个 slab 的字节——**在 100 GB 的文件上是 100 GB 的 seek**。

于是读取循环变成：

```cpp
// hybrid_persist.cpp:334-341（block 循环内）
const std::optional<pc::RestoredBlock> block = index_->restore_host_block(
    parent, saved_block.hash, saved_block.tokens, saved_block.extra);
if (!block) { continue; }
reader.seek(payload + static_cast<std::uint64_t>(index) * slab_bytes);
reader.bytes(slab(block->slab), slab_bytes);
```

**`seek` 到"这一块在文件里的位置"，只读那一块。** 文件可以是 200 GB，实际读的可能只有 8 块。

而且 `reader.read()`（返回"已读字节数"）**不计 seek 越过部分**（08 号报告 `:140` 的注释）——这使"读了多少"这个进度指标反映真实 IO，而不是文件偏移：

```cpp
// hybrid_persist.cpp:140
// Bytes read, not counting what seeks passed over.
```

> **分段布局 + 随机访问 = 小 tier 只付它要的那部分代价。** 而"先表后数据"是让这件事成立的前提：没有表，你连"第 N 块在哪"都不知道。

## 三、save 的顺序不变量：父必须先于子

```cpp
// hybrid_persist.cpp:203-214
std::unordered_map<std::uint32_t, std::int32_t> file_index;
file_index.reserve(nodes.size());
for (const pc::NodeRef node : nodes) {
    const pc::BlockIdentity identity = index_->block_identity(node);
    writer.value(identity.parent.valid() ? file_index.at(identity.parent.index) : kRootIndex);
    ...
    file_index.emplace(node.index, static_cast<std::int32_t>(file_index.size()));
}
```

**`file_index.at(...)` 用的是 `at` 不是 `[]`**——父不在 map 里就抛 `std::out_of_range`。这是把"父必须先写"这个不变量做成**运行期断言**，而不是写在注释里靠人记住。

**它依赖 `collect_persistable` 已经按 parents-first 顺序返回节点。**

```cpp
// hybrid_persist.cpp:227-231（快照记录）
const pc::SnapshotView view = index_->snapshot(snapshot);
if (view.host_slabs.size() != host_layout_.image_slabs + (view.tail_len != 0 ? 1U : 0U)) {
    throw std::logic_error("a saved snapshot's Host slabs do not match its tail");
}
```

快照的 slab 数也要核对——**写入前先验证"这个对象的形状和几何一致"**，避免写出一个读回来会算错的记录。

且快照的 `anchor` 也用 `file_index.at()`：

```cpp
writer.value(view.anchor.valid() ? file_index.at(view.anchor.index) : kRootIndex);
```

**anchor 必须已经作为 block 写过**——同样靠 `at` 兜住。

## 四、三段式校验：为什么"缺文件""别人的文件""几何不对"要分开

```cpp
// hybrid_persist.cpp:250-253
if (index_->stats().nodes != 0 || index_->stats().snapshots != 0) {
    throw std::logic_error("hybrid prefix cache load requires an empty cache");
}
Reader reader(path);
if (!reader.open()) {
    out.message = "no saved prefix cache at " + path.string() + " yet; it is written at shutdown";
    return out;
}
```

**① 文件不存在**：`out.message` 说明"还没写过，关机时才写"，`ok = false` 但**不是错误**——首次启动的正常情况。

```cpp
// hybrid_persist.cpp:258-263
std::array<char, 8> magic{};
reader.bytes(magic.data(), magic.size());
if (magic != kMagic || reader.value<std::uint32_t>() != kVersion) {
    out.message = "not a prefix cache file of this format version";
    return out;
}
```

**② 不是这个格式 / 版本不同**：一句话说清，直接返回。

```cpp
// hybrid_persist.cpp:264-280
const auto fingerprint_bytes = reader.value<std::uint32_t>();
if (fingerprint_bytes > (1U << 20U)) {
    out.message = "prefix cache file header is damaged";
    return out;
}
std::string saved(fingerprint_bytes, '\0');
reader.bytes(saved.data(), saved.size());
if (saved != fingerprint) {
    out.message = "saved prefix cache belongs to a different model, KV format or build";
    return out;
}
if (reader.value<Geometry>() != geometry_of(host_layout_)) {
    out.message = "saved prefix cache has a different Host geometry";
    return out;
}
```

**③ 指纹不匹配 / 几何不匹配**：两句不同的 message——"属于别的模型、KV 格式或构建" vs "Host 几何不同"。

**为什么 geometry 要单独一句**：08 号报告的判据是「改它之后旧字节是**被误读**（→指纹）还是只是**位置错位**（→几何）」。
- 指纹不对 = **字节的含义变了**（换模型、换 KV dtype），读了就是垃圾。
- 几何不对 = **字节的含义没变，但边界位置变了**（`slab_bytes`、`image_bytes`、page stride），读了会切错地方。

两类故障的排查方向完全不同，所以不能合成一句"文件不兼容"。

**而且这三段都在 `phase` 之外**：

```cpp
// hybrid_persist.cpp:266-267
// Opened once the header matches, so a missing or foreign file shows only the reason it was
// not restored. A file damaged part way through completes it with the bytes that were read.
std::optional<StartupPhaseScope> phase;
```

**`phase.emplace(...)` 在三段校验全过之后才执行**（`:324`）。什么意思：**一个不属于我的文件，不该在启动进度条上占一格。** 只有确认"这确实是我的缓存、我要开始读了"，才对外声明"正在加载前缀缓存"。

> **对外声明"我在做什么"之前，先确认"这件事确实要做"。** 否则用户在启动日志里看到"加载前缀缓存 0%…失败"，而那根本不是他的格式。

## 五、分配前的上界校验

```cpp
// hybrid_persist.cpp:290-295
const std::uint64_t file_bytes = std::filesystem::file_size(path);
const std::uint64_t slab_bytes = host_layout_.slab_bytes;
const auto node_count          = reader.value<std::uint32_t>();
const auto snapshot_count      = reader.value<std::uint32_t>();
// Every entry takes at least one slab of the file: bounds the tables before allocating.
if (static_cast<std::uint64_t>(node_count) + snapshot_count > file_bytes / slab_bytes) {
    throw std::runtime_error("prefix cache file header is damaged");
}
```

**在 `std::vector<SavedBlock> blocks(node_count)` 之前**先算：每个 entry 至少占一个 slab，所以 `node_count + snapshot_count` 不可能超过 `file_bytes / slab_bytes`。

一个损坏的 header 可能声称 `node_count = 4e9`，`std::vector<SavedBlock>` 会试图分配 1.1 TB —— **在分配之前挡住，而不是 catch `bad_alloc`。**

> 这是**外部输入的尺寸字段必须先用文件实际大小约束**的标准动作。注意它用的是"每项至少一个 slab"这个松弛上界，不是精确值——**精确值需要读表，而读表需要先分配**。松弛上界刚好够用。

## 六、结构不变量在 load 侧重验

```cpp
// hybrid_persist.cpp:298-307（block 循环）
const SavedBlock& block = blocks[index];
block.parent      = reader.value<std::int32_t>();
...

if (block.parent >= static_cast<std::int32_t>(index) || block.parent < kRootIndex) {
    throw std::runtime_error("prefix cache file block order is damaged");
}
```

**`parent < index`**——"父序号一定在子之前"。save 侧靠 `file_index.at()` 保证，load 侧**再验一遍**：

> save 保证了这件事，但 load 读的是**可能被改过的文件**。**写入侧的不变量，读取侧必须重验——因为读取侧面对的不是自己的写操作，是磁盘上的字节。**

同样的重验在快照上：

```cpp
// hybrid_persist.cpp:311-315
if (snapshot.tail_len >= pc::kBlockTokens ||
    snapshot.anchor >= static_cast<std::int32_t>(node_count) ||
    snapshot.anchor < kRootIndex) {
    throw std::runtime_error("prefix cache file snapshot is damaged");
}
```

`tail_len < 64`（超过一块就不是 tail 了）、`anchor < node_count`（不能指到表外）。

以及 kind 的枚举范围：

```cpp
// hybrid_persist.cpp:318-322
const auto kind_byte = reader.value<std::uint8_t>();
if (kind_byte > static_cast<std::uint8_t>(pc::SnapshotKind::Boundary)) {
    throw std::runtime_error("prefix cache file snapshot kind is damaged");
}
```

**反序列化枚举必须先验范围再 cast**——C++ 里任意 `uint8_t` cast 成 enum 都是合法值，后面 `switch` 没有 default 就是 UB。

## 七、尺寸恒等式：用算术而不是读 footer 来判截断

```cpp
// hybrid_persist.cpp:326-329
const std::uint64_t payload = reader.position();
if (file_bytes != payload + slabs * slab_bytes + kFooterBytes) {
    throw std::runtime_error("prefix cache file is truncated or damaged");
}
```

**读完表就立刻用算术核对总长度**，而不是先读 footer。所以：

- 文件被截断 → 这里就抛（不需要读到 footer 才知道）
- 多出尾巴 → 同样抛

`kFooterBytes = 8 + 4 + 4 = 16`（`hybrid_persist.cpp:36`）。

**footer 的校验在最后**，且校验两个计数：

```cpp
// hybrid_persist.cpp:388-396
reader.seek(payload + slabs * slab_bytes);
std::array<char, 8> footer{};
reader.bytes(footer.data(), footer.size());
if (footer != kFooter || reader.value<std::uint32_t>() != node_count ||
    reader.value<std::uint32_t>() != snapshot_count) {
    throw std::runtime_error("prefix cache file footer is damaged");
}
```

**footer 是"写完整了"的证据**：magic 在前说明"这是这个格式"，footer 在后说明"写到最后一个字节了"。计数重复一遍是为了抓"表被改过但 footer 没跟上"。

> 三段防线各管一件事：**长度恒等式管"截断"**（算术可判，不必读到底）、**footer 管"完整性"**（写到了最后一个字节）、**CRC 类校验不存在**（这个格式不校验内容，因为 slab 的内容正确性由块身份与 token 判等保证）。

## 八、级联跳过：被跳过的块带走它的后代

```cpp
// hybrid_persist.cpp:331-340
std::vector<std::optional<pc::NodeRef>> restored(node_count);
for (std::uint32_t index = 0; index < node_count; ++index) {
    const SavedBlock& saved_block = blocks[index];
    if (!plan.blocks[index] ||
        (saved_block.parent != kRootIndex &&
         !restored[static_cast<std::size_t>(saved_block.parent)])) {
        continue;
    }
    ...
}
```

**两个跳过条件**：① `plan` 没选它（tier 装不下，08 号报告的 GDSF 贪心）② 它的父没恢复成功。

第二条是关键：**一个块的 Host 语义是"从根到这个块的连续路径"**（08 号报告的 `collect_persistable` 是"可恢复闭包"）。父没恢复，子恢复出来没有父指针指向它——**变成一个无法通过前缀命中到达的孤儿块**。

```cpp
// hybrid_persist.cpp:354-357（快照循环）
const SavedSnapshot& saved_snapshot = snapshots[index];
if (!plan.snapshots[index] ||
    (saved_snapshot.anchor != kRootIndex &&
     !restored[static_cast<std::size_t>(saved_snapshot.anchor)])) {
    continue;
}
```

快照同理：anchor 没恢复，这个快照就是**指向空气的引用**。

**因为 `parent < index` 且 `anchor < node_count`，一遍正向遍历就能完成级联判断**——父一定在子之前被处理过，`restored[parent]` 一定已经有值。**这就是"表必须父先子后"的第二个理由**（第一个是 save 侧的 `file_index.at`）。

> **顺序不变量同时服务于三个目标**：save 侧能算出父的文件序号、load 侧能一遍正向完成级联、以及 `restored[]` 数组不需要递归或拓扑排序。**一个不变量三次收租。**

## 九、取消点：每个 slab 之前，加最后一个 slab 之后

```cpp
// hybrid_persist.cpp:216-232（save 的 slab 段）
for (const pc::NodeRef node : nodes) {
    check_abandoned();
    writer.bytes(slab(index_->node(node).host_slab), host_layout_.slab_bytes);
}
for (const pc::SnapshotRef snapshot : snapshots) {
    for (const std::uint32_t slab_id : index_->snapshot(snapshot).host_slabs) {
        check_abandoned();
        writer.bytes(slab(slab_id), host_layout_.slab_bytes);
    }
}
writer.bytes(kFooter.data(), kFooter.size());
writer.value(...); writer.value(...);
writer.finish();
out.bytes = writer.written();
// Abandoned after the last slab: the previous file still stays.
check_abandoned();
std::filesystem::rename(temporary, path);
```

**取消点密度 = 每个 slab 一次**（不是每字节）。粒度选择：slab 是文件的最小可寻址单位，取消的代价是"重写一个 slab"。

**最后一个 slab 之后还有一次 `check_abandoned()`，在 `rename` 之前。** 注释一句话说清：

> `// Abandoned after the last slab: the previous file still stays.`

**含义**：全部字节都写完了、只差改名——这时候取消仍然**保留旧文件**。因为没有 rename，`path` 上还是上一次成功保存的文件。

**这个设计的价值**：整个 save 过程中，`path` 指向的文件**始终是一份完整的、可加载的旧快照**。任何时刻被打断（进程杀了、盘满了、用户取消），损失只是这次的新内容，不是上一份好数据。

```cpp
// hybrid_persist.cpp:233-241
} catch (const std::exception& error) {
    // The writer is closed by now, so the temporary file can be deleted on every platform.
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    out.message = error.what();
    return out;
}
```

catch 里先关（`Writer` 是栈对象，析构已关）再删——**Windows 不能删一个还打开着的文件**（08 号报告的记录）。

## 十、load 的失败处置：`clear()` 全清

```cpp
// hybrid_persist.cpp:404-412
} catch (const std::exception& error) {
    if (phase) { phase->complete(reader.read()); }
    // Partially restored entries may hold unread slab bytes: drop them all.
    clear();
    out         = HybridPersistResult{};
    out.message = error.what();
    return out;
}
```

**为什么必须全清而不是"保留已成功的"**：

> `// Partially restored entries may hold unread slab bytes: drop them all.`

一个块被 `restore_host_block` 登记进 index 之后，它在 Host tier 里**占着 slab**。而 slab 的字节要到 `reader.bytes(slab(...))` 才填。如果这中间抛了，index 里就有一个"已登记但内容未读入"的块——**它的 hash 是对的（表里读来的），token 也是对的，于是后续的前缀命中会返回它，但它的 KV 字节是垃圾。**

**这类"元数据正确、数据未初始化"的状态是最危险的一种**：所有检查都通过，只有数值是错的。所以处置是**全清**——宁可丢掉这次加载的全部成果。

> 注意 `out = HybridPersistResult{}` 这行：**连 `saved_blocks` / `required_host_bytes` 这些"文件里有什么"的统计也清掉**。因为调用方可能拿 `required_host_bytes` 去决定"要不要加大 tier 再试一次"——一个失败的加载不该留下这种建议。

## 十一、可引用数字

| 项 | 值 | 位置 |
|---|---|---|
| magic / footer | `NINFHPC1` / `NINFEND1` | `hybrid_persist.cpp:32-33` |
| 版本 | 3 | `:34` |
| 块记录 | 284 B（4+8+8+256） | `:64-70` |
| footer 长度 | 16 B（8+4+4） | `:36` |
| 指纹长度上限 | 1 MiB | `:265` |
| 三段校验 | magic+version / fingerprint / geometry | `:258-280` |
| 尺寸恒等式 | `file_bytes == payload + slabs*slab_bytes + 16` | `:327` |
| 取消点密度 | 每 slab 一次 + rename 前一次 | `:219-231` |
| 枚举范围校验 | `kind_byte > Boundary` 即拒 | `:319` |

---

## 十二、六条可搬走的做法

1. **表前置、数据后置**：让"决定读哪几块"发生在任何 slab IO 之前，小 tier 只付它要的那部分代价；`read()` 不计 seek 越过部分，进度指标才诚实。
2. **写入侧的不变量，读取侧必须重验**：`parent < index` 在 save 靠 `at()` 保证，在 load 靠范围检查再验一遍——读取侧面对的不是自己的写操作。
3. **顺序不变量一次收三份租**：save 能算父的文件序号、load 能一遍正向做级联、`restored[]` 不需要拓扑排序。
4. **校验要分段且每段给不同的处置**：身份/指纹/几何三类故障的排查方向不同，合成一句"不兼容"等于抹掉信息。而且三段全过之后才对外声明"正在加载"。
5. **全程用临时文件 + rename，取消点涵盖最后一个字节之后**：`path` 永远指向一份完整的旧快照，任何时刻被打断只损失新内容。
6. **"元数据对、数据未初始化"必须全清**：块已登记进 index 但 slab 未读入时抛异常，最危险的残留状态就是它——hash 和 token 都对，命中会返回垃圾字节。处置是全清，连"文件里有什么"的统计也清掉。

> 与 08 号报告的关系：**08 讲这个格式"装什么"（块身份五元组、指纹 vs 几何的判据、GDSF 恢复计划），14 讲它"怎么读写"（分段、三段校验、级联跳过、临时文件与取消点）。** 格式设计的难点从来不在字段定义，而在这四件事：读多少、验什么、坏了怎么办、被打断怎么办。
