# 08 · prefix 层：块身份、Host tier 几何、落盘与恢复选择

> **底座口径（2026-11-17 补）**：本文读的是 `iamwavecut/ninfer-all` 分支线 @`6cc1f9fa`（2026-11-04）的源码树。该线**不是上游主干**——`Neroued/ninfer` master 与它已分叉（`compare` 实测 `diverged, ahead_by=523, behind_by=1895, merge_base=f76e19c0fbd0`），且 2026-11-04 之后本线与上游都改过本文描述的部分机制（已核到 4 处，见 `notes/2026-11-17-ninfer上游与分支血缘核验.md`）。**引用本文任何断言请连带 sha 与日期。**

> 承接 06（物理页）与 07（逻辑页 + suspend + 租约）。本篇是**最接近 KVMem 的那一层**：块怎么被认出、Host tier 怎么切、落盘怎么保证原子、装不下时选谁恢复。
> 源码：`kvmem-study/ninfer-src/src/models/qwen3_5/program/prefix/{block_keys,hybrid_host_layout,hybrid_cache,hybrid_persist}.{h,cpp}` + `src/runtime/prefix_cache/prefix_index.{h,cpp}`，commit `6cc1f9fa…`。

---

## 一、块身份是五元组，哈希只是索引，token 才是证据

```cpp
// hybrid_persist.cpp:64-70（落盘记录的块身份）
struct SavedBlock {
    std::int32_t parent = kRootIndex;   // 显式父指针，从不靠猜
    std::uint64_t hash  = 0;            // lookup_hash
    std::uint64_t extra = 0;            // vision/lora 累积维度
    std::array<TokenId, pc::kBlockTokens> tokens{};   // 64 个 token 原文
};
```

**一条记录 8+4+8+8+256 = 284字节，其中 256 字节是 token 原文**。这钱花得很有讲究：

- `hash` 只负责**找候选**（hash 表桶）；
- `extra` 负责**区分"token 相同但上下文不同"**（同一段文本前面挂不同图片，KV 不一样）；
- `tokens` 负责**最终判等**——非加密哈希会碰撞，碰撞了必须能当场否掉。

> 这就是 03 号报告里 KVMem「判定走内容 proof 链而非 prompt_cache_key」的**更严格版本**：KVMem 用 exact prefix + digest 比对；NInfer 直接把 64 个 token id 原文写进文件，代价是每块 256 字节，换来"跨重启也能逐字复核"。**拿存储换可验证性，是这一层最清晰的取舍。**

## 二、`extra` 是前缀累积量，不是块内属性

```cpp
// block_keys.cpp:83-100
std::uint64_t cumulative = seed;                  // lora seed 或 0
for (block = 0; block < blocks; ++block) {
    const std::uint64_t end = (block + 1) * kBlockTokens;
    while (next < ranges.size() && ranges[next].begin < end) {
        cumulative = accumulate_vision(cumulative, ranges[next].key);
        ++next;                                    // 只吸收"已开始"的 vision range
    }
    extras[block] = cumulative;
}
```

**一张图出现在第 3 块，第 3、4、5…块全部携带它的身份。** 这不是实现偷懒，是 RadixAttention 语义的必然：KV 是前缀的函数，一张图改变了它之后**所有** token 的 KV，所以身份必须向后污染。

对照：如果只把 vision key 记在"包含它的那一块"，那么"同一段文本 + 不同图片"的前缀会在图片块之后合并，复用到错误的 KV。

`extras` 只有在 `!vision_items.empty() || seed != 0` 时才分配——**没有额外维度的普通文本走零开销路径**（`extras` 为空 vector，`block_lookup_hashes` 直接用 token）。

## 三、种子是 ASCII 拼出来的模型名

```cpp
// block_keys.cpp:19-20, 33
constexpr std::uint64_t kVisionExtraSeed = 0x6e696e666572ULL;      // "ninfer"
constexpr std::uint64_t kLoraExtraSeed   = 0x6e696e6665722d6cULL;  // "ninfer-l"
std::uint64_t key = 0x6e696e6665722d76ULL;                          // "ninfer-v"
```

`6e 69 6e 66 65 72` = `n i n f e r`。**把哈希种子写成模型名前 6 字节**：既是合格的 48 位常量（足够打散），又能在十六进制 dump 里一眼看出"这是 ninfer 的 seed"。

> ⚠️ 实操坑：`cat`/`grep` 把 `0x6e696e6665722d76ULL` 渲染成过 `0x6e696e6665722d76ULL` 这种乱码。**遇到可疑字面量必须 `od -c` 复核**，别拿渲染结果当证据。

混合函数是 splitmix64 的双乘子变体（`block_keys.cpp:13-17`）：
`state ^= value + 0x9e3779b97f4a7c15 + (state<<6) + (state>>2); state *= 0xbf58476d1ce4e5b9; return state ^ (state>>31);`
逐字节喂 `content_digest`。**非加密，够快，靠 token 原文兜底碰撞。**

## 四、frontier 有结构性恒等式

```cpp
// runtime/prefix_cache/prefix_index.h:292-295
// frontier must equal 64 * (anchor depth + 1) + tail.size() (tail.size() for the root anchor).
```

**frontier 不是自由记账的数字，它由 anchor 深度和 tail 长度唯一决定**，`publish_snapshot` 会校验。

这一条的工程价值：任何"frontier 冲突"都变成结构错误，而不是需要人工核对的软状态。03 号报告里 KVMem 的 `accepted_through == frontier` 判定，在这里对应成**frontier 本身的构造恒等式**——更硬。

## 五、Host tier 是统一 slab 池：块和状态镜像抢同一批字节

```cpp
// hybrid_host_layout.h:12-24
struct HybridHostLayout {
    HostKVPageLayout text;                    // Main text 页布局
    std::optional<HostKVPageLayout> backend;  // backend 池存在时
    std::size_t backend_offset = 0;
    std::size_t slab_bytes = 0;
    std::size_t block_payload_bytes = 0;      // 一个块 bundle 实际传输的字节（不含 padding）
    std::size_t image_bytes = 0;
    std::uint32_t image_slabs = 0;
};
```

三条几何规则：

1. **slab 大小统一**，一个 slab = 一个 KV block bundle（text page + host layout + backend page）。
2. **一个 StateImage 拆到 `image_slabs` 个连续 slab，snapshot tail 再占一个**。
3. 于是注释里那句关键设计：**「blocks and state images compete for the same bytes under the index's eviction policy instead of a fixed split」**——不分固定配额，统一按驱逐策略争。

> 这跟我 04 号报告里计划的「B（预算）× ctx」扫描直接相关：**NInfer 的 Host tier 是一个单一字节池，不是一个"块区 + 状态区"**。所以扫 `--kvmem-host-mib` 时，块的命中与状态镜像的命中会**互相挤占**，曲线不会是单调的——这是我原计划里没预料到的交互项。

**预算的拒绝规则很狠**（`hybrid_host_layout.h:30-34`）：
- 零预算 = 关闭 Host tier；
- 非零预算必须**至少装得下一个完整 snapshot（image + tail）加一个 block**，否则直接拒。
- 理由一句话：「a smaller tier could never back a snapshot」——**装不下最小可恢复单元的 tier 是纯浪费，不如关掉**。

> 这是"配置校验应该拒绝无意义配置"的范例。我见过太多系统接受 `cache_size=100MB` 然后永远 miss。

## 六、三条归属契约与三条流纪律

`hybrid_cache.h:33-46` 类头注释，逐条抄：

1. **block id 拥有**「一个已提交满 Main 页上的一个非写者引用」（有 backend 池时还拥有对应 backend 页）；**snapshot tail 用同一套 id 空间**。
   → 「非写者引用」= `writer_references == 0` 是插入前缀缓存的前提。写者还活着的页不能被别人复用。
2. **snapshot 拥有**「一个不可变 StateImage」；它在 Device 上时 index 报 device slot，**它的 Host 副本住在 index 自有的 slab 里**。所以：
   - 丢掉 Device 副本 = 释放 StateImage（Host 副本还在）；
   - 恢复 = 把 slab 拷进**调用方预留的** Device slot（不是自己抢）。
3. **流纪律**：
   - Host 写跑在 **transfer stream**，排在生产者已入队工作之后；
   - Host 恢复跑在**专用 restore stream**，从而与 write-through 流量**和计算**重叠。

> 第 3 条是「分层召回不阻塞 decode」的全部实现根据。06 号报告里 `evictable_kv_pool` 的「lease/close 不 quiesce stream」是设备侧的一半，这里是 host 侧的一半。

## 七、18 个计数器就是验收面

```cpp
// hybrid_cache.h:56-74
admissions  snapshot_hits  reused_tokens  blocks_inserted
blocks_reattached  blocks_duplicate  taps_created  taps_skipped
endpoints_created  host_image_writes  host_block_writes
host_image_restores  host_block_restores(含 prefetched)  prefetched_blocks
host_tail_restores  host_write_bytes  host_restore_bytes  evicted_blocks
```

两个细节值得抄：

- **`blocks_reattached` 与 `blocks_duplicate` 分开计**：复用既有节点 vs 重复插入被拒，是两种完全不同的健康度信号。合成一个计数器就再也分不清"命中好"和"插入策略坏"。
- **`host_block_restores` 注释明确「prefetched blocks included」**，同时单列 `prefetched_blocks`——**总量与其中预取部分都可反推**，不需要事后猜。

> 这正是我 04 号报告 U 系列 UNKNOWN 里「怎么采指标」的答案：`context_cache.hybrid` snapshot 里读这些字段即可，不需要自己插桩。

## 八、落盘：表在前、slab 在后，是为了"少装也能选"

```cpp
// hybrid_persist.cpp:7-10
// Layout (version 3): the header, the block table (parents first), the snapshot table, then the
// slab bytes of every block and of every snapshot in table order, then the footer. The tables come
// first so a Host tier smaller than the file can choose what to restore before reading any slab.
```

**文件布局本身承载了一条策略**：元数据全部前置 → 恢复时可以先读表、算出 `plan_host_restore`、再 `seek` 到被选中的那几个 slab，**没被选中的 slab 一个字节都不读**。

`Reader` 的计数器注释印证了这一点（`:140`）：「Bytes read, **not counting what seeks passed over**」。所以 `out.bytes` 报的是真实 I/O 量，不是文件大小。

> 「设计文件格式时把选择所需的信息前置」——这是可流式/可部分加载格式的通用手法，比"整文件读进内存再筛"省一个数量级。

## 九、原子保存：临时文件 + rename，取消点选在 slab 之间

```cpp
// hybrid_persist.cpp:156-241 全路径
if (!pending_.empty() || restore_.open || !landing_.empty()) {
    throw std::logic_error("hybrid prefix cache save requires idle transfers");   // ① 保存不许和传输并发
}
std::filesystem::path temporary = path; temporary += ".tmp";
try {
    Writer writer(temporary);
    ... 写 magic/version/fingerprint/geometry/表 ...
    for (node : nodes)     { check_abandoned(); writer.bytes(slab(...), slab_bytes); }
    for (snapshot: snaps)  { for (slab_id : host_slabs) { check_abandoned(); ... } }
    writer.bytes(kFooter); writer.value(nodes.size()); writer.value(snapshots.size());
    writer.finish();
    check_abandoned();                       // ② 最后一个 slab 之后还可以取消
    std::filesystem::rename(temporary, path); // ③ 唯一提交点
} catch (const std::exception& error) {
    std::filesystem::remove(temporary, ignored);   // ④ 任何失败都删临时文件，旧文件保持原状
    out.message = error.what(); return out;
}
```

四道纪律：

1. **保存要求传输空闲**（`pending_` 空、`restore_` 未开、`landing_` 空）——落盘不是并发安全的传输操作，直接拒。
2. **取消点在 slab 之间**，粒度是"一个 slab"。注释 `:231`：「Abandoned after the last slab: the previous file still stays.」——**最后一个数据块写完再取消，也还是保留旧文件**，绝不产生"看起来完整但内容不全"的新文件。
3. **rename 是唯一提交点**（同目录 rename 原子）。
4. **catch 里先 remove 临时文件再返回**，且注释特意说明「The writer is closed by now, so the temporary file can be deleted on every platform」——**Windows 上打开的文件不能删**，所以先 `finish()`（关闭）再 catch 才删得掉。这是个真跨平台坑。

返回的是 `HybridPersistResult` 而不是抛异常（`:174` `kSaveAbandoned` 走 `out.message` 返回）：**"被用户取消""没 Host tier""指纹不匹配"都是正常结果，不是异常**；只有内部不变量破坏才 throw。

## 十、指纹 = 模型 + KV/state 格式 + 二进制 + 几何

```cpp
// hybrid_persist.cpp:1-5
// Nothing in it is trusted unless the fingerprint (model artifact, KV and state formats, binary)
// and the Host geometry match; a damaged or foreign file loads nothing.
```

指纹四要素 + 独立校验的 `Geometry`（`slab_bytes / image_bytes / block_payload_bytes / text_page_stride / backend_page_stride / backend_offset / image_slabs`）。

**几何为什么必须单独校验而不是塞进指纹**：指纹是"产物身份"，几何是"这次启动的配置"。同一个二进制、同一个模型，换了个 `--host-kv-mib` 就换了 `image_slabs`——**文件里的 slab 边界全错位**。合进指纹会导致"配置一改就得换文件"，分开校验则能给出精确的拒绝理由，且 `geometry ==` 用 `default` 比较（`hybrid_persist.cpp:54-56`）零手写成本。

> 对照 07 号报告：`SnapshotConfig` 把 RoPE 参数放进指纹（因为旋转每个 key），这里把几何放进单独结构（因为只影响布局）。**同一个文件格式里，两类"会影响字节解释的东西"用了不同层级的校验**——判断标准是"改它之后，旧字节是被误读（→指纹）还是只是位置错位（→几何）"。

## 十一、加载：级联跳过，任何错误全清

```cpp
// hybrid_persist.cpp:355-372
// Restored index ids of the file's blocks; absent when a block was not chosen or did not
// restore, so its descendants and the snapshots on it are skipped too.
if (!plan.blocks[index] || (saved_block.parent != kRootIndex && !restored[parent])) { continue; }
```

**级联规则**：父块没恢复 → 子块自动跳过；anchor 没恢复 → 挂在上面的 snapshot 自动跳过。因为前缀树的结构决定了「父缺则子无意义」。

表顺序由加载时强校验保证（`:309` `throw "prefix cache file block order is damaged"`，`:1457-1458` `if (parent >= index) throw "saved prefix cache blocks are not parents first"`）：**父的下标必须严格小于子**。这一条同时让"单趟构建树"成为可能。

```cpp
// hybrid_persist.cpp:409-415  任何异常 →
catch (const std::exception& error) {
    if (phase) { phase->complete(reader.read()); }
    // Partially restored entries may hold unread slab bytes: drop them all.
    clear();
    out = HybridPersistResult{};
    out.message = error.what();
    return out;
}
```

**部分恢复的条目里可能挂着"读了一半的 slab"**，所以全清。与 07 号报告 `resume` 的 catch、06 号报告 `lease()` 的中途失败契约是同一条：**要么全成要么全退**。

另：`load` 要求**缓存为空**（`:258` `throw "hybrid prefix cache load requires an empty cache"`）——不支持合并进活 tier。

## 十二、装不下时选谁：GDSF 密度的无时间基贪心

```cpp
// runtime/prefix_cache/prefix_index.h:510-521
// Chooses what an empty Host tier of `host_slabs` restores from a saved one: everything when it
// fits. Otherwise snapshots are taken greedily by the GDSF density of §9.3 without its
// time-dependent base — (1 + hits) times the prefill a restore saves from the root, per byte of
// image, tail and anchor path. Each takes its own slabs and the path blocks not yet chosen;
// one that does not fit is passed over for later ones that may. Blocks are chosen only on a
// chosen snapshot's path, so no restored block is dead.
```

密度公式（`prefix_index.cpp:1482-1489`）：

```
saved   = prefill_seconds(0, frontier) − restore_seconds(image_bytes)
size    = image_bytes + (有 tail ? block_bytes : 0) + path_blocks × block_bytes
density = (1 + hits) × max(0, saved) / max(size, 1)
```

排序：密度降序，**平局时 frontier 大的优先**（`std::stable_sort`，`:1492-1497`）。选择循环（`:1499-1516`）：

```cpp
std::uint64_t left = host_slabs;
for (const std::uint32_t index : order) {
    need = snapshot.slabs + (anchor 路径上尚未选中的块数);
    if (need > left) { continue; }        // 跳过，继续看后面的（不是 break）
    ...
    left -= need;
}
```

五个可直接复用的设计点：

1. **去掉时间基**：GDSF 原本按访问频率随时间的衰减定价，但**恢复是一次性决策**，没有"在缓存里待多久"这个维度。所以只留 `(1+hits)`。**照搬论文公式会引入一个没有意义的项。**
2. **`max(0, saved)`**：如果某个 snapshot 的恢复成本高于重算成本（小 frontier + 大传图），saved 为负——**钳零而不是让它去挤占别的项**。负价值条目不参与排序。
3. **`need` 含未选中的路径块**：选一个 snapshot 必须连带它整条 anchor 路径。不这么算就会"选了但恢复不了"。
4. **装不下就 `continue` 不 `break`**：后面的小条目可能装得下。**贪心的正确写法是跳过而不是中断。**
5. **只选被选 snapshot 路径上的块**：注释明说「no restored block is dead」——**不恢复任何"没人用"的块**。这保证了 Host tier 里没有纯占位的孤儿。

> 第 5 条跟我 04 号报告要采的 `evicted_blocks` 直接相关：如果恢复策略保证不用孤儿，那么"恢复后立刻驱逐"就不会发生，测出的 miss 都是真的容量不足。

## 十三、`collect_persistable` 的范围：可恢复闭包，不是全部缓存内容

```cpp
// runtime/prefix_cache/prefix_index.h:302-306
// What a saved Host tier keeps: every Host-resident snapshot whose anchor path is Host-resident
// throughout, and exactly those paths, parents before children. Other Host content is dead
// (no snapshot can resume through it) or unreachable after a restart.
```

**三个边界各有一条理由**：① 只有 Host 驻留的 snapshot 能存（Device-only 的没字节可写）；② anchor 路径必须**全程** Host 驻留（中间缺一块就断链）；③ 其余 Host 内容要么 dead 要么重启后不可达。

> 这是"持久化范围 = 可恢复闭包"的精确表述。大多数缓存落盘实现会把"所有缓存条目"写下去，然后加载时发现一半用不上。

## 十四、可直接引用的数字

| 项 | 值 | 来源 |
|---|---|---|
| 块大小 | 64 tokens（`pc::kBlockTokens`） | `block_keys.cpp:75` |
| 落盘块记录 | 284 字节，其中 256 = token 原文 | `hybrid_persist.cpp:64-70` |
| 文件 magic / footer / version | `NINFHPC1` / `NINFEND1` / 3 | `hybrid_persist.cpp:32-34` |
| 计数器数 | 18 | `hybrid_cache.h:56-74` |
| Host slab 预算下限 | 一个 image（`image_slabs`）+ 一个 tail + 一个 block | `hybrid_host_layout.h:30-34` |
| 指纹长度上限 | 1 MiB（读文件时拒绝超长） | `hybrid_persist.cpp:277` |
| 排序 | 密度降序，平局 frontier 降序 | `prefix_index.cpp:1492-1497` |
| 前缀缓存文件个数 | 每次 save 一个完整文件（整 tier 快照式） | `hybrid_persist.cpp:1-10` |

---

## 十五、对 KVMem 定位的最后一块拼图

把 06/07/08 合起来看，「ninfer 本身已有前缀缓存」这件事彻底清楚了：

| 能力 | NInfer 原生 | KVMem 加上去的 |
|---|---|---|
| Host KV 存储 | ✅ `host_replica` 槽位 + pinned slab 池 | 换成可驱逐 + 可选 SSD 的外部存储 |
| 前缀块身份 | ✅ hash + extra + 64 token 原文 | 不改（复用既有结构） |
| Host tier 驱逐 | ✅ GDSF 密度 + GDSF inflation | 不改 |
| 落盘 | ✅ `--prefix-cache-file`，进程内重启复用 | 不改这个，另做跨进程/跨会话的**内容检索召回** |
| 跨进程共享 | ❌（文件是独占快照） | ✅ 这是 KVMem 的真增量 |
| 按内容检索召回 | ❌（只有前缀命中） | ✅ 01 号报告的核心结论 |
| 全历史保留 | ❌（Host tier 有界，超了驱逐） | ✅ RetainAll |

**所以 KVMem 的定位可以精确成一句话**：NInfer 已经有一套完整的"有界前缀缓存 + 落盘"；KVMem 把有界换成无界（Host RAM 全留），把"前缀命中"换成"内容检索召回"，把"独占文件"换成"跨会话共享"。**它没有重写引擎，它换的是三个策略参数级别的决策——但那三个决策恰好是 arch 层面的。**

> 这也解释了为什么补丁主体是 +16651/−1250 这么大：要改的不是"有没有接口"，而是**每一个假设了"Host tier 有界"的路径**（驱逐、定价、几何、预算、统计、快照）。07 号报告里 `settle` 而不是 fail 的降级路径、08 号报告里 `plan_host_restore` 的贪心选择——在"无界"世界里全都不需要了。**改一个核心假设的代价，就是要把所有围绕它做的补偿机制都拆掉。**
