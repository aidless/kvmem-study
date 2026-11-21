# 13 · Disk spill：什么时候写、写哪些、什么时候停

> **底座口径（2026-11-17 补）**：本文读的是 `iamwavecut/ninfer-all` 分支线 @`6cc1f9fa`（2026-11-04）的源码树。该线**不是上游主干**——`Neroued/ninfer` master 与它已分叉（`compare` 实测 `diverged, ahead_by=523, behind_by=1895, merge_base=f76e19c0fbd0`），且 2026-11-04 之后本线与上游都改过本文描述的部分机制（已核到 4 处，见 `notes/2026-11-17-ninfer上游与分支血缘核验.md`）。**引用本文任何断言请连带 sha 与日期。**

> 09 号报告讲的是磁盘层的**存储结构**（内容寻址、4 KiB 槽、65/25/10 配额）。这篇讲**决策**：什么时候决定把 KV 写到盘上、写哪几段、遇到什么就停下。
> 源码：`src/models/qwen3_5/program/storage/disk_tier.cpp`（`plan_owner_spill :117-186`、`progress_owner_spill :192-320`、`spill_owner_to_disk :322-334`、`spill_released_owner :336-339`、`flush_disk_tier :341-352`、`disk_restorable_frontier :354-390`）+ `context.cpp` 的 `release_sequence_kv_strict`（`written_to_disk` 门控）。
> 一句话结论：**spill 不是"缓存满了往外倒"，它只有两个确定时点——释放前、关机时；执行方式是一份预编清单从新到旧逐批提交，遇到第一个"字节不可得"的页就整条停。**

---

## 一、只有两个（加一个）触发点

```cpp
// context.cpp（release_sequence_kv_strict 内）
if (!written_to_disk) { spill_released_owner(sequence); }
```

```cpp
// disk_tier.cpp:336-339
void ProgramImpl::spill_released_owner(const SequenceState& sequence) noexcept {
    if (!disk_kv) { return; }
    spill_owner_to_disk(sequence, std::chrono::steady_clock::now() + kReleaseSpillBudget);
}
```

```cpp
// disk_tier.cpp:341-352
void ProgramImpl::flush_disk_tier() noexcept {
    if (!disk_kv) { return; }
    // Shutdown writes the catalogued continuations still resident, so a restart can restore the
    // conversations that were live; pages already on disk are skipped.
    const auto deadline = std::chrono::steady_clock::now() + kShutdownSpillBudget;
    for (std::uint32_t index = 0; index < continuation_capacity; ++index) {
        if (continuation_slots[index].role != ContinuationSlotRole::Catalogued) { continue; }
        if (std::chrono::steady_clock::now() > deadline) { break; }
        spill_owner_to_disk(continuation_states[index], deadline);
    }
}
```

**三个时点，各自有明确的预算**：

| 时点 | 预算 | 范围 | 理由 |
|---|---|---|---|
| 释放一个 owner 之前 | `kReleaseSpillBudget = 20s` | 这一个 owner | 内存马上还回去，**这是最后一次能读它的机会** |
| 关机 | `kShutdownSpillBudget = 60s` | 所有 `Catalogued` continuation，共享一个死线 | 让重启能恢复"活着的会话" |
| （隐式）已写过盘 | — | 跳过 | `written_to_disk == true` 时不重复写 |

**注意关机那一个：所有 continuation 共享同一个 deadline，不是一个 60s。** `for` 循环里 `if (now() > deadline) break;`——**机器关机的时间预算是全局的，不是每项的**。如果写成每项 60s，200 个 continuation 最坏要 3.3 小时。

> 这是"超时预算"最常见的实现错误：**deadline 是绝对时刻还是每项时长**。写成每项时长，总时间就随条目数线性增长，而调用方（关机流程）以为自己有一个上界。

## 二、清单先编好：`plan_owner_spill` 的选点哲学

```cpp
// disk_tier.cpp:117-124
ProgramImpl::DiskOwnerSpill ProgramImpl::plan_owner_spill(const SequenceState& sequence) const {
    DiskOwnerSpill spill;
    if (!disk_kv || !sequence.kv) { return spill; }
    const auto digest_limit = static_cast<std::uint32_t>(sequence.prefix_digests.size());
    const std::uint32_t frontier =
        std::min({sequence.text_kv_valid, sequence.execution_frontier, digest_limit});
    if (frontier == 0) { return spill; }
```

**frontier 取三者最小**：`text_kv_valid`（KV 已提交到哪）、`execution_frontier`（算到哪）、`prefix_digests.size()`（摘要算到哪）。

第三个最容易被忽略：**摘要没算到的地方，磁盘上就定位不到**——内容寻址的前提是摘要存在。所以可写范围被摘要长度硬限。

```cpp
// disk_tier.cpp:125-137
const auto add_chain = [&](DiskKVKind kind, const KVAddressSpaceStore& addresses,
                           KVAddressSpaceHandle address, std::uint32_t chain_frontier) {
    const std::uint32_t pages =
        std::min(kv_pages_for_frontier(chain_frontier), addresses.mapped_pages(address));
    for (std::uint32_t page = 0; page < pages; ++page) {
        spill.items.push_back(DiskSpillItem{
            .kind = kind, .frontier = std::min((page + 1U) * kPageTokens, chain_frontier),
            .page = page});
    }
};
```

**每页一个 item，key 是 `min((page+1)*64, frontier)`**——即"这一页覆盖到的末尾"。这是 09 号报告「每 frontier 一个摘要」的直接后果：**页是按它的末端 key 的**。

### 四类选点

```cpp
// disk_tier.cpp:138-152
add_chain(DiskKVKind::MainKV, *text_kv_addresses, sequence.kv->text, frontier);
const std::uint32_t backend_frontier =
    std::min(backend_frontier_at(speculative_backend, frontier), backend_kv_valid(sequence));
const bool backend = sequence.kv->backend && backend_kv_addresses && backend_kv_pages;
if (backend && backend_frontier != 0) { add_chain(DiskKVKind::BackendKV, ...); }
if (sequence.endpoint_valid && sequence.execution_frontier == frontier &&
    state_store->valid(sequence.state.read)) {
    spill.items.push_back(DiskSpillItem{.kind = DiskKVKind::StateImage, .frontier = frontier,
                                        .state = sequence.state.read});
}
```

**① 主链全写**：从第 0 页到 frontier 覆盖的页，**一页不漏**。因为 09 号报告的恢复逻辑「停在第一个缺页处」——漏一页，后面全白写。

**② backend 链用 `backend_frontier_at(speculative_backend, frontier)`**：投机解码的 draft 前缀和主链不是同一个 frontier（draft 可能更短或更长），所以它的末页由 `backend_frontier_at` 换算。

**③ endpoint StateImage 的条件里有 `execution_frontier == frontier`**——这是个严格相等：

```cpp
sequence.endpoint_valid && sequence.execution_frontier == frontier
```

意思是**只有"算到的地方正好是摘要也覆盖到的地方"才存这个状态**。如果算得比摘要远（摘要没跟上），这个 StateImage 对应的 frontier 在磁盘上无法被寻址，存了也没法恢复。**条件苛刻是为了不存无法索引的东西。**

### ④ 检查点：rewrite + 三个最早的 anchor

```cpp
// disk_tier.cpp:154-166
std::vector<std::pair<std::uint32_t, StateImageHandle>> checkpoints;
if (sequence.rewrite_checkpoint.valid && sequence.rewrite_state &&
    state_store->valid(*sequence.rewrite_state) && sequence.rewrite_checkpoint.frontier != 0 &&
    sequence.rewrite_checkpoint.frontier < frontier) {
    checkpoints.emplace_back(sequence.rewrite_checkpoint.frontier, *sequence.rewrite_state);
}
std::vector<std::pair<std::uint32_t, StateImageHandle>> anchors;
for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
    if (anchor.frontier != 0 && anchor.frontier < frontier && state_store->valid(anchor.state)) {
        anchors.emplace_back(anchor.frontier, anchor.state);
    }
}
std::sort(anchors.begin(), anchors.end(),
          [](const auto& left, const auto& right) { return left.first < right.first; });
anchors.resize(std::min(anchors.size(), kSpilledAnchors));
checkpoints.insert(checkpoints.end(), anchors.begin(), anchors.end());
```

`kSpilledAnchors = 3`，**取排序后最早的三个**。`disk_tier.cpp:24` 的注释解释了为什么是"最早"而不是"最近"：

> `// a follow-up turn re-renders the conversation, so boundaries inside generated text rarely match and the early, stable boundaries are the ones a later prompt shares.`

**后续一轮会把整段对话重新 render 一遍，生成文本内部的边界很少能对上**——所以值得存的是**早期、稳定**的边界（用户 prompt 结束处、第一轮结束处），不是最近的。

> 这个判断可以推广：**当"未来会不会用得上"取决于"未来会怎么访问"，就选那些访问路径结构决定的边界，而不是时间上最近的。** LRU 在这里是错的启发式。

### ⑤ 页内边界要双写

```cpp
// disk_tier.cpp:167-183
for (const auto& [checkpoint, state] : checkpoints) {
    // The chain keys a page by its end, so a boundary inside a page needs that page again
    // under its own frontier.
    if (checkpoint % kPageTokens != 0) {
        spill.items.push_back(DiskSpillItem{.kind = DiskKVKind::MainKV,
                                            .frontier = checkpoint,
                                            .page = checkpoint / kPageTokens});
    }
    const std::uint32_t backend_checkpoint = backend_frontier_at(speculative_backend, checkpoint);
    if (backend && backend_checkpoint % kPageTokens != 0) {
        spill.items.push_back(DiskSpillItem{.kind = DiskKVKind::BackendKV,
                                            .frontier = backend_checkpoint,
                                            .page = backend_checkpoint / kPageTokens});
    }
    spill.items.push_back(
        DiskSpillItem{.kind = DiskKVKind::StateImage, .frontier = checkpoint, .state = state});
}
```

**一个页内的 checkpoint 会把那一页再写一遍，key 换成 checkpoint 自己。**

代价是写放大：同一页字节最多写 5 次（主链一次 + 4 个 checkpoint 各一次）。收益是**可恢复性**——checkpoint 落在 64 的中间时，磁盘上必须存在一条以它为末页的链，否则"按 checkpoint 恢复"找不到起点。

> **为可恢复性付的写放大，是这类系统里最正当的一类冗余。** 因为它换的不是性能，而是"这个功能到底存不存在"。

## 三、批处理：同族连续 ≤ 32 页，StateImage 独占一批

```cpp
// disk_tier.cpp:246-259（StateImage 分支）
if (first.kind == DiskKVKind::StateImage) {
    const std::size_t bytes = state_images->host_layout().image_bytes;
    ...
    if (const auto host = state_store->host_view(first.state)) {
        spill.batch_bytes.emplace_back(host->data, bytes);      // Host 驻留：零拷贝引用
    } else {
        HostStateImageView staging = disk_state_staging_pool->writable_view(*disk_state_staging);
        state_images->copy_to_host(state_store->physical_slot(first.state), staging, transfer_streams);
        synchronize(transfer_streams);                          // Device 驻留：先拷到 staging
        spill.batch_bytes.emplace_back(staging.data, bytes);
    }
    spill.batch_ids.push_back(first_id);
    ++spill.next;
    break;                                                      // ← 一批只有一个 StateImage
}
```

**KV 页可以攒批（≤ `kDiskStagingPages` = 32），StateImage 一批只有它自己。** 因为 StateImage 是 `image_bytes` 大小的连续 blob，塞进 32 页的批里会让 `try_submit` 的 bytes 语义混乱。

**Device 驻留的页要先 `copy_to_host` 到 staging，然后把指针回填**：

```cpp
// disk_tier.cpp:303-313
if (!device_pages.empty()) {
    pages.physical_pool().copy_to_host(
        device_pages, staging.subview(0, static_cast<std::uint32_t>(device_pages.size())),
        transfer_streams);
    synchronize(transfer_streams);
    for (std::size_t slot = 0; slot < device_items.size(); ++slot) {
        const auto view = staging.subview(static_cast<std::uint32_t>(slot), 1);
        spill.batch_bytes[device_items[slot]] = {view.data(), stride};
    }
}
```

**`device_items` 是"占位索引"**：扫描时对 device 页先 `spill.batch_bytes.emplace_back()`（空 `span`），记下它在 vector 里的下标；一次批量拷贝完成后回填真实指针。

**为什么要记下标而不是就地拷**：`copy_to_host` 支持批量（一次 DMA 拷 32 页），逐页拷会退化成 32 次小 DMA。而 `batch_bytes` 里需要的是**拷贝完成后**的 staging 地址——所以先占位、后回填。

> 这个"先占位后回填"的模式在零拷贝/异步场景里反复出现：**收集阶段只决定形状，填充阶段才需要地址。** 混在一起写就必须放弃批量。

## 四、三重停止条件，且"停止"有自己的语义

```cpp
// disk_tier.cpp:281-296（批内扫描）
const DiskKVIdentity id = disk_identity(sequence.prefix_digests, item.frontier);
if (spill.batch_ids.empty() || !disk_kv->contains(id, item.kind)) {
    if (item.page >= addresses.mapped_pages(address)) {
        spill.stopped = true;  break;                                  // ① 页未映射
    }
    const LogicalKVPageHandle logical = addresses.logical_page(address, item.page);
    if (pages.committed_columns(logical) < item.frontier - item.page * kPageTokens) {
        spill.stopped = true;  break;                                  // ② 列不足
    }
    if (pages.host_replica_current(logical)) {
        const HostKVPageReplica& replica = pages.host_replica(logical);
        const auto view = host_kv_extents->view(replica.extent).subview(replica.page_offset, 1);
        spill.batch_bytes.emplace_back(view.data(), stride);
    } else if (pages.device_resident(logical)) {
        device_items.push_back(spill.batch_bytes.size());
        device_pages.push_back(pages.physical(logical));
        spill.batch_bytes.emplace_back();
    } else {
        spill.stopped = true;  break;                                  // ③ 字节不可得
    }
    spill.batch_ids.push_back(id);
} else {
    ++spill.deduplicated;                                               // ④ 已在盘上
}
```

四个分支，三个停止：

| 条件 | 含义 |
|---|---|
| `page >= mapped_pages` | 地址空间没映射到这一页 |
| `committed_columns < frontier − page*64` | 这一页还没填到这个 frontier（07 号报告的 `committed_columns`） |
| 既非 host replica current 也非 device resident | 字节不在任何地方可读 |

**为什么"停止"而不是"跳过"**：

```cpp
// disk_tier.cpp:225-228（批处理注释）
// Items already on disk are skipped; a chain whose bytes are unavailable ends the spill,
// since nothing after the gap could be restored.
```

**一个洞之后的全部内容都不可恢复。** 09 号报告说恢复「停在第一个缺页处」——那么在一个洞之后再写 100 页，是 100 页永远不会被读到的垃圾。所以**遇到洞就整条停**。

> 这是"链式结构 + 单向恢复"的必然结论：**写入端必须和恢复端的读法对齐，否则后半段是纯浪费。** 一个按前缀链恢复的系统里，"部分成功"没有意义——只有"从根到一个连续前缀"有意义。

### `stopped` 的第二个语义：不再排队，但已提交的必须落地

```cpp
// disk_tier.cpp:193-198
if (spill.stopped) {
    // A stopped spill queues nothing more; only what the bridge already holds must land.
    spill.batch_ids.resize(spill.batch_submitted);
    spill.batch_bytes.resize(spill.batch_submitted);
}
```

**停止不抛弃在途的批次**：把待提交列表截断到"已提交数"，然后照常等它们落地。因为 `try_submit` 已经把 staging 的字节**借出去**了（09 号报告：`try_submit` 借出的字节在 ticket 离开 Pending 前不能变）——如果这时直接返回，调用方会释放 staging，在途的 DMA 就读到已回收的内存。

**这是"取消"最容易错的地方：取消的任务不是"不用管了"，是"必须等它把手里的缓冲还回来"。**

## 五、去重在提交前查

```cpp
// disk_tier.cpp:237-242
if (disk_kv->contains(first_id, first.kind)) {
    ++deduplicated;
    ++next;
    continue;
}
```

**`contains` 查的是 09 号报告那个 128 位摘要**，所以在**磁盘已经有一模一样的字节**时直接跳过——跨会话也跳（这是 09 号报告"同一内容在两个会话去重成一页"的写入侧实现）。

**查在提交前，不是批内查**：批一旦组成（`batch_bytes` 是多页的指针数组），中途去掉一页就要重排 staging。

## 六、失败与超时的收尾

```cpp
// disk_tier.cpp:322-334
void ProgramImpl::spill_owner_to_disk(const SequenceState& sequence,
                                      std::chrono::steady_clock::time_point deadline) noexcept {
    if (!disk_kv || !sequence.kv) { return; }
    try {
        DiskOwnerSpill spill = plan_owner_spill(sequence);
        while (!progress_owner_spill(sequence, spill)) {
            if (std::chrono::steady_clock::now() > deadline) { spill.stopped = true; }
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    } catch (...) {
        // The owner's memory is released after this returns; nothing may still read it.
        disk_kv->wait_idle();
    }
}
```

**超时只是"停止"，不是"异常"**——`spill.stopped = true` 后循环会继续跑到"已提交的都落地"。**因为超时不等于可以走人：在途 DMA 还拿着 staging 的地址。**

`catch (...)` 里 `disk_kv->wait_idle()` 那行注释是全部的要点：

> `// The owner's memory is released after this returns; nothing may still read it.`

**这个函数是 `noexcept`，返回后调用方立刻释放 owner 的内存。** 所以任何异常路径都必须先确保磁盘层不再持有指向该内存的指针。

> 这就是 07 号报告「fail-fast vs fail-soft 分界」在 spill 上的实例：**能抛的地方抛（`plan_owner_spill` 里 vector 分配可能 `bad_alloc`），但 `noexcept` 边界上必须把在途状态收干。**

## 七、反向查询：`disk_restorable_frontier`

```cpp
// disk_tier.cpp:354-390
std::optional<std::uint32_t>
ProgramImpl::disk_restorable_frontier(const RequestBasePlanImpl& base,
                                      const PreparedPromptData& prompt,
                                      std::uint32_t prompt_tokens) const {
    if (!disk_kv || !disk_kv_restore || prompt.has_media() || prompt_tokens < 2 ||
        base.prefix_digests.size() < prompt_tokens) {
        return std::nullopt;
    }
    std::vector<std::uint32_t> frontiers = disk_kv->live_state_frontiers();
    std::sort(frontiers.begin(), frontiers.end(), std::greater<>());
    ...
    const auto chain_present = [&](DiskKVKind kind, std::uint32_t frontier) {
        for (std::uint32_t end = kPageTokens; end <= frontier; end += kPageTokens) {
            if (!disk_kv->contains(disk_identity(base.prefix_digests, end), kind)) { return false; }
        }
        return frontier % kPageTokens == 0 ||
               disk_kv->contains(disk_identity(base.prefix_digests, frontier), kind);
    };
    for (const std::uint32_t frontier : frontiers) {
        if (frontier == 0 || frontier >= prompt_tokens) { continue; }
        if (!disk_kv->contains(disk_identity(base.prefix_digests, frontier), DiskKVKind::StateImage) ||
            !chain_present(DiskKVKind::MainKV, frontier)) { continue; }
        const std::uint32_t backend_frontier = backend_frontier_at(speculative_backend, frontier);
        if (backend && backend_frontier != 0 && !chain_present(DiskKVKind::BackendKV, backend_frontier)) {
            continue;
        }
        return frontier;
    }
    return std::nullopt;
}
```

这是写入侧的对偶：**不问"该写哪些"，问"盘上哪条链是完整的"**。

四个判据：① 有 `StateImage` ② 主链每页都在（含页内边界那一次）③ backend 链每页都在 ④ `frontier < prompt_tokens`（只恢复到 prompt 之内，超出的是生成文本，不会命中）。

`frontiers` **降序**遍历，返回第一个满足的——**优先用最长的那段**。`frontier % kPageTokens == 0 ||` 那个短路正是写入侧页内双写的对偶：**整页边界不需要额外那一条，页内边界需要。**

还有两个前置否决：`prompt.has_media()`（图像内容寻址不稳定，08 号报告说视觉 prompt 靠 token 判等不成立）、`prompt_tokens < 2`。

## 八、可引用数字

| 项 | 值 | 位置 |
|---|---|---|
| 释放时 spill 预算 | 20 s | `disk_tier.cpp:28` |
| 关机 spill 总预算 | 60 s（**全局共享，非每项**） | `disk_tier.cpp:30` |
| 存几个 anchor | 3，且取**最早** | `disk_tier.cpp:24,165` |
| 批深 | `kDiskStagingPages = 32` 页 | `disk_tier.cpp:290` |
| 轮询间隔 | 200 µs | `disk_tier.cpp:330` |
| 停止条件数 | 3（未映射 / 列不足 / 字节不可得） | `disk_tier.cpp:282-296` |
| 页内边界写放大 | 最多 1 + 4 = 5 次/页 | `disk_tier.cpp:167-183` |

---

## 九、六条可搬走的做法

1. **超时预算是绝对 deadline 不是每项时长**：关机流程里 N 个条目共享一个 deadline，否则总时长随条目数线性增长。
2. **写端必须和读端的恢复语义对齐**：读端"停在第一个缺页"，写端就必须"遇洞即停"——部分成功在链式结构里没有意义。
3. **取消不是"不用管了"，是"等它把缓冲还回来"**：`stopped` 后仍要等在途批次落地，否则 DMA 读到已回收内存。
4. **选检查点用"访问路径结构"而非"时间最近"**：后续一轮会重新 render 对话，生成文本内部的边界很少对上，值得存的是早期稳定边界（用户 prompt 结束处）。LRU 在这种场景是错的启发式。
5. **为可恢复性付写放大是正当的**：页内 checkpoint 把该页按自身 frontier 再写一遍（最多 5×），换的是"这个 checkpoint 能不能被恢复"。
6. **收集阶段只决定形状，填充阶段才要地址**：device 页先占位后回填，才能一次批量 DMA 拷 32 页；混在一起写就必须退化成逐页拷贝。

> 与 09 号报告合起来看：**09 讲"盘上怎么放"（内容寻址、自描述槽、固定配额），13 讲"什么时机放、放到哪停"（两个时点、遇洞即停、与恢复语义对齐）。** 存储格式可以很通用，但写入策略必须和恢复策略成对设计——这是两个文件合起来才讲得清的事。
