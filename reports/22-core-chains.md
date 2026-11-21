# 22 · 三条主链的内部机制：句柄、会计、故障

> 底座：`kvmem-study/ninfer-src` 快照（`6cc1f9fa`），Qwen3.5 Program 实现。
> 范围：三份此前只读了半份的文件 —— `storage/context.cpp`（1889 行）、`prefix/hybrid_cache.cpp`（785 行）、`storage/state_store.h`（867 行）。
> 方法：逐行读完三份，只记 01–21 号报告**没有**展开的机制。已确立的事实（页大小 64、reserve→materialize 两级、StateImage 要么完整要么不存在、准入是报价）不重复；11 号的 fork 三段式与 `StateReadOwnership`、08 号的块身份与 Host slab 几何、12 号的租约阶梯也不复述，本文只补它们在**这三个文件内部**的接缝。

## 一、三份文件的分工，以及"谁能验证谁"

三条链各有自己的可验证对象，而且验证主体不是同一个东西：

- `context.cpp` 里每个公开方法的第一件事几乎都不是干活，而是**验句柄**。它同时管两套句柄体系：lane 上的 Sequence 句柄（`context.cpp:18-31`）和 continuation slot 句柄（`context.cpp:33-39`），还有 shared prefix（`context.cpp:41-47`）、capture offer（`context.cpp:49-63`）、pending transaction（`context.cpp:84-100`）。这五套验证的输入都不只是"id 在不在表里"。
- `state_store.h` 的句柄是另一个更简单的体系：`(owner 指针, index, generation)` 三元组，`StateImageHandle` 只存这三个私有字段（`state_store.h:30-32`），`valid()` 要求 `owner == this` 且 `index` 在界内且 `role != Free` 且 `generation` 匹配（`state_store.h:193-197`）。
- `hybrid_cache.cpp` 没有句柄，它管的是**可重入的 CUDA 对象所有权**：block id（`hybrid_cache.cpp:117-139`）、snapshot→StateImage 绑定（`hybrid_cache.cpp:249-262`）、Host slab 分片（`hybrid_cache.cpp:266-269`）、pending write / landing 批次（`hybrid_cache.cpp:334-482`）。

一句话：`context` 验的是"这条 lane 在这个 epoch 上还有效吗"，`state_store` 验的是"这个 StateImage 还是我给出的那个实例吗"，`hybrid_cache` 验的是"这批拷贝的 pin 还挂得稳吗"。三层的**故障模式因此完全不同**——前两层用句柄 + epoch 让陈旧访问安全失败，第三层用流序 + 事件让并发安全成立。

## 二、句柄为什么要 owner + index + generation 三件套

`state_store.h` 的设计把这个答案写在类型定义里：一个 `StateImageHandle` 只有 `owner_`、`index_`、`generation_` 三个字段（`state_store.h:30-32`），构造是私有的，只有 store 自己能造（`state_store.h:34`）。

关键在于 `generation_` 在**每次释放时前进，且跳过 0**：

```cpp
object.role          = StateImageRole::Free;
object.content_epoch = 0;
if (++object.generation == 0) { ++object.generation; }
free_objects_[free_object_count_++] = handle.index_;
```
（`state_store.h:702-705`）

`allocate` 时对象被重置，但**保留**旧 generation 作为新句柄的世代（`state_store.h:760` 的 `.generation = object.generation`）。于是同一个 `index` 槽位被复用后，旧句柄的 `generation_` 对新句柄的 `generation_` 差 1，`valid()` 返回 false（`state_store.h:196`）。跳过 0 是因为 generation 从 1 起步（`state_store.h:734` 的 `= 1`），保持"非零=已发过"的不变量。

`context.cpp` 的 lane epoch 是同一套思路，但更省事——它只存 epoch 数组，`invalidate_lane` 只是 `++lane_epochs[lane]` 后跳过 0（`context.cpp:102-106`）。slot 的 generation 同理，`retire_continuation_slot` 里 `++slot.generation` 后跳过 0（`context.cpp:266` 附近）。

这两处"跳过 0"不是防御性代码，是**句柄可复用的前提**：0 被保留为"从未分配/已失效"的哨兵，任何 `generation == 0` 的句柄一律无效。

## 三、引用会计不是"数个数"，是"剩余量协议"

这是本文最想强调的跨模块契约。`state_store.h` 里 `checkpoint_references` 是一个 uint32 计数器（`state_store.h:741`），但 context 侧**从不直接读它来做决定**，它做的是"我如果释放掉属于我的那部分，剩下的还能不能安全释放"。

`context.cpp:155-165` 的 `validate_state` 是第一个实例：

```cpp
const std::uint32_t owned = owned_checkpoint_references(sequence, handle);
const std::uint32_t total = state_store->checkpoint_references(handle);
if (owned > total ||
    (owned != 0 && state_store->role(handle) != StateImageRole::CheckpointImmutable)) {
    return false;
}
return !release_object || total != owned ||
       state_store->can_release_after_checkpoint_references(handle, owned);
```
（`context.cpp:158-165`）

三层含义叠在一起：`owned > total` 是"我宣称拥有的比实际多"——逻辑错误；`owned != 0` 时该对象必须是 immutable——否则这条序列根本不该持有 checkpoint 引用；`total == owned` 时，释放我这份之后引用归零，必须走 `can_release_after_checkpoint_references` 确认对象本身可回收。

`can_release_after_checkpoint_references` 自己的判据是（`state_store.h:716-730`）：若 `released_references > checkpoint_references` 直接 false；若两者**不相等**直接 true——注意这里，**剩下的引用还大于零，所以显然还不能回收**；只有相等（即释放后归零）时才继续检查 `source_pins == 0 && !destination_pinned && !has_pending_replica`。

这个"剩余量协议"在 fork abort 路径上还有一个更精细的孪生兄弟：`can_release_source_after_fork_abort`（`state_store.h:457-474`）和 `can_release_destination_after_fork_abort`（`state_store.h:476-490`）。注释点明了它的意图：

> abort_fork releases exactly the pin represented by this binding. Any other source pin still protects the object and therefore prevents terminal owner settlement.
> （`state_store.h:464-465`）

也就是说：**"abort 掉这个 fork"只归还它自己那一个 pin，不保证对象能回收**——`source_pins != 1` 时直接 false（`state_store.h:466`）。这避免了"一个 fork 失败就误删了另一个 fork 还在用的源"。

结论：整套引用会计的设计对象是**归属**（这个引用是谁的）而不是**总量**（一共有几个）。总量只用于最后一个判据：归零之后能不能动。

## 四、三档释放梯度：strict / best_effort / skip

`context.cpp` 里几乎所有"释放"操作都有三档，这不是重复，是**故障契约的分层**：

| 档 | 判据失败时 | 用于 |
|---|---|---|
| `strict` | 抛异常或 `std::terminate` | 正常路径，错误必须暴露 |
| `best_effort` | 全 `catch` 吞掉 | 清理路径，不能因失败阻断流程 |
| `skip` | 直接 `return` | 前置条件不满足，整体不执行 |

实例：

- **state 释放三兄弟**：`release_active_sequence_state_strict`（`context.cpp:1269-1329`，任何异常 terminate）、`release_sequence_state_strict`（`context.cpp:1331-1399`，`fork_pending` 直接 terminate）、`release_sequence_state`（`context.cpp:1401-1465`，best-effort 全 catch，判据只留 `releasable = valid(handle)`）。
- **shared references 释放**：`release_active_shared_references_strict`（`context.cpp:1126-1136`，`refs == 0` 且 role 非 live 即 terminate）vs 非 strict 版（`context.cpp:1467-1477`，直接跳过）。
- **lane 清理**：`clear_lane_strict`（`context.cpp:1138-1153`）失败即 throw；`clear_execution_failure_lanes`（`context.cpp:1155-1166`）有事务就整体跳过。

最典型的是 `clear_execution_failure_lanes` 那句注释：

> A concurrent resource transaction may pin or inspect these active owners. Engine-wide cleanup aborts that transaction before releasing lanes, preserving the only safe order.
> （`context.cpp:1155-1158` 附近）

代码上就一行 `if (has_context_transaction()) return;`——**有事务在跑，就一个字节都不动**。这不是偷懒，是把"清理"的时机权交给"事务"，因为事务可能正持有这些 lane 的 pin；先清 lane 再 abort 事务会让事务解引用已释放对象。

`release_active_shared_references_strict` 与它的 best-effort 兄弟的差别也值得注意：strict 版在 `refs == 0` 且 `role` 不是 live 时直接 terminate（`context.cpp:1126-1136`），best-effort 版只 `continue`。理由是 strict 版跑在"序列主动清除"的路径上，此时 refs 归零而 role 已非 live 意味着**共享引用的生命周期已经结束**，状态不一致必须炸；best-effort 版跑在批量收尾，允许残留。

## 五、压力（pressure）的两个护栏

`context.cpp` 里有两处"制造压力"的代码，都带着明确的反直觉护栏。

**护栏一：压力只作用于候选的完整峰值，不作用于已 clamp 的缺口。** `materialization_deficit`（`context.cpp:553-560`）先算 `physical_occupancy() + admission.demand.physical_peak_additional` 再与容量相减，注释说：

> Pressure is relative to this candidate's real peak. Treating every dimension as scarce would forbid Device-to-Host demotion even when Host capacity is available.
> （`context.cpp:556-558` 附近）

`guided_materialization_deficit`（`context.cpp:562-574`）更进一步，注释：

> Pressure acts on the candidate's complete peak, not on its already-clamped deficit. Applying a demotion directly to a zero Host deficit would otherwise manufacture Host pressure even when the arena has ample slack and steer the heuristic toward unnecessary destruction.
> （`context.cpp:565-568` 附近）

代码上它是先把 `pressure.added` 加进 `physical_peak_additional`、减去 `pressure.removed`（`context.cpp:571-573`），再算 `required - capacity`。如果反过来"直接在零缺口上施加 demotion"，零缺口 + demotion = 负数，会被 `positive_resource_difference` 钳成 0 —— 于是 demotion 看起来"没起作用"，但**副作用是它确实释放了 Device 页**，净效果是"无谓地破坏了缓存"。护栏把它改成"先投影到完整峰值再钳"，让 demotion 只有在真的存在 host 压力时才被采纳。

**护栏二：`physical_peak_fits` 用"减法"防加法溢出。** `context.cpp:576-595` 判断候选峰值是否装得下时，不是 `used + added <= capacity`，而是 `added <= capacity && used <= capacity - added`。这个形状在 `owner_exclusive_resources` 的 KV 字节累加里也一样（`context.cpp:375-384`）：

```cpp
if (entitlement < mapped ||
    entitlement - mapped >
        std::numeric_limits<std::uint32_t>::max() - device_pages) {
    throw std::logic_error("owner active KV entitlement is inconsistent");
}
device_pages += entitlement - mapped;
```

两个 uint32 相加前先确认右边不溢出。这不是过度设计——`entitlement - mapped` 是 active lane 的"已预留未映射"量，理论上受 `capacity` 约束，但一旦 entitlement 记账被污染（比如 double-free），这个减法会下溢成巨大正数，直接 `+=` 会静默绕过所有容量检查。

## 六、hybrid lane 不持有"独占基线"

`active_exclusive_baseline`（`context.cpp:496-513`）里有一行非常关键的分支：

```cpp
// A Hybrid lane's entitlement is its admission quote, not exclusive page ownership.
if (hybrid_prefix_cache()) { return out; }
```
（`context.cpp:498-499` 附近）

`out` 是默认构造的空 `ActiveExclusiveBaseline`。意思是：**一旦启用了 hybrid prefix cache，hybrid 模式的 lane 就不参与"独占页归属"的基线记账**。

为什么？因为 hybrid 模式下，一块物理页的归属由 `hybrid_prefix_cache` 的块引用计数管理（`hybrid_cache.cpp:141-173` 的 retain/release），不由 active lane 的独占页记账管理。如果这里还去数 exclusive 页，就会**双记账**——同一块页既算 hybrid 块的引用，又算 active lane 的独占资源。

`credit_active_ownership_transfers`（`context.cpp:515-530`）是配套：它把"释放 active owner 前后 exclusive 资源的**正差**"计入 `request.active_resources`。注意它 catch 全部异常——因为这是**账务**，失败不能阻断 release 主流程。正差意味着"这个 owner 释放后，整体占用下降了"，这份下降量要记到 request 账上，供后续准入报价使用。

## 七、共享页只算一次，Host 字节要过 epoch 守卫

`owner_exclusive_resources(SequenceState)` 的 `add_kv` lambda（`context.cpp:343-396`）里，对每块逻辑页先问 `address_references`，`> 1` 就跳过：

```cpp
// A shared logical page contributes to aggregate occupancy once. Releasing this
// address cannot free either replica while another address still references it,
// so it is not part of this owner's exact transition effect.
```
（`context.cpp:357-360` 附近）

注释把理由说全了：共享逻辑页对**总占用**只贡献一次；释放这个地址不会释放任何副本，只要还有另一个地址引用它。所以它不属于"这个 owner 的精确转移效果"——而 release 的记账算的正是转移效果。

`host_kv_prefix_bytes`（`context.cpp:771-805`）里的 epoch 守卫是另一条跨层契约：

```cpp
// A destructive private rewrite changes this tail page's content epoch, so its old
// Host replica cannot remain part of active entitlement.
```
（`context.cpp:793-795` 附近）

代码上：如果 `selected_columns != committed_columns(logical)` 就跳过这块页的字节统计。含义是——尾页如果被 private rewrite 改写过（content epoch 前进），旧的 Host 副本已经**不再代表当前 entitlement**，不能算进 active 的字节账。这是 07 号报告"epoch + committed_columns"在**准入字节会计**里的落点：epoch 不只是安全机制，它还决定"这一页的 Host 字节还归不归我"。

## 八、backend_frontier 公式，与 graft 的零 digest 隐身

`checkpoint_summary`（`context.cpp:807-850`）里有一个随后端而变的 frontier：

```cpp
const std::uint32_t backend_frontier =
    backend_ == KvBackend::Mtp   ? frontier - 1
    : backend_ == KvBackend::DFlash ? frontier
    : 0;
```
（`context.cpp:826-829` 附近）

三个后端三种算法：MTP 减 1、DFlash 用 frontier、其余用 0。这是投机后端与 checkpoint 索引的跨模块契约——MTP 的最后一个 draft token 不产生 KV，所以它的有效 frontier 比逻辑 frontier 少 1；DFlash 没有这个偏移。

`shared_prefix_summary`（`context.cpp:911-954`）里有一段更巧妙的：

```cpp
// Grafts have identity==nullptr; use a zero-digest shortlist key with correct frontier
// so the summary passes validation. The digests being zero keeps it invisible to the
// prefix-index matching loop.
```
（`context.cpp:926-928` 附近）

"graft" 是从 host 恢复出来、没有原始 identity 的 shared prefix。它需要出现在 continuation summary 里（好让校验通过），但又**不能让 prefix-index 匹配循环看见它**（否则会被当成可复用的前缀去匹配别的序列）。解法：给它一个全零 digest 的 shortlist key——校验路径认 frontier 就过，匹配循环按 digest 查就永远查不到。**用零值做隐身**，这是个漂亮的取舍：不引入新的"是否 graft"标志位，只靠 digest 恰好为零。

## 九、执行失败清理与事务的互斥（见第四节）

这一节和第四节是同一件事的两面，单独拎出来强调契约方向：`clear_execution_failure_lanes`（`context.cpp:1155-1166`）在 `has_context_transaction()` 时**整体跳过**，而不是"只清没有事务 pin 的那些 lane"。理由注释写得很清楚（`context.cpp:1156-1159` 附近）：并发事务可能正 pin 着或检查这些 active owner，引擎级清理必须先 abort 那个事务再释放 lane，"保留唯一安全的顺序"。

这句话定义了**两个清理路径的优先级**：事务的 abort 由事务自己负责，lane 的清理让位。如果反过来先清 lane，事务 abort 时会解引用已 `retire` 的 lane 句柄，拿到的是新 generation 的空对象——虽然句柄安全，但事务的状态机已经错乱（它以为 lane 还在，实际已经换了世代）。

代码上就一句 `if (has_context_transaction()) { return; }`（`context.cpp:1158`），前面两行是注释（`context.cpp:1156-1157`）。函数体里对每个 lane 做的是 `clear_lane_best_effort` + `invalidate_lane`（`context.cpp:1161-1164`），并跳过越界 lane 与 continuation 超容量的 lane（`context.cpp:1159-1160`）。

## 十、hybrid_cache 的 pin 生命周期：reserve → retain → adopt → release

`hybrid_cache.cpp` 的 pin 不是"引用计数加减"那么简单，它是一条四段流水线，每段都带失败回滚。

**第一段：retain（准入时 pin）。** `retain`（`hybrid_cache.cpp:141-165`）先检查 `can_retain_reference && device_resident && committed_columns >= columns`，然后做一件反直觉的事：

```cpp
// A committed full page (or a finishing sequence's tail) receives no further writes.
// Clearing the writer flag lets later readers pin it as an immutable source.
if (store.writer_references(page) != 0) { store.set_writer(page, false); }
```
（`hybrid_cache.cpp:157-159`）

**清 writer 标志**。一个页只要 committed 且完整，就不再接受写；把 writer 清掉，后来的读者就能把它当**不可变源**来 pin。这是"页被多个读者共享"的前提——如果 writer 还在，读者 pin 的时候得担心写者会不会改它。

**第二段：insert_block 的 adopts 判据。** `hybrid_cache.cpp:192-195`：

```cpp
// Only a new node or a host-only node adopts the caller's page; a Device-resident duplicate
// leaves it private to the caller.
const bool adopts = !existing || index_->node(*existing).device == CopyState::Absent;
```

如果已经有一个 Device-resident 的重复节点，调用者给的页**不被采纳**，保持私有（调用者自己处理）。只有新节点或 host-only 节点才"接管"调用者的页。这个判据决定了 `blocks_` 表里登记的是谁的页——登记错了会导致后续 release 时引用计数错乱。

**第三段：restore 的 pin 归属转移。** `land_restore`（`hybrid_cache.cpp:618-633`）里的注释：

```cpp
// The admission's own pins end at activation; the copies' sources and destinations stay
// pinned by the batch until it lands.
for (const NodeRef node : restore_.nodes) { index_->pin_node(node); }
```
（`hybrid_cache.cpp:621-623` 附近）

准入阶段的 pin 在 activation 就结束了，但**拷贝的源和目标由批次自己 pin 住，直到它落地**。这是两套 pin 的交接：准入的 pin 管"能不能恢复"，批次的 pin 管"拷贝在飞行中别被换掉"。

**第四段：落地与释放。** `finish_landing`（`hybrid_cache.cpp:680-688`）对每个 node `complete_device_fill` + `unpin_node`；`release_device_block`（`hybrid_cache.cpp:764-769`）在驱逐时 `release` + `free_block_id` + `++counters_.evicted_blocks`。

## 十一、hybrid_cache 的事件与批次的 FIFO 语义

`poll`（`hybrid_cache.cpp:452-469`）和 `drain`（`hybrid_cache.cpp:471-482`）是两套完成检测，注释点明了为什么可以这么写：

```cpp
// Every write shares the transfer stream and every restore the restore stream, so each list
// completes in submission order.
```
（`hybrid_cache.cpp:453-454` 附近）

所有 write 共用 transfer stream，所有 restore 共用 restore stream——**每个链表按提交顺序完成**。所以 `poll` 只需查队首事件是否 ready，ready 就 `publish_write` 弹一个；`drain` 直接 `cudaEventSynchronize` 队首。这是 07 号"单流 FIFO 换掉依赖图"在 hybrid_cache 里的具体实现：不维护事件依赖图，靠"同一流上顺序即依赖"。

`settle_prefetch`（`hybrid_cache.cpp:651-661`）有一个更快的收尾：

```cpp
// One restore stream completes batches in submission order: the last prefetch batch's final
// event covers every batch submitted before it.
```
（`hybrid_cache.cpp:652-653` 附近）

要找"所有 prefetch 批次都完成了"，不需要逐个等——**等最后一个 prefetch 批次的 final event** 就够了，因为它在单流上，前面的批次都排在它前面。

`recycle_events`（`hybrid_cache.cpp:690-701`）的 OOM 降级也值得记：

```cpp
} catch (...) {
    // Out of memory while recycling: release the events instead.
```
（`hybrid_cache.cpp:694-695` 附近）

事件回收（push 回 `spare_events_`）时如果 OOM，就**降级为直接 `cudaEventDestroy`**——宁可丢事件池也不能让恢复流程因为回收失败而炸。

`order_after`（`hybrid_cache.cpp:321-332`）的注释是事件池可重用的依据：

```cpp
// The wait captured the recorded work; the event may be re-recorded.
spare_events_.push_back(ready);
```
（`hybrid_cache.cpp:330-331`）

`cudaStreamWaitEvent` 只是**捕获**了 producer 当时记录的工作，event 本身之后可以被 re-record。所以 event 用完直接还池，不需要"作废"。

## 十二、StateImageStore 的双自由栈与 RAII transfer

**双自由栈。** `StateImageStore` 维护两个独立的空闲结构：`free_objects_`（逻辑槽位，大小 `logical_capacity`）和 `free_device_slots_`（物理 Device 槽位，大小 `device.slot_count()`）。容量约束在构造里写死了三条（`state_store.h:105-110`）：

```cpp
if (logical_capacity == 0 || device.slot_count() <= 0 ||
    logical_capacity < static_cast<std::uint32_t>(device.slot_count()) ||
    (host != nullptr && logical_capacity < static_cast<std::uint32_t>(device.slot_count()) +
                                               host->capacity())) {
    throw std::invalid_argument("StateImageStore capacity is inconsistent");
}
```

逻辑容量必须 ≥ Device 槽数（每个 Device 副本都要有逻辑身份），有 host 时还必须 ≥ Device 槽数 + host 容量（Host 副本也要有逻辑身份）。两个栈的初始顺序是**逆序填充**的（`state_store.h:111-117`），所以 `free_objects_.back()` 先拿到的是 index 0——这个细节不关键，但说明"槽位分配顺序"是被刻意固定的，可复现。

`device_occupied`（`state_store.h:133-135`）有个微妙点：`free_device_slots_.size() - free_device_count_`。`free_device_slots_.size()` 是**容量**，`free_device_count_` 是**当前空闲数**，相减就是已占用数。但 `free_device_slots_` 是个 `vector<int32_t>`，它从不收缩——只是用 `free_device_count_` 当栈顶指针。这个设计让"归还槽位"是 O(1) 的 `free_device_slots_[free_device_count_++] = slot`（`state_store.h:770-772`），不需要从 vector 中间删除。

**RAII transfer。** `StateImageTransfer` 的析构函数（`state_store.h:835-837`）：

```cpp
inline StateImageTransfer::~StateImageTransfer() {
    if (owner_ != nullptr) { owner_->abort_transfer(std::move(*this)); }
}
```

**transfer 对象销毁时自动 abort**。这意味着如果调用者 `reserve_device_to_host` 拿到一个 transfer，中间抛异常没来得及 `publish_transfer`，析构函数会替它回滚——`abort_transfer`（`state_store.h:661-689`）归还 pending slot、减 source_pins、清 destination_pinned、把 HostToFork 的 destination 退回 ReservedDestination。

`valid_transfer`（`state_store.h:804-811`）是 abort 的安全阀：它要求 `transfer.id_ != 0` 且 `destination.transfer_id == transfer.id_`。id 是 `next_transfer()` 单调递增且跳过 0 的（`state_store.h:779-782`）。这保证了**一个 transfer 对象只能被消费一次**——`consume`（`state_store.h:819-822`）把 owner 和 id 都清零，第二次 abort 时 `valid_transfer` 返回 false，直接 `consume` 后返回，不会二次归还 slot。

`content_epoch` 与 `generation` 是**两个独立的计数器**：`content_epoch` 在每次内容变化（freeze/activate/adopt/fork 等）时前进（`state_store.h:774-777` 的 `next_epoch`），用于"内容变了"的语义；`generation` 在每次释放时前进（`state_store.h:704`），用于"句柄失效"的语义。两者都跳过 0，但用途完全不同——epoch 比较的是"内容新旧"，generation 比较的是"句柄是否还是当初那个"。

## 可引用数字与常量

| 量 | 值 | 出处 |
|---|---|---|
| `kHostChunkBytes`（Host slab 单次 pin 上限） | 4 GiB | `hybrid_cache.cpp:29` |
| `kFullPage` | `kPagedKVPageSize` = 64 | `hybrid_cache.cpp:27` |
| 构造容量约束（有 host） | `logical_capacity ≥ device_slot_count + host_capacity` | `state_store.h:107-108` |
| `materialization_pins` 判据 | 交易源页 + 所有未释放 victim | `context.cpp:65-82` |
| `owner_exclusive_resources` 共享页判据 | `address_references > 1` 跳过 | `context.cpp:357-360` |
| `active_exclusive_baseline` hybrid 分支 | 启用 hybrid 时返回空基线 | `context.cpp:498-499` |
| `materialization_deficit` 压力基准 | 候选完整峰值 | `context.cpp:553-560` |
| `guided_materialization_deficit` 护栏 | 完整峰值而非已 clamp 缺口 | `context.cpp:562-574` |
| `physical_peak_fits` 溢出防护 | `used ≤ capacity - added` | `context.cpp:576-595` |
| `backend_frontier` | MTP=frontier−1, DFlash=frontier, 其他=0 | `context.cpp:826-829` |
| `clear_execution_failure_lanes` 事务让位 | 有事务即整体跳过 | `context.cpp:1155-1166` |
| lease 三档 extra_tokens | `margin` / `2·page`(128) / `page`(64) | `context.cpp:1647` |
| `device_kv_lease_settlement_tokens` 返回值 | `max(max(1,slack), forced)` | `context.cpp:1662-1685` |
| 句柄 generation 起步 | 1（0 保留为哨兵） | `state_store.h:734` |
| content_epoch / generation 跳过 0 | 各自 `if (++x == 0) ++x` | `state_store.h:774-777`, `state_store.h:704` |
| `StateImageTransfer` 析构 | 自动 `abort_transfer` | `state_store.h:835-837` |
| `recycle_events` OOM 降级 | 改直接 `cudaEventDestroy` | `hybrid_cache.cpp:694-698` |
| `settle_prefetch` 收尾 | 等最后一个 prefetch 批次 final event | `hybrid_cache.cpp:651-661` |

## UNKNOWN 清单

以下是三份文件中**没有逐行读透**或**跨文件未追到调用方**的地方，如实标注：

1. **`state_store.h:600-601` 的 `destination_pinned = true` 与 `role = ActiveMutable`**：`begin_host_fork` 的 destination 在 copy 之前就被设为 ActiveMutable（不是 ReservedDestination）。11 号报告说 fork 三段式 destination 是 ReservedDestination→ActiveMutable，这里似乎在 `begin` 阶段就提前转了。是否是"host fork 特例"未追到调用方确认。
2. **`context.cpp:1647` 的三档 extra_tokens 的 `margin` 具体数值来源**：源码里 `margin` 是局部变量，其值取决于 `kv_lease_cushion_pages` 与页大小，我没有追到 `kv_lease_cushion_pages` 的配置注入点。
3. **`hybrid_cache.cpp:439-450` 的 `publish_write` 与 `poll` 的调用时序**：我读了 `poll`/`drain`，但没有追 `publish_write` 是否只在 `poll` 里调用，还是也有独立调用路径。
4. **`context.cpp:1868-1886` 的 `ordered_reset` 的"三个 frontier"具体是哪三个**：我确认它清 pos/rope/MTP position 与三个 frontier，但三个 frontier 的字段名没有逐一核对。（已核对：`sequence.text_kv_valid`、`sequence.mtp_kv_valid`、`sequence.dflash_context_frontier` 三个，`context.cpp:1883-1885`。）
5. **`hybrid_cache.cpp:189-223` 的 `insert_block` 返回值 `InsertResult` 的三个字段**（`device_attached`/`inserted`/`blocks_reattached` 等）与 counters 的对应关系：我只确认了 `blocks_duplicate`/`blocks_inserted`/`blocks_reattached` 三个计数器，没有确认 19 号报告的"18 个计数器"里这三个与另外 15 个的完整映射。
6. **`context.cpp:1027-1124` 的 `can_clear_lane_strict` 的 `ForkEndpoint` 三态**：我读到 enum `{None, Source, Destination}`（`context.cpp:1063`）以及 fork abort 时对 source/destination 走不同 can_* 门，但**没有逐条展开**三个状态下每个 can_* 门的具体判据。
7. **`state_store.h:330-347` 的 `split_device_replica_identity`**：11 号第七节已覆盖，但我没有追它在 context 侧的调用点，无法确认"Both 源的零拷贝分身份"在完整链路里的触发条件。
8. **`hybrid_cache.cpp:336-390` 的 `start_snapshot_host_write`**：我读了主体，但 `host_layout_.image_slabs` 与 `host_layout_.image_bytes` 的几何关系（ slabs 数 × slab_bytes 是否等于 image_bytes）没有逐值核对。