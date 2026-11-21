# 11 · StateImage：一个身份、三个平面、两把锁

> **底座口径（2026-11-17 补）**：本文读的是 `iamwavecut/ninfer-all` 分支线 @`6cc1f9fa`（2026-11-04）的源码树。该线**不是上游主干**——`Neroued/ninfer` master 与它已分叉（`compare` 实测 `diverged, ahead_by=523, behind_by=1895, merge_base=f76e19c0fbd0`），且 2026-11-04 之后本线与上游都改过本文描述的部分机制（已核到 4 处，见 `notes/2026-11-17-ninfer上游与分支血缘核验.md`）。**引用本文任何断言请连带 sha 与日期。**

> 06–10 讲的是 KV 页（token 的 KV）。这篇讲**另一类状态**：linear-attention 的 recurrent state、conv 状态、DFlash 草稿环——统称 `StateImage`。
> 源码：`src/models/qwen3_5/program/storage/state_store.h 867 行`（本篇逐行读完全文）、`prefix/hybrid_cache.cpp 785 行`（读 `:285-340` 事件排序、`:439-500` poll/drain、`:551-662` 分层恢复与预取结算）、`prefix/hybrid_cache.h` 的 `HybridRestoreLayer`。
> 一句话结论：**KV 页是"可以按列补齐"的（07 号报告的 `committed_columns`），StateImage 是"要么完整要么不存在"的**——所以它需要一整套独立的身份/副本/转移机制，而这套机制的设计比 KV 侧更干净。

---

## 一、三个平面：逻辑身份 ≠ 副本

```cpp
// state_store.h:97-118
StateImageStore(qwen3_5::StateImageDevicePool& device, qwen3_5::HostStatePool* host,
                std::uint32_t logical_capacity)
    ...
    if (logical_capacity == 0 || device.slot_count() <= 0 ||
        logical_capacity < static_cast<std::uint32_t>(device.slot_count()) ||
        (host != nullptr && logical_capacity < static_cast<std::uint32_t>(device.slot_count()) +
                                                   host->capacity())) {
        throw std::invalid_argument("StateImageStore capacity is inconsistent");
    }
```

类头注释一句话定调：

> `// Program-private logical StateImage ownership. Logical identity is independent from Device and Host replicas; published raw Tensor views are reconstructed only from an active Device binding.`

**三个平面**：

| 平面 | 载体 | 数量关系 |
|---|---|---|
| 逻辑身份 | `StateImageHandle{owner, index, generation}` | `logical_capacity` 个槽 |
| Device 副本 | `device_slot`（`std::optional<int32_t>`） | `device.slot_count()` 个 |
| Host 副本 | `host_slot`（`std::optional<HostStateSlotHandle>`） | `host->capacity()` 个 |

**构造期的不等式是整个设计的锚**：`logical_capacity >= device.slot_count() + host->capacity()`。

也就是说 **逻辑身份数 ≥ 两份物理副本数之和**。为什么是 ≥ 不是 ≤？因为**同一个逻辑身份最多同时拥有一个 Device 副本和一个 Host 副本**，而"一个接受 Host-only、一个接受 Device-only、一个两者都有"都需要独立身份槽。这个不等式保证**最坏情况下（每个物理槽各占一个身份）也装得下**。

对比 06 号报告的 `DeviceKVPagePool`：那里是"页 = 一个物理对象"，身份和存储是同一个东西。**这里身份先于存储存在**——一个身份可以暂时两份副本都没有（转移中），也可以只有一份。**这是"状态"和"页面"的根本差别：页面可以按列补齐，状态不能。**

## 二、四态角色机

```cpp
// state_store.h:38-43
enum class StateImageRole : std::uint8_t {
    Free,
    ActiveMutable,
    CheckpointImmutable,
    ReservedDestination,
};
```

四个状态，两条正交的轴：

- **可变性**：`ActiveMutable`（正在被 decode 写） vs `CheckpointImmutable`（已冻结，可被多个读方共享）
- **占位**：`ReservedDestination`（已占槽但内容未就绪） vs `Free`

状态转移全部是显式函数，每个都把自己需要的全部前置条件写进一个 `if`：

```cpp
// state_store.h:171-179  freeze：ActiveMutable → CheckpointImmutable
if (object.role != StateImageRole::ActiveMutable || !object.device_slot ||
    object.source_pins != 0 || object.destination_pinned || has_pending_replica(object)) {
    throw std::logic_error("StateImage active image is not freezable");
}
object.content_epoch = next_epoch();      // ← 冻结 = 换 epoch
object.role          = StateImageRole::CheckpointImmutable;
```

```cpp
// state_store.h:182-190  thaw：CheckpointImmutable → ActiveMutable
if (object.role != StateImageRole::CheckpointImmutable || !object.device_slot ||
    object.checkpoint_references != 0 || object.source_pins != 0 ||
    object.destination_pinned || has_pending_replica(object)) {
    throw std::logic_error("StateImage checkpoint is not thawable");
}
object.role = StateImageRole::ActiveMutable;   // ← 解冻不换 epoch：字节没变
```

**`freeze` 换 epoch，`thaw` 不换。** 这是全篇最值得记的一处语义判断：

- **冻结**是"从这里开始这份字节对外承诺不变" → 新 epoch（此后任何写都会变成违反承诺）。
- **解冻**是"我要继续写这份字节" → **字节此刻没变**，所以 epoch 不变；但它一旦被写，下一次 freeze 才会换 epoch。

**epoch 记的是"内容版本"，不是"状态版本"。** 这与 07 号报告里 `epoch` 必须配 `committed_columns` 一起校验完全同源：**身份字段只覆盖它真正承诺的东西。**

## 三、两把锁：`source_pins` 与 `destination_pinned`

```cpp
// state_store.h:731-743（Object 结构）
struct Object {
    std::uint32_t generation    = 1;
    std::uint64_t content_epoch = 0;
    std::optional<std::int32_t> device_slot;
    std::optional<qwen3_5::HostStateSlotHandle> host_slot;
    std::optional<std::int32_t> pending_device_slot;
    std::optional<qwen3_5::HostStateSlotHandle> pending_host_slot;
    std::uint64_t transfer_id           = 0;
    std::uint32_t checkpoint_references = 0;
    std::uint32_t source_pins           = 0;
    bool destination_pinned             = false;
    StateImageRole role                 = StateImageRole::Free;
};
```

**两个方向各一把锁，不能合用一个**：

- **`source_pins`（计数）**：多少方正在**读**这份字节。可叠加（多个 fork 同时读同一 checkpoint）。
- **`destination_pinned`（布尔）**：这份字节**本身**正在被写。最多一个写者，所以是布尔不是计数。

合用一个计数的坏处：`pins > 0` 分不清"有读者"和"有写者"，而这两件事的处置完全相反——有读者要推迟回收，有写者要拒绝任何其他访问（包括读：正在被写的目的地不能作为另一个 fork 的源）。

**`pending_*` 是第三类占位**：已分配物理槽但字节还在路上。于是：

```cpp
// state_store.h:744-747
[[nodiscard]] static bool has_pending_replica(const Object& object) noexcept {
    return object.pending_device_slot.has_value() || object.pending_host_slot.has_value() ||
           object.transfer_id != 0;
}
```

**`has_pending_replica` 出现在本文件每一个前置条件里**——freeze、thaw、release、drop_device_replica、drop_host_replica、can_recycle_checkpoint_destination、activate_reset、require_copied_destination、`split_device_replica_identity`、`begin_fork`……

> 这是**"in-flight 状态必须能被打断的实现看见"**的教科书做法。一个在途拷贝的源/目的都不能被回收、冻结、复用——把这条检查做成一个函数，然后**在每个状态转移的第一行调用它**，而不是分散写。

## 四、fork 三段式：begin / commit / abort

```cpp
// state_store.h:400-424
[[nodiscard]] StateImageSelectors begin_fork(StateImageHandle source, StateImageHandle destination) {
    ...
    if (source == destination || source_object.role != StateImageRole::CheckpointImmutable ||
        !source_object.device_slot ||
        source_object.source_pins == std::numeric_limits<std::uint32_t>::max() ||
        destination_object.role != StateImageRole::ReservedDestination ||
        !destination_object.device_slot || destination_object.source_pins != 0 ||
        destination_object.destination_pinned || has_pending_replica(destination_object)) {
        throw std::logic_error("StateImage fork binding is invalid");
    }
    ++source_object.source_pins;                    // ← 锁源（可叠加）
    destination_object.destination_pinned = true;   // ← 锁目的（独占）
    destination_object.role               = StateImageRole::ActiveMutable;
    destination_object.content_epoch      = source_object.content_epoch;   // ← 继承 epoch
    return {.source      = *source_object.device_slot,
            .destination = *destination_object.device_slot};
}
```

**三件事一次发生**：锁源、锁目的、**继承 epoch**。

继承 epoch 的意义：分叉出来的新身份**装的还是同一份字节**（拷贝之后才可能不同）。所以"epoch 相同"在 begin_fork 之后、第一次写之前是**真的**——这不是记账偷懒，是准确描述。

```cpp
// commit_fork：只释放两把锁，不改角色（目的保持 ActiveMutable）
--source_object.source_pins;
destination_object.destination_pinned = false;

// abort_fork：释放锁 + 目的退回 ReservedDestination + epoch 归零
--source_object.source_pins;
destination_object.destination_pinned = false;
destination_object.role               = StateImageRole::ReservedDestination;
destination_object.content_epoch      = 0;     // ← 目的从未被写过，epoch 归零是对的
```

**abort 把 epoch 归零是对的**：目的槽里的字节是**上一任主**留下的垃圾，不是这次 fork 的内容。归零后任何残留的校验都会判不等（07 号报告的 `epoch == content_epoch` 双字段校验在此处救场）。

**`begin_fork` 的源前置里有一条 `source_pins == UINT32_MAX`**——防计数溢出。这个检查在这个文件里出现了三次（`begin_fork`、`reserve_device_to_host`、`begin_host_to_device`、`begin_host_fork`）。**溢出检查出现在每个自增之前，不是只在一个地方。**

## 五、三种转移方向，第三种是关键

```cpp
// state_store.h:55-59
enum class StateTransferDirection : std::uint8_t {
    DeviceToHost,
    HostToDevice,
    HostToFork,
};
```

前两个是普通的双向搬运（降级 / 提升）。**第三个 `HostToFork` 才是这套设计的点睛**：

```cpp
// state_store.h:601-643  begin_host_fork
// 前置：源是 CheckpointImmutable 且只有 Host 副本（!source_object.device_slot 不强制，但 !host_slot 会被拒）
const std::optional<std::int32_t> target = take_device_slot();   // ← 目的占一个 Device 槽
...
destination_object.pending_device_slot = *target;
destination_object.destination_pinned  = true;
destination_object.role                = StateImageRole::ActiveMutable;
destination_object.content_epoch       = source_object.content_epoch;
++source_object.source_pins;
device_->copy_from_host(host_->view(*source_object.host_slot), *target, stream);   // ← Host → 目的槽
```

**Host 副本直接拷进 fork 的目的地，源完全不落地到 Device。**

对比用 `HostToDevice` 做同一件事要的代价：

| 路径 | 源需要的 Device 槽 | 拷贝次数 | 结束状态 |
|---|---|---|---|
| `HostToDevice` + `split` | **1**（先提升源） | 1 | 源 DeviceOnly，目的无 |
| `HostToFork` | **0** | 1 | 源仍 HostOnly，目的 DeviceMutable |

也就是说：**一个 HostOnly 的 checkpoint 可以被 N 个会话并发 fork 而完全不需要临时占用 Device 槽**。这是"冷状态预热"的正确形状——**预热的代价是一次拷贝，不是一次占用**。

> 这与 08 号报告的「Host 恢复跑专用 restore stream、与 write-through 和计算重叠」是同一族：**冷数据进热路径的路径必须短，且不能为了搬运而占一份宝贵的热资源。**

## 六、`publish_transfer(keep_source_replica)`：一个开关决定提升还是复制

```cpp
// state_store.h:645-670
void publish_transfer(StateImageTransfer&& transfer, bool keep_source_replica) {
    ...
    case StateTransferDirection::DeviceToHost:
        source.host_slot = source.pending_host_slot;   source.pending_host_slot.reset();
        if (!keep_source_replica) { return_device_slot(*source.device_slot); source.device_slot.reset(); }
        ...
    case StateTransferDirection::HostToDevice:
        source.device_slot = source.pending_device_slot;  source.pending_device_slot.reset();
        if (!keep_source_replica) { host_->release(*source.host_slot); source.host_slot.reset(); }
        ...
    case StateTransferDirection::HostToFork:
        destination.device_slot = destination.pending_device_slot;  destination.pending_device_slot.reset();
        ...
}
```

**同一个转移函数，`keep_source_replica` 一个布尔决定两种语义**：

- `false` → **降级/提升**（边搬边丢，副本数不变，省空间）
- `true` → **复制**（搬完两边都在，副本数 +1，抗风险）

这比写两个函数好：**"搬"这个动作在两种语义下完全相同，只有收尾不同**。把差异压成一个布尔参数，避免两份几乎一样的代码漂移。

> 注意 `HostToDevice` 那一支 `!keep_source_replica` 时的失败处理：`if (host_ == nullptr || !host_->release(...)) throw std::logic_error("StateImage Host replica release failed at publication")`。
> **这是唯一一个"在 publish 阶段还可能抛"的地方**——因为"释放 Host 副本"这个动作可能失败，而 publish 本应是纯记账。注释没写，但意图清楚：**释放失败意味着 Host 池状态和我们的账对不上，这是"不可能状态"，宁可以 terminate 之外的方式炸出来，也不能假装成功。**

## 七、`split_device_replica_identity`：Both 源的零拷贝分身份

```cpp
// state_store.h:330-352
// Transfers the Device replica to a new logical identity while the old immutable identity
// keeps its Host replica. This is the zero-copy preserving branch for a Both source.
void split_device_replica_identity(StateImageHandle source, StateImageHandle destination) {
    ...  // 前置：源 = CheckpointImmutable 且两份副本都在；目的 = ReservedDestination 且两份都没有
    destination_object.device_slot   = source_object.device_slot;      // 指针搬家
    destination_object.content_epoch = source_object.content_epoch;
    destination_object.role          = StateImageRole::ActiveMutable;
    source_object.device_slot.reset();                                  // 源只剩 Host
}
```

**当源有两份副本时，分身份是零拷贝的**：把 Device 副本的**所有权**给新身份，源留下 Host 副本。

一次 `std::optional` 的移动，把"两份副本"变成"两个身份各持一份，字节一份都没动"。

> 这是本系列里**唯一一处真正意义上的零拷贝分叉**。06 号报告的 `evictable_kv_pool` 是"零拷贝借出"，08/10 的 fork 仍然要拷一次（Host→Device）。这里因为**源恰好有一份可丢的冗余副本**，才做得到完全不搬。
> **前置条件里"源必须两份副本都在"不是巧合——它就是这个优化的物理前提。** 换个说法：**零拷贝的代价是"你必须先有一份冗余"。**

## 八、副本丢弃的对称性：只能丢"另一份还在"的那份

```cpp
// state_store.h:296-320
[[nodiscard]] bool drop_device_replica(StateImageHandle handle) noexcept {
    ...
    if (object.role != StateImageRole::CheckpointImmutable || !object.device_slot ||
        !object.host_slot || object.source_pins != 0 || ...) { return false; }
    return_device_slot(*object.device_slot);
    object.device_slot.reset();
    return true;
}

[[nodiscard]] bool drop_host_replica(StateImageHandle handle) noexcept {
    ...
    if (host_ == nullptr || object.role != StateImageRole::CheckpointImmutable ||
        !object.device_slot || !object.host_slot || ...) { return false; }
    ...
}
```

**两个函数都要求 `device_slot && host_slot` 同时存在**——即**只有在丢完之后仍有另一份副本时才允许丢**。

这就是"永不丢失最后一份副本"的不变量，用**前置条件**实现，而不是用"先检查有没有别处"的调用方纪律。**把不变量做成被调方的门，调用方就不需要记住它。**

## 九、`recycle_checkpoint_destination`：原地回收但保住 epoch

```cpp
// state_store.h:349-380
[[nodiscard]] bool can_recycle_checkpoint_destination(StateImageHandle handle) const noexcept {
    ...
    return object.role == StateImageRole::CheckpointImmutable &&
           object.device_slot.has_value() && !object.host_slot &&
           object.checkpoint_references == 1 && object.source_pins == 0 &&
           !object.destination_pinned && !has_pending_replica(object);
}

[[nodiscard]] std::uint64_t recycle_checkpoint_destination(StateImageHandle handle) {
    ...
    const std::uint64_t epoch    = object.content_epoch;   // ← 把 epoch 交出去
    object.checkpoint_references = 0;
    object.role                  = StateImageRole::ReservedDestination;
    return epoch;
}

void restore_recycled_checkpoint(StateImageHandle handle, std::uint64_t content_epoch) {
    ...
    object.content_epoch         = content_epoch;          // ← 拿回来
    object.checkpoint_references = 1;
    object.role                  = StateImageRole::CheckpointImmutable;
}
```

**用途**：一个 checkpoint 只有**我自己**一个引用（`checkpoint_references == 1`）、且只有 Device 副本、且没人读 → 我可以把它临时变回"空目的地"，用它做一次新的物化（省一次 malloc），完事再把 epoch 还回去。

**为什么必须交出 epoch 而不是留在对象里**：变成 `ReservedDestination` 之后，槽里的字节随时可能被覆盖；若 epoch 还挂在对象上，任何"epoch 相等 → 内容可信"的检查都会在覆盖之后误判。**把 epoch 交给调用方保管 = 让"这段时间内容不可信"成为类型可见的事实。**

这是**用一个返回值承载"不可信窗口"**——比加一个 `bool dirty` 干净得多，因为 dirty 标志的清除时机又是个问题，而 epoch 的恢复是**结构性的**（要么还被调用方拿着，要么已经还回来了）。

> 呼应 07 号报告：`destructive_truncate` 要求"无 host 副本"。这里 `can_recycle_checkpoint_destination` 也要求 `!object.host_slot`。**同一条纪律：能被就地覆盖的对象，必须只有一份副本。**

## 十、单流 FIFO 换掉了依赖图

```cpp
// hybrid_cache.cpp:452-470
void HybridPrefixCache::poll() {
    // Every write shares the transfer stream and every restore the restore stream, so each list
    // completes in submission order.
    while (!pending_.empty()) {
        const cudaError_t status = cudaEventQuery(pending_.front().done);
        if (status == cudaErrorNotReady) { break; }
        CUDA_CHECK(status);
        publish_write(pending_.front());
        pending_.pop_front();
    }
    while (!landing_.empty()) {
        const cudaError_t status = cudaEventQuery(landing_.front().layers.back());
        if (status == cudaErrorNotReady) { break; }
        ...
    }
}
```

**注释那行是全部的设计理由**：「Every write shares the transfer stream and every restore the restore stream, so each list completes in submission order.」

于是 `poll()` 只需要**查队首**，队首没完成就停——不需要为每个在途项维护"它依赖谁"的图。`drain()` 同理（`cudaEventSynchronize` 队首，pop，循环）。

**代价**：所有写必须串行在同一流上（失去并行度）；**收益**：完成顺序 = 提交顺序，队列即状态机。

> 与此对照 `settle_prefetch`（`:651-661`）：
> ```cpp
> // One restore stream completes batches in submission order: the last prefetch batch's final
> // event covers every batch submitted before it.
> for (auto batch = landing_.rbegin(); batch != landing_.rend(); ++batch) {
>     if (batch->prefetch) { CUDA_CHECK(cudaEventSynchronize(batch->layers.back())); break; }
> }
> poll();
> ```
> **只需同步"最后一个预取批的最后一个层事件"，就已覆盖它之前提交的所有批。** 没有单流 FIFO 这个前提，这里就得同步 N 个事件（或算传递闭包）。

**"用调度约束换状态机简化"是一个常被低估的工程决策。** 单流的带宽损失是可测的；依赖图的复杂度是不可测的（bug 面）。

## 十一、分层恢复：prelude 之后按层发事件

`submit_restore`（`:551-617`）是 06 号报告"层序是 memcpy 参数"那句话的完整实现：

```cpp
// Prelude: what activation reads (the snapshot tail) and what a pass reads outside the layer
// stack (backend KV, the continuation hidden, DFlash local state).
enqueue_from_host(restore_.tail_copies, restore_stream_);
if (!restore_.copies.backend_pages.empty()) { ... copy_from_host_records(...); }
if (restore_.image) { ... copy_from_host_segments(..., Kind::Rest, ...); }
restore_.prelude = record();            // ← 第一道闸：激活所需的全部到位

// Then each layer's own state, in forward order, so a pass can start on the first layers
// while later ones are still arriving.
for (const HybridRestoreLayer& layer : layers_) {
    ...
    restore_.layers.push_back(record());   // ← 每层一个事件
}
```

两段结构：

1. **prelude**：`tail`（激活立刻要读的）+ backend KV + continuation hidden + DFlash 局部状态。**"不在层栈里、但一次 pass 会读"的东西全在这里。**
2. **逐层**：按 `HybridRestoreLayer` 的**正序**发拷贝，**每层记一个事件**。

`land_restore`（`:618-634`）只等 prelude：
```cpp
CUDA_CHECK(cudaStreamWaitEvent(consumer, restore_.prelude, 0));
```

**消费者只要 prelude 就够开工了**，后续层的到达由计算流自己按层等（事件在 `restore_.layers` 里）。

> 这就是"分层召回真的能省时间"的机制：**不是少拷，而是让第 1 层的计算和第 20 层的拷贝重叠。** 06 号报告说"层序是 memcpy 参数不是调度决策"——现在可以补一句：**层序是 memcpy 参数，而每层一个事件才是调度决策。**
> 而且前置有硬断言（`:494-500`）：`if (view.device != CopyState::Absent || view.host != CopyState::Resident) throw "hybrid restore source block is not host-only"` —— **恢复只从"Device 上没有、Host 上有"的块读**，避免"恢复一个已经在 Device 上的块"。

## 十二、`StateReadOwnership`：借用要标出来

```cpp
// state_store.h:815-836
enum class StateReadOwnership : std::uint8_t {
    // The primary binding owns the direct StateImage lifetime.
    Primary,
    // The same active lineage retains the source through an optional checkpoint reference.
    LineageCheckpoint,
    // A retained private or shared owner outside this active lineage retains the source.
    ExternalOwner,
};

struct ActiveStateBinding {
    StateImageHandle read;
    StateImageHandle write;
    bool fork_pending                 = false;
    StateReadOwnership read_ownership = StateReadOwnership::Primary;

    // A non-primary read source is pinned only for the pending Fork. Its allocation belongs to a
    // surviving same-lineage checkpoint or external owner, never to the primary binding.
    [[nodiscard]] bool borrows_read() const noexcept { return read_ownership != StateReadOwnership::Primary; }

    // An external owner can be a retained private endpoint whose lifetime has no StateImage
    // checkpoint reference, so refcount zero alone does not authorize this sequence to release it.
    [[nodiscard]] bool read_has_external_owner() const noexcept { return read_ownership == StateReadOwnership::ExternalOwner; }
};
```

**读源可能不属于我。** 这句话的后果写在注释里：**"refcount zero alone does not authorize this sequence to release it"**。

三种归属各自意味着不同的释放权：
- `Primary`：我是主人，引用归零就能释放。
- `LineageCheckpoint`：同一 lineage 的 checkpoint 还拿着，我不能释放。
- `ExternalOwner`：**要么是我retained 的私有端点，要么是别人共享给我的——引用计数归零也不能证明它该被释放。**

> 注意 `borrows_read()` 与 `read_has_external_owner()` 的关系：前者是"我不拥有"，后者是"连计数都不可信"。**两个谓词分别对应"要不要 pin"和"能不能靠计数判断"两个不同的问题。**
> 这与 03 号报告 KVMem 的「restore 恒 fork、源不消费」是同一条纪律的两层实现——**读的一方永远不推断写的一方的所有权。**

## 十三、可引用数字与事实

| 项 | 值 | 来源 |
|---|---|---|
| 逻辑身份容量下界 | `device.slot_count() + host->capacity()` | `state_store.h:107-112` |
| 角色状态数 | 4（Free / ActiveMutable / CheckpointImmutable / ReservedDestination） | `state_store.h:38-43` |
| 转移方向数 | 3（DeviceToHost / HostToDevice / **HostToFork**） | `state_store.h:55-59` |
| 每个 Object 的计数/标志 | `checkpoint_references` / `source_pins` / `destination_pinned` / `transfer_id` / 2 个 `pending_*` | `:731-743` |
| Host chunk | `kHostChunkBytes = 4 GiB` | `hybrid_cache.cpp:29` |
| 每层一个事件 | `restore_.layers`（`HybridRestoreLayer` 正序） | `hybrid_cache.cpp:600-616` |
| 预取结算 | 只同步"最后一个预取批的最后一层事件" | `hybrid_cache.cpp:651-661` |

---

## 十四、六条可搬走的设计做法

1. **身份容量 ≥ 物理副本总和**：让"同一身份最多一份 Device + 一份 Host"这个约束在容量上可满足，就不需要运行时的容量恐慌。
2. **两把锁不合并**：读者用计数（可叠加），写者用布尔（独占）。合成一个 `pins` 会丢掉"这是读还是写"这个决定性信息。
3. **`has_pending_replica()` 进每个前置条件**：in-flight 必须被所有状态转移看见，做成一个函数、每处第一行调用。
4. **`freeze` 换 epoch、`thaw` 不换**：epoch 是"内容版本"不是"状态版本"。身份字段只承诺它真正承诺的东西。
5. **`HostToFork` 让冷状态预热不占热资源**：拷贝一次，不占 Device 槽。零拷贝分身份（`split_device_replica_identity`）的物理前提是**源有一份冗余副本**。
6. **单流 FIFO 换掉依赖图**：完成顺序 = 提交顺序，队列即状态机。带宽损失可测，依赖图 bug 不可测。

> 与前五份连起来看，这篇补上了缺失的一环：**06–10 处理的是"可以按列补齐的存储"，本篇处理"必须原子存在或不存在的状态"**。前者靠 `committed_columns` 允许部分就绪，后者靠"永不丢最后一份副本"保证要么完整要么不存在——**两个极端，各自诚实。**
