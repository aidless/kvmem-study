# 07 · NInfer 上层三层：逻辑页双副本、suspend 契约、租约阶梯

> **底座口径（2026-11-17 补）**：本文读的是 `iamwavecut/ninfer-all` 分支线 @`6cc1f9fa`（2026-11-04）的源码树。该线**不是上游主干**——`Neroued/ninfer` master 与它已分叉（`compare` 实测 `diverged, ahead_by=523, behind_by=1895, merge_base=f76e19c0fbd0`），且 2026-11-04 之后本线与上游都改过本文描述的部分机制（已核到 4 处，见 `notes/2026-11-17-ninfer上游与分支血缘核验.md`）。**引用本文任何断言请连带 sha 与日期。**

> 承接 06 号报告（`paged_kv_cache` 物理层）。本篇读的是它上面三层：**逻辑页层**（`kv_store.h` 2099 行）、**suspend/snapshot 层**（`residency.cpp` 168 行全读 + `session_snapshot.cpp` 1175 行）、**租约/admission 层**（`context.cpp` 1889 行）。
> 源码树同 06：`kvmem-study/ninfer-src/`，commit `6cc1f9fa438b7094685ae417dfa40e1ca324e617`。

---

## 一、一个逻辑页同时挂两个副本，八个正交计数

```cpp
// src/models/qwen3_5/program/storage/kv_store.h:28-53
class HostKVExtentCapability { owner_, index_, generation_ };   // 与设备侧同构的三元组
struct HostKVPageReplica {
    HostKVExtentCapability extent;
    std::uint32_t page_offset       = 0;
    std::uint32_t membership_node   = max;   // 在 extent 内是第几个成员
    std::uint64_t content_epoch     = 0;
    std::uint32_t committed_columns = 0;     // 0..64，可部分填满
};
```

`LogicalKVPageStore::Page` 上并存的计数（都在 `kv_store.h`，各有 getter）：

| 字段 | 语义 | 归零条件 |
|---|---|---|
| `device_replica` / `host_replica` / `pending_device_replica` | 三个副本槽位（后者是 restore 中转） | 见下 |
| `content_epoch` | 内容版本，任何内容变更都 `next_epoch()` | 永不 |
| `committed_columns` | 已提交的列数（≤ `kPagedKVPageSize`=64），**可部分** | truncate |
| `references` / `active_references` | 地址空间引用数（fork 出来的分支共享同一页！） | 地址释放 |
| `writer_references` | 0 或 1 | 写者结束 |
| `protected_columns` | 前缀保护下限，truncate 不得低于它 | 显式 clear |
| `source_pins` | 拷贝在飞的源钉住计数 | unpin |
| `destination_pinned` | restore 目的地已预留 | publish/abort |

**「一页两副本」才是 KV 分层的本体**：`device_resident()` 与 `host_resident()` 各查各的（`kv_store.h:438-445`），一页可以同时有设备副本和 Host 副本，也可以只有 Host 副本（这正是「KV 历史全在 host」的状态）。

> **对 KVMem 的定位修正**：KVMem 的 patch 不是"发明了 host KV"——引擎本来就有 `host_replica` 槽位和 `HostKVExtentStore`。它做的是把这个 `HostKVExtentCapability` 指向的 extent store **换成可驱逐、可落盘的外部实现**。这就是为什么它能「只借不建」（`ninfer-kvmem.patch:8554-8557` `WindowMemoryBackend`）——接口早就留好了。

---

## 二、谓词/动作分离：`can_*` 全 noexcept，动作全 throw

每个状态转移都成对存在：

```cpp
[[nodiscard]] bool can_destructive_truncate(...) const noexcept;   // 只判，不抛
void destructive_truncate(...);                                     // 判不过就 throw
```

实测 `kv_store.h` 里这类配对至少 11 组（`can_pin_source`/`pin_source`、`can_attach_host_replica`/`attach_host_replica`、`can_release_reference`/`release_reference`、`can_destructive_truncate[+_inactive][+_after_host_release]`/`destructive_truncate[+_inactive]`…）。

**为什么这个拆法值钱**：admission 与资源规划要在 noexcept 路径上反复问"能不能"（不能就换策略，不抛异常），而真正执行时判不过才是**程序错误**，该炸。

> 与我自己 AGENTS.md 第 8/21/37 条同源：**判定与执行分开，判定必须可被反复调用且零副作用**。区别只在于这里的判定结果喂给调度器，我的喂给门禁。

---

## 三、内容证明链的引擎级根：`epoch` + `committed_columns`

每一次跨副本、跨层的引用都带**期望 epoch 和期望列数**，进去先对：

```cpp
// kv_store.h:748-753
bool can_attach_host_replica(handle, epoch, coverage) const noexcept {
    return !page.host_replica && epoch == page.content_epoch && coverage == page.committed_columns;
}
// kv_store.h:497-503  restore 前置
if (page.host_replica->content_epoch != page.content_epoch ||
    page.host_replica->committed_columns != page.committed_columns) {
    throw std::logic_error("logical KV Device restore is not reservable");
}
// kv_store.h:538-544
bool host_replica_current(handle) const noexcept {
    return page.host_replica && page.host_replica->content_epoch == page.content_epoch &&
           page.host_replica->committed_columns == page.committed_columns;
}
```

**为什么必须是两个字段而不是一个 epoch**：页可以**部分填满**。`commit_coverage()`（`kv_store.h:645-653`）要求 `columns` 单调不减（`throw std::invalid_argument("logical KV committed coverage is not monotonic")`）。所以「epoch 相同」不足以保证副本可用——同一 epoch 下覆盖从 32 列涨到 64 列，host 副本还是旧的 32 列。

**这就是 03 号报告里那条「判定走内容 proof 链而非 prompt_cache_key」的引擎级根据**：如果引擎自己在页层就已经用「epoch + 覆盖」双字段防错，上层就不可能只靠一个 key 判对。

---

## 四、Host extent 的生命周期耦合：不能单独释放一页

```cpp
// kv_store.h:735-745  release_reference 的最后一个引用归零时
page.device_replica.reset();
// A Host extent owns its page capability until the extent is released as one atomic
// allocation. Keep the zero-reference logical descriptor alive long enough for
// HostKVExtentStore to detach that replica; without a Host replica the descriptor can
// be reclaimed immediately.
if (!page.host_replica) { release_descriptor(handle, page); }
```

Host 侧分配单位是 **extent（一批页）**，一次原子释放。所以：

- 单页的引用归零**不能**立刻回收描述符——哪怕它的 device 副本已经没了，只要 host 副本还挂在一个活 extent 上，描述符就得留着等 extent 整体释放。
- 反过来 `detach_host_replica()`（`:784-798`）在解除副本后，若 `references==0 && source_pins==0 && !device_replica` 才真正 `release_descriptor`。

**这是真实代价**：Host 侧分配粒度粗 → **零引用页会被"最后释放的那一页"拖住**。这也是为什么 `HostKVExtentStore` 里有一整套 "partition each affected extent once, release all zero-reference runs, and republish retained runs with generation-checked capabilities" 的碎片整理（`host_kv_store.h:373-375`）。

> 任何做"批量分配 + 单页回收"的人都会撞上这个：**分配粒度决定回收延迟**。代码选择的是保留描述符 + 事后重分区，而不是允许 extent 局部释放。

---

## 五、fail-fast 与 fail-soft 的分界线很清楚

- **抛异常**（可恢复，调用方能降级）：所有 `can_* → throw std::logic_error/invalid_argument` 的转移。
- **`std::terminate()`**（不可恢复，状态已不可信）：`rebind_host_replica`（`kv_store.h:767-782`）。它跑在地址空间拆除的 noexcept 路径上，**任何一条不变量对不上就说明内存状态已经坏了**——继续跑比死掉更危险。同样 `host_kv_store.h` 里两处 `std::terminate()`（`prepare()` 与 `append` 的 `if (extent.state != ExtentState::Free)`、`if (node == kInvalidIndex)`）。

**判据**：noexcept 路径上出现不可能状态 = 内在损坏 → 死；可抛路径上出现前置不满足 = 调用顺序错 → 抛。这个二分法在整个代码库里一致。

---

## 六、破坏性截断要求排他

```cpp
// kv_store.h:655-666
void destructive_truncate(handle, columns) {
    if (columns > page.committed_columns || columns < page.protected_columns ||
        page.references != 1 || page.writer_references != 1 || page.source_pins != 0 ||
        page.destination_pinned || page.host_replica) {
        throw std::invalid_argument("logical KV page is not destructively truncatable");
    }
    ...
}
```

六个前置：**我是唯一引用、我是唯一写者、没人正在拷贝我、没有 restore 在途、且我还没有 Host 副本**。

最后一条最容易被忽略：**只要 host 副本存在，就不能就地截断**——因为截断会改 epoch，那份 host 副本立刻变成"旧内容的孤儿"。要么先 `drop` 掉 host 副本，要么走 fork。

> 「回滚/截断需要排他」是分布式系统老话，但这里给出了它的**具体字段清单**，可以直接抄成检查表。

---

## 七、suspend 的六道前置 + 两条清理路径（全文 168 行）

```cpp
// src/models/qwen3_5/program/residency.cpp:120-133
if (!suspendable.enabled) { throw std::logic_error("Qwen3.5 Program was not built with suspendable device memory"); }
for (const RequestControl& request : requests) {
    if (request.lifecycle != Lifecycle::Empty || request.prefill) {
        throw std::logic_error("model suspend requires an idle Program");
    }
}
if (has_context_transaction() || pressure_planning_active_ || (vision_broker && vision_broker->window_open())) {
    throw std::logic_error("model suspend requires an idle Program");
}
if (disk_kv) { disk_kv->wait_idle(); }
// Every stream of every rank, transfers and Vision included, not only the compute streams.
for (std::size_t rank = 0; rank < device.size(); ++rank) { const RankBinding bound(device, rank); CUDA_CHECK(cudaDeviceSynchronize()); }
// Loaned pages have no backing: they return before the live bytes are copied.
if (kv_expert_loan) { kv_expert_loan->recall_all(); }
```

六道门，每道都是一个具体的"别的线程/别的子系统还在动内存"的可能：

1. 每个 request 必须是 `Lifecycle::Empty` 且无 prefill 在飞
2. 无 context 事务在跑
3. 无压力规划在跑
4. vision 窗口必须关
5. `disk_kv->wait_idle()`——磁盘层自己也要静下来
6. **每个 rank 的每一条 stream 都 `cudaDeviceSynchronize()`**（注释特意点明"包括传输流与 Vision 流，不只是计算流"）
7. `kv_expert_loan->recall_all()`——借出去的页没有 backing，必须在拷贝活字节**之前**收回来

第 7 道是顺序敏感的：**loan 回收必须在 capture 之前**，否则 capture 读到的地址范围里有"没有 backing 的页"，写进快照就是垃圾。

### 失败复原：释放全部再抛

```cpp
// residency.cpp:96-118
} catch (...) {
    // A partial restore holds memory the device may need for the retry; give all of it back.
    try { release_device_state_backing(); } catch (...) {}
    throw;
}
```

`resume` 路径同样（`:145-155`）：restore 失败 → 释放全部 → 重抛。**和 06 号报告里 `lease()` 的契约是同一条**："中途失败把每颗送回原映射再抛"。整个代码库对"部分完成"的处理只有一种：**要么全成，要么全退，中间态不许留给调用方**。

---

## 八、快照绑定什么、拒绝什么、能降级什么

`session_snapshot.cpp:25-45` 的格式注释写得比多数论文摘要还清楚：

```
// A snapshot is the complete host image of one catalogued continuation: its resident prefix
// (ledger, exact identity and shortlist digests), its checkpoint directory (endpoint, rewrite
// checkpoint, long anchors), a deduplicated table of the StateImages those checkpoints name, and
// the paged Text and backend KV payload in logical page order. The KV payload covers every
// checkpoint frontier, because every checkpoint lies at or below the committed frontier.
```

四条可抄的判据：

1. **绑定必须精确匹配，不许"重新解释字节"**：
   `session_snapshot.cpp:33-36` "A snapshot binds to the exact model binding and execution configuration of the server that wrote it; restore rejects any mismatch rather than reinterpreting bytes; that includes the RoPE scaling (YaRN / position interpolation), which rotates every stored key."
   → RoPE scaling 变化会**旋转每一个已存储的 key**，所以它必须进配置指纹（`SnapshotConfig` 里 `yarn_factor_bits` / `yarn_native_context` / `interpolation_factor_bits` / `interpolation_threshold`，`:200-203`）。
2. **明文列出"两个 TP 切法可以有相同 plane 数但不同 stride"**：
   `:227-228` "The planes' payload sizes: two divisions of the KV heads over tensor-parallel ranks can have the same plane count and page stride." → 所以 plane 数量和 stride 都不足以判等，还要 `plane_digest`（`:229` `plane_digest(const HostKVPageLayout&)`）。
3. **降级规则写死"哪些能丢、哪个不能丢"**：
   `:36-39` "Restore degrades gracefully: an optional checkpoint whose StateImage fits neither the Device nor the Host state pool, or whose anchor ordinal exceeds this server's anchor capacity, is dropped, while the endpoint is mandatory."
4. **aborted prefill 的语义**：
   `:355-358` "A continuation salvaged from an aborted prefill keeps the whole prompt in its ledger and identity while its KV, state and endpoint stop at the salvaged cursor; its session is that executed prefix, not the unexecuted prompt tail."
   → 「被打断的 prefill」不是一个模糊状态，它有精确语义：**ledger 记全量 prompt，KV/state/endpoint 停在已执行前缀**。

### 完整性校验的威胁模型

```cpp
// session_snapshot.cpp:322-325
// Integrity check over the whole snapshot, stored as a trailing u64. A bit flip that keeps every
// length and domain valid would otherwise restore silently corrupted KV or state. Word-wise, so a
// multi-GB snapshot hashes at memory speed; not cryptographic, since the files are trusted local
// state and the threat is corruption, not tampering.
```

**这一句就是「按威胁模型选机制」的范例**：威胁是**位翻转**不是**篡改**，所以要的是"能发现任意 bit 变但不必抗碰撞"的 word-wise 哈希（多 GB 快照内存速度跑完），而不是 SipHash/Blake3。换成密码学哈希只会让 resume 慢十倍，威胁模型一点不变。

> 对照：03 号报告里 KVMem 的 checkpoint 用 checksum 而非签名，同一逻辑。

版本演进（`kSessionSnapshotVersion = 5`，`:44-46`）：v3 加 RoPE profile + text plane 尺寸；v4 加 word-wise prefix shortlist digests；v5 加 LoRA adapter key。**每次都是"多一个能影响 KV 字节含义的维度"，而不是"多一个方便字段"**。

---

## 九、租约阶梯：按算术判，不按异常判

```cpp
// program/storage/context.cpp:1595-1604
// A full window first; when the pool cannot spare one, a step-sized extension still leaves
// the cushion. No rung may exceed its pool or what one address can map (the cushion would
// otherwise overshoot the context limit), so the reservation can only fail on space.
// The ladder ends at one page group: a pool that can only spare its own allocation
// granularity would otherwise settle with free page groups that no coarser rung can use.
...
// Only a thin lease grows: extending a whole one would ask its pool for space the other lease
// needs, and the one reservation that fails settles both.
```

```cpp
// context.cpp:1630-1632
// A saturated pool re-enters this ladder every round until the engine resumes the lease;
// a rung that cannot fit is found by arithmetic, not by an exception per rung, and is
// applied to both pools or neither.
```

三条设计约束：

1. **阶梯自顶向下，最细一档 = 一个 page group**（"a pool that can only spare its own allocation granularity would otherwise settle with free page groups that no coarser rung can use"）——保证不会出现"池子里还剩几页但没人能用"的尾部分散。
2. **两池全做或全不做**（`:1505-1506` "Both pools or neither: a backend pool that cannot follow returns the text entitlement it just grew, so the sequence never holds one pool's resize without the accounting for it."）——**这是"跨资源原子性"的极简实现**：先成功的一方主动退回，而不是搞两阶段提交。
3. **失败靠算术不靠异常**（`:1630-1632`）——每个 rung 先算能不能装，不能就换下一档，不用 try/catch 当控制流。

### settle：让本轮不失败，把请求写进账

```cpp
// context.cpp:1653-1656
// The pool cannot extend this lease. The sequence settles where the lease still covers a
// step, so no launch can fail coverage mid-round. A space settlement records what its steps
// asked for, so the engine can give it retained cache pages and resume growth before the
// request's budget is bounded.
```

**"settle 而不是 fail"**：装不下更大的 lease 就退到"还能覆盖一步"的位置，**保证本轮 launch 不会因覆盖不足失败**；同时记录这一轮实际想要什么，让引擎下一轮拿保留缓存页还它。这是"降级但不断供 + 记录欠账"。

---

## 十、压力启发式不许自己制造压力

这是这次读到的**最反直觉的一条**：

```cpp
// context.cpp:555-567
// Pressure is relative to this candidate's real peak. Treating every dimension as scarce
// would forbid Device-to-Host demotion even when Host capacity is available.
// Pressure acts on the candidate's complete peak, not on its already-clamped deficit. Applying
// a demotion directly to a zero Host deficit would otherwise manufacture Host pressure even
// when the arena has ample slack and steer the heuristic toward unnecessary destruction.
```

两句话各修一个真 bug：

1. **把每一维都当稀缺** → 即使 Host 有余量也禁止 Device→Host 下沉（策略自我封死）。
2. **对"已夹紧后的缺口"施加压力** → 对一个本来就零缺口的维度施加压力，**会凭空制造出 Host 压力**，把启发式推向"不必要的销毁"。

> 这跟我 AGENTS.md 第 37 条「假阳守卫」是同一个病：**测量/判据本身改变了被测量的东西**。这里更狠——不是误报，是启发式**把自己的响应当成了新的输入**。任何带反馈的资源策略（包括我做的配额门、限流器）都要问一句："我的响应会不会变成下一轮的输入？"

---

## 十一、清理顺序：先 abort 事务，再释放 lane

```cpp
// context.cpp:1156-1157
// A concurrent resource transaction may pin or inspect these active owners. Engine-wide
// cleanup aborts that transaction before releasing lanes, preserving the only safe order.
```

**先中止引用者，再释放被引用者**。反向顺序会让在飞的事务拿到已经释放的句柄（虽然 generation 检查会挡住，但那一层应该只挡 bug，不该挡正常路径）。

---

## 十二、顺带确认的两处小事实

- **共享逻辑页对占用只贡献一次**（`context.cpp:357-359`）："A shared logical page contributes to aggregate occupancy once. Releasing this address cannot free either replica while another address still references it, so it is not part of this owner's exact transition effect." → **fork 出来的分支共享页，占用按页算不按引用算**。任何"按分支求和的显存估算"都会高估。
- **Graft 的不可见性靠零 digest**（`context.cpp:926-928`）："Grafts have identity == nullptr; use a zero-digest shortlist key with the correct frontier so the summary passes validation. The digests being zero keeps it invisible to the prefix-index matching loop." → 用一个合法但永不匹配的 key 让"不该被复用"的东西通过校验且永不被命中。**这是"用一个合法值表达不可用"的干净做法**，比到处 `if (is_graft) skip` 强。

---

## 十三、可直接引用的数字

| 项 | 值 | 来源 |
|---|---|---|
| 逻辑页存储 | 2099 行 | `wc -l kv_store.h` |
| Host extent store | 715 行 | `host_kv_store.h` |
| context（租约/admission） | 1889 行 | `context.cpp` |
| session_snapshot | 1175 行 | `session_snapshot.cpp` |
| residency（suspend） | 168 行（全读） | `residency.cpp` |
| 快照格式版本 | 5 | `session_snapshot.cpp:47` |
| 快照 anchor 上限 | 64 | `:49` `kMaximumSnapshotAnchors` |
| 逻辑页计数字段 | 8 个正交计数（见第一节表） | `kv_store.h` |
| suspend 前置门 | 7 道（6 类 + 每 rank stream 同步） | `residency.cpp:120-133` |
| 破坏性截断前置 | 6 条 | `kv_store.h:657-663` |
