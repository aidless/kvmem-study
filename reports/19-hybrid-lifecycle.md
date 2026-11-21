# 19 · hybrid_program.cpp 后半段：一个快照从 tap 到 endpoint 的完整生命周期

> 底座：`kvmem-study/ninfer-src/src/models/qwen3_5/program/prefix/hybrid_program.cpp` 1610 行，commit `6cc1f9fa438b7094685ae417dfa40e1ca324e617`（kvmem `backends/versions.json` 钉住）。
> 本报告接 10 号报告（`:214-1180` 准入报价与跨请求合并），读 `:1174-1610`——块发布、tail 拷贝、tap 捕获、supersede、lane 释放、endpoint 发布与统计。
> 方法：只引用能指到 `path:line` 的事实；机制意图取源码注释原文，UNKNOWN 如实标注。

## 一、一句话

前半段解决「从哪个快照恢复、能不能恢复」；后半段解决「恢复之后怎么把新的前缀和快照**还回缓存里**，且还回去的动作不能挡住正在跑的 decode」。核心是一条不对称的纪律：**写缓存（tap / endpoint / host write）全部允许失败并静默跳过，释放 KV 与状态槽则 `std::terminate()` 兜底**——因为前者是优化，后者是正确性。

## 二、块发布：三池交集，DFlash2 draft ring 不算

`hybrid_publish_blocks`（`:1174-1203`）先算可发布前沿：

```cpp
std::uint32_t frontier = sequence.text_kv_valid;
if (sequence.kv->backend) { frontier = std::min(frontier, backend_kv_valid(sequence)); }
frontier = std::min(frontier, static_cast<std::uint32_t>(sequence.ledger.size()));
const std::uint32_t full = frontier / kBlock;
```

三者的交集：**文本 KV 有效位、backend KV 有效位、token ledger 长度**。注释给了一条边界划分：「Every enabled pool must hold the whole block. The DFlash2 draft ring is StateImage content, not block content, so it does not bound block publication.」

这句话把 DFlash2 的 draft ring 明确归进 StateImage 而不是块内容——它不参与块前沿。这解释了 06 号报告「DFlash2 draft ring 是 StateImage 内容」在发布层的落点：分类不是命名偏好，它决定哪个前沿被约束。

循环逐块推进（`while (lane.path.size() < full)`）：以 `lane.path_hash` 为父哈希、`tokens` 为 64 个 token、`extra` 为前缀累积的身份位，调用 `pc::block_lookup_hash` → `hybrid_->insert_block(parent, hash, tokens, extra, pages)` → 把 `result.node` 压入 `lane.path`，并把 `lane.path_hash = hash` 滚动到新哈希。**链是一条 hash 链，每块的身份包含父哈希**，所以插入顺序就是前缀顺序。

`extra` 的取法有个细节：块索引 < `lane.prompt_extras.size()` 时取对应项，否则取 `lane.trailing_extra`——尾部不足一整块的区域用累积的尾值。

## 三、tail：一份快照的最后一页是**复制**出来的

`hybrid_copy_tail`（`:1205-1232`）做一次完整的 tail 页复制：

1. `hybrid_make_room(1U, backend_pool ? 1U : 0U)` 先腾位；
2. 两池各 `reserve(1)`，任一失败就返回 `nullopt`（先前成功的不回滚在这里——由 reservation 的析构负责）；
3. `materialize_transfer_destination(*reservation, columns)` 拿到目标页；
4. `copy_page(source, copy, device.stream)` 在 **compute stream**（不是 transfer stream）上复制；
5. `register_tail_copy(copy, columns)` 登记。

注释：「Each tail owns its bundle, so evicting any cache entry returns exactly one page per pool.」——**tail 是自己拥有的一整束页**，这保证了驱逐一个缓存条目恰好归还每池一页。如果 tail 与 block 共享页，驱逐就不能按条目归还，会计数失衡。

`hybrid_make_room`（`:669-681`）的循环是个两态机：先 `evict_device_blocks(1)`，返回 0 说明没有可驱逐块；若还有在途传输则 `drain()` 让它们落地变成可驱逐块；既无可驱逐又无在途才 false。这与 10 号报告的 `make_room` 是同一套逻辑在 tail 路径的复用。

`hybrid_reclaim_device_kv`（`:1548-1571`）是压力路径上的同一循环，只是目标是 `(target = available + need)`，一次也只驱一块；`released != 0` 时 `advance_resource_revision()` 通知上游资源变化。

## 四、tap 捕获：先抢槽位，再复制状态，任何一步失败都只记 skip

`hybrid_publish_pending`（`:1234-1316`）把一个 pending tap 变成已发布快照。它的前置判断值得逐字看：

```cpp
const bool anchored = lane.path.size() >= full;
const bool tail_block_ready = tail == 0 || lane.path.size() > full;
```

- `anchored`：锚点块（第 `full` 块）已在 path 里；
- `tail_block_ready`：若 tap 在块中间（`tail != 0`），还需要**下一块**也发布——因为 tail 要复用那一块的页，块本身必须已完全提交。

finishing 时允许一个例外：如果 lane 自己的 `text_kv_valid >= tap.frontier` 且 backend 追平，就用自己尚未发布的 partial page 作为 `tail_source`（`:1256` 注释：「The finishing lane's own partial page holds the tail; it is copied before the lane releases it.」）。这是 lane 结束前最后一次把内部状态兑现成缓存快照的机会。

不发布的两种情况都只 `drop`：`!anchored`（且非 finishing）和 `!root_snapshot_allowed(lora_key, full)`。后者是 10 号报告已经讲过的「根节点不携带 block extra，LoRA 身份对第 0 块内快照不可见」——同一个谓词在这里第二次设卡，证明它不是只在准入期检查一次就完事。

`hybrid_capture_tap`（`:1318-1354`）的顺序是**先 acquire 再 copy**：

```
index.acquire_device_slot(estimate_priority(last_capture, frontier, frontier % kBlock != 0))
  → state_store->reserve_destination()
  → state_images->copy_slot(write, image, device.stream)
  → publish_copied_checkpoint(*image)
  → lane.pending.push_back({frontier, image, slot, boundary})
```

任何一步失败都释放已拿到的资源再抛或记 skip；`catch (...)` 里先 `state_store->release(*image)` 再 `index.release_device_slot(*slot)` 然后重抛——**资源顺序与获取顺序相反**。

`boundary=true` 的 tap 入 pending 但不设 `lane.tap_snapshot`（`:1307-1310`），因此不参与后续 supersede。这正是 10 号报告 `SnapshotKind::Boundary` 的「引用计数式撤销权」在 lane 侧的落地：**合并造出的快照不会因为本 lane 继续前进而被撤销**。

`hybrid_after_prefill_chunk`（`:1356-1385`）决定 tap 何时真正触发。Exact tap 要求 `tap.position == cursor`；flexible tap 在 `final_chunk_next = prompt_tokens - cursor <= prefill_chunk` 时提前落。跳过条件三条：在 vision exclusion 内、`cursor <= last_capture`、非 exact 且距上次 capture 不足 `kMinimumTapSeparation`。注释明确 flexible 的语义：「A flexible tap lands at a chunk boundary, possibly past it.」——tap 的 frontier 可能大于 planned position，但一定是块边界或 chunk 边界。

## 五、supersede：谁的快照可以收回

两个 supersede 函数共享同一个模式——**先检查有效性，再收回**：

`hybrid_supersede_resume`（`:1388-1398`）：

```cpp
if (lane.disposable || !lane.resume_snapshot.valid() ||
    frontier <= lane.resume_frontier || !index.valid(lane.resume_snapshot)) return;
index.supersede(lane.resume_snapshot);
```

注释：「No request continues a disposable one, so the snapshot it resumed from still serves the requests that share it (the other reads of one decision request).」

这是 disposable 语义的关键：一次性决策读取完就走，**不能因为它自己前进而把复用源收回**——还有别的读取在同一决策上。`frontier <= resume_frontier` 的检查则保证只有「本 lane 已经走过了恢复点」才收回。

`hybrid_supersede_tap`（`:1400-1408`）同理，操作 `lane.tap_snapshot` / `lane.tap_frontier`。两个 `valid()` 检查（`snapshot.valid()` 与 `index.valid(snapshot)`）是两道不同的门：前者是句柄非空，后者是 index 里还在。缺任何一道都可能对一个已经不在 index 的快照调用 supersede。

`hybrid_finish_lane` 里在 `publish_snapshot` **之前**调 `hybrid_supersede_resume`（`:1475` 注释）：「The lineage moves past its source here, so a slot it needs comes from its own lineage before another conversation's snapshot.」——**先释放源头，再从池里抢 slot**，这样新快照优先拿到自己血脉里的槽位，不会挤掉别的会话。

## 六、lane 释放：host 写穿必须在 release_path 之前

`hybrid_release_lane`（`:1417-1443`）是 `noexcept`，有两处 `std::terminate()`。顺序：

1. 若 `restore_ticket != 0`，先 `order_after_restore(ticket, device.stream)`。注释解释为什么需要：「The state slot the lane returns may still be a destination of its restore (a lane can end before its first prefill pass waits for the copies)」——**lane 可能在第一次 prefill 等拷贝之前就结束了**，此刻 restore 还在往要归还的槽里写。
2. 释放所有 pending tap 的 image 和 slot；
3. 若 `state.active`，先 `start_block_host_writes(path, device.stream, device.transfer_stream)`，**再** `index.release_path(path)`。

第 3 条的顺序是整套机制的命门，注释原文：「back them on the Host tier first, in one batch, so eviction drops Device copies instead of losing nodes」。翻译：path 上的块一旦去掉 pin 就变可驱逐；如果不同时把它们复制到 Host，驱逐就只是丢掉 Device 副本而**丢掉节点本身**。批量（`in one batch`）是为了减少 transfer stream 的提交次数。host 写失败被 `catch (...) {}` 吞掉——写穿是优化，失败只意味着这些块以后要重算，不影响正确性。

`std::terminate()` 只在最外层 `catch (...)`（`:1441`）。这是 07 号报告「fail-fast vs fail-soft 分界」在 lane 生命周期的对应：**noexcept 路径遇到不可能状态就终止**。

## 七、endpoint 发布：十道门，catch 全吞

`hybrid_finish_lane`（`:1445-1546`）是整条链的终点。成为 endpoint 必须同时满足十个条件（`:1462-1475`）：

| # | 条件 | 含义 |
|---|---|---|
| 1 | `endpoint && request.publish_continuation` | 请求声明这是可延续的结束点 |
| 2 | `kv_releasable` | 两池 `can_release_after_deactivate` 都为真 |
| 3 | `backend_caught_up` | backend 追平；masked draft 时额外要求 `dflash_context_frontier == frontier` |
| 4 | `frontier != 0` | 空快照没有意义 |
| 5 | `root_snapshot_allowed(lora_key, anchor_blocks)` | LoRA 身份对根块不可见时禁止 |
| 6 | `frontier >= deepest_snapshot + kMinimumTapSeparation` | 与最深 tap 保持最小间距 |
| 7 | `frontier <= ledger.size()` | 前沿不越过 token ledger |
| 8 | `path.size() >= anchor_blocks` | 锚点块都已发布 |
| 9 | `!fork_pending && read == write` | 状态没有分叉、读写同一个 image |
| 10 | `role(write) == ActiveMutable` | image 角色正确 |

第 2 条有个关键注释：「Moving the endpoint image into the index commits this lane to the strict KV release below, which cannot report failure; a release that would fail keeps the image here so the ordinary strict clear can refuse it and the Engine recovers.」——**先确认能释放，再转移 image 所有权**；因为转移之后只能走 `release_active_sequence_kv_strict`，它不返回失败。这是「所有权转移是不可逆操作」的显式防御。

发布动作：`freeze(image)` → `register_tail`（若有 partial tail；backend columns 单独算）→ `index.publish_snapshot(anchor, frontier, tail_tokens, tail_id, slot, SnapshotKind::Endpoint)`。

`published.created` 时：`attach_image` → `image_moved = true` → `sequence.state = {}` → `++endpoints_created` → **再** `hybrid_supersede_resume`（注释「Superseded first, so the Host write below can take its slabs」）→ `start_snapshot_host_write`。顺序仍是「先释放旧槽，再用新槽写 host」。`!created` 时 `thaw(image)`——别人已经发布了同样的快照，本 lane 的 image 退回可变状态。

整个 `try` 的 `catch (...)` 注释只有一句：「Terminal publication is optional; a failure keeps whatever was published and releases the lane normally.」然后无条件 `hybrid_release_lane(lane)`。

最后分两路：`image_moved == false` → `clear_lane_strict`；`true` → `release_active_sequence_kv_strict` + `retire_continuation_slot` + `request.retire()`。注释：「The release was checked before the image moved, so its strict invariants hold here.」——前面第 2 道门的检查在这里兑现。

## 八、activate 的三条入口守卫与两个来源

`hybrid_activate`（`:805-1173`）入口：

```cpp
if (lane >= max_concurrency || lane_epochs[lane] != quote.destination_epoch ||
    requests[lane].lifecycle != Lifecycle::Empty ||
    active_continuations[lane] < continuation_capacity || requests[lane].prefill ||
    !transaction.state) {
    throw std::logic_error("hybrid admission destination is not free");
}
if (reuse != 0 && tail != 0 && snapshot.tail_device_copy == CopyState::Absent) {
    throw std::logic_error("hybrid admission tail is not on the Device");
}
```

第二道守卫对应 08 号报告的 tail 契约：**partial tail 必须在 Device 上**，Host 副本不可替代——因为激活后第一次 decode 就要读它，不能等恢复。

状态的三个来源（`:854-866`）：

- `reuse == 0`：`state_store->activate_reset(state, device.stream)`，全新零状态；
- `transaction.state_restored == true`：`activate_copied(state)`，Host restore 已经落在目标槽；
- 否则：从 `hybrid_->image(quote.snapshot)` 在 device 上 `copy_slot` 到目标槽，再 `activate_copied`。

KV 的原则（注释原文）：「cached full pages are shared; a partial tail is copied into a private page.」——完整页共享，部分页复制成私有。这与 `hybrid_copy_tail` 的「tail 自己拥有整束」是同一条不变量在不同层的表现。

`rebuild_work = quote.root_rebuild_work`（`:929` 注释）：「The generated-output rebuild ledger continues from the root recipe exactly as Legacy materialization installs it; it does not depend on the reuse source.」——**生成输出的重建账本从 root recipe 继续，与从哪个快照复用无关**。这是一条正交性声明：复用路径影响 prefill 成本，不影响生成时的重算逻辑。

tap 规划（`:1020-1060`）三条决策都很有意思：

1. 从 Endpoint 恢复时，删掉 `GenerationOpener` 类型的 hint。理由写在注释里：「client echoes generated turns token for token (agent loops, preserved reasoning): its own endpoint will serve the next turn, so the generation opener is not worth the prefill split an exact tap costs. Otherwise the history is re-rendered (for example with reasoning stripped) and the next turn diverges at the opener.」——**客户端行为决定要不要在生成起点插 tap**：回显原文就不需要，重新渲染就需要。
2. branched read 的 `tap_tokens = n + 1U`（多一个 token），因为分支跟随 frontier，显式 boundary 是下一次读同一状态的恢复点。
3. disposable 请求删掉 `position > named`（最后一个显式 boundary）之后的所有 tap。理由：「No request continues a disposable one: past the last boundary its client named, the prompt is its own (a decision read's question), so no snapshot there serves another request, and an exact tap would only cost a prefill pass.」

`progress_hybrid_materialization`（`:627-666`）的注解把 Host 恢复的调度讲清楚了：「A Host restore does not hold the admission: the lane starts now and its Device work queues behind the copies, activation behind the restore's prelude and each layer of its first prefill pass behind that layer's copies (§6.5).」——**恢复不阻塞准入**：lane 立即启动，Device 工作排在拷贝之后，激活排在 restore prelude 之后，首轮 prefill 逐层排在对应层的拷贝之后。这是 08 号报告「逐层恢复是 memcpy 参数不是调度决策」在 activate 侧的实现。

取消路径里还有一句：「The abort returns the state slot a landing restore may still be writing」——`abort` 前要 `order_after_restore` 等落地中的 restore 完成，否则归还的槽仍在被写。

## 九、统计数据面

`hybrid_stats`（`:1573-1608`）把两组计数合成对外结构：

- **index 侧 6 项**：`nodes` / `snapshots` / `device_resident_blocks` / `device_evictable_blocks` / `host_free_slabs` / `free_device_slots`，外加 config 的 `host_slabs` 与 host_layout 的 `host_slab_bytes`；
- **cache 侧 18 项**：`admissions` / `snapshot_hits` / `reused_tokens` / `blocks_inserted` / `blocks_reattached` / `blocks_duplicate` / `taps_created` / `taps_skipped` / `endpoints_created` / `host_image_writes` / `host_block_writes` / `host_image_restores` / `host_block_restores` / `host_tail_restores` / `host_write_bytes` / `host_restore_bytes` / `evicted_blocks`；
- **index 侧三个丢数计数**：`host_snapshot_evictions` / `host_dead_reclaims` / `unbacked_node_losses`。

第三组是 04 号实验计划真正该采的「丢了多少」指标：前两组告诉你做了多少事，这三个告诉你**失去了什么**。`unbacked_node_losses` 尤其关键——它度量那些既不在 Device 也没写回 Host 就被驱逐的节点，是「写穿来不及」的直接读数。

## 十、可引用的数字与常量

| 项 | 值 | 出处 |
|---|---|---|
| 块大小 | 64 token | `paged_kv_cache.h:20`（06 号报告） |
| `kMinimumTapSeparation` | 见 `prefix_index.h`（未在本次逐字读取） | `hybrid_program.cpp:1383`、`:1468` 引用 |
| tail 复制所在 stream | `device.stream`（compute），host 写穿在 `device.transfer_stream` | `:1226`、`:1437` |
| endpoint 发布门 | 10 道（见第七节表） | `:1462-1475` |
| tap 跳过原因 | 3 条（exclusion / 不前进 / 间距不足） | `:1381-1384` |
| lane 释放的 `std::terminate` | 1 处（最外层 catch） | `:1441` |
| 状态来源 | 3 种（zero / device copy / host restore） | `:854-866` |
| 对外统计字段 | 18 + 6 + 3 = 27 项 | `:1573-1608` |

## 十一、四条跨报告的连接

1. **与 06 号（物理层）**：`hybrid_copy_tail` 的两池 reserve → materialize → 复制，正是 06 号「reserve 只占容量、materialize 才绑物理页」在 prefix 层的应用；`copy_page` 在 compute stream 上完成，印证「零拷贝的前提是 free 页写前不读」。
2. **与 08 号（prefix 层）**：`insert_block` 的父哈希链 + token 原文落盘（284 B/块）解释了为什么 `block_lookup_hash` 只能找候选；`snapshot` 的 Host 副本住 index 自有 slab 解释了 `start_snapshot_host_write` 为什么排在 supersede 之后。
3. **与 10 号（准入）**：`hybrid_make_room` 的「evict → drain → 再 evict」与准入 `fits()` 的 `evictable_on_path` 扣除是同一套会计的两端；`root_snapshot_allowed` 在准入与发布两次设卡。
4. **与 13 号（disk spill）**：`hybrid_release_lane` 的「先 host 写穿再 release_path」与 disk spill 的「遇洞即停」是同一条不变量在 L2 / L3 上的不同表现——**可恢复性优先于回收速度**。

## 十二、UNKNOWN 与未读

- `kMinimumTapSeparation` 的具体数值未在本次逐字读取（只见到引用），按铁律 2 不填。
- `hybrid_activate` 的 `:870-1020` 段（KV 地址空间创建、共享页/复制 tail 的具体绑定）与 `:1060-1173` 段（vision / DFlash prefill 初始化）只抽样读了关键守卫，未逐行；这两段分别连到 vision 排他与 DFlash2 masked backend，是下一层的候选。
- 18 号报告的 host tier 六行实测只验证了形状（host=0 时写计数为 0 等），本报告给出的统计字段可用于 04 号实验矩阵的逐计数器对账——该矩阵尚未跑。
