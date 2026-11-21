# 10 · 准入：一条有价格的报价，一次有边界的合并

> **底座口径（2026-11-17 补）**：本文读的是 `iamwavecut/ninfer-all` 分支线 @`6cc1f9fa`（2026-11-04）的源码树。该线**不是上游主干**——`Neroued/ninfer` master 与它已分叉（`compare` 实测 `diverged, ahead_by=523, behind_by=1895, merge_base=f76e19c0fbd0`），且 2026-11-04 之后本线与上游都改过本文描述的部分机制（已核到 4 处，见 `notes/2026-11-17-ninfer上游与分支血缘核验.md`）。**引用本文任何断言请连带 sha 与日期。**

> 06→07→08→09 是"东西怎么存"。这篇是"**请求来了，决定复用多少**"——存储栈的入口。
> 源码：`src/models/qwen3_5/program/prefix/hybrid_program.cpp 1610 行`（本篇逐段读引 `:214-1180`）、`src/runtime/prefix_cache/prefix_index.h`（`SnapshotKind` / `TapPlacement` / 计数器）、`prefix/hybrid_cache.cpp` 的 `poll` / `drain` / `settle_prefetch`。
> 一句话结论：**NInfer 把前缀复用做成"报价"（quote）而不是"判定"（decision）**——准入返回的不是"能不能"，而是"这次请求带多少服务单位、复用多少 token、还剩多少可驱逐页"，**并且这个报价可以被延后（Readiness::TemporarilyBlocked）而不是被拒绝**。

---

## 一、报价的形状：五态出口，零拒绝

```cpp
// hybrid_program.cpp:340-370（精简）
HybridAdmissionQuote out;
out.destination = destination;
// Coalescing only defers a quote that could otherwise be admitted now.
if (plan.allow_prefix_reuse && prompt.identity.reusable && (selected || fits(nullptr, {})) &&
    hybrid_await_sibling(prompt, selected ? selected->frontier : 0U, destination.value)) {
    out.readiness = runtime::Readiness::TemporarilyBlocked;
    out.summary   = quote->summary;
    return out;                                   // ← 不是拒绝，是"等你兄弟"
}
if (selected) {
    quote->snapshot                       = selected->snapshot;
    quote->reuse_frontier                 = selected->frontier;
    quote->summary.reusable_prompt_tokens = selected->frontier;
    quote->summary.prefix_reuse_path      = reuse_path_for(index.snapshot(selected->snapshot).kind);
} else if (fits(nullptr, {})) {
    quote->summary.reusable_prompt_tokens = 0;     // 没命中但装得下 → 从零开始
    quote->summary.prefix_reuse_path      = PrefixReusePath::Root;
} else if (hybrid_->transfers_pending()) {
    // Blocks pinned only by their in-flight Host writes become evictable once those land.
    hybrid_->drain();
    return hybrid_quote(prompt, base, destination);   // ← 排空后重新报价（递归）
} else {
    out.readiness = runtime::Readiness::TemporarilyBlocked;
    out.summary   = quote->summary;
    return out;                                   // ← 装不下且无可排空 → 稍后重试
}
```

**五个出口，没有一个叫 "reject"**：

| 出口 | 条件 | 语义 |
|---|---|---|
| `Ready` + 有候选 | 找到且装得下 | 复用 `selected->frontier` 个 token |
| `Ready` + 无候选 | `fits(nullptr, {})` | 从根开始，`reusable=0` |
| `TemporarilyBlocked` | 兄弟请求正在做同一段物化 | 等它，不重复劳动 |
| `TemporarilyBlocked` | 装不下且无在途传输 | 等驱逐/等释放 |
| 递归重报价 | 装不下但**有**在途 Host 写 | `drain()` 后重跑整个报价 |

最后一条最值得学：**"'装不下'有两种原因——真的没页了，和'页正被在途写钉着、写完就能驱逐'。"** 前者只能等，后者可以**当场排空**。用 `transfers_pending()` 区分，用**递归重跑整个报价**（而不是局部打补丁）保证状态自洽。

> 注意 `hybrid_await_sibling` 的调用被夹在 `(selected || fits(nullptr, {}))` 里——**「只有本来就能准入的报价才允许被延后」**。否则一个装不下的请求去等兄弟，会永远等下去。**延后必须建立在"本可立即满足"之上，这是防死锁的第一性判据。**

## 二、`fits()`：一句会计把"我自己的 pin"扣出预算

```cpp
// hybrid_program.cpp:252-283
const auto fits = [&](const pc::MatchCandidate* candidate, std::span<const pc::NodeRef> path) {
    const std::uint32_t frontier = candidate != nullptr ? candidate->frontier : 0U;
    const std::uint32_t restored =
        candidate != nullptr ? candidate->host_only_blocks +
                                   (candidate->tail && !candidate->tail_on_device ? 1U : 0U)
                             : 0U;
    // Entries this admission pins stop being evictable: its unpinned Device-resident path
    // blocks and the snapshot's Device-resident tail.
    std::uint32_t evictable_on_path = 0;
    for (const pc::NodeRef node : path) {
        const pc::NodeView view = index.node(node);
        if (view.pins == 0 && view.device == pc::CopyState::Resident) { ++evictable_on_path; }
    }
    if (candidate != nullptr && candidate->tail && candidate->tail_on_device &&
        index.snapshot(candidate->snapshot).pins == 0) {
        ++evictable_on_path;
    }
    const std::uint64_t evictable = index.device_evictable_blocks() - evictable_on_path;
    const std::uint64_t text_need = static_cast<std::uint64_t>(plan.text_kv_page_entitlement) -
                                    frontier / kBlock + restored;
    if (text_need > text_kv_pages->physical_pool().available_pages() + evictable) {
        return false;
    }
    ...
};
```

四个乘在一起的正确性细节，任意一个搞错就是"算得下但跑起来 OOM"：

1. **`text_need = entitlement − frontier/64 + restored`**：整段 prompt 的页额度减去候选已提供的块数，加上**必须从 Host 恢复的块数**。复用不是免费的——Host-only 块要占一页。
2. **`restored` 里 tail 单独 +1**：`candidate->tail && !candidate->tail_on_device` → 尾部不满一块，Device 上没有，恢复时要单独占一页。
3. **`evictable_on_path` 必须从可用里扣掉**：这些块现在是"未 pin 且驻留 Device"，看起来可驱逐；**但这次准入会把它们 pin 住**，pin 完就驱逐不了。所以 `evictable = 全部可驱逐 − 我要 pin 的那部分`。
   > **这是全场最精妙的一行。** 朴素写法 `available + evictable >= need` 会算出一个"自己吃自己"的假可行：我要 pin 的块被算进了我可以驱逐的池子里。
4. **backend 池独立再算一遍**：`backend_need = backend_entitlement − hybrid_backend_frontier(frontier)/64 + restored`，两个池都必须够，否则 `false`。

**这条 `fits` 是"预测自己会占用多少"的教科书写法**：把「我要动的对象」从「我能动的资源」里先减掉，再判断供需。

## 三、哪里不能切：三条排除谓词

```cpp
// hybrid_program.cpp:293-300（候选过滤）
std::erase_if(match.candidates, [&](const pc::MatchCandidate& candidate) {
    return candidate.filling_blocks != 0 ||                                  // ① 还没填完
           inside_vision(candidate.frontier, ranges) ||                       // ② 切在图像中间
           !root_snapshot_allowed(prompt.lora_key, candidate.path_blocks);    // ③ 根块 + adapter
});
```

三个谓词各有一条硬理由：

**① `filling_blocks != 0`** —— 块还在被别的请求填。填完才能作为前缀复用（否则拿到半块 KV）。

**② `inside_vision`**（`:63-67`）：
```cpp
return range.begin < frontier && frontier < range.end;   // 严格双侧
```
**不能在一个 vision token range 的中间切开。** 图像占位 token 的 KV 互相依赖（一个 `<|image_pad|>` 的 KV 取决于整张图的 patch 序列），切在中间等于半个图像的条件——数值上无意义。两侧端点可以切（`begin` 之前 / `end` 之后）。

**③ `root_snapshot_allowed`**（`:55-61`）——**引擎自己承认的一个盲点**：
```cpp
// A snapshot inside the first block hangs from the root, whose key carries no block extra, so the
// adapter key cannot tell its prompts apart: only prompts without adapter scales keep or resume
// from one.
return anchor_blocks != 0 || lora_key == std::array<std::uint64_t, 2>{};
```
根节点的 key 不携带 block extra（08 号报告的 `extra` 是第一块之后才累积的），所以 **LoRA 身份对"第 0 块内的快照"不可见** → 带 adapter 的请求禁止复用这种快照。

> 这段注释的价值不在代码，在**它把设计缺陷写在了使用点旁边**。三行注释说清了：什么情况下成立、为什么、以及代价（带 adapter 的请求失去一个复用机会）。

## 四、预报要先落地，不然会选到更浅的源

```cpp
// hybrid_program.cpp:311-318
if (hybrid_->prefetch_landing() && filling(match)) {
    // A prefetch (§6.6) is copying blocks this prompt resumes over: they land within a batch's
    // copy time, and skipping them would pick a shallower source.
    hybrid_->settle_prefetch();
    match = index.match(lookup, prompt.block_hashes, prompt.block_extras, lookup_tokens);
}
```

**如果正在落地的预取恰好覆盖这个 prompt 要复用的块，就等它落地再重新匹配。** 不等的话：那些块现在 `filling_blocks != 0` → 被谓词①滤掉 → 只能选一个更浅的候选 → **白丢复用**。

判断"值得等"的依据是 `kPrefetchBatchBlocks`（`:43-45`）：
```cpp
// Blocks one prefetch batch copies (§6.6): about 20 ms of PCIe for 27B INT8, the longest a quote
// may wait for blocks it is about to resume over.
constexpr std::uint32_t kPrefetchBatchBlocks = 256;
```
**256 块 ≈ 20 ms 的 PCIe（27B INT8）** —— 又一个带具体硬件的实测数。**"最多等一个批次的拷贝时间"是个有物理依据的界，不是拍脑袋的 timeout。**

## 五、候选排序：首选 + 其余按序，第一个装得下就用

```cpp
// hybrid_program.cpp:322-337
const pc::AdmissionChoice choice = index.choose(match, lookup_tokens);
std::vector<std::size_t> order;
if (choice.candidate) { order.push_back(*choice.candidate); }
for (std::size_t i = 0; i < match.candidates.size(); ++i) {
    if (!choice.candidate || i != *choice.candidate) { order.push_back(i); }
}
for (const std::size_t i : order) {
    const pc::MatchCandidate& candidate = match.candidates[i];
    const std::span<const pc::NodeRef> path(match.path.data(), candidate.path_blocks);
    if (fits(&candidate, path)) { selected = candidate; break; }
}
```

**`choose()` 给"最好"的，但"最好"可能装不下** → 退回按候选顺序试，第一个装得下的用。

这是**"首选 + 回退"而不是"过滤后取最优"**：容量约束是动态的（取决于此刻可用页），不能作为 `choose()` 的输入（否则每次容量变动都要重算排序）。**把"价值判断"和"可行性判断"分成两段，是这里最干净的地方。**

## 六、合并等待：一个真正的经济学判据

这是整个文件里最复杂也最有价值的函数（`hybrid_await_sibling`，`:477-579`）。

**动机**：两个请求共享同一段长前缀，A 正在 prefill，B 刚到。若 B 也自己 prefill 一遍，同一段 token 被算两次。更好的做法是**让 B 等 A 把那段 prefill 完并发布快照，B 直接复用**。

**判据（`:531-543`）**：

```cpp
// Waiting saves this request's prefill of the shared tokens, beyond the pass it runs
// either way; a new tap costs the sibling one split. The sibling's remaining prefill
// to the snapshot is the wait, doubled because decode rounds of other lanes interleave
// with it.
const double saved = hybrid_cost_.prefill_seconds(reuse, n - reuse) -
                     hybrid_cost_.prefill_seconds(wait->target, n - wait->target);
const double predicted =
    2.0 * hybrid_cost_.prefill_seconds(prefill.cursor, wait->target - prefill.cursor);
if (saved > (wait->plan_tap ? hybrid_cost_.chunk_seconds : 0.0) &&
    predicted <= hybrid_coalesce_wait_seconds_) {
    consider(*wait);
    continue;
}
```

三个量，两个门：

- **`saved`** = 自己 prefill 到 n 的代价 − 等到 target 再 prefill 剩下的代价 = **等能省下的时间**。
- **成本** = 若需要在兄弟上新增一个 tap，兄弟要多切一刀（`chunk_seconds`）。**收益必须超过这个成本才值得等。**
- **`predicted`** = 兄弟从当前游标 prefill 到 target 的代价 × **2.0**。
  > **乘 2 的理由写在注释里：「decode rounds of other lanes interleave with it」**——兄弟的 prefill 不是独占执行的，中间会插入其他 lane 的 decode。这是个经验系数，但**它是显式写出来的经验系数**，不是藏在魔数里。
- **门**：`saved > cost`（值得）且 `predicted <= hybrid_coalesce_wait_seconds_`（等得起）。

**两个前置条件本身就是过滤器**（`:478-479`、`:494-499`）：
```cpp
if (hybrid_coalesce_wait_seconds_ <= 0.0 || !prompt.vision_items.empty()) { return false; }   // 视觉 prompt 不参与合并
...
// Vision placeholders are equal tokens for different media, so token equality proves nothing there.
if (!prefill.prompt.vision_items.empty()) { continue; }
```
**视觉 prompt 双向排除合并**：`<|image_pad|>` 占位 token 对不同图片是一样的，**token 相等在那里证明不了任何事**。（与 08 号报告 `extra` 累积同源——同一类"占位 token 不携带身份"的问题，在两处各自做了防御。）

**切点对齐**（`:519-521`）：
```cpp
// A new tap goes on the block boundary below the divergence: a frontier inside a block
// publishes only once the sibling completes that block.
const std::uint32_t aligned = shared / kBlock * kBlock;
```
**新增的 tap 必须落在分歧点下方最近的块边界上**——块内的 frontier 要等兄弟把这一块填满才能发布。

**优先级：优先复用兄弟已计划的精确 tap**，除非"从它 prefill 到新 tap"比"多切一刀"更贵（`:524-536`）。**能不改兄弟的计划就不改。**

## 七、`boundary` 标记：分歧点不许被任何一方收回

```cpp
// hybrid_program.cpp:565-577
// The snapshot the waiting request resumes from is where two conversations diverge: it is
// published as a boundary, so neither lineage supersedes it.
...
if (/* 已有精确 tap */) { std::prev(at)->boundary = true; }
else if (best->plan_tap) { state.taps.insert(at, PlannedTap{.position=..., .placement=Exact, .boundary=true}); }
for (HybridPendingTap& tap : state.pending) { if (tap.frontier == best->target) { tap.boundary = true; } }
```

配合 `prefix_index.h:57-64` 的枚举注释：

```cpp
enum class SnapshotKind : std::uint8_t {
    Tap,
    Endpoint,
    OutputBoundary,
    // A tap at a boundary that later prompts may share across conversations: a client-named
    // breakpoint, the end of the tools or leading System/Developer block, or the divergence of
    // concurrent requests. A lineage continuing past it supersedes it only while no other
    // conversation has continued from it.
    Boundary,
};
```

**「A lineage continuing past it supersedes it only while no other conversation has continued from it」**——这是**引用计数式的撤销权**：

- 普通 `Tap`：我自己往下走，可以覆盖它（`supersede`）。
- `Boundary`：**只要别的会话从它继续过，我的 lineage 就不能收回它。**

**如果没有这条，合并就是造一个立刻被销毁的快照**：A 请求把自己的 tap 发布出来让 B 用，A 继续生成 → A 的 supersede 逻辑把自己的旧 tap 收回 → B 手里拿的是死引用。`boundary=true` 是**让"跨请求共享"在所有权模型里合法**的那一步。

> 这条与 07 号报告的 `destructive_truncate` 六前置、03 号 KVMem 的 `restore 恒 fork 不消费` 是同一族纪律：**共享的读方必须能阻止写方收回，且这个阻止必须是显式的所有权声明，不是时序上的侥幸。**

## 八、复用改变调度成本：服务单位要重算

```cpp
// hybrid_program.cpp:359-373
// Scheduler service units: one per nominal prefill chunk, one per Vision item use (which may
// shorten a chunk) and one per further generated token.
const std::uint32_t reuse  = quote->reuse_frontier;
const std::uint32_t suffix = n - reuse;
std::uint64_t units        = suffix == 0 ? 1ULL : 1ULL + (suffix - 1ULL) / prefill_chunk;
if (plan.vision_control_plan) {
    for (const auto& item : plan.vision_control_plan->items) {
        if (item.token_end > reuse) { ++units; }
    }
}
if (quote->summary.effective_output_tokens != 0) {
    units += quote->summary.effective_output_tokens - 1ULL;
}
units += branch_service_work(plan.branches.get(), prefill_chunk);
quote->summary.service_work_quanta = units;
```

**`units` 从 `suffix = n − reuse` 算，不是从 `n` 算。** 也就是说：**命中缓存让这个请求在公平调度里更便宜**，它占的调度份额变小 → 别的请求得到更多算力。

三个细节：
1. `suffix == 0` 时仍记 **1** unit（完全命中也要占一格，否则可以无限堆积）；
2. vision item 若跨过 reuse 点，**每个 +1**（图像处理会切碎 chunk）；
3. `effective_output_tokens − 1`（第一个 decode token 已含在 prefill 那格里）。

> **缓存命中不能只在延迟上体现，还要在公平性核算上体现。** 否则命中缓存的请求会按"未命中成本"占份额，挤掉真正需要算力的请求——这是批处理系统里非常常见的一处漏算。

## 九、提交侧：三池交集决定"一块算不算完成"

```cpp
// hybrid_program.cpp:1180-1190（hybrid_publish_blocks）
std::uint32_t frontier = sequence.text_kv_valid;
// Every enabled pool must hold the whole block. The DFlash2 draft ring is StateImage content,
// not block content, so it does not bound block publication.
if (sequence.kv->backend) { frontier = std::min(frontier, backend_kv_valid(sequence)); }
frontier = std::min<std::uint32_t>(frontier, static_cast<std::uint32_t>(sequence.ledger.size()));
```

**`frontier = min(text_kv_valid, backend_kv_valid, ledger.size())`** —— 可发布的前沿取**所有启用池的交集**。

这是 07 号报告 `committed_columns` 在程序层的对应物：**一页"可复用"要求它的每一个副本都完整**。并且注释精确划了边界：**DFlash2 的 draft ring 是 StateImage 内容不是块内容，所以不约束块发布**——"哪些池约束这个对象"是个必须逐个说明的判断，不能一律取 min。

驱逐循环（`:669-681`）：
```cpp
while (text_pool.available_pages() < text_pages ||
       (backend_pool != nullptr && backend_pool->available_pages() < backend_pages)) {
    if (hybrid_->index().evict_device_blocks(1) != 0) { continue; }   // 一次一块，够了就退出循环
    if (!hybrid_->transfers_pending()) { return false; }              // 驱逐不动且无在途 → 真没戏
    hybrid_->drain();                                                 // 有在途 → 排空让它们变可驱逐
}
```
**每次只驱逐一块**（不是一次驱逐到够），因为驱逐本身有代价，够了就停。**"驱逐不动"和"暂时驱逐不动"分开处理**——同第一节的报价逻辑一致。

## 十、本篇可引用数字

| 项 | 值 | 来源 |
|---|---|---|
| 预取批大小 | 256 块 | `hybrid_program.cpp:44` |
| 预取批的物理代价 | **≈20 ms PCIe（27B INT8）** | `hybrid_program.cpp:43-44` 注释 |
| 合并等待的预测放大系数 | **2.0**（"decode rounds of other lanes interleave"） | `hybrid_program.cpp:540-542` |
| 合并排除 | 视觉 prompt（双向）、`vision_items` 非空 | `:478, :497-498` |
| 切点对齐 | `shared / kBlock * kBlock` | `:519-521` |
| 服务单位 | `1 + (suffix-1)/prefill_chunk`，`suffix = n − reuse` | `:362-364` |
| 块发布前沿 | `min(text_kv_valid, backend_kv_valid, ledger.size())` | `:1183-1187` |
| 驱逐粒度 | 1 块/次 | `:675` |

---

## 十一、五条可搬走的设计做法

1. **准入返回"报价"而不是"判定"**：报价里带服务单位、复用 token 数、可驱逐页数——上层调度器能拿它做公平性决策。"能不能"这个二元答案浪费了准入时算出来的全部信息。
2. **延后必须建立在"本可立即满足"之上**（`(selected || fits(nullptr,{})) && await_sibling(...)`）。否则延后会变成死等。**这是防死锁的结构性判据，不是超时兜底。**
3. **预测自己占用的对象，要先从可支配资源里扣掉**（`evictable_on_path`）。朴素写法会算出"自己吃自己"的假可行。
4. **共享的读方要能阻止写方收回**（`boundary`）。做成显式所有权声明（枚举值 + 引用计数语义），不要依赖时序。
5. **缓存命中要进公平性核算**（`units` 用 `suffix`）。只在延迟上体现命中，会挤掉真正需要算力的请求。

> 六份报告连起来看，NInfer 的纪律可以压成一句：**每个"免费"的能力背后都有一处显式的记账**——页配额（06）、副本列数（07）、slab 归属（08）、字节预算（09）、服务单位（10）。**没有一处依赖"应该够用"。**
