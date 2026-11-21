# 12 · 租约阶梯：从"能不能扩"到"答案变短"

> **底座口径（2026-11-17 补）**：本文读的是 `iamwavecut/ninfer-all` 分支线 @`6cc1f9fa`（2026-11-04）的源码树。该线**不是上游主干**——`Neroued/ninfer` master 与它已分叉（`compare` 实测 `diverged, ahead_by=523, behind_by=1895, merge_base=f76e19c0fbd0`），且 2026-11-04 之后本线与上游都改过本文描述的部分机制（已核到 4 处，见 `notes/2026-11-17-ninfer上游与分支血缘核验.md`）。**引用本文任何断言请连带 sha 与日期。**

> 07 号报告留下了一句没兑现的话：「租约阶梯按算术判不按异常判」。这一篇把它读完了，结论是——**我当时的表述半对**，真正的结构比"算术 vs 异常"复杂一层，而且更值得学。
> 源码：`src/models/qwen3_5/program/storage/context.cpp:1571-1830`（`ensure_sequence_kv_lease` 144 行、`device_kv_lease_settlement_tokens`、`device_kv_lease_shortfall`、`resume_device_kv_lease`）、`src/runtime/engine/engine_core.h:2955-3000`（`apply_device_kv_lease_settlements`）、`src/runtime/engine/context_cache/resource_manager.h:1296-1360`（`reclaim_device_kv_for_lease`）、`:2291-2305`（可回收判据）。
> 一句话结论：**Device KV 租约不是"够 / 不够"的二值判定，是一条四级降级链**——扩租约 → 抢保留缓存 → 缩答案预算 → 以 `finish_reason=length` 诚实收尾。**任何一级都不允许"launch 失败"。**

---

## 一、为什么需要一条阶梯而不是一个判定

推理的每一步 decode 都要往 KV 里追加。如果"追加装不下"这件事以 `cudaError` 的形式暴露，那它出现在**已经算了一半的 kernel 里**——除了崩掉没有别的处置。

所以引擎把"装不装得下"提前成一次**租约（lease）**：进入 decode 之前先声明"我这个序列从现在起最多要用到 X 页"，之后每一步都在租约内，不会失败。

问题变成：**声明的时候池子不够，怎么办？**

三种可能的答案：

| 答案 | 后果 |
|---|---|
| 拒绝这个请求 | 用户看到一个明明有 24GB 显存的机器说"内存不够" |
| 用满再说，装不下就崩 | 最差 |
| **降级到"还能工作"的最小状态，并让收尾理由诚实** | 引擎的选择 |

## 二、只有 decode 可以扩租约

```cpp
// context.cpp:1575-1580
if (!context_cache.kv_lease_growth) { return; }
// Only a decode step may extend the lease: materialization during admission has to leave the
// sequence holding exactly the entitlement its plan declared.
RequestControl& request = requests[sequence.lane];
if (request.lifecycle != Lifecycle::Active) { return; }
```

**这条注释是全篇的骨架**：准入期（admission）的物化**必须**让序列恰好持有它计划里声明的 entitlement，不许偷偷扩。

为什么这条如此重要——回到 04 号实验计划的语境：**预算 B 的可信性依赖于"规划期和运行期用同一把尺"**。如果准入阶段自己就能扩租约，那"我按 B=2048 页规划，实际用了 3000 页"就会发生，B×ctx 扫描的每一条曲线都不再可解释。

**扩租约的权力只给 decode，是因为 decode 是"已经在跑、必须跑完"的阶段**——准入还能退回去重选策略，decode 不能。

## 三、只有"thin"的租约才增长

```cpp
// context.cpp:1589-1597
const std::uint32_t backend_ceiling = std::min(capacity, request.lease_ceiling + kv_lease_backend_allowance_tokens());
const std::uint32_t reach = kv_pages_for_tokens(backend_ceiling);
const bool main_thin =
    text_pages < reach && kv_pages_for_tokens(main_tokens) + cushion > text_pages;
const bool backend_thin = sequence.kv->backend.has_value() && backend_tokens != 0 &&
                          backend_pages < reach &&
                          kv_pages_for_tokens(backend_tokens) + cushion > backend_pages;
if (!main_thin && !backend_thin) { return; }
```

**两个条件同时成立才算 thin**：① 租约还没到 `reach`（还没拿满本该拿的）② 这一步要用的量 + 垫子已经超过现有租约（真的快不够了）。

```cpp
// context.cpp:1622-1626
// Only a thin lease grows: extending a whole one would ask its pool for space the other lease
// needs, and the one reservation that fails settles both.
const auto targets = [&](std::uint32_t extra_tokens) { ... };
```

**为什么"满的租约"不许再增长**：两个序列共用一个池，如果 A 已经拿满还去抢，B 就饿死。这条注释半句话写清了整个公平性前提——**"抢"的许可只对"确实不够的人"开放。**

> 这和 10 号报告的服务单位公平性核算同源：**资源分配的公平性不靠调度器的善意，靠"申请者必须先证明自己需要"这个前置。**

## 四、`target()` 的三重 clamp：先把非空间失败全排掉

```cpp
// context.cpp:1607-1610
const auto target = [reach](std::uint32_t cap, std::uint32_t pages, std::uint32_t wanted) {
    return std::min(cap, std::max(pages, std::min(reach, wanted)));
};
```

三个上界：`cap`（池物理容量 / 地址空间可映射页数，取更小者）、`reach`（ceiling + draft allowance）、以及不低于 `pages`（现有的）。

```cpp
// context.cpp:1613-1620（targets 内部）
.main = main_thin ? target(std::min(text_kv_pages->physical_pool().capacity_pages(),
                                    text_kv_addresses->page_capacity()),
                           text_pages,
                           kv_lease_pages_for_tokens(...))
                  : text_pages,
```

**为什么先 clamp 一遍？** 看紧随其后的注释：

> `// No rung may exceed its pool or what one address can map (the cushion would otherwise overshoot the context limit), so the reservation can only fail on space.`

**"让预留只可能因空间失败"** —— 这是一个失败归因设计。因为后面要 catch `std::bad_alloc`，如果 rung 有可能因为"超出地址空间"而失败，那个 `bad_alloc`（或别的异常）就会和"真的没空间"混在一起，`lease_space_limited` 这个标志就不可信，而 `lease_space_limited` 又是"能不能 resume"的判据。

> **凡是要用一个布尔承载"原因是 X"的地方，必须先把所有非 X 的原因在更早的地方排除掉。** 否则那个布尔迟早说谎。

## 五、三档阶梯，且必须收到一页组

```cpp
// context.cpp:1666-1673
const std::uint32_t page = static_cast<std::uint32_t>(kPagedKVPageSize);
for (const std::uint32_t extra_tokens : {kv_lease_growth_margin_tokens(), 2U * page, page}) {
    try {
        if (grow(targets(extra_tokens))) { return; }
    } catch (const std::bad_alloc&) { space_limited = true; }
}
```

三档**降序**尝试：一整窗（`margin_tokens`）→ 两页 → 一页。

```cpp
// context.cpp:1602-1606
// The ladder ends at one page group: a pool that can only spare its own allocation
// granularity would otherwise settle with free page groups that no coarser rung can use.
```

**最后一级必须落到"一页组"（64 token）**。理由很实在：如果阶梯最低只到两页，而池子恰好只剩一页的空间，那么它会 settle——**手里有空闲页组，但没有任何更粗的 rung 能用它**。多要一档一页，收益是"在碎片化的池上多榨出一步的余量"。

> 这条对任何"分级申请"的设计都成立：**阶梯的末级必须等于分配器的最小粒度**。否则末级之下的空间等于不存在。

## 六、修正 07 号："算术判"与"异常兜底"是两层，不是二选一

07 号报告我写的是「按算术判不按异常判」。原文其实两样都有：

```cpp
// context.cpp:1648-1663  grow()
// A saturated pool re-enters this ladder every round until the engine resumes the lease;
// a rung that cannot fit is found by arithmetic, not by an exception per rung, and is
// applied to both pools or neither.
if (!text_kv_addresses->can_resize_entitlement(sequence.kv->text, wanted.main) ||
    (sequence.kv->backend && !backend_kv_addresses->can_resize_entitlement(
                                 *sequence.kv->backend, wanted.backend))) {
    space_limited = true;
    return false;
}
resize_sequence_kv_entitlement(sequence, wanted.main, wanted.backend);
```

**分工是清楚的**：

| 手段 | 覆盖面 | 何时发生 |
|---|---|---|
| `can_resize_entitlement()` 算术预判 | 常态 | **每一轮、每一档**都走（池饱和时每步都重进阶梯） |
| `catch (std::bad_alloc&)` | 罕见 | 算术说能装、实际 `resize` 失败 |

**为什么不能只靠异常**——注释前半句就是答案：「A saturated pool re-enters this ladder every round」。池饱和时，**每步 decode 都会重跑这个三档阶梯**，如果每档都靠抛异常来发现"装不下"，那异常就成了常态路径：每步 3 次 throw/catch，每次都构造栈回溯。异常是给"意外"用的，不能当控制流。

**为什么不能只靠算术**——`can_resize_entitlement` 是"账面上装得下"，而 `resize` 真的会向池申请；两者之间没有原子性。算术过了、实际失败，是必须兜住的。

> 所以 07 号那句话该改成：**「常态用算术判（因为每轮都走），异常只作兜底（因为账面与实际之间没有原子性）」**。这不是文字游戏——**任何高频路径上的"用异常表达预期内结果"都会变成性能事故**，而任何"只信预判不信实际"都会在竞态下漏。

## 七、两池全做或全不做

```cpp
// context.cpp:1648-1657
if (!text_kv_addresses->can_resize_entitlement(sequence.kv->text, wanted.main) ||
    (sequence.kv->backend && !backend_kv_addresses->can_resize_entitlement(
                                 *sequence.kv->backend, wanted.backend))) {
    space_limited = true;
    return false;
}
```

**两个池都要能扩，否则一个都不扩。** 因为一次 decode 同时写 main KV 和 backend KV，只扩一个的话下一步照样装不下——**但已经占掉的空间不会自己还回来**。

注释里的「and is applied to both pools or neither」和 08 号报告 `context.cpp:1505-1506` 的"两池全做或全不做"是同一条纪律的第二次出现。**在同一个文件里出现两次的纪律，说明它是这个引擎的硬约束，不是局部优化。**

## 八、settle：承认不够，但保证下一步能跑

```cpp
// context.cpp:1675-1682
// The pool cannot extend this lease. The sequence settles where the lease still covers a
// step, so no launch can fail coverage mid-round. A space settlement records what its steps
// asked for, so the engine can give it retained cache pages and resume growth before the
// request's budget is bounded.
request.lease_settled        = true;
request.lease_space_limited  = space_limited;
request.lease_minimum_target = targets(page);
```

**三个字段各有各的读者**：

| 字段 | 语义 | 谁读 |
|---|---|---|
| `lease_settled` | 这个序列处于"租约不再增长"状态 | `settlement_tokens` / `shortfall` 的准入判据 |
| `lease_space_limited` | 原因是**空间**（不是别的） | 只有它为真才允许 resume |
| `lease_minimum_target` | 连"最小一步"都还差多少页 | `shortfall` 算缺口 |

`targets(page)` 存下来的是**最小一档的目标**——这就是"我还差多少才能恢复正常"的量。

## 九、引擎侧的第四级：缩答案预算，让 finish_reason 诚实

```cpp
// engine_core.h:2955-2961（apply_device_kv_lease_settlements 的头注释，全文）
// A sequence whose Device KV lease can no longer grow finishes at the frontier its lease
// covers. When the pool ran out of space, retained cache is released first so the answer can
// continue; only a lease that still cannot take its smallest step has its remaining budget
// bounded, so that finish carries the request's generation limit reason instead of the
// sequence running past its lease and failing a launch.
```

这段注释是整个机制的说明书，四级降级链完整地在这里：

```cpp
// engine_core.h:2975-2993
if (const auto shortfall = instance_.program->device_kv_lease_shortfall(*request->sequence)) {
    const std::uint32_t released = resources_.reclaim_device_kv_for_lease(
        *instance_.program, shortfall->main_pages, shortfall->backend_pages);   // ← 第三级：抢保留缓存
    if (instance_.program->resume_device_kv_lease(*request->sequence)) {
        if (released != 0) { publish_diagnostic(..., DiagnosticLevel::Info, "extended: released %u ..."); }
        continue;                                                              // ← 抢到了，继续跑
    }
    if (!request->lease_shortfall_reported) {
        request->lease_shortfall_reported = true;
        publish_diagnostic(..., DiagnosticLevel::Warning, "... still short %u main / %u backend pages for its smallest step, so the answer ends early with finish_reason length", ...);
    }
}
// ← 第四级：缩预算
const std::uint32_t control = request.output.control_suffix_tokens();
const std::optional<std::uint32_t> limit =
    instance_.program->device_kv_lease_settlement_tokens(*request->sequence, control);
if (limit) { request->budget->cap_remaining(*limit); }
```

**四个可学的细节**：

**① 诊断分级如实反映处置**：抢到 → `Info`（"extended"），抢不到 → `Warning`（"ends early with finish_reason length"）。而且 `lease_shortfall_reported` 保证**每个请求只报一次** —— 池饱和时这个函数每步都跑，不设标志会把日志刷爆。

**② 降级的方向是"缩答案"不是"报错"**。`cap_remaining(limit)` 让请求的剩余预算变小，于是 finish 走正常路径，`finish_reason = length`。**用户看到的是"答案变短了"，不是"服务不可用"。**

**③ 缩多少由 `device_kv_lease_settlement_tokens` 算**：

```cpp
// context.cpp:1704-1718
// The settle keeps the widest single step and the caller's forced control span unused, so
// every launch it still licenses stays inside the lease.
const std::uint32_t step = std::max(widest_verify_window() + 1U, forced_span_tokens);
const std::uint32_t covered = std::min(backend ? tokens_for_pages(backend_entitlement) : UINT32_MAX,
                                       tokens_for_pages(text_entitlement));
const std::uint32_t frontier = state.execution_frontier;
const std::uint32_t slack    = covered > frontier + step ? covered - frontier - step : 0U;
const std::uint32_t forced   = forced_span_tokens == 0 ? 0U : forced_span_tokens + 1U;
return std::max(std::max(1U, slack), forced);
```

`step` 取**最宽一次投机验证窗口 + 1**——注意是 `widest_verify_window()`，不是当前窗口。**预算是按最坏情况发的**，因为下一次 launch 可能真的用最宽窗口。

`covered` 取**两池租约的更小者**——两个池里先耗尽的那个决定总覆盖。

`max(1, slack, forced)` 兜底：**至少 1 token**。缩到 0 会让请求"还没开始就结束"，那是个更难解释的状态。

**④ 第四级与第三级的顺序不能换**：先抢缓存（快、无损），抢不到才缩答案（有损）。**在有损手段之前，先用完所有无损手段。**

## 十、`reclaim_device_kv_for_lease`：一次一个、最老优先、跳过有主的

```cpp
// resource_manager.h:1296-1360（结构）
std::uint32_t reclaim_device_kv_for_lease(Program& program, std::uint32_t main_pages,
                                          std::uint32_t backend_pages) {
    ...
    while (freed_main < main_pages || freed_backend < backend_pages) {
        const bool need_main    = freed_main < main_pages;
        const bool need_backend = freed_backend < backend_pages;
        std::optional<Victim> victim;
        std::tuple<std::uint64_t, std::uint64_t> oldest{max, max};
        const auto consider = [&](bool shared, std::uint32_t slot, std::uint64_t recency,
                                  std::uint64_t id, auto pages) {
            if (!((need_main && pages.main != 0) || (need_backend && pages.backend != 0))) { return; }
            const std::tuple<std::uint64_t, std::uint64_t> key{recency, id};   // ← 二级排序键
            if (key < oldest) { oldest = key; victim = Victim{...}; }
        };
        for (每个 private catalog slot) {
            if (entry.state != Catalogued || !entry.handle || private_has_active_edge(slot)) { continue; }
            consider(false, slot, owner_recency_epoch(entry), entry.id, program.retained_device_kv_pages(*entry.handle));
        }
        for (每个 shared catalog slot) {
            if (!shared_automatic_reclaimable(entry, slot)) { continue; }
            consider(true, slot, shared_recency_epoch(entry), entry.id, program.retained_device_kv_pages(*entry.handle));
        }
        if (!victim) { break; }
        const bool evicted = victim->shared ? reclaim_automatic_shared_prefix(program, victim->slot)
                                            : release_private_owner(program, victim->slot);
        if (!evicted) { break; }
        freed_main += victim->main;  freed_backend += victim->backend;  ++released;
    }
    return released;
}
```

**六个决策点**：

1. **`while` 而不是一次算完** —— 每次只驱一个，驱完重扫。因为 `retained_device_kv_pages` 要现算（`owner_exclusive_resources` 可能抛，接口是 `noexcept` 且异常时返回 `{}`，保守地不声称可回收）。**"一次一个 + 重算"比"预扫全表再批量驱"更容易正确**：每次驱逐都改变了别人的账。

2. **victim 排序键是 `(recency, id)` 二元组** —— recency 相同（同一 epoch 发布）时用 id 决出唯一受害者。**没有这个 tie-break，同一批次的多个候选会因遍历顺序不同而随机选一个，不可复现。**

3. **`need_main` / `need_backend` 参与筛选** —— 只找"能补上当前缺口那一种"的 victim。补 backend 缺口的不会去驱逐只占 main 页的条目。

4. **`private_has_active_edge(slot)` 跳过** —— 有活跃 lane 正指向的 catalog 条目不能抢。**这是"缓存 vs 活跃请求"的唯一分界线**：活跃请求占用的部分不是缓存。

5. **`program.retained_device_kv_pages(handle)`** 而不是读台账字段 —— 页数**现算**（`owner_exclusive_resources`）。台账可能滞后，现算是唯一可信的。

6. **`if (!evicted) break`** —— 驱不掉就退出，不无限循环。**失败退出而不是重试**：如果 `release_private_owner` 拒绝了（比如有 pin），重试一万次还是拒绝。

## 十一、可回收的四条件，与"扩展者不回收"

```cpp
// resource_manager.h:2291-2296
[[nodiscard]] bool shared_automatic_reclaimable(const SharedCatalogEntry& entry,
                                                std::uint32_t slot) const {
    return entry.state == SharedCatalogState::Catalogued && entry.handle &&
           entry.transaction_pins == 0 && shared_active_edge_count(slot) == 0 &&
           !entry.explicit_credit;
}
```

四个条件：**已上架** + **有句柄** + **无事务 pin** + **无活跃边** + **非人工保留（`explicit_credit`）**。

`explicit_credit` 是最有意思的一条：**有些条目是被人显式"记了信用"的，自动回收不许碰。** 这给运维留了一个"这个我保证它有价值"的出口——没有它，自动回收会把刚预热好的共享 prefix 在最坏的时刻抢走。

```cpp
// resource_manager.h:2298-2305
// A shared prefix the capturing request extends serves every request that capture could and
// more, so it is never reclaimed for that capture: the slot would narrow to one conversation.
[[nodiscard]] bool shared_reclaimable_for(const SharedCatalogEntry& entry, std::uint32_t slot,
                                          std::span<const PrefixShortlistKey> extended) const {
    return shared_automatic_reclaimable(entry, slot) &&
           std::find(extended.begin(), extended.end(), entry.summary.checkpoint.shortlist_key) == extended.end();
}
```

**一个"被捕获请求扩展过"的共享 prefix，对它自己的捕获者永不回收**——因为它服务的请求集是捕获者的**超集**，抢了它等于把共享前缀窄化成一段对话。

> 这条判据的形状值得单独记：**当 A 的能力严格包含 B 的能力时，为 B 牺牲 A 永远是净亏。** 判断"该不该驱逐"时，先问"受害者是不是比受益者更通用"。
> 数学上它就是 `extended ∋ entry.key` 的一个反向检查——**被扩展者不能成为扩展者的受害者。**

## 十二、可引用数字

| 项 | 值 | 位置 |
|---|---|---|
| 阶梯档数 | 3（`margin_tokens` / `2×64` / `64`） | `context.cpp:1667` |
| 最后一级 | 1 页组 = 64 token | `context.cpp:1666` |
| 两池策略 | 全做或全不做 | `context.cpp:1648-1657` |
| `target()` 上界 | 3 重（池容量 / 地址空间页容量 / reach） | `context.cpp:1607-1610` |
| settle 后最小保证 | `max(1, slack, forced)` token | `context.cpp:1718` |
| step 定义 | `max(widest_verify_window()+1, forced_span)` | `context.cpp:1709` |
| 降级级数 | 4（扩 → 抢缓存 → 缩预算 → length 收尾） | `engine_core.h:2955-2961` |
| 诊断频率 | 每请求 1 次（`lease_shortfall_reported`） | `engine_core.h:2979` |
| 回收粒度 | 一次一个 victim，驱完重扫 | `resource_manager.h:1302` |
| victim 排序键 | `(recency, id)` 二元组 | `resource_manager.h:1322` |

---

## 十三、六条可搬走的做法

1. **高频路径不用异常表达预期内结果**：池饱和时每步重进阶梯，算术预判是常态路径，`bad_alloc` 只兜"账面与实际之间没有原子性"这一层。
2. **失败归因先清理竞争原因**：`target()` 三重 clamp 的唯一目的是让"预留只可能因空间失败"，从而让 `space_limited` 这个布尔不说谎。
3. **分级申请的末级必须等于分配器最小粒度**，否则末级之下的空闲等于不存在。
4. **降级链先无损后有损**：扩 → 抢缓存（无损）→ 缩预算（有损）→ 诚实收尾。终态是"答案变短 + `finish_reason=length`"，不是"服务不可用"。
5. **驱逐用"一次一个 + 重算"**，不要预扫全表批量驱；排序键必须带 tie-break（`id`），否则同批次候选随遍历顺序漂移。
6. **被扩展者不能成为扩展者的受害者**：能力严格包含时，为窄的牺牲宽的是净亏。驱逐前先问"受害者是不是比受益者更通用"。

> 一句话收束这篇与 06–11 的关系：**06 讲了页怎么借、07 讲了副本怎么记账、08 讲了前缀怎么判等、09 讲了磁盘怎么寻址、10 讲了准入怎么报价、11 讲了状态怎么原子转移，12 讲的是"这些都算完还不够时，系统怎么体面地缩小自己"。** 前六篇是容量工程，这一篇是**降级工程**——而一个系统真正的健壮性，全在降级链的形状里。
