# 15 · `stage_plan.h`：多卡分层不是均分，是先解容量上界再求均衡

> **底座口径（2026-11-17 补）**：本文结论只依赖 `stage_plan.h` 的整数运算与模型文件实数，**不受**分支线变更影响（该文件在 `6cc1f9fa` 与上游 HEAD 均存在）。背景见 `notes/2026-11-17-ninfer上游与分支血缘核验.md`。

> 04 号实验计划里 B（page groups）是自变量、要扫。这篇解释 **B 的上界是怎么来的**——它不是配出来的，是解出来的：先解出"所有 stage 同时能装下的最大页组数"，再在能达到这个数的所有切法里挑最均衡的那个。
> 源码：`src/core/stage_plan.h` 329 行（`StagePlan :29-146`、`StageBudget/StageSolution/LayerCost :150-166`、`StageCosts/prefix_costs :168-187`、`saturating_add/mul :188-200`、`stage_fits :202-213`、`feasible :217-240`、`solve_stage_plan :246-329`）。
> 一句话结论：**`solve_stage_plan` 是两级精确解——第一级用倍增+二分求"容量"的极大值（feasibility 对页组数单调），第二级用 min-max DP 求"均衡"的最优值（在所有达到极大容量的切法里）。两个目标字典序，都不靠启发式。**

---

## 一、前提：一个 stage 拥有完整的层

文件头注释（`:3-8`）先把语义钉死：

```
// A stage owns its layers whole: weights, the KV cache of its attention layers and the recurrent
// state of its GDN layers all live on the stage's device. That is what lets a stage run its layers
// with the same kernels as a single GPU and cross to the next stage once, at a layer boundary.
//
// A one-stage plan is the identity mapping, which keeps every consumer a no-op on one GPU.
```

**三样东西必须一起搬迁**：权重、该层 attention 的 KV 缓存、GDN（linear attention）层的 recurrent state。

**为什么不能拆**：拆了之后 stage 就没法用和单卡一样的 kernel——KV 在别的卡上，每次 attention 都要跨设备取。而"整层拥有"之后，**跨设备只发生在层边界，每个边界恰好一次**：

```cpp
// stage_plan.h:126-129
// True when the residual stream must move to another stage after this layer. Exactly one per
// stage boundary, which is what makes a layer split tolerant of a slow link.
[[nodiscard]] bool crosses_after(std::uint32_t global_layer) const {
    if (global_layer + 1 >= layer_count_) { return false; }
    return placement(global_layer).stage != placement(global_layer + 1).stage;
}
```

> `// which is what makes a layer split tolerant of a slow link.`
>
> **对慢链路容忍的原因不是"传得少"，是"传的次数确定且最少"**：34 层切 2 段 = 1 次传输，而不管每段多少层。pipeline parallel 的全部价值在这个不变量上。

以及单卡时必须是恒等映射（`single_stage()`），这样所有跨设备代码在单卡上是 no-op——**不为多卡场景在单卡路径上加分支**。

## 二、`StagePlan` 的三条构造不变量

```cpp
// stage_plan.h:46-59
StagePlan(std::uint32_t layer_count, std::vector<std::uint32_t> boundaries)
    : layer_count_(layer_count), boundaries_(std::move(boundaries)) {
    if (layer_count_ == 0) { throw std::invalid_argument("stage plan needs at least one layer"); }
    if (boundaries_.empty()) { throw std::invalid_argument("stage plan needs a boundary"); }
    if (boundaries_.back() != layer_count_) {
        throw std::invalid_argument("stage plan must cover every layer");
    }
    std::uint32_t previous = 0;
    for (const std::uint32_t end : boundaries_) {
        if (end <= previous) {
            throw std::invalid_argument("stage plan stages must each own at least one layer");
        }
        previous = end;
    }
}
```

① **必须覆盖每一层**（`back() == layer_count`）② **严格升序**（`end > previous`，所以每个 stage 非空）。

第二条的注释（`:36-38`）说清了它的性质：

```
// boundaries[i] is the exclusive end of stage i: strictly ascending, last == layer_count. A
// stage cannot be empty; a device that would hold nothing is a planning mistake, not a
// configuration.
```

> `// a device that would hold nothing is a planning mistake, not a configuration.`
>
> **把"空 stage"当成错误而不是允许**——因为一个不装东西的设备仍然占着一个通信跳，仍然增加一次跨设备传输，却什么也没分担。允许它就等于允许用户写出"用 8 卡跑成 7 卡"的配置。

`from_layer_counts`（`:63-82`）接受 `--stage-layers 30,34` 这种显式指定，**求和必须精确相等**，且报错信息带具体数字：

```cpp
throw std::invalid_argument("stage layer counts sum to " + std::to_string(total) +
                            " but the model has " + std::to_string(layer_count) + " layers");
```

`even`（`:85-95`）把余数摊到**前面的** stage（`(layer_count * (stage+1)) / stages` 的整数除法自然产生这个效果）。

> 报错带数字这件事值得单说：**"配置不对"和"你写了 60 层但这个模型有 34 层"是两个不同质量的错误信息。** 前者要用户自己去数，后者直接指出差在哪。

## 三、成本模型：一个 stage 的三项开销

```cpp
// stage_plan.h:150-161
struct StageBudget {
    // Bytes free on the device before anything of the model is placed on it.
    std::uint64_t available_bytes = 0;
    // Bytes the stage needs whatever layers it owns: the embedding on the first stage, the head and
    // prefill buffers on the last, workspace, graph memory.
    std::uint64_t fixed_bytes = 0;
    // KV bytes per page group the stage holds whatever layers it owns: MTP's layer on the first.
    std::uint64_t kv_bytes_per_page_group = 0;
};
```

**`fixed_bytes`**：不管装哪几层都要占的——第一段的 embedding、最后一段的 head 与 prefill buffer、workspace、graph memory。

**`kv_bytes_per_page_group`**：不管装哪几层都要占的**每页组** KV——比如 MTP 的那一层固定落在第一段。

于是判定函数：

```cpp
// stage_plan.h:203-213
inline bool stage_fits(const StageCosts& costs, const StageBudget& budget, std::uint32_t begin,
                       std::uint32_t end, std::uint64_t page_groups) {
    std::uint64_t used = budget.fixed_bytes;
    used = saturating_add(used, costs.weight_prefix[end] - costs.weight_prefix[begin]);
    used = saturating_add(used, costs.state_prefix[end] - costs.state_prefix[begin]);
    const std::uint64_t kv = saturating_add(costs.kv_prefix[end] - costs.kv_prefix[begin],
                                            budget.kv_bytes_per_page_group);
    used = saturating_add(used, saturating_mul(kv, page_groups));
    return used <= budget.available_bytes;
}
```

四行相加：`fixed` + 权重（区间前缀和差） + recurrent state（同） + `(该段每页组 KV + 固定每页组 KV) × 页组数`。

**注意 `kv_bytes_per_page_group` 在括号里、乘的是 `page_groups`**——它不是每层都加的固定项，是**每个页组加一次的固定项**。所以 `page_groups = 0` 时它完全消失。

**前缀和的设计**（`:174-184`）：`weight_prefix` / `kv_prefix` / `state_prefix` 三个数组，`[begin, end)` 的成本 = `prefix[end] - prefix[begin]`，O(1) 查询。

> 这是区间成本 DP 的标准预处理：**把 O(n) 的区间求和变成 O(1)，让 O(stages × layers²) 的 DP 变得在毫秒级可行。** 没有它就得在每个 DP 转移里求一次和。

## 四、安全算术：溢出必须读成"放不下"

```cpp
// stage_plan.h:188-193
inline std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) {
    return a > std::numeric_limits<std::uint64_t>::max() - b
               ? std::numeric_limits<std::uint64_t>::max()
               : a + b;
}

// stage_plan.h:195-200
inline std::uint64_t saturating_mul(std::uint64_t a, std::uint64_t b) {
    return (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
               ? std::numeric_limits<std::uint64_t>::max()
               : a * b;
}
```

注释（`:201-202`）说明了为什么：

```
// Whether layers [begin, end) fit on `budget` while holding `page_groups` page groups. Saturating,
// so a huge page-group count reads as "does not fit" instead of wrapping around to "fits".
```

**`saturating_mul` 挂在乘法上不是洁癖**：`kv * page_groups` 里 `page_groups` 是二分搜索的探针值，`high *= 2` 会一路乘到 `1 << 62`。真做乘法会回绕成一个**小**数，于是 `used <= available` 成立，`feasible` 返回 true——**二分搜索会朝"更多页组"的方向走到底，最后一个运行期 OOM。**

> `(a != 0 && b > MAX / a)` 这个写法本身就是"除法代替乘法来检测乘法溢出"：**先除后比，不会溢出。** 这也解释了为什么 `a != 0` 必须先判（除零）。

## 五、`feasible`：可达性 DP，明确拒绝贪心

```cpp
// stage_plan.h:217-240
inline bool feasible(const StageCosts& costs, std::span<const StageBudget> budgets,
                     std::uint32_t layer_count, std::uint64_t page_groups) {
    // reachable[end]: the first `stage` stages can own exactly layers [0, end).
    std::vector<char> reachable(layer_count + 1, 0);
    reachable[0] = 1;
    for (std::size_t stage = 0; stage < budgets.size(); ++stage) {
        std::vector<char> next(layer_count + 1, 0);
        for (std::uint32_t begin = 0; begin < layer_count; ++begin) {
            if (reachable[begin] == 0) { continue; }
            for (std::uint32_t end = begin + 1; end <= layer_count; ++end) {
                if (!stage_fits(costs, budgets[stage], begin, end, page_groups)) { break; }
                next[end] = 1;
            }
        }
        reachable = std::move(next);
    }
    return reachable[layer_count] != 0;
}
```

`reachable[end]` = "前 `stage` 个 stage 恰好拥有层 `[0, end)`"。每轮把可达状态推到下一轮。最终 `reachable[layer_count]` 表示全部层被恰好切完。

**为什么不用贪心**（`:215-216` 的注释，值得逐字读）：

```
// Exact rather than greedy. Filling early stages as full as possible is not safe once
// every stage must own a layer: it can leave a small middle stage no choice but to take a layer it
// cannot hold, when a slightly different cut would have fit. The instance is tiny (layers x stages
// squared), so the search costs nothing.
```

**贪心反例的形状**：前面几段尽量装满 → 中间的小 stage 只剩一个"必须接一层"的位置，而那一层它装不下 → 整体判不可行。但稍微少给第一段一层，中间段就能接两层里便宜的那个，整体可行。

> `// The instance is tiny (layers x stages squared), so the search costs nothing.`
>
> **这句话是"什么时候可以精确解"的正确判据**：不是"精确解总是更好"，而是 **当实例规模小到精确解的成本可以忽略时，精确解严格优于启发式**。34 层 × 4 段 = 34²×4 ≈ 4600 次判定，微秒级。

**单调性剪枝**：`if (!stage_fits(...)) break;` —— 因为 `end` 递增时区间只会变长、成本只会变高（`prefix[end] - prefix[begin]` 对 `end` 单调增），所以**第一个不 fit 的 `end` 之后全都不 fit**。这把内层循环从 O(layers) 的平均情况降到"能装多少就试多少"。

## 六、`solve_stage_plan`：两级精确解

```cpp
// stage_plan.h:246-260
inline StageSolution solve_stage_plan(std::span<const LayerCost> layers,
                                      std::span<const StageBudget> budgets) {
    if (layers.empty()) { throw std::invalid_argument("stage solver needs at least one layer"); }
    if (budgets.empty()) { throw std::invalid_argument("stage solver needs at least one stage"); }
    if (budgets.size() > layers.size()) { throw std::invalid_argument("more stages than layers"); }
    const auto layer_count = static_cast<std::uint32_t>(layers.size());
    const detail::StageCosts costs = detail::prefix_costs(layers);

    if (!detail::feasible(costs, budgets, layer_count, 0)) {
        throw std::runtime_error(
            "the model's weights do not fit on the given devices even with no KV cache");
    }
```

### 第一级：容量极大值

```cpp
//  Largest page-group count that is still feasible. Feasibility is monotone in the count.
constexpr std::uint64_t kUnbounded = std::numeric_limits<std::uint64_t>::max();
const bool any_kv = costs.kv_prefix.back() != 0 ||
                    std::any_of(budgets.begin(), budgets.end(), [](const StageBudget& budget) {
                        return budget.kv_bytes_per_page_group != 0;
                    });
std::uint64_t best = kUnbounded;
if (any_kv) {
    std::uint64_t low  = 0;
    std::uint64_t high = 1;
    while (high < (std::uint64_t{1} << 62) &&
           detail::feasible(costs, budgets, layer_count, high)) {
        low = high;
        high *= 2;
    }
    while (low + 1 < high) {
        const std::uint64_t middle = low + (high - low) / 2;
        if (detail::feasible(costs, budgets, layer_count, middle)) { low = middle; }
        else { high = middle; }
    }
    best = low;
}
```

**先倍增找上界、再二分**（exponential + binary search）。这是"在无界整数域上找单调谓词的极大值"的标准做法：倍增给出 `[low, high)` 区间，二分在区间内收敛。

**`(1 << 62)` 的上界**：给 `high *= 2` 一个不溢出的天花板。加上 `saturating_mul`，即使逼近这个数也不会回绕。

**`any_kv` 的门**：若所有层 `kv_bytes_per_page_group` 都是 0、且所有 budget 的 `kv_bytes_per_page_group` 也是 0，则**容量与页组数无关** → `best = kUnbounded`（`StageSolution` 的注释说 `Unbounded (the maximum value) when no layer has a KV cost`），后续用 `at = 0` 做 DP。

**这一步就是 B 的上界**。命令行的 `--max-batch` 之类只能**往下压**，不能超过它。

### 第二级：均衡 min-max DP

```cpp
// Among the splits that hold `best` page groups, minimise the fullest stage. Exact DP over
// (stages placed, layers placed): the minimum achievable maximum utilisation.
const std::uint64_t at = any_kv ? best : 0;
...
auto utilisation = [&](std::size_t stage, std::uint32_t begin, std::uint32_t end) -> double {
    if (!detail::stage_fits(costs, budgets[stage], begin, end, at)) { return kInfinity; }
    const long double used =
        static_cast<long double>(budgets[stage].fixed_bytes) +
        static_cast<long double>(costs.weight_prefix[end] - costs.weight_prefix[begin]) +
        ... ;
    const long double capacity = static_cast<long double>(budgets[stage].available_bytes);
    return capacity > 0 ? static_cast<double>(used / capacity) : kInfinity;
};
```

**DP 状态 `dp[stage][end]` = "用 `stage+1` 个 stage 覆盖 `[0, end)` 时，最满那个 stage 的最小利用率"**，转移取 `max`（最满的），外层取 `min`（最小化）：

```cpp
const double value = std::max(dp[stage][begin], u);
if (value < dp[stage + 1][end]) {
    dp[stage + 1][end]   = value;
    from[stage + 1][end] = begin;
}
```

**这是经典的 min-max 划分 DP**：目标是"让最大的那块尽量小"。`std::max` 在转移里（继承上一段的最满值、与本段取最大），`<` 在比较里（选更小的）。

**`long double` 不是多余**：`used` 是几十 GB 的字节数，`used / capacity` 若用 `uint64` 整数除会全部变成 0 或 1，DP 就没有区分度了。用 `long double` 保留小数份额。

> **"最均衡"是第二目标，不是与容量并列的目标**——`at = best` 固定之后才求均衡。如果反过来（先均衡再容量），一个漂亮的均分可能只装 100 页组，而牺牲一点均衡能装 800。

### 回溯

```cpp
std::vector<std::uint32_t> boundaries(stages);
std::uint32_t end = layer_count;
for (std::size_t stage = stages; stage > 0; --stage) {
    boundaries[stage - 1] = end;
    end                   = from[stage][end];
}
return StageSolution{StagePlan(layer_count, std::move(boundaries)), best};
```

从最后一层往回走 `from` 表，得到每段的边界。返回 `StageSolution{plan, best}` —— **切法与它对应的容量一起返回**，调用方不需要再算一遍。

## 七、对 04 号实验计划的直接后果

这是本篇报告与本项目实验设计的连接点：

1. **B 的上界是解出来的，不是配的。** 04 号计划里"扫 B = 64/128/256/512…"必须以 `solve_stage_plan` 返回的 `page_groups` 为天花板——超过它就 OOM 而不是变慢。
2. **B 与切法耦合。** 不同的 `boundaries` 给出不同的 `best`。所以我如果要固定 B 扫别的变量，必须**先固定切法**（用 `--stage-layers` 显式指定），否则求解器会为每个 B 选不同切法，把"切法效应"和"B 效应"混在一起。
3. **`stage_fits` 是纯算术，无 CUDA 依赖。** 这意味着我**不需要 GPU 就能算出"这张卡最多能装多少页组"**——在 Phase B 拿不到 GPU 主机的现在，这个函数可以在本机直接跑（给 `StageBudget{available_bytes, fixed_bytes, kv_bytes_per_page_group}` 填数即可）。
4. **均衡目标会在 B 接近上界时主导结果。** B 小时切法自由、可以很均衡；B 逼近 `best` 时切法被容量锁死，最满的卡必然接近 100%。所以 B×ctx 曲线在接近上界处会有拐点。

## 八、可引用数字

| 项 | 值 | 位置 |
|---|---|---|
| 求解器规模 | O(stages × layers²)，34 层×4 段 ≈ 4.6e3 次判定 | `:218` 注释 |
| 倍增上界 | `1 << 62` | `:275` |
| 均衡 DP 精度 | `long double` 中间量，`double` 存储 | `:299-305` |
| `StagePlan` 不变量 | 3 条（覆盖全层 / 严格升序 / 非空） | `:46-59` |
| 成本项 | 3 类前缀和 + 2 类固定项 | `:150-161` |

---

## 九、六条可搬走的做法

1. **双目标要字典序，不要加权**：先解"最大容量"（可判定、单调、能二分），再在等价类里解"最均衡"（min-max DP）。加权合成会把两个不同量纲的东西（页组数 vs 利用率）混在一起，且无法回答"我这个 B 是不是天花板"。
2. **体积小就精确解**：`// The instance is tiny … so the search costs nothing` —— 判断标准是实例规模，不是"精确解是否更优雅"。贪心在"每段非空"约束下会给出假不可行。
3. **安全算术必须挂在乘法上**：溢出回绕会让"不可能"读成"完全可行"，二分搜索于是朝错误方向走到底，最后变成运行期 OOM。
4. **成本用前缀和，判定用区间差**：O(n) 预处理把 O(layers) 的区间求和变成 O(1)，让 O(stages × layers²) 的 DP 落到毫秒级。
5. **把"空单元"当错误而不是配置**：一个不装东西的设备仍占一个通信跳，允许它等于允许"用 8 卡跑成 7 卡"。
6. **切法与它的容量一起返回**：`StageSolution{plan, page_groups}` 让调用方不必知道求解器内部用了哪个 `at`；同时也提醒调用方——**这两个值是一体的，单独引用其中一个是错的**。

> 与 06 号报告的关系：06 讲 `plan_device_kv_page_pool` 是**运行时**按切法给每个 rank 分配 KV 池（`span<LayoutBuilder* const>` + `plane_rank`），15 讲**规划期**先决定切法与容量上界。**先是静态的 `stage_plan` 给出 B，再是 `kv_loan`/`paged_kv_cache` 在这个 B 里做动态分配。** 两者的接口就是 `StageSolution::page_groups`。
