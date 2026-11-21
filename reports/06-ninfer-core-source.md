# 06 · NInfer 引擎本体：KV 页池的真实实现（源码级，非补丁级）

> **底座口径（2026-11-17 补）**：本文读的是 `iamwavecut/ninfer-all` 分支线 @`6cc1f9fa`（2026-11-04）的源码树。该线**不是上游主干**——`Neroued/ninfer` master 与它已分叉（`compare` 实测 `diverged, ahead_by=523, behind_by=1895, merge_base=f76e19c0fbd0`），且 2026-11-04 之后本线与上游都改过本文描述的部分机制（已核到 4 处，见 `notes/2026-11-17-ninfer上游与分支血缘核验.md`）。**引用本文任何断言请连带 sha 与日期。**

> 数据来源：本地解包的 **固定上游源码树** `kvmem-study/ninfer-src/`，commit `6cc1f9fa438b7094685ae417dfa40e1ca324e617`（即 kvmem `backends/versions.json` 钉的那个 base），tarball 16,577,465 bytes / 解包 94MB，来自 `https://codeload.github.com/iamwavecut/ninfer-all/tar.gz/6cc1f9fa…`。
> 与 01–05 号报告的区别：那五份读的是 kvmem 侧 `backends/patches/*.patch`，**看到的是"加了什么"**；本篇读的是引擎本体，**看到的是"为什么只能这么加"**。所有结论带 `path:line`。

---

## 一、页大小 64 是编译期常量，不是配置

```cpp
// kvmem-study/ninfer-src/src/core/paged_kv_cache.h:20
inline constexpr std::int32_t kPagedKVPageSize = 64;
```
```cpp
// kvmem-study/ninfer-src/src/ops/kernel/paged_kv_address.cuh:8-13
inline constexpr int kPagedKVPageShift = 6;
inline constexpr int kPagedKVPageMask  = kPagedKVPageSize - 1;
static_assert(kPagedKVPageSize == (1 << kPagedKVPageShift));
```

**结论**：decode kernel 用 `position >> kPagedKVPageShift` 取物理页号、`position & kPagedKVPageMask` 取页内偏移（`paged_kv_address.cuh:71-75` `paged_kv_physical_page`）。

这解释了 kvmem 文档里那句「ninfer 后端 block 固定 64（一块=一页）」：**不是懒得做成可配，而是页大小被编进了 kernel 的移位常量，改它等于重编译全部 attention/decode kernel**。任何想做"可调 block size"的人到这一行就会停手。

> 面试落点：判断一个参数是「设计选择」还是「架构约束」，看它是 runtime config 还是被 `static_assert` 钉进 kernel 的常量。前者能调参，后者只能重编。

---

## 二、两级分配：reserve（容量权）→ materialize（物理页）

```cpp
// paged_kv_cache.h:210-247  DeviceKVPageReservation：move-only，只持 pages_ 计数，没有物理页 id
// paged_kv_cache.h:305      [[nodiscard]] std::optional<DeviceKVPageReservation> reserve(std::uint32_t pages) noexcept;
// paged_kv_cache.h:316-319  void materialize(DeviceKVPageReservation&, std::uint32_t target_page_count,
//                            std::vector<DeviceKVPageLease>& destination,
//                            std::optional<DeviceKVPageHandle> preferred_predecessor = std::nullopt);
```

分两级的理由写在代码里：reserve 只承诺**容量**（admission 阶段就能判能不能接这个请求），materialize 才绑定**物理页号**（真正要写 KV 时）。中间这段时间预留占着账不占页。

**这就是 KVMem「预算 B」能作为可声明契约的技术根**：请求进来先 `reserve(B)`，失败即拒（抛 `ContextCacheExhausted`，见 `paged_kv_cache.cpp:513-520`），成功则后续一定物化得出来——不会跑到一半 OOM。

### 预算的唯一强制点：`cover()`

```cpp
// paged_kv_cache.cpp:463-482
bool DeviceKVPagePool::cover(std::uint32_t extra) noexcept {
    const auto free_run_pages = [&] {
        return std::uint64_t(capacity_pages()) - lent_pages_ - loaned_pages_ - allocated_pages_;
    };
    const std::uint64_t need = std::uint64_t(reserved_pages_) + extra;
    if (need <= free_run_pages()) { return true; }
    if (!recall_ || loaned_pages_ == 0) { return false; }
    try { recall_(static_cast<std::uint32_t>(need - free_run_pages())); }
    catch (...) { return false; }
    return need <= free_run_pages();
}
```

四个计数必须同时满足，任何一个算漏都会让预算失真：`capacity`（物理上限）、`lent`（借给窗口了）、`loaned`（借给 expert cache 但可召回）、`allocated`（已物化）。**不足时先调 `recall_` 钩子强行要回 loaned 页，要回失败才拒**——这是"宁可踢掉别人的缓存，也不让当前请求失败"的取向。

> 值得抄的工程细节：配额判定集中在一个纯函数里，返回 bool 而非抛异常，`reserve` 再转成 `optional`。好处是 `noexcept` 路径可测、可断言，且调用方不会在"能不能"这一层看到异常。

### 句柄防悬挂：per-page generation

```cpp
// paged_kv_cache.h:160-179  DeviceKVPageHandle { owner_, index_, generation_ }
// paged_kv_cache.cpp:671-683 valid_handle / physical_index
```
每页一个 `page_generations_`（h:399），句柄带 generation，页被释放再分配后旧句柄 generation 不匹配即失效。**这是 CUDA 侧 use-after-free 的结构性防线**——不是靠调用者小心，是靠句柄不可伪造（构造函数全 private，只 friend 给池和 Lease）。

---

## 三、`materialize` 的连续性偏好不是优化，是可借性的前提

```cpp
// paged_kv_cache.cpp:549-566
std::optional<std::int32_t> preferred;
if (preferred_predecessor) { preferred = physical_index(*preferred_predecessor) + 1; }
else if (!destination.empty()) { preferred = destination.back().index_ + 1; }
// 先试 preferred 所在 run 的后缀，够则取；不够再 find_if 找任意 count >= run.count 的 run；
// 最后才把一个请求拆到多个 run 上。
```

为什么不直接第一个合适 run 拿走？因为 **lend 要求连续**：

```cpp
// paged_kv_cache.h:271-274
// Device bytes one plane devotes to a run of consecutive pages. Page-major geometry only:
// a run is one contiguous block there, which is what makes a run lendable. Lending exists for a
// single device, so a pool spanning several ranks refuses it.
[[nodiscard]] KVPlaneByteRange plane_page_range(std::size_t plane, std::int32_t first_page, std::uint32_t count) const;
```

**推论（这份报告最重要的新发现）**：分配策略直接决定"有多少内存可借"。碎片化的页池即使空闲页总数够，也借不出去——因为 VMM 只能 unmap 整颗 granule、且必须落在**连续**页里。所以 `materialize` 的连续性偏好同时服务两个目的：decode 的 strided 拷贝，和窗口的可借性。

---

## 四、lend 与 loan 是两种不同强度的"借"

```cpp
// paged_kv_cache.h:286-296
void lend_pages(std::int32_t begin, std::uint32_t count);    // 整段移出循环，内存交出去
void return_pages(std::int32_t begin, std::uint32_t count);
void loan_pages(std::int32_t begin, std::uint32_t count);    // 可召回借出
void recall_pages(std::int32_t begin, std::uint32_t count);
void set_recall(std::function<void(std::uint32_t pages)> recall);
[[nodiscard]] std::uint32_t unpromised_free_pages() const noexcept;
```

注释把差别写清楚了（h:291-296）：

- **lend**：run 从 free runs 里彻底移除，admission 看不见这块内存 → 用于「窗口租一段显存跑别的」。
- **loan**：页**仍算 usable/available**，admission 与容量规划都当它还在 → 用于「expert cache 蹭空闲 KV 内存」。
- **loan 的上限是 `unpromised_free_pages()`**（h:298-299「Free pages beyond what the reservations will materialize」），`loan_pages` 第一行就断言（cpp:436-438 `throw std::invalid_argument("Paged KV loan takes pages a reservation needs")`）。

> 这是"能借多少"的唯一正确算法：不是"空闲多少借多少"，而是"**空闲减去未来一定会物化的预留**"。少减这一项，profiling 时就会出现「明明有空闲页，请求却 OOM」。

---

## 五、批量溢出/回灌：把 N 次 memcpy 合成 1 次 strided 传输

```cpp
// paged_kv_cache.h:369-390
void copy_to_host_records(std::span<const DeviceKVPageHandle> source,
                          std::span<std::byte* const> records,
                          std::span<const std::uint32_t> record_groups,
                          const HostKVPageLayout& layout, RankStreams streams = {}) const;
// 三个约束（h:375-380 注释）：
//  ① 连续物理页且 record 以恒定 pitch 前进 → 合成一次 strided transfer
//  ② 必须落在同一个 pinned allocation 内（由 record_groups 标注）
//  ③ 距离不得超过 device 的 max copy pitch
void copy_from_host_records(..., std::size_t plane_begin, std::size_t plane_end, RankStreams);  // h:386-390
```

**两个可抄的设计**：

1. `record_groups` 是**显式的分配归属标注**，不是靠地址猜。跨 pinned 区块的 strided 拷贝会静默出错（CUDA 对 pitch 有上限），代码选择"宁可不合并"，而不是"合并后祈祷"。
2. 按 `[plane_begin, plane_end)` 分段的回灌版本，注释说目的是「让调用方把它消费方需要的层（模型的 per-layer planes）排在前面」（h:384-385）。**这就是 03 号报告里「逐层召回」在拷贝层的落点**——分层不是调度层的花活，是 memcpy 顺序的参数。

---

## 六、可驱逐 arena：租借期间**不 quiesce stream**

```cpp
// src/core/evictable_kv_pool.h:16-30
// ① 每个 store 在自己今天用的偏移上绑定 → plane 指针、block table、已捕获的 decode graph 全部保持有效
// ② 借出去只含 free 页，且「no page is read before the append kernel writes it」→ close 时只重映射，不拷任何数据
// ③ 「Neither lease() nor close() quiesces a stream」→ 窗口打开期间 decode 照跑
```

这三句是**整个 KV 虚拟化能"零拷贝"的全部理由**，而且依赖的是一条 kernel 侧不变量：**free 页在被写之前绝不会被读**。

这条不变量在同一份代码里被**第二次**使用：

```cpp
// src/models/qwen3_5/program/residency.cpp:49-52
// Free KV pages are written before they are read, so a suspend skips them; on an idle server
// they are most of the persistent arena.
std::vector<std::vector<ByteExtent>> holes(device.size());
const auto add_holes = [&](const DeviceKVPagePool& pool) {
    for (const auto& extent : pool.free_extents()) { ... }
};
```

**suspend 时把空闲页当"洞"从快照里挖掉**，用同一个 `free_extents()`（h:280-288）。空闲服务器上这些洞是持久 arena 的绝大部分——所以 suspend 快照能小得多。

> 这是最值得学的一条：**一条不变量支撑两个特性**（可借性 + 快照瘦身）。设计时先找到这种"既安全又省事"的不变量，比堆两个独立机制强得多。

### suspend 保持地址不变

```cpp
// src/core/evictable_kv_pool.h:96-102
// Model suspend: every physical piece goes back to the device while the home addresses stay
// reserved, so the pointers and graphs that captured them stay valid.
void release_backing();
void restore_backing();
```
`release_backing` 释放物理 backing 但**保留虚拟地址预留**，所以 CUDA Graph 捕获的指针不失效。`restore_backing` 重新映射后的内容未定义，等 owner 重写。

---

## 七、故障注入是设计的一部分（不是测试后补的）

```cpp
// src/core/evictable_kv_pool.h:117-122
enum class LeaseFault : std::uint8_t {
    None,
    Unmap,   // before the piece leaves its home mapping
    Overlay, // after the home unmap, before the overlay mapping exists
    Access,  // after the overlay mapping exists, before the piece counts as lent
};
```

三个断点精确落在「单颗 granule 搬移」的**三个阶段之间**，且 `lease()` 的契约是：**中途失败会把每一颗都送回原映射再抛**（h:110-113「A failure part-way through returns every piece to its home mapping before it throws, so a refused lease leaves the arena exactly as it found it and the caller may try again」）。

**这跟我自己 AGENTS.md 第 8/21/37 条是同一套纪律**：门必须能红（这里=租借必须能失败且失败后状态可复原），失败路径必须是显式可注入的枚举，而不是靠 `#ifdef` 或随机故障。

---

## 八、跨平台的真实雷：Windows 上 pinned host 会吃掉显存预算

```cpp
// src/core/host_kv_clamp.h:11-23
// The pinned host KV cache competes with the model for *device* memory on Windows (WDDM maps a
// pinned host allocation into the GPU's address space and charges it against the same budget)
inline std::size_t clamp_host_kv_reservation_bytes(std::size_t requested_bytes,
                                                   std::size_t free_device_bytes,
                                                   std::size_t minimum_stride) noexcept {
    constexpr std::size_t kPinnedHostReserveBytes = 1ULL << 30;
    const std::size_t budget = free_device_bytes > kPinnedHostReserveBytes
                                    ? (free_device_bytes - kPinnedHostReserveBytes) / 2 : 0;
    if (requested_bytes > budget) { return budget - (budget % minimum_stride); }
    return requested_bytes;
}
```

**在 Windows 上，`cudaHostAlloc` 出来的 pinned 内存会被 WDDM 映射进 GPU 地址空间并计入显存预算**。所以「把 KV 历史放 host RAM」这个方案在 Windows 上并不免费——它要占显存，而且代码直接把它砍到 `(free_device - 1GiB) / 2`。

**这条对任何做 CPU offload 的人都是硬约束**：Linux 上 "host KV 免费" 的心智模型搬到 Windows 会直接 OOM，而且现象是"显存莫名其妙被吃了"。函数纯算术、无 CUDA 依赖，注释写明配套测试在 `tests/host/test_host_kv_clamp.cpp`。

---

## 九、借还计划：从最高页开始借

```cpp
// src/core/kv_loan.h:10-17  KVLoanPlan { runs, granules, bytes }
// src/core/kv_loan.h:19-21  kMaxLoanUnitPages = 256
// src/core/kv_loan.h:81-95  plan_kv_loan(...)
```

三个判据：

1. **只收完全落在借用页内的 granule**（h:44-46「Only granules that lie entirely inside the borrowed pages are collected, so a granule shared with a live page is never unmapped」）——共享 granule 永不被 unmap。
2. **每 granule 需要超过 256 页的 plane 不借**（h:19-21）：那些是极小的 scale plane，借出去省不了多少字节，注释算过账「cost the loan nothing but a few per cent of yield」。
3. **从最高的空闲页开始借**（h:88-92「the highest pages go first, which is where the allocator leaves the pool free the longest」）——按"这块空闲多久了"排序，而不是按大小。这是 LRU 思想落在物理页布局上。

---

## 十、源码里还有文档没说全的东西

| 发现 | 位置 | 与文档的关系 |
|---|---|---|
| MoE expert cache 借 KV 空闲内存 | `src/core/kv_expert_loan.h` / `kv_expert_loan.cpp`（`DeviceKVPagePool` 的消费方之一） | 文档说 ninfer 后端只覆盖 Dense（MoE 走 PolyStrata），但**引擎本体已有 expert loan 基础设施**——说明"只在 Dense"是 kvmem 后端适配层的选择，不是引擎能力边界 |
| 多 rank 位置切分 + 每 rank 独立 execution table | `paged_kv_cache.h:130-140` `plan_device_kv_page_pool(std::span<LayoutBuilder* const>, std::span<const std::size_t> plane_rank, ...)`、`:143-147` `plan_kv_execution_tables` | 「DFlash2 双卡按层拆分」的底层：pool 的句柄与页号 rank 无关，上层（admission/prefix reuse/context cache）完全不知道有几张卡 |
| disk tier 预算 65/25/10 | `src/core/disk_kv_bridge.h:64` | 比 03 号报告的 `--kvmem-disk-mib` 单参数细：总量按 main/backend/state 三分（无 backend 时 85/15） |
| 分层可行性判定 | `src/core/stage_plan.h:197-260` `stage_fits` / 全部 budget 上的切分可行性 | 「DFlash2 按层拆分」的规划侧：判 `[begin,end)` 层在 `budgets[stage]` 上能不能装下 `page_groups` |

上层三个大文件（本次只数了行数，未逐行读）：`program/storage/context.cpp` 1889 行、`program/prefix/hybrid_program.cpp` 1610 行、`program/storage/session_snapshot.cpp` 1175 行。

---

## 十一、这次源码级阅读**推翻/修正**了此前五份报告的哪些说法

1. **「block 固定 64」不是软件限制而是编译期常量**——此前只能从文档转述，现在知道改动代价是重编 kernel（`paged_kv_address.cuh:8-13` 的 `static_assert`）。
2. **「预算 B」的强制点只有一处**：`cover()`（`paged_kv_cache.cpp:463-482`）。此前说"B+R 可声明契约"，现在能指到具体那个函数和它用的四个计数器。
3. **03 号报告说逐层召回的代价主要在拷贝**——源码确认拷贝层专门开了 `[plane_begin, plane_end)` 的重载（`paged_kv_cache.h:386-390`），即"层序"是 memcpy 的参数而非调度决策。
4. **新增一条 05 号报告（Radiance/AMD）没提的跨 ISA 差异**：`host_kv_clamp.h` 的 Windows/WDDM 约束与 GPU 厂商无关但与**驱动模型**有关，ROCm/HIP 侧有没有对应约束本次未验（UNKNOWN）。

---

## 十二、可直接引用的数字（全部来自本次本地字节）

| 项 | 值 | 来源 |
|---|---|---|
| 源码树 commit | `6cc1f9fa438b7094685ae417dfa40e1ca324e617` | tarball 路径名 |
| tarball 大小 | 16,577,465 bytes | `ls -la /tmp/ninfer-all.tar.gz` |
| 解包大小 | 94 MB | `du -sh kvmem-study/ninfer-src` |
| KV 核心文件行数 | `paged_kv_cache.h` 560 / `.cpp` 1412 / `paged_kv_storage.h` 123 / `paged_kv_window.cpp` 44 / `paged_kv_window.cuh` 36 / `paged_kv_address.cuh` 105 | `wc -l` |
| 页大小 | 64（编译期常量） | `paged_kv_cache.h:20` |
| 预算计数器 | capacity / lent / loaned / allocated / reserved 五个 | `paged_kv_cache.cpp:463-482`、`paged_kv_cache.h:259-265` |
| hosted 夹紧 | `(free_device − 1GiB) / 2`，向下取整到整页 | `host_kv_clamp.h:19-22` |
| 借还最大单位 | `kMaxLoanUnitPages = 256` 页/granule | `kv_loan.h:19` |
| disk tier 预算分配 | 65/25/10（无 backend 时 85/15） | `disk_kv_bridge.h:64` |
