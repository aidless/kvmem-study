# KVMem spread 检查点策略机制拆解

> 研究对象：github.com/kvmem/kvmem（KV 内存虚拟化层，host RAM + SSD 分层），spread 检查点策略（v0.19.1 特性）。
> 方法：只读精读 `docs/checkpoint-policy.md`（498 行）、`docs/checkpoint-policy-validation-20261025.md`（610 行）、`docs/ninfer-cross-owner-validation-20261029.md` 与 `kvmem/`、`tools/`、`backends/patches/` 源码。所有论断带相对仓库根的 `path:line`；源码未覆盖处如实标注。
> 日期：2026-11-10。

---

## 1. TL;DR

spread 的本质：把「检查点保留」从 legacy 的「只留会话末梢」扩展为「沿 prompt 前缀按 interval 留一组带角色的覆盖点（root / anchor / periodic / tail / input / endpoint），RAM 元数据常驻、payload 按需下沉 SSD、restore 时按需读回」。**复用判定完全内容驱动**（exact token 前缀 + media/position identity + accepted frontier + native KV/stamp proof），client key（`prompt_cache_key` / `promptCacheKey` / `conversation_id`）全部降级为诊断元数据——这就是「不改 OpenCode 分支协议」的代价所在：KVMem 放弃了用 key 建立可变所有权，改用匿名门与显式绑定分支来兜住风险。legacy 保持默认是因为 spread 的收益集中在 middle-edit（实测 −28~−30%），普通请求在 SSD 路径反而回退（switch back +59.6%）。

---

## 2. legacy vs spread：定义、配置、验证范围

### 2.1 定义

- **legacy（默认）**：`KVMEM_CHECKPOINT_POLICY` 未设或 `=legacy` 时 `enabled=false`，检查点只服务于当前会话的 continuation/末梢恢复；`conversation_id`/`prompt_cache_key` 等别名承担轻量路由语义（`docs/checkpoint-policy.md:3-6`；`kvmem/src/host/checkpoint_policy.cpp:7-16`）。
- **spread（实验性）**：`enabled=true`，为 llama.cpp 与 NInfer Qwen3.5 路径加「历史 coverage」：每条 native history version 保留至多 `COUNT` 个覆盖点，覆盖 root（system prefix）、observed LCP anchor、周期性 prefill chunk、tail、最后 mutable message 前的 input boundary 与 committed endpoints（`docs/checkpoint-policy.md:3-6, 251-257`；`kvmem/include/kvmem/checkpoint_policy.hpp:27-33` `CoveragePurpose{Periodic,Tail,Root,Anchor,Input,Endpoint}`）。
- PolyStrata/Radiance 有各自 native 策略，不受 spread 变量控制（`docs/checkpoint-policy.md:3-6, 30-35`）。

### 2.2 配置（`docs/checkpoint-policy.md:168-185`，解析在 `kvmem/src/host/checkpoint_policy.cpp:7-29`）

| 变量 | 默认 | 含义 |
|---|---|---|
| `KVMEM_CHECKPOINT_POLICY` | `legacy` | `legacy \| spread` |
| `KVMEM_CHECKPOINT_INTERVAL` | 16384 | 可选捕获的 token 间距 |
| `KVMEM_CHECKPOINT_MIN_GAP` | 0 | 0=禁用近邻稀疏化；64 为已测实验值 |
| `KVMEM_CHECKPOINT_COUNT` | 24 | 每 native history version 保留 1–256 |
| `KVMEM_CHECKPOINT_MIB` | 4096 | 逻辑保留预算 |
| `KVMEM_CHECKPOINT_HOT` | min(4,count) | RAM 目标张数 |
| `KVMEM_CHECKPOINT_HOST_MIB` | 1024 | 每 lane 检查点 Host 预算 |

spread 额外变量在 legacy 下被忽略；NInfer 另有 `KVMEM_CHECKPOINT_EXTRA_RECOMPUTE_TOKENS`（默认 0 禁用，16 为实验候选），与 MIN_GAP>0 互斥（launcher 拒绝双活跃阈值，`docs/checkpoint-policy.md:189-225`）。

### 2.3 验证范围（`docs/checkpoint-policy.md:445-491`；`docs/checkpoint-policy-validation-20261025.md`）

- 无 GPU：`content_cache_test`（逆序并发发布、native proof 拒绝、源生命周期、group 公平、origin coverage）、`kvmem-checkpoint-policy-test` / `kvmem-checkpoint-image-test`（framing/checksum/截断/满盘/取消/源生命周期/预留退款）。
- 回归脚本：`tests/content-cache-regression.py`（六重叠对循环：共享/不同/缺失 key）、`tests/checkpoint-coverage-regression.py`（tail/middle/front 编辑、会话切换、**实际 SSD 读**）、`tests/session-restore-regression.py --checkpoint-policy spread`。实际 restore 证据以 `KVMEM_HISTORY_RESTORE` 为准，API cache-hit 计数不算（`docs/checkpoint-policy.md:491`）。
- GPU 认证：NInfer RTX 5050（Qwen3.5-4B Q4_K_M + 27B 24 层 offload）、llama.cpp 0.8B Q8_0 + 4B MTP3，全过 oracle 数值容差（27B 135 次采样 max err 0.00063167055，容差 0.002；`docs/checkpoint-policy-validation-20261025.md:256-307`）。

---

## 3. 源码实现地图

### 3.1 关键函数/结构体清单

**公共层（kvmem/include, kvmem/src/host）**

| 位置 | 内容 |
|---|---|
| `kvmem/include/kvmem/checkpoint_policy.hpp:12-24` | `CheckpointPolicy{enabled, extra_recompute_tokens, interval=16384, min_gap, count=24, bytes=4GiB, hot_count=4, host_bytes=1GiB}` |
| `kvmem/include/kvmem/checkpoint_policy.hpp:27-33` | `CoveragePurpose` + `CoverageInfo{frontier,bytes,used,identity,purpose,protections}` |
| `kvmem/include/kvmem/checkpoint_policy.hpp:41-45` | `checkpoint_anonymous_continuation(lcp,input_end,accepted_end)`：`boundary=min(input_end,accepted_end)`，放行 `lcp>=boundary` 或 `boundary-lcp<=64`。注释：消耗性 restore 不得把共享 root 误认匿名 continuation；更深编辑需显式 owner |
| `kvmem/include/kvmem/checkpoint_policy.hpp:60-164` | `CheckpointArchive<Record>`：`remember()`（!enabled 直接 return false；pointer identity 去重；同 identity 变 frontier → throw "checkpoint identity changed frontier"；同 frontier 显式替换）、`prune()`（扣 fixed 执行 lease 后调 `retain_checkpoint_coverage`）、`select(limit,valid)`（返回 frontier≤limit 的最深合法记录，**不读 SSD**）、`nearby()`（MIN_GAP 稀疏化的 replay 替代查询） |
| `kvmem/src/host/checkpoint_policy.cpp:7-29` | `CheckpointPolicy::environment()`：env 解析，legacy 短路 |
| `kvmem/src/host/checkpoint_policy.cpp:31-115` | `retain_checkpoint_coverage`：按 frontier 排序、min_gap 稀疏化、超预算驱逐 cost=`(right-left)/max(interval,newest-left)` 最小者（软保护 Anchor/Root/Input 角色，平局 LRU） |
| `kvmem/include/kvmem/content_cache.hpp:23-43` | `rank_content_candidates(candidates,new_user_query,max=16)`：过滤 `!frontier || (!new_query && !continuation)`，排序 frontier↓ → cold_bytes↑ → last_used↓。这是跨会话候选排序的单一入口 |
| `kvmem/include/kvmem/content_cache.hpp:56-63` | `content_snapshot_covers(incoming,previous)`：非空非零 immutable-origin 集合包含关系（NInfer 版本归并的判据） |
| `kvmem/include/kvmem/content_cache.hpp:91-135` | `select_content_publication`（同 identity→Deduplicated / 空位→Inserted / 满→LRU Replaced）与 `select_content_eviction`（先多版本组的冗余版本，再 last_used） |
| `kvmem/include/kvmem/cache_owner.hpp:9-35` | `CacheOwnerHints` + `resolve_cache_owner`：校验 ≤256 字节、`prompt_cache_key` 与 `promptCacheKey` 必须一致（否则 throw "prompt_cache_key and promptCacheKey must agree"），优先级 conversation_id > prompt_cache_key > session_id。头注释：「Owners identify mutable conversation lineages. **They do not authorize reuse**; native token/media/position, accepted-state and Query proofs remain required」 |
| `kvmem/include/kvmem/checkpoint_image.hpp:9-35` | `KVMCKPT1` 帧：magic `0x3154504b434d564b` + identity（payload 对象指针值，防错位读）+ span 数/长度 + 尾部 8 字节 hash；读时逐项 `expect`，truncated/length/checksum mismatch → `SnapshotCorrupt` |

**llama.cpp 侧（tools/）**

| 位置 | 内容 |
|---|---|
| `tools/llama-kvmem-server.cpp:826-831` | Chat Completions 解析 `prompt_cache_key`/`promptCacheKey`/`kvmem.conversation_id` → `resolve_cache_owner` |
| `tools/kvmem-conversation-store.h:47-58` | `kvmem_store_match`，含 `allow_bound_edit`（注释「spread policy: an explicit owner may replace its suffix」） |
| `tools/kvmem-conversation-store.h:100-126` | `kvmem_store_keep_cap=min({lcp,live_row,eval_end-(spec_ok?0:1)})`；`kvmem_store_checkpoint`=ckpt_rows 中 ≤keep_cap 最大 row；`kvmem_store_continuation=(rows-lcp)<=max(0,last_n_gen)+64`（绝对 token 数门，非比例） |
| `tools/kvmem-conversation-store.h:170-251` | `kvmem_store_select`（client_id 仅 `(void)client_id` 诊断）与 `kvmem_store_select_shared`（接入 `rank_content_candidates`：continuation→extend，否则 fork(source)，reason=extend/shared_prefix） |
| `tools/kvmem-lane-conversations.h:625,669` | `plan_locked` 组 matches 调 `kvmem_store_select_shared`；`describe()` 设 `result.match.allow_bound_edit = source.mm_archive.policy().enabled` |
| `tools/llama-kvmem-server.cpp:2090-2098` | **发现权开关**：`discover = !cr.msgs.empty() && cr.msgs.back().role=="user"`——只有新 user turn 才允许发现冷历史前缀；tool continuation（末条 role=="tool"）一律 `discover=false` |
| `tools/llama-kvmem-driver.cpp:119-142` | `conversation_match`：`allow_bound_edit=(active?st.mm_archive:conv.mm_archive).policy().enabled`；ckpt_rows 合并 live `mm_checkpoints` 与 archive 中 identity 条目的 `record.row` |
| `tools/llama-kvmem-driver.cpp:1338-1372` | `run_prefill_multimodal` 检查点选择：`keep=prefix_restore_limit(...)` → `mm_checkpoints.select` 与 `mm_archive.select` 取 row 更大者 → `llama_driver_read_checkpoint`（**按需读 SSD**）→ `multimodal_restore`；读失败 erase 该条回退更早 row；全败 → clear 从 0 重算 |
| `tools/llama-kvmem-driver.cpp:847-905` | `checkpoint_tier`：RAM hot→SSD cold 的降级写（victim=!shared && !execution_held && 最旧 LRU；`atomic_store(&source->backing, job.destination)` 原子切换） |
| `tools/llama-kvmem-driver.cpp:993-1025` | `llama_driver_read_checkpoint`：cold→`read_checkpoint_image` 到 staging（计 cold_reads），hot→borrowed lease+memcpy（计 hot_reads） |

**NInfer 侧（backends/patches/）**

| 位置 | 内容 |
|---|---|
| `backends/patches/ninfer-checkpoint-policy.patch`（1106 行） | `KvmemPrefixCheckpoint` 加 `cold_tag/cold_payload_bytes` 三态；冷读 `kvmem::read_checkpoint_image(cold_state, cold_tag, ...)`；`restore_memory_prefix` 加 `window.coverage.truncate(frontier)`（rewind 撤销后续状态）；匿名 continuation 门（patch L706-726） |
| `backends/patches/ninfer-content-cache.patch`（1241 行，2026-11-04 内容快照版） | `canonical_snapshot_input()`：要求 `cp.identity.stamp.valid && stamp.session==window.stamp.session`、`state_bytes()==image_bytes`、`prefix_identity.size()>=end`、`graph.accepted_through==end`、每页 `archive[page]/host_versions` 覆盖；`snapshot_input_matches`＝exact token 前缀相等 && `prefix_identity.matches`；`snapshot_content_group()` 选最深可比组；`memory_restorable_history` 重写为内容驱动单循环（删除 session_key 所有权过滤），候选转 `ContentCacheCandidate` 后 `rank_content_candidates`；`fork_memory_history` 恒 fork、restore 不消费源；`reserve_memory_history` 删除「fresh 请求按 key 失效同 key histories」逻辑（注释：fresh 请求绝不失效其他独立发布的内容快照）；`openai_responses_request.cpp` 删除 `use_prompt_cache_key/update_session_index`（session_key 不再驱动 retention/index） |

### 3.2 数据流文字图

```
请求进入
  │
  ├─[1] key 解析（纯诊断）
  │    Chat Completions: body.prompt_cache_key / promptCacheKey / kvmem.conversation_id
  │      → tools/llama-kvmem-server.cpp:826-831 → cache_owner.hpp resolve_cache_owner
  │    Responses: store:false + prompt_cache_key（docs/checkpoint-policy-validation-20261025.md:247-249）
  │    产出: conversation_id 字符串——只进 monitor/日志，不建可变 slot
  │
  ├─[2] 候选发现（RAM 元数据 only，不读 SSD）
  │    新 user turn（discover=true, server.cpp:2090-2091）
  │      → llama: lane matches → kvmem_store_select_shared（kvmem-conversation-store.h:228-251）
  │              → rank_content_candidates（frontier↓/cold_bytes↑/last_used↓，content_cache.hpp:23-43）
  │      → NInfer: memory_restorable_history 遍历有界 native 目录（window.cpp）
  │              → ContentCacheCandidate{frontier,cold_bytes,publication_order,continuation}
  │    tool continuation（末条 role=="tool"）: discover=false → 只走 native continuation proof
  │
  ├─[3] native restore proof（逐候选验证，通过才可 restore）
  │    exact token 前缀 + prefix_identity（media/position/grid）+ accepted_through==frontier
  │    + session/incarnation stamp + KV 页覆盖 + state_bytes==image_bytes
  │    （checkpoint-policy.hpp:41-45 匿名门；ninfer-content-cache.patch canonical_snapshot_input；
  │      docs/checkpoint-policy.md:37-42）
  │
  ├─[4] fork / extend（restore 不消费源）
  │    continuation → extend 原会话；否则 fork：新私有执行分支，丢弃 edited suffix，
  │    retained source 保持有效（checkpoint-policy.md:51-57, 388-389；
  │      ninfer-content-cache.patch fork_memory_history）
  │
  ├─[5] 按需读 SSD（只有被选中的候选）
  │    llama: llama_driver_read_checkpoint（driver.cpp:993-1025）→ read_checkpoint_image
  │    NInfer: cold_tag → read_checkpoint_image（ninfer-checkpoint-policy.patch L214）
  │    KVMCKPT1 帧 + identity + checksum 校验（checkpoint_image.hpp:9-35）
  │    读失败: llama 回退更早合法 row；NInfer 已准入事务回退干净全 prefill
  │    （docs/checkpoint-policy.md:423-438）
  │
  └─[6] 发布与分层
       执行快照 turn 后进 CheckpointArchive::remember（RAM metadata + hot payload）
       超预算 → checkpoint_tier / write_checkpoint_image 下沉 SSD（hot→cold 原子切换）
       目录（frontier/bytes/identity/cold 字节数）常驻 RAM；重启后冷记录不能 restore
       （checkpoint-policy.md:162-166, 398-421；driver.cpp:847-905）
```

---

## 4. 三个问题

### 4a. 回退/编辑后复用的正确性怎么保证？

**答案：不是内容哈希，也不是单靠 generation 绑定，而是一条 native restore proof 链。** 文档明确「Hashes 与 LCP 仅候选过滤器；任意 LCP 不是可恢复 recurrent state」（`docs/checkpoint-policy.md:40-42`）。proof 要求（`docs/checkpoint-policy.md:37-42`；`backends/patches/ninfer-content-cache.patch` `canonical_snapshot_input()`）：

1. **exact token 前缀**：候选 frontier 内 prompt token 逐个相等（`snapshot_input_matches`）；
2. **media/position identity**：`prefix_identity.matches`——嵌套 VisionItem、三 position 轴、grid/rewrite frontier 的身份匹配（PolyStrata 审计曾抓出漏 media fingerprint 的缺陷，`docs/checkpoint-policy-validation-20261025.md:561-610`）；
3. **accepted frontier**：`cp.graph.accepted_through == end`——检查点必须命名一个真正 evaluated 过的 frontier，`checkpoint_policy.hpp` 的 `remember()` 对「同 identity 变 frontier」直接 throw "checkpoint identity changed frontier"；
4. **session/incarnation stamp**：`identity.stamp.session == window.stamp.session`——跨进程重启或换 model instance 的快照在 stamp 层就被拒；
5. **KV/statistics 覆盖**：每页 `archive[page]/host_versions` 覆盖校验、`state_bytes()==image_bytes`，恢复不完整 coverage 的候选被拒绝（验证记录 rows 65/95 精确字节通过、不完整 coverage 拒绝，`docs/checkpoint-policy-validation-20261025.md:123-133`）；
6. **图像完整性**：SSD payload 走 `KVMCKPT1` 帧，identity（payload 指针值）+ 逐 span 长度 + 尾部 checksum，错位读/截断/损坏 → `SnapshotCorrupt`（`kvmem/include/kvmem/checkpoint_image.hpp:9-35`）。

**回退/编辑场景的语义**：restore 永远 fork 出**新私有执行分支**，只在新分支丢弃 edited suffix（`window.coverage.truncate(frontier)`，`ninfer-checkpoint-policy.patch` restore_memory_prefix），**retained source 不被消费、保持有效**（`docs/checkpoint-policy.md:51-57, 388-389`；`ninfer-content-cache.patch` 注释「directory hit 从不消费源，包括 continuation」）。所以「回退后旧检查点还能用」的保证来自：检查点本身是 immutable accepted state（写后不替换，替换需 native dominance proof，`docs/checkpoint-policy.md:59-64, 89-94`），新请求只是站在它上面重新分叉；「编辑后还能用」来自前缀 proof——编辑点之前的 token/media/position 前缀仍逐项相等。generation 绑定体现在「restore 绑定 source generation，worker 不能改源或发布 engine state」（`docs/checkpoint-policy.md:398-421`），但它是隔离手段而非匹配键。

**实测佐证**：middle edit 在 legacy 0、spread 2074/2176（llama 7K fixture / NInfer），且 SSD 与 RAM 结果一致（真实冷读）；front edit 两策略都为 0（安全全重算）（`docs/checkpoint-policy-validation-20261025.md:140-156, 227-254`）。

### 4b. 为什么「无需修改 OpenCode 的分支协议」？KVMem 做了什么妥协？

**OpenCode 侧**：发送的还是原有请求体——Chat Completions 的 `prompt_cache_key`/`promptCacheKey`、Responses 的 `store:false` + `prompt_cache_key`，没有新增请求模式（`docs/checkpoint-policy.md:3-6`；验证用 **real OpenCode 1.19.5 unchanged-body** 请求跑通，`docs/ninfer-cross-owner-validation-20261029.md:10,21`）。

**KVMem 侧的妥协（这是关键）**：

1. **放弃 key 的所有权语义**。所有 client key 被降级为「诊断元数据」：`resolve_cache_owner` 只做归一与校验，头注释明言 owners identify lineages 但 **do not authorize reuse**（`kvmem/include/kvmem/cache_owner.hpp:7-8`）；`kvmem_store_select` 里 client_id 是 `(void)client_id`（`tools/kvmem-conversation-store.h:170-223`）；NInfer 侧删除了 `use_prompt_cache_key/update_session_index` 与按 session_key 找 publication target / fresh 请求失效同 key histories 的整套逻辑（`ninfer-content-cache.patch` reserve_memory_history 大删）。
2. **parent/child 同 key 不再隐含共享/继承**：同 key 不授权替换、序列化或抑制发布；缺失、不同、共享 key 走**同一条内容策略**（`docs/checkpoint-policy.md:10-18`）。这要求跨会话候选发现完全由内容驱动——代价是必须新增 `rank_content_candidates` + immutable-origin coverage（`content_snapshot_covers`）这套纯内容机制来替代原本 key 能提供的路由。
3. **用门来补偿失去 key 所有权后的风险**：
   - 匿名 continuation 门 `checkpoint_anonymous_continuation`：匿名（无显式 owner）请求只允许在 accepted input boundary 附近（≤64 token）消耗性 restore，更深编辑需显式 owner（`checkpoint_policy.hpp:41-45`）——这是首版事故的直接产物（无关短 prompt 通过 common-root checkpoint 消费了旧长历史，switch-back 掉到 29 token，`docs/checkpoint-policy-validation-20261025.md:315-338`）；
   - tool continuation 丧失发现权（`discover=false`，`tools/llama-kvmem-server.cpp:2090-2091`），只走 native continuation proof + append（`docs/checkpoint-policy.md:44-49`）；
   - 显式 key 才授权 suffix 替换（llama `allow_bound_edit`＝policy().enabled，`tools/kvmem-lane-conversations.h:669`；NInfer 注释「显式 session keys 授权编辑」，`ninfer-checkpoint-policy.patch:706-726`）——middle-edit 收益只对绑定会话开启。

一句话：**KVMem 把「会话身份」从协议层搬进了内容层**，OpenCode 的分支/parent-child 协议因此无需感知 KV 层的存在；付出的代价是 KVMem 自己维护匿名门、发现权规则和跨 owner 内容归组。

### 4c. legacy 默认而非 spread，说明 spread 有什么代价或风险？

文档自己给出的裁决是「middle-edit 收益支持 spread 保持 opt-in」（`docs/checkpoint-policy-validation-20261025.md:371-375`），具体代价/风险：

1. **普通请求性能回退（SSD 路径）**。受控计时（NInfer 5050，3 次轮换、每次新 worker）：Separate question 0.401→0.548s（SSD +36.6%）、Switch back 0.342→0.547s（SSD **+59.6%**）、Document 32.507→33.856s；收益集中在 middle edit 32.382→22.452/23.158s（RAM −30.7%/SSD −28.5%）（`docs/checkpoint-policy-validation-20261025.md:340-375`）。llama 侧大多 ±10% 内（`L376-413`）。即 spread 是**用普通路径的小额税换编辑场景的大额收益**，工作负载不偏编辑时不划算。
2. **内存与 IO 成本**：每 lane 1GiB 检查点 Host 预算 + hot 图像 + SSD 写带宽；27B 单条 retained record ≈149.9 MiB（峰值 945,732,680 字节贴着 1024MiB 预算，`docs/checkpoint-policy-validation-20261025.md:297-307`）。Host 预算不足以保证保留有用前缀：host128-ram-reclaim2 fixture 里 switch-back 只恢复 28 token、front edit 0（`L165-187`）。
3. **行为语义变化**：middle edit 对显式绑定会话从「拒绝」变「替换后缀」——这是新能力但也是新攻击面/新语义（legacy 用户可预期的 fork 门被放开，`docs/checkpoint-policy-validation-20261025.md:140-156, 207-223`）。
4. **正确性风险的历史证据**：首版匿名测试事故（消费旧长历史）；metadata 计费审计缺陷（`metadata_bytes()` 漏 ResidentPrefixIdentity，53K token identity 截断后四条向量仍占 689,000 字节，`L561-610`）；MIN_GAP=64 稀疏化实测使 switch-back 回退 +16.8~21.8% 且不保证正确性意义上的更便宜（`L436-559`）——这些都需要额外门与审计才能兜住。
5. **文档的显式门槛**：「改默认需独立代表性性能与质量认证（含 27B 与更长上下文）」（`docs/checkpoint-policy.md:493-498`）——即官方认为现有验证还不足以承担默认开关的后果。

---

## 5. 对 agent 长期记忆系统的启示

把 agent 会话持久化下沉到 KV/推理层（或任何「前缀缓存即记忆」的系统），这套机制给出五条可借鉴点与三个坑：

**可借鉴**

1. **身份与匹配解耦**。client 侧的 session id 只当诊断标签，复用判定全部走内容 proof（exact 前缀 + 结构身份 + accepted frontier）。agent 记忆系统同理：别让 conversation_id 决定「能不能复用」，让它只决定「给谁记账」——这样分支、fork、编辑、换客户端都天然正确。
2. **immutable accepted state + fork-not-consume**。检查点写后不可变，restore 永远开新私有分支、源不消费。这是 agent 会话树（回退/regenerate/branch）能安全建立在共享存储上的最低要求：任何「记忆条目」都应是 append-only 的 accepted state，消费即 fork。
3. **分层元数据常驻、payload 按需**。目录（frontier/identity/字节数）永远在 RAM，图像在 SSD 且按需读——候选排序永远只碰元数据。长期记忆索引也应如此：检索/打分不碰重负载，选中才物化。
4. **可驱逐但有角色的 retention**。覆盖点带 purpose（root/anchor/periodic/tail/input/endpoint），驱逐按「信息增益/间隔」打分并软保护关键角色，而非纯 LRU。agent 记忆等价物：system prompt 锚点、最新 frontier、用户画像 root 各有保留权重。
5. **把验证建在「实际 restore 证据」上**。KVMem 坚持以 `KVMEM_HISTORY_RESTORE`（native 实际前缀复用）为准，API 层 cache-hit 计数只是 admission quote（`docs/checkpoint-policy.md:137-160, 491`）。agent 记忆评测同理：以「这次推理真的少算了多少 prefill」为准，不以「记忆系统说命中了」为准。

**坑**

1. **匿名发现的误消费**是最大风险：无 owner 的短请求顺着 common root 消费别人的长历史（首版事故）。等价物：agent 的闲聊回合不该加载长项目记忆——需要显式的「匿名门」（边界距离/深度阈值），而这道门的参数（64 token）是事故驱动拍出来的，不是推导出来的。
2. **计费遗漏会让容量承诺失效**：一次 metadata 计费审计漏掉一个 identity 结构就差 689KB/条，Host 预算瞬间失真。记忆层的「token/字节预算」必须对每个常驻结构（索引向量、身份元数据、页表）保守计费并定期审计。
3. **普通路径税 vs 编辑路径收益**：spread 默认关是因为普通请求在 SSD 路径 +36~59%。记忆系统若为「改写历史」场景优化，要先测量日常请求付了多少税——收益集中在长尾场景时，opt-in + 场景开关比全局默认更诚实。

---

## 附：本次精读覆盖的文件

`docs/checkpoint-policy.md`、`docs/checkpoint-policy-validation-20261025.md`、`docs/ninfer-cross-owner-validation-20261029.md`、`kvmem/include/kvmem/{cache_owner,checkpoint_policy,content_cache,checkpoint_image}.hpp`、`kvmem/src/host/checkpoint_policy.cpp`、`tools/{llama-kvmem-server.cpp,kvmem-conversation-store.h,kvmem-lane-conversations.h,llama-kvmem-driver.cpp}`、`backends/patches/{ninfer-content-cache.patch,ninfer-checkpoint-policy.patch,llamacpp-checkpoint-policy.patch}`。

源码未覆盖项：PolyStrata/Radiance 的 native spread 等价实现细节（其 patch 不在本仓库 backends/patches 下，文档仅给边界表）；`rank_content_candidates` 在 Radiance 路径的调用点（未逐行核对）。
