# 把 agent 会话持久化下沉到 KV cache 层

做 agent 框架的人和做推理系统的人，长期在解决同一个问题，只是站在不同的层。框架这一层叫会话持久化：多轮对话、分支、regenerate、编辑历史里的某一条消息，要求状态能回退、能复用、跨会话还在。推理系统这一层叫 KV cache 复用：同样的 prompt 前缀出现过一次，prefill 就该少算一次。两边的工程答案分头演进了很久——框架自己存消息列表，推理引擎做前缀缓存，中间靠一个 session id 或者 `prompt_cache_key` 勉强对上号。

KVMem（单机 KV 内存虚拟化层）的 spread 检查点策略把这两层捏到了一起。先说它在主流方案里的定位：LMCache、Mooncake、vLLM Tiered Offloading、SGLang HiCache 全部围绕前缀复用命中率设计，KVMem 走全历史保留（RetainAll，`docs/memory-backend-contract.md:307`）加按需检索召回（query 取真实用户内容末尾 ≤512 tokens，`docs/multi-backend-differences.md:41`），也是五者中唯一单机、无 daemon、无 RDMA 的方案。spread 是它的实验性检查点策略（v0.19.1）。我精读了它的设计文档、验证记录和后端补丁源码，这篇文章把机制、正确性保证和代价实数讲清楚。所有引用相对 kvmem 仓库根，读者可以按图索骥。

## 一、agent 场景为什么难住检查点

legacy 策略（默认的那个）里，检查点只服务于当前会话的 continuation 和末梢恢复，`conversation_id`/`prompt_cache_key` 这些别名承担轻量路由语义（`docs/checkpoint-policy.md:3-6`；`kvmem/src/host/checkpoint_policy.cpp:7-16`）。这套在「下一个请求接着上一个请求」的直线会话里够用。

agent 工作负载把三件事同时塞了进来。第一是分支与回退：用户 regenerate，或者从历史某一点开新支，请求的前缀停在旧会话的中段。第二是编辑：改写历史中间一条消息，前缀在编辑点之前完全相同、之后作废。第三是跨会话：几天后带着同样的 system prefix 和部分历史回来，进程可能重启过，client 可能换过。

这三个场景的公共要求是：一条很深的历史前缀必须能被新请求安全地站上去。拿真实的 agent 负载举例：一个 coding agent 跑到第 40 轮，用户在第 12 轮的 tool 结果后面补了一句更正然后重新生成。此刻请求前缀的第 1 到第 11 轮与旧会话逐字相同，第 12 轮起作废。理想行为是复用前 11 轮的 KV、只重算第 12 轮之后；legacy 的检查点帮不上忙，它既没有在中段留覆盖点，也没有判定「这 11 轮相同」的机制。

麻烦在于什么叫「安全」。文档里有一句关键提醒：hash 与 LCP 只能当候选过滤器用，一条任意 LCP 远够不上可恢复的 recurrent state（`docs/checkpoint-policy.md:40-42`）。两段 token 序列逐字节相同，假如权重不同、media 处理路径不同，或者其中一段根本没被完整 evaluate 过，拿它的 KV 来用就是错的。所以正确性问题靠内容哈希或者 generation 号都解决不了——spread 的答案是一条完整的 proof 链，后面细说。

## 二、身份与匹配解耦

spread 最反直觉的决定，是把 client 传来的所有 key 全部降级为诊断元数据。`resolve_cache_owner` 的头注释写着：owners 标识可变的会话谱系，但 they do not authorize reuse；native token/media/position、accepted-state 与 Query proofs 仍然是必需的（`kvmem/include/kvmem/cache_owner.hpp:7-8`）。落到代码上，这个函数只做归一化与校验——限 256 字节以内，`prompt_cache_key` 与 `promptCacheKey` 同时出现必须一致，否则直接 throw（`kvmem/include/kvmem/cache_owner.hpp:9-35`）；解析出的 owner 字符串只进 monitor 和日志。llama.cpp 侧的候选选择里，client_id 的下场是 `(void)client_id`（`tools/kvmem-conversation-store.h:170-223`）。NInfer 侧走得更远，patch 直接删掉了 `use_prompt_cache_key/update_session_index`，连同「fresh 请求按 key 失效同 key histories」的整套逻辑，注释给的理由是 fresh 请求绝不失效其他独立发布的内容快照（`backends/patches/ninfer-content-cache.patch`）。

配套的语义约定在 `docs/checkpoint-policy.md:10-18`：同 key 不授权替换、序列化或抑制发布；key 缺失、不同、共享，走同一条内容策略。

为什么下手这么狠？因为 key 是 client 说的话，而 client 不可信也不完整。OpenCode 的 parent/child 分支协议、各家客户端自己发明的 session id，语义互相冲突；parent/child 同 key 也未必隐含共享或继承。key 一旦参与复用判定，agent 框架的每一次分支语义变化都要推理引擎跟着改。把 key 降级之后，OpenCode 一个字节都不用动。验证记录里用 real OpenCode 1.19.5 的 unchanged-body 请求跑通了整个链路（`docs/ninfer-cross-owner-validation-20261029.md:10,21`）。

代价当然转移到了 KVMem 自己身上：原本 key 提供的路由功能，需要一套纯内容机制补回来。跨会话候选的排序收敛在 `rank_content_candidates` 这一个入口：过滤掉无 frontier 且非 continuation 的候选，按 frontier 降序、cold_bytes 升序、last_used 升序排，最多 16 个（`kvmem/include/kvmem/content_cache.hpp:23-43`）；NInfer 版本归并的判据是 immutable-origin 集合的包含关系（`kvmem/include/kvmem/content_cache.hpp:56-63`）。

## 三、内容 proof 链

候选通过排序只是拿到入场券，能否 restore 由 proof 链说了算（`docs/checkpoint-policy.md:37-42`；`backends/patches/ninfer-content-cache.patch` 的 `canonical_snapshot_input()`）：

1. **exact token 前缀**：候选 frontier 内的 prompt token 逐个相等（`snapshot_input_matches`）；
2. **media/position identity**：`prefix_identity.matches`，嵌套 VisionItem、三条 position 轴、grid/rewrite frontier 的身份都要对上；
3. **accepted frontier**：`graph.accepted_through == end`，检查点必须命名一个真正 evaluate 过的 frontier。`remember()` 对「同 identity 却变了 frontier」直接 throw "checkpoint identity changed frontier"（`kvmem/include/kvmem/checkpoint_policy.hpp:60-164`）；
4. **session/incarnation stamp**：`identity.stamp.session == window.stamp.session`，跨进程重启或换 model instance 的快照在这一层就被拒；
5. **KV 页覆盖**：每页 `archive[page]/host_versions` 覆盖校验，`state_bytes()==image_bytes`，恢复不完整 coverage 的候选被拒。验证记录 rows 65/95 是精确字节通过、不完整 coverage 拒绝的实测（`docs/checkpoint-policy-validation-20261025.md:123-133`）；
6. **图像完整性**：SSD payload 走 `KVMCKPT1` 帧，magic + identity（payload 指针值，防错位读）+ 逐 span 长度 + 尾部 8 字节 hash，截断或校验失败抛 `SnapshotCorrupt`（`kvmem/include/kvmem/checkpoint_image.hpp:9-35`）。

这套链路的每一环都在堵一种具体的错。媒体指纹这一环就是审计抓出来的——PolyStrata 审计曾发现实现漏了 media fingerprint（`docs/checkpoint-policy-validation-20261025.md:561-610`）。

## 四、一次请求的完整旅程

把上面的部件串起来。一个请求进来，走六步。

第一步，key 解析。Chat Completions 的 `prompt_cache_key`/`promptCacheKey`/`kvmem.conversation_id` 进 `resolve_cache_owner`（`tools/llama-kvmem-server.cpp:826-831`），产出只进 monitor 和日志，不建任何可变 slot。第二步，候选发现，全程只碰 RAM 元数据。新 user turn 才开发现权（`tools/llama-kvmem-server.cpp:2090-2091`），llama 侧经 lane matches 汇入 `kvmem_store_select_shared`（`tools/kvmem-conversation-store.h:228-251`），再进 `rank_content_candidates` 排序；NInfer 侧由 `memory_restorable_history` 遍历有界 native 目录转成候选。第三步，逐候选做 native restore proof，就是上一节那六环。第四步，fork 或 extend：continuation 顺延原会话，其余 fork 新私有分支、丢 edited suffix、源保持有效（`docs/checkpoint-policy.md:51-57, 388-389`）。第五步，只有被选中的候选才碰 SSD：llama 走 `llama_driver_read_checkpoint`（`tools/llama-kvmem-driver.cpp:993-1025`），读入 `KVMCKPT1` 帧并校验；读失败则 erase 该条回退更早合法 row，全部失败就 clear 从 0 重算（`docs/checkpoint-policy.md:423-438`）。第六步，发布与分层：执行快照过 turn 后进 `CheckpointArchive::remember`（RAM 元数据加 hot payload），超预算经 `checkpoint_tier` 下沉 SSD，hot→cold 原子切换（`tools/llama-kvmem-driver.cpp:847-905`）；目录常驻 RAM，重启后冷记录不能 restore（`docs/checkpoint-policy.md:162-166, 398-421`）。

注意第二步和第五步的分离：排序与淘汰的决策全部发生在元数据层，SSD 读是选中之后才付出的成本。这套「元数据常驻、payload 按需」的分层，是后面记忆系统启示的第三条的出处。

## 五、restore 恒 fork，源不消费

proof 通过之后，restore 的语义是唯一确定的：永远 fork 出一个新私有执行分支，只在新分支里丢弃 edited suffix（NInfer 的 `restore_memory_prefix` 里对应 `window.coverage.truncate(frontier)`，`backends/patches/ninfer-checkpoint-policy.patch`），retained source 保持有效、不被消费（`docs/checkpoint-policy.md:51-57, 388-389`；`ninfer-content-cache.patch` 的注释写明 directory hit 从不消费源，包括 continuation）。

我认为这是整个设计里正确性价值最高的一条。回退后旧检查点还能用，依据在于检查点本身是 immutable accepted state——写后不替换，替换需要 native dominance proof（`docs/checkpoint-policy.md:59-64, 89-94`）；新请求只是站在它上面重新分叉，谁都没动它。编辑后还能用，依据在于前缀 proof——编辑点之前的 token/media/position 前缀逐项相等。generation 绑定也有：restore 绑定 source generation，worker 不能改源、不能发布 engine state（`docs/checkpoint-policy.md:398-421`），但它承担的是隔离职责，匹配这件事已经完全交给内容。这两件事分开之后，agent 会话树（回退、regenerate、branch）才能安全地建在共享存储上：任何消费者都无法通过读路径破坏别人正在用的状态。

实测数据支撑这个语义的收益。middle edit 场景，legacy 的实际恢复为 0，spread 在 llama 7K fixture 和 NInfer 上分别是 2074 和 2176，且 SSD 与 RAM 结果一致（真实冷读，以 `KVMEM_HISTORY_RESTORE` 为准而非 API cache-hit 计数）（`docs/checkpoint-policy-validation-20261025.md:140-156, 227-254`）。front edit 两个策略都是 0——安全全重算，proof 链在该拒绝的地方拒绝了。

## 六、兜底三件套

放弃 key 所有权后，风险要靠门来兜。spread 配了三道：

**匿名 continuation 门**。无显式 owner 的请求，只允许在 accepted input boundary 附近做消耗性 restore：`boundary=min(input_end, accepted_end)`，放行条件是 `lcp>=boundary` 或 `boundary-lcp<=64`（`kvmem/include/kvmem/checkpoint_policy.hpp:41-45`）。这道门是事故的直接产物——首版匿名测试里，无关短 prompt 顺着 common root checkpoint 消费了一条旧长历史，switch back 掉到 29 token（`docs/checkpoint-policy-validation-20261025.md:315-338`）。64 这个数字是事故驱动拍出来的，没有推导。

**tool continuation 无发现权**。发现权开关只有一个判据：末条消息 role 为 user 才允许发现冷历史前缀；tool continuation 一律 `discover=false`（`tools/llama-kvmem-server.cpp:2090-2091`），只能走 native continuation proof 加 append（`docs/checkpoint-policy.md:44-49`）。agent 的工具循环恰恰是误发现的高发区——工具回包触发的请求形状上和用户请求接近，内容上却完全不同。

**显式 key 才授权 suffix 替换**。middle-edit 收益只对绑定会话开启：llama 侧 `allow_bound_edit = source.mm_archive.policy().enabled`（`tools/kvmem-lane-conversations.h:669`；字段注释见 `tools/kvmem-conversation-store.h:47-58`），NInfer 侧注释写明显式 session keys 授权编辑（`ninfer-checkpoint-policy.patch:706-726`）。匿名请求永远走 fork，没有替换别人后缀的资格。

三道门合起来回答了一个问题：当系统决定信任内容而非 key 时，谁来限制内容的误用。答案是把信任分级——匿名最少，continuation 只能顺着走，显式 owner 换来编辑权。

## 七、代价实数，以及 legacy 为什么还是默认

spread 至今保持 opt-in，文档自己给的裁决是 middle-edit 收益支持它保持 opt-in（`docs/checkpoint-policy-validation-20261025.md:371-375`）。数字如下，受控计时来自 NInfer RTX 5050，3 次轮换、每次新 worker（`docs/checkpoint-policy-validation-20261025.md:340-375`）：

- Separate question：0.401→0.548s，SSD 路径 +36.6%
- Switch back：0.342→0.547s，SSD 路径 +59.6%
- Document：32.507→33.856s
- Middle edit：32.382→22.452s（RAM −30.7%）/ 23.158s（SSD −28.5%）

llama 侧大多在 ±10% 以内（`docs/checkpoint-policy-validation-20261025.md:376-413`）。规律很清楚：收益集中在 middle edit，普通请求在 SSD 路径上交税。工作负载不偏编辑时，这道开关就是净亏。

还有一类代价不在计时里，是行为语义的变化。middle edit 对显式绑定会话从「拒绝」变成了「替换后缀」，这是新能力，也是新语义：legacy 用户可预期的 fork 门被放开了，替换的边界条件（谁算 owner、替换到哪一格）从此成为系统协议的一部分（`docs/checkpoint-policy-validation-20261025.md:140-156, 207-223`）。实验性策略承担得起这种语义试探，默认开关承担不起。

内存账同样要算。每 lane 检查点 Host 预算默认 1024 MiB（`kvmem/include/kvmem/checkpoint_policy.hpp:12-24`），27B 模型单条 retained record ≈149.9 MiB，实测峰值 945,732,680 字节，贴着预算上限（`docs/checkpoint-policy-validation-20261025.md:297-307`）。预算不足时保留质量直线下滑：host128-ram-reclaim2 fixture 里 switch-back 只恢复 28 token、front edit 为 0（`docs/checkpoint-policy-validation-20261025.md:165-187`）。MIN_GAP=64 的近邻稀疏化实测让 switch-back 回退 +16.8~21.8%，且在正确性意义上更便宜这一点没有保证（`docs/checkpoint-policy-validation-20261025.md:436-559`）。还有一个审计教训：metadata 计费漏掉 ResidentPrefixIdentity，53K token 的 identity 截断后四条向量仍占 689,000 字节，容量承诺当场失真（`docs/checkpoint-policy-validation-20261025.md:561-610`）。

文档末尾写了改默认的门槛：需要独立代表性性能与质量认证，含 27B 与更长上下文（`docs/checkpoint-policy.md:493-498`）。官方自己也认为现有验证还不足以承担默认开关的后果。

## 八、拿去建 agent 记忆系统：五条启示与三个坑

把视野拉回 agent 长期记忆。KVMem 这套机制可借鉴的点，按我读下来的重要性排：

1. **身份与匹配解耦**。session id 只当诊断标签，复用判定全走内容 proof。对应到记忆系统：conversation_id 决定给谁记账，决定不了能不能复用。分支、fork、编辑、换客户端因此天然正确。
2. **immutable accepted state + fork-not-consume**。记忆条目应当是 append-only 的 accepted state，消费即 fork。这是回退/regenerate/branch 能安全建立在共享存储上的最低要求。
3. **元数据常驻，payload 按需**。目录（frontier/identity/字节数）永远在 RAM，SSD 图像被选中才读，候选排序从不碰重负载——llama 侧的检查点选择路径（`tools/llama-kvmem-driver.cpp:119-142, 1338-1372`）在选中之前都不读 SSD。记忆索引同理：检索与打分只碰元数据，选中才物化。
4. **带角色的 retention**。覆盖点带 purpose（Root/Anchor/Periodic/Tail/Input/Endpoint，`kvmem/include/kvmem/checkpoint_policy.hpp:27-33`），驱逐按间隔增益打分并软保护关键角色（`kvmem/src/host/checkpoint_policy.cpp:31-115`）。system prompt 锚点、最新 frontier、用户画像 root 各有保留权重，纯 LRU 给不了这个。
5. **以实际 restore 证据为准**。KVMem 坚持 `KVMEM_HISTORY_RESTORE`（native 实际前缀复用）为验证标准，API 层 cache-hit 计数只算 admission quote（`docs/checkpoint-policy.md:137-160, 491`）。记忆评测同理：指标是这次推理真的少算了多少 prefill，记忆系统自报的命中率不算数。

三个坑同样值钱。其一，匿名发现的误消费：闲聊回合加载长项目记忆就是它的等价问题，需要显式匿名门，而门的参数往往来自事故而非推导。其二，计费遗漏让容量承诺失效：索引向量、身份元数据、页表都要保守计费并定期审计，漏一个常驻结构就是 689,000 字节/条级别的失真。其三，普通路径税 vs 编辑路径收益：为长尾场景优化的系统，先测日常请求付了多少税，opt-in 加场景开关比全局默认更诚实。

## 这套设计对谁有用

agent 框架作者：如果你在为会话树设计持久化——回退、regenerate、分支、编辑历史——KVMem 给出了完整的参照系。身份与匹配解耦让你的框架免于和推理引擎协商 session 语义；immutable + fork-not-consume 让共享存储上的分支操作天然安全；三道兜底门示范了信任分级怎么做，包括那道参数来自事故而非推导的 64-token 门。真实系统里不少阈值就是这样来的，承认它比假装推导出来更体面。

推理引擎维护者：如果你的前缀缓存要跨会话、跨 client 复用，proof 链的六环（exact 前缀、media/position identity、accepted frontier、stamp、页覆盖、checksum）可以直接当 checklist 用；`prompt_cache_key` 在你的系统里是授权凭证还是诊断标签，决定了你替 client 承担了多少它从没答应过的风险。在开任何「更聪明的保留策略」之前，先跑一遍普通请求的计时。KVMem 把 +59.6% 的 switch-back 税和 −28.5% 的编辑收益并排写进文档，这个动作本身比 spread 策略更值得抄。
