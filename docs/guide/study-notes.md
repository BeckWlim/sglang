# SGLang 完整架构源码阅读笔记

> 阅读基线：`131bd51b01d96de247fbd1966fb45bdbecbb80b5`（2026-08-01）
>
> 当前完成范围：KV Cache 子系统，以及 Request 在 Scheduler 中的状态流转和 L1/L2 缓存生命周期。
>
> 组织原则：先建立 KV Cache 的组件边界，再沿 Request 生命周期串联 Scheduler、RadixTree、内存池和 HiCache。

## 1. KV Cache 子系统

### 1.1 实现谱系与最重要的职责边界

SGLang 中的 RadixTree 是 token 前缀索引结构，`RadixCache` 是使用这棵树管理 GPU KV 的具体实现。`HiRadixCache` 直接继承 `RadixCache`，在相同 token 前缀拓扑上增加 Host KV、异步搬运和可选 Storage Backend。新版 `UnifiedRadixCache` 则把 Full Attention、SWA、Mamba/SSM 等多类状态组织到统一树中，并可通过 `init_hicache()` 激活分层能力。

| 运行实现 | 跨请求前缀匹配 | L1 GPU | L2 Host | L3 Storage |
|---|---:|---:|---:|---:|
| `ChunkCache` | 否 | 请求私有 | 否 | 否 |
| `SWAChunkCache` | 否 | 请求私有，并提前释放窗口外 SWA KV | 否 | 否 |
| 普通 `RadixCache` | 是 | 是 | 否 | 否 |
| `HiRadixCache` | 是 | 是 | 是 | 可选 |
| `UnifiedRadixCache` | 是 | 是 | 启用 HiCache 后支持 | 配置 backend 后支持 |

这里必须区分“节点能表达什么”和“运行实例真正管理什么”。`python/sglang/srt/mem_cache/radix_cache.py:217-268` 中的 `TreeNode` 同时定义了：

```python
self.value = None          # L1 GPU slot/page indices
self.host_value = None     # L2 Host slot/page indices
self.hash_value = None     # 各逻辑 page 的链式 hash
```

这个类是普通 `RadixCache` 与 `HiRadixCache` 共同使用的节点 schema。普通 `RadixCache.__init__()` 仅持有 GPU allocator，节点的 `host_value` 保持为空，因此实际管理范围是 L1。其 `match_prefix()` 返回 GPU `device_indices`，并为了统一 `MatchResult` 接口令 `last_host_node == last_device_node`，见 `radix_cache.py:355-413` 和 `base_prefix_cache.py:166-204`；该模式下的实际命中层级仍为 L1。

普通 RadixCache 的淘汰同样是 L1-only：`radix_cache.py:571-599` 释放节点的 GPU KV 后直接 `_delete_leaf(x)`。只有 `HiRadixCache` 才在 `hiradix_cache.py:81-178` 创建 Host Pool 与 Controller，通过 `hiradix_cache.py:835-865` 完成 L1→L2 backup，通过 `hiradix_cache.py:1335-1406` 完成 L2→L1 load-back。于是同一节点才真正可能处于 L1-only、L1+L2 或 L2-only 状态。

工厂选择位于 `mem_cache/registry.py`。标准配置下普通模型返回 `RadixCache`。显式启用 hierarchical cache 后，普通 MHA/MLA 模型使用传统 `HiRadixCache`；Hybrid/DSA 或设置 `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1` 时，通常创建 `UnifiedRadixCache` 并调用 `init_hicache()`。因此 HiCache 分层能力可以由 `HiRadixCache` 或启用分层组件的 `UnifiedRadixCache` 承载。

### 1.2 L1/L2 本地前缀树与 L3 内容寻址

L1 与 L2 使用独立物理 KV pool，但共享同一棵本地 token RadixTree。节点的 `value` 和 `host_value` 是精确本地地址，所以本地树可以直接回答 GPU 连续命中到哪里、Host 连续命中又延伸到哪里。

L3 使用另一套协作索引。节点中的 `hash_value` 是生成 L3 内容地址的 key 或 anchor；Mooncake、HF3FS、NIXL 等 Storage Backend 负责 `hash → KV page` 的实时存在性、物理位置和对象生命周期：

```text
本地 HiRadixTree
├── token prefix → TreeNode
├── TreeNode.value → L1 GPU slots
├── TreeNode.host_value → L2 Host slots
└── TreeNode.hash_value → L3 page hash 链

Storage Backend
└── page hash → KV page 的存在性与数据
```

HiRadix 在 storage 或 KV events 开启时计算节点 hash，见 `hiradix_cache.py:1938-1953`。写 L3 时，`hiradix_cache.py:911-935` 把 Host slots、token key、hash 列表和可选 prefix keys 交给 `HiCacheController.write_storage()`；Controller 最终调用 backend 的 `batch_set`。查询时，`managers/cache_controller.py:1023-1045` 根据最后一个本地 hash 和剩余 tokens 生成后续 page hashes，再调用 `storage_backend.batch_exists()`。

因此 HiCache 负责 L3 key 生成、查询编排和回填，本地 HiRadixTree 保存前缀锚点，Storage Backend 保存 L3 权威状态。普通 RadixTree 给出最长本地前缀与必要锚点；Scheduler、HiCacheController 和 Storage Backend 负责异步 I/O、Host 分配、超时、失败回收和跨 rank 同步。

### 1.3 Page 粒度、查询、重建与淘汰

当 `page_size>1` 时，`RadixKey.page_aligned()`、`match()` 和 `child_key(page_size)` 保证匹配长度、节点分裂和插入边界都按 page 对齐，相关实现位于 `radix_cache.py:139-214`。压缩树中的一个节点可以包含多个连续 page，并在公共前缀的 page 边界处分裂。

L3 同样以 page 为逻辑寻址单位。`mem_cache/utils.py:106-135` 使用 prior hash 形成位置敏感的链式 key：

```text
H0 = hash(page0)
H1 = hash(H0, page1)
H2 = hash(H1, page2)
```

相同 token page 处于不同历史前缀后通常产生不同 key，因为其 KV 本来就依赖此前上下文。一个包含多个 page 的节点会保存 hash 列表；Mooncake backend 还可能把一个逻辑 page 展开成 K/V 或多个 head shard 物理对象，只有所需对象全部存在时才算逻辑 page 命中。

模型的 layer-wise 切分运行沿用上述抽象。可以把 KV Cache 理解为 `layer × token` 的二维结构：RadixTree 和链式 hash 沿 token 维度把连续列切成 page；逐层执行、逐层加载以及 Pipeline Parallel 则沿 layer 维度切分行。`radix_cache.py:355-425` 的本地匹配和插入根据 token sequence、可选 `extra_key` 与 `page_size` 决定逻辑前缀；`mem_cache/utils.py:106-112` 的 L3 hash 接口接收 `token_ids`、`prior_hash` 和 `page_size`。因此保持 KV 数值语义的 Piecewise CUDA Graph、逐层模型执行和逐层权重加载继续复用 token 维 RadixTree 与 page hash。

一个逻辑 page 的 payload 则包含当前执行单元所需的各层 KV。普通单 stage 可以将其理解为该 page 上所有 attention layer 的 K/V；PP 部署中，每个 stage 只持有自己负责的 local layers。Mooncake 仍复用相同的 token page hash，但会在物理对象键中加入模型或 backend tag、TP/PP rank 以及 K/V 分量：`storage/mooncake_store/mooncake_store.py:419-430` 构造模型命名空间，`550-577` 构造 TP/PP suffix，`1240-1267` 将一个逻辑 hash 展开成实际查询对象。于是这里应区分两层标识：

```text
逻辑 page：chain_hash(token page)
物理对象：namespace + page hash + TP/PP shard + K/V 或其他 pool 分量
```

当前 Radix/HiCache 在树节点上表达 page 级聚合命中状态。某个 rank 或 stage 只有在逻辑 page 所需物理分量完整存在时，才把它作为统一连续前缀使用；TP 等并行 rank 通过一致的实际可用长度共同调度。`hicache_design.mdx:87-90` 与 `hiradix_cache.py:1463-1471` 展示了用 `all_reduce(MIN)` 收敛 L3 命中长度的路径。PP 层间的关键控制状态则由 `hiradix_cache.py:242-298` 传播，使各 stage 对请求前缀采用一致决策。

Layer-wise load 改变的是 page payload 的就绪时机。
`cache_controller.py:790-817` 在独立 load stream 中按层执行 `load_to_device_per_layer()`，每完成一层便记录对应 event；
attention backend 读取某层 buffer 时在 `memory_pool.py:2233-2263` 等待该层 event。
因此同一个已被 Radix 判定命中的 token 前缀，可以在 layer N 计算时并行加载 layer N+1，以 layer event 为粒度推进模型执行。
索引仍给出统一的 token 命中范围，layer event 只控制这个范围内的 KV 何时可由各层消费。

Hybrid/SWA/Mamba/DSA 等模型进一步体现了逻辑 page 与物理对象之间的一对多映射。
`hybrid_cache_controller.py:602-625` 仍以主 KV 的 token page hash 确定 anchor 命中长度，
再通过 `PoolTransfer` 搬运 SWA、Mamba state、Indexer 或压缩 KV 等 sidecar；
某些 sidecar 甚至有不同的物理 page size，例如 `hybrid_pool_assembler.py:315-384` 中 DeepSeek V4 的主 KV 使用普通 `page_size`，SWA pool 使用 `swa_page_size`。
更准确的结论是：主 token page 仍是 Radix 与连续前缀的索引单位，但它可以展开为多个具有不同层映射、布局、物理粒度和命中策略的缓存对象。

缓存隔离由 KV 内容或编码是否发生变化决定。
模型权重/revision、LoRA、位置编码、KV dtype/量化格式、layer skipping、动态路由、压缩策略或具有差异的 TP/PP 布局发生变化时，需要使用独立的缓存命名空间。
本地 Radix 可用 `RadixKey.extra_key` 隔离 adapter 或 cache version；
但 `RadixKey.hash_page()` 和 Controller 的 L3 链式 hash 本身主要来自 tokens，Mooncake 依靠 served model name、`extra_backend_tag` 和 rank suffix 隔离物理对象。
因此部署侧应给任何会改变 KV 数值语义或物理编码的配置分配不同 storage namespace/tag。
判断准则可以压缩为：只改变执行与搬运时序的 layer-wise 优化可以共用索引；
改变 KV 语义或对象格式的优化必须隔离 L3 数据。

请求通过 `Req.init_next_round_input()` 调用 `tree_cache.match_prefix()`，见 `managers/schedule_batch.py:1244-1292`。
普通 Radix 只返回 L1；HiRadix 的 `match_prefix()` 位于 `hiradix_cache.py:1701-1732`，返回连续的 L1 前缀及其后的连续 L2 前缀。
Scheduler 在 `managers/scheduler.py:2514-2537` 根据二者计算本地未命中的 suffix，再从 `last_hash` 开始生成候选 page hashes。

Controller 会统一计算候选 suffix 的 hash，再按 `STORAGE_BATCH_SIZE` 分批查询。
批次内部可以批量检查多个 key，但遇到第一个不完整 page 后停止后续批次。
接口返回“从 suffix 开头起连续存在的 page 数”，以最长连续前缀表达命中结果。

L3 命中后，Controller 先按实际 hit count 分配 L2 Host slots，再通过 `batch_get_v1()` 或等价接口把数据读入 Host。
I/O 成功并完成 rank 同步后，`hiradix_cache.py:1775-1816` 的 `_insert_helper_host()` 才创建或补齐 host-only 节点：

```python
new_node.value = None
new_node.host_value = host_value.clone()
new_node.hash_value = hash_value
```

之后 `init_load_back()` 再把 L2 前缀恢复到 L1。完整路径是：

```text
本地 L1/L2 match
    → 对 suffix 查询 L3
    → L3 KV 读入 L2
    → 重建/补齐 host-only Radix 路径
    → L2 load-back 到 L1
    → 模型消费 GPU KV
```

HiRadix 的 L1 淘汰若已有 Host backup，只清除 `value`，节点降级为 L2-only，见 `hiradix_cache.py:1232-1249`。
L2 也需淘汰时，`hiradix_cache.py:1300-1333` 释放 host slots 并从父节点删除 host leaf，必要时向根级联清理。
本地淘汰与 Mooncake 对象生命周期相互独立，因此 L1、L2 和本地树节点释放后，L3 对象仍可继续存在。
未来请求仍能从 root、`last_hash=None` 重算 hash，查询 L3 并重建本地树。
Mooncake 独立淘汰对象后，本地 `hash_value` 继续充当查询锚点，下一次通过 `batch_exists()` 获取权威存在状态。

当前查询在首个 L3 miss 处停止，后续 page 仍可能保存在 Storage Backend 中。
假设 `P0` 命中、`P1` miss、`P2/P3` 存在，系统完全可以一边计算 `P1`，一边加载 `P2/P3`，在后续计算消费完整历史前合并 page table。
但当前 Storage API、Controller、树插入和 Scheduler 都只表达一个最长连续前缀。
支持 gapped reuse 需要多区间命中表示、分段 prefill、提前分配和提交多段 page table、多个 I/O 的锁和失败回收，以及考虑 kernel、PCIe/NVLink、HBM 和 SM 竞争的 cost model。
当前 HiCache 已对连续前缀使用后台 I/O、独立 load stream 和 layer event 重叠搬运与计算，见 `cache_controller.py:782-835`、`996-1088`。

### 1.4 PD 分离中的 KV 生产与消费

默认 PD 模式下，Prefill 是 prompt KV 的汇集与补全节点：它先查询本地 L1，若启用 HiCache 则继续利用 L2/L3，再对首个 miss 后的 suffix 执行模型计算。
最终 Prefill GPU 中形成可供本请求使用的完整 prompt KV，然后通过 PD transfer 写入 Decode 预分配的 GPU slots。
因此传给 Decode 的 KV 可能一部分来自 Prefill 缓存恢复，另一部分来自本轮计算；默认 Decode 将二者作为统一的 prompt KV 消费。

```text
请求 tokens
    → Prefill L1/L2/L3 match
    → 计算剩余 suffix
    → Prefill GPU 中形成完整 prompt KV
    → P→D transfer
    → Decode 请求级 GPU KV
    → 逐 token 生成
```

这里还必须区分 L3 read 与 write。`batch_exists/batch_get` 命中可以加速当前 Prefill；`batch_set` 把新计算 KV 写入 L3，主要收益属于未来请求，本轮反而承担额外 I/O。也要区分 Mooncake 的两个角色：`--disaggregation-transfer-backend mooncake` 是当前请求的 P→D 在线传输，`--hicache-storage-backend mooncake` 是供未来请求按 page hash 查询的 L3，二者的目的、索引和生命周期不同。

`arg_groups/pd_disaggregation_hook.py:54-83` 规定，Decode 未设置 `--disaggregation-decode-enable-radix-cache` 时会被强制 `disable_radix_cache=True`，使用 ChunkCache。默认 `disaggregation/decode.py:1013-1046` 令 `prefix_len=0`、`total_prefix_len=0`，`decode.py:1476-1614` 为请求预分配 slots，所以 Prefill 通常传输整个 prompt 范围。

显式开启 Decode Radix 后，`decode.py:552-566` 执行 Decode 本地 prefix match，Prefill 传输 `[total_prefix_len, origin_input_len)`。同时设置 `--enable-hierarchical-cache` 时，Scheduler 令 `enable_decode_hicache=True`，见 `managers/scheduler.py:412-417`，Decode 通过 `decode_hicache_mixin.py` 查询自己的 L1/L2/L3。这是一条实验性可选路径，支持范围受 speculative decoding、HiSparse、fake backend、Decode DCP 及部分 Hybrid 模型组合约束。

另一条可选路径是 `--disaggregation-decode-enable-offload-kvcache`。`decode_kvcache_offload_manager.py` 把 Decode 新生成的 page 异步写入临时 Host Pool，再写到共享 L3；storage ack 后释放 Host slots。它可以与默认 ChunkCache 共存，用途是让未来 Prefill 复用多轮输出；当前 Decode 的 L3 查询能力由 Decode Radix/HiCache 配置决定。

因此默认 PD 的职责可以概括为：Prefill 负责从 L1/L2/L3 恢复并计算缺失 KV，Decode 主要通过 P→D 消费这些 prompt KV；Decode 直接查询 L3 是可选实验性模式，Decode 向 L3 异步回写则是另一条面向未来请求的路径。

### 1.5 ChunkCache 的请求生命周期、SWA 回收与私有 DRAM

`mem_cache/chunk_cache.py:35-112` 中的 `ChunkCache` 采用请求私有 KV：`match_prefix()` 返回空结果，`insert()` 和 `evict()` 是 no-op，公共 `evict_from_tree_cache()` 在 `mem_cache/common.py:105-130` 直接返回。其资源管理围绕请求生命周期回收展开。

请求完成或中止时，`common.py:132-175` 的 `release_kv_cache()` 调用 `ChunkCache.cache_finished_req()`，释放 `[cache_protected_len, effective_kv_committed_len)` 的请求自有 KV，再释放 speculative 等场景的 overallocated tail，最后归还 `ReqToTokenPool` 行并令 `req.kv=None`。Chunked prefill 中间暂停时，`cache_unfinished_req()` 把已有 indices 保存到同一请求的 `prefix_indices`，供下一 chunk 在请求私有缓存上继续。

Decode 内存不足时，RadixCache 可以先淘汰可回收树叶；ChunkCache 通过 retraction 回收请求私有空间。`schedule_batch.py:2668-2723` 按策略撤回一个或多个活跃请求，并通过 `release_kv_cache(..., is_insert=False)` 释放整段请求 KV；最后一个请求仍无法完成下一步分配时，源码将其中止。

`SWAChunkCache` 继承 ChunkCache，沿用请求私有资源管理，并增加 Sliding Window Attention 的确定性回收。Hybrid SWA 模型由 `SWATokenToKVPoolAllocator` 维护 Full 与 SWA 两个 pool，并通过 `full_to_swa_index_mapping` 关联物理 slots；`mem_cache/common.py:47-95` 的 `free_swa_out_of_window_slots()` 按 page 对齐释放窗口外 SWA slots，而 Full Attention KV 保留到请求结束。全 SWA 模型使用 `PureSWAChunkCache`，结束时只回收仍处于占用状态的 slots。

当前仓库通过 PD Decode retract 提供请求私有 DRAM 暂存机制，类名沿用现有 cache/allocator 组件。`schedule_batch.py:1593-1610` 的 `Req.offload_kv_cache()` 调用 allocator 的 `get_cpu_copy()`，把各层 KV 按 copy chunk 保存到 `req.kv_cache_cpu`；GPU KV 随后释放。请求恢复时，`disaggregation/decode.py:690-738` 重新预分配 GPU slots，调用 `load_kv_cache()` 拷回并删除 CPU snapshot。

```text
被 retract 的同一 Decode 请求
    → GPU KV 复制为 req.kv_cache_cpu
    → 释放原 GPU KV
    → 等待恢复
    → 分配新的 GPU slots
    → CPU snapshot 拷回
    → 删除 req.kv_cache_cpu
```

这是按 request identity 恢复的私有快照：每个请求持有自己的 CPU 副本，恢复后释放该副本。`memory_pool.py:2174-2215` 的实现前后执行 device synchronize，形成同步 retraction snapshot。HiCache 的共享 L2 则提供统一 Host 容量、跨请求 prefix 复用、Host LRU 和 page hash。Decode async offload manager 中的 Host Pool 承担 L1→L3 临时 staging，storage ack 后立即释放。

### 1.6 部署判断与当前架构边界

标准启动使用普通 RadixCache。长公共前缀、大规模 Prefill、Host 内存充足或需要跨实例共享 Mooncake/HF3FS/NIXL 时，可以选择 HiCache。它同时引入 Host 内存、GPU↔CPU 搬运、hash、异步 I/O、淘汰和失败管理；L2/L3 回载延迟也会影响首 token 时间，因此需要结合重复率、节省的 Prefill 计算、带宽和 SLO 判断。

PD 部署中，Prefill 按工作负载选择普通 Radix 或 HiCache；Decode 默认使用 Chunk/SWAChunk；希望减少重复 P→D 传输时可实验性开启 Decode Radix；需要 Decode 自己查询 L1/L2/L3 时再叠加 hierarchical cache；只希望多轮输出进入共享 L3 时使用 Decode offload manager。上述选择共同组成能力组合矩阵，“Decode 使用 Mooncake”只是其中一个维度。

当前子系统以最长连续前缀为核心抽象。本地树、Host 层和 Storage 层共同维护连续可执行历史。这一设计带来首 miss 停止、page-aligned 分裂和 L3 回填后重建节点等行为；非连续 reuse 则需要更复杂的 Scheduler 与内存管理表示。


## 2. Request 在 Scheduler 中的状态流转与缓存生命周期

本章沿一条生成请求在 Scheduler 中的实际执行顺序展开。
`run_event_loop()` 只是创建调度 stream 的入口包装；
真正持续运行的是 `dispatch_event_loop()` 选择出的 `event_loop_normal()` 或 `event_loop_overlap()`。
请求进入 `waiting_queue` 后完成前缀查询和调度准入，随后构造 L1 索引并执行 prefill；
prefill 结果被提交到 RadixTree，未结束的请求进入 `running_batch` 持续 decode；
请求结束时提交最后一段 KV，释放请求行和非缓存 KV。
启用 HiCache 后，同一条路径在前缀查询阶段增加 L2 命中和 L2→L1 回填，在缓存提交后增加 L1→L2 备份。

下面的图保留理解主架构所需的函数边界。阅读时先沿 `event_loop_normal()` 理解串行语义；
`event_loop_overlap()` 的阶段相同，只是把 CPU 调度与 GPU forward 重叠。

```text
run_scheduler_process()
  -> Scheduler.run_event_loop()
  -> dispatch_event_loop()
  -> event_loop_normal() / event_loop_overlap()
       |
       | recv_requests()
       | process_input_requests()
       |   -> handle_generate_request()
       |   -> waiting_queue.append(req)
       |
       | get_next_batch_to_run()
       |   -> get_new_batch_prefill()
       |   -> for req in waiting_queue:
       |        init_next_round_input()               # 构造 tokens，匹配 prefix
       |          -> tree_cache.match_prefix()
       |        PrefillAdder.add_one_req()            # 准入，选择整段或 chunk
       |          -> optional init_load_back()        # L2 -> L1
       |   -> ScheduleBatch.init_new()
       |   -> ScheduleBatch.prepare_for_extend()
       |        -> alloc_for_extend()                 # request row + suffix slots
       |
       | run_batch()                                  # model forward
       | process_batch_result()
       |   -> prefill 完成后进入 running_batch
       |   -> prepare_for_decode() / decode 循环
       |   -> finished: release_kv_cache()
       |
       +-- retract: 释放请求 KV 后回到 waiting_queue
```

这条链通过同一个可变 `Req` 对象传递核心数据：`init_next_round_input()` 把 `MatchResult` 写入 `req.prefix_indices`、`req.last_node`、`req.host_hit_length` 等字段；
紧随其后的 `PrefillAdder.add_one_req()`、`ScheduleBatch.prepare_for_extend()` 和 `alloc_for_extend()` 继续读取这些字段。

初次阅读可以先沿普通自回归路径理解主线。
dLLM、Mamba、SWA、speculative、PP/PD 和多模态分支扩展输入构造、缓存布局或执行方式，并沿用“接收请求 → 选 batch → forward → 处理结果”的 Scheduler 主架构。
其中 dLLM 是按 mask block 去噪的 diffusion LLM 专用路径。

Scheduler 通过请求所处的容器与 `Req` 字段共同表达状态：位于 `waiting_queue` 表示等待 prefill，位于 `chunked_req` 表示 prefill 仍在分块推进，
位于 `running_batch.reqs` 表示进入 decode，`finished_reason` 非空表示结束，`is_retracted=True` 表示释放 KV 后等待重新调度。

### 2.1 Request 的生命周期字段

`Req` 定义在 `python/sglang/srt/managers/schedule_batch.py:721`。与本章主线直接相关的字段如下：

| 字段 | 生命周期含义 |
|---|---|
| `rid` | 请求身份 |
| `origin_input_ids` | 原始 prompt tokens |
| `output_ids` | Scheduler 已接收的生成 tokens，按轮次追加 |
| `full_untruncated_fill_ids` | `origin_input_ids + output_ids` 的当前完整序列 |
| `extend_range` | 本次 prefill/extend 实际计算的逻辑 token 区间 |
| `prefix_indices` | 已命中并可供本轮 forward 使用的 L1 KV slots |
| `last_node` | 当前请求锁定的最后一个 L1 Radix 节点 |
| `last_host_node` | 当前 token 前缀到达的最后一个 L2 节点 |
| `best_match_node` | HiCache 本轮 L2→L1 回填的目标节点 |
| `host_hit_length` | 位于 L2、尚需恢复到 L1 的连续 token 数 |
| `req_pool_idx` | `ReqToTokenPool.req_to_token` 的请求行编号 |
| `kv` | 请求当前分配的 KV 范围信息，核心字段是 `kv_allocated_len` |
| `kv_committed_len` | 已经完成 forward、具有有效 KV 的序列长度 |
| `cache_protected_len` | 已经由 RadixTree 持有并受节点锁保护的 L1 前缀长度 |
| `finished_reason` | 请求结束标记 |
| `is_retracted` | 请求曾因资源不足退出 running batch，等待重新 prefill |

其中有三种长度：

```text
cache_protected_len
    RadixTree 已持有的前缀，由树节点生命周期负责回收

kv_committed_len
    已完成模型 forward 的 KV 长度

req.kv.kv_allocated_len
    allocator 已经预留的 KV 长度，可能包含尚未提交的预分配尾部
```

它们满足的常见关系是：

```text
cache_protected_len <= kv_committed_len <= kv_allocated_len
```

### 2.2 请求进入 waiting_queue

生成请求在 `Scheduler.handle_generate_request()` 中构造 `Req`，完成参数、token、grammar 和多模态输入处理后，经 `Scheduler._add_request_to_queue()` 进入等待队列，入口位于 `python/sglang/srt/managers/scheduler.py:2539`。

普通单体调度模式中的状态变化是：

```text
Req(
    req_pool_idx=None,
    kv=None,
    prefix_indices=[],
    finished_reason=None,
)
    |
    v
scheduler.waiting_queue.append(req)
```

此时请求只占用 Scheduler 元数据；`ReqToTokenPool` 行与新的 L1 KV slot 会在缓存查询和调度准入之后分配。

每轮调度由 `Scheduler.get_next_batch_to_run()` 驱动。它先处理上轮 batch，再调用 `get_new_batch_prefill()` 尝试从 `waiting_queue` 选择新请求；本轮 prefill 为空时，调用 `update_running_batch()` 推进 decode。该顺序使新 prefill 在满足资源预算时优先形成 batch，既有 decode 请求保留在 `running_batch` 中。

### 2.3 Request 构造前缀查询并形成 L1 命中

Scheduler 遍历 `waiting_queue` 时，在 `python/sglang/srt/managers/scheduler.py:3133` 调用：

```python
req.init_next_round_input(self.tree_cache)
```

`Req.init_next_round_input()` 位于 `schedule_batch.py:1194`。它先调用 `_refresh_fill_ids()`，使：

```text
full_untruncated_fill_ids = origin_input_ids + output_ids
```

普通请求走 `_refresh_fill_ids()`；dLLM 请求使用专用的 block 输入构造，阅读普通调度主线时可以跳过。

然后构造查询参数：

```python
RadixKey(
    token_ids=req.full_untruncated_fill_ids,
    extra_key=req.extra_key,
    limit=req._compute_max_prefix_len(input_len),
)
```

`RadixKey` 的主体是 token 序列，`extra_key` 提供 LoRA、cache salt 等缓存命名空间。默认 `limit=input_len-1`，保留至少一个 token 执行 forward 并产生 logits。`page_size>1` 时，`match_prefix()` 再通过 `page_aligned()` 把可匹配长度向下对齐到完整 page。

查询前的 embed override 和 SWA 代码用于在特殊缓存布局下限制可复用范围。SWA 是 Sliding Window Attention：SWA 层只保留最近窗口的 KV，Hybrid 模型则同时维护 Full Attention 与 SWA 缓存；两类路径共用同一个 `Req` 类型。

普通 `RadixCache.match_prefix()` 位于 `python/sglang/srt/mem_cache/radix_cache.py:355`。它沿 token 前缀树查找最长连续前缀，并拼接沿途节点的 `value`：

```text
RadixTree token path

root
  |
  +-- key=[A B] value=[32 33]
                    |
                    +-- key=[C D] value=[40 41]

request key       = [A B C D E]
device hit        = [A B C D]
prefix_indices    = [32 33 40 41]
last_node         = node([C D])
```

查询结果被写回 `Req`：

```python
req.prefix_indices = match_result.device_indices
req.last_node = match_result.last_device_node
req.last_host_node = match_result.last_host_node
req.best_match_node = match_result.best_match_node
req.host_hit_length = match_result.host_hit_length
req.cache_protected_len = len(req.prefix_indices)
```

`init_next_round_input()` 把 `match_result` 直接写入 `Req`，然后 Scheduler 在同一个 `for req in waiting_queue` 循环中立刻调用 `PrefillAdder.add_one_req(req, ...)`。因此后续模块消费的是 `req.prefix_indices` 等持久字段。

`prefix_indices` 已经是可直接读取的 L1 slot vector。L1 hit 直接复用 TreeNode 保存且仍处于占用状态的 slots。

Radix 节点负责 token prefix 到物理 slot 的映射：

```text
TreeNode.key       = token segment
TreeNode.value     = 该 segment 对应的 L1 GPU slot IDs
```

TreeNode 保存 slot ID，实际 K/V tensor 位于当前 worker 的本地 KV pool；attention backend 使用 slot ID 访问：

```text
(local_layer_id, slot_id, local_kv_head, head_dim, K/V)
```

### 2.4 HiRadixTree 将命中范围延伸到 L2

`HiRadixCache` 复用同一棵 token RadixTree，并给节点增加 `host_value`。节点在 L1 和 L2 中的状态由两个字段表达：

```text
L1-only:  value=[GPU slots], host_value=None
L1+L2:    value=[GPU slots], host_value=[Host slots]
L2-only:  value=None,        host_value=[Host slots]
```

`HiRadixCache.match_prefix()` 位于 `python/sglang/srt/mem_cache/hiradix_cache.py:1701`。它遍历完整的连续 token 路径，收集仍在 L1 的节点 `value`，并统计其后连续 L2-only 节点的长度：

```text
request tokens      [A B | C D | E]
                     L1    L2    miss

node [A B]          value=[32 33]
                     host_value=[100 101]

node [C D]          value=None
                     host_value=[102 103]

match result:
    device_indices  = [32 33]
    host_hit_length = 2
    last_device_node= node [A B]
    last_host_node  = node [C D]
```

Scheduler 随后把该请求交给 `PrefillAdder.add_one_req()`。资源预算通过并锁住当前 L1 前缀后，`python/sglang/srt/managers/schedule_policy.py:1159` 调用：

```python
new_indices, req.last_node = self.tree_cache.init_load_back(...)
req.prefix_indices = torch.cat([req.prefix_indices, new_indices])
req.cache_protected_len = len(req.prefix_indices)
```

`HiRadixCache.load_back()` 在 `hiradix_cache.py:1335` 完成以下构造：

```text
TreeNode.host_value
    |
    | HiCacheController.load(host_indices)
    v
TokenToKVPoolAllocator 分配目标 L1 slots
    |
    | enqueue Host -> GPU copy
    v
TreeNode.value = new device_indices
    |
    v
req.prefix_indices += new device_indices
```

例如 L2 slots `[102, 103]` 被恢复到新分配的 L1 slots `[70, 71]`：

```text
before load-back:
    req.prefix_indices = [32 33]
    node [C D].value   = None

after load-back:
    req.prefix_indices = [32 33 70 71]
    node [C D].value   = [70 71]
```

`HiCacheController.load()` 负责分配目标 L1 slots 并把搬运操作加入队列。Scheduler 在构造 `ScheduleBatch` 后通过 `tree_cache.ready_to_load_host_cache()` 启动搬运；`HiCacheController.start_loading()` 位于 `python/sglang/srt/managers/cache_controller.py:782`，它按本地 layer 执行 `load_to_device_per_layer()`。每层完成后发布 layer event，模型执行到该层时等待对应事件。对 Scheduler 而言，L2 命中在进入 forward 前已经转换成普通的 `prefix_indices`，后续 batch 构造只处理统一的 L1 slot vector。

### 2.5 调度准入与 L1 索引构造

`PrefillAdder.add_one_req()` 的核心目标是：根据已经命中的 prefix，判断当前 `Req` 能否加入本轮 prefill batch，并决定本轮处理完整 suffix 还是只处理一个 chunk。当前 batch 的候选元素就是 `Req`；通过准入的请求被追加到 `self.can_run_list`，随后 `ScheduleBatch.init_new(can_run_list, ...)` 令它们成为 `ScheduleBatch.reqs`。

其普通模型逻辑可以压缩为：

```text
计算 uncached input 和未来 output 的容量需求
    -> 检查 KV、SWA、chunk 和 batch-size 预算
    -> 锁住 matched prefix
    -> 可选 init_load_back()，将 L2 prefix 恢复到 L1
    -> 设置 extend_range（完整 suffix 或一个 chunk）
    -> can_run_list.append(req) 并更新 batch budget
```

容量估算把未缓存输入、未来生成 token 和 allocator 开销合并成一个保守需求，用来避免当前 batch 过度承诺 KV 内存。Hybrid SWA 额外检查窗口缓存池；只有 SWA 层的 KV 被窗口限制，Full Attention 层仍保留完整历史。L2 命中时，`init_load_back()` 准备目标 L1 slots 并排队 Host→GPU 搬运，再把返回的 slots 追加到 `req.prefix_indices`。

完整 suffix 能放入预算时，`set_extend_range()` 覆盖全部剩余输入；否则在启用 chunked prefill 时只覆盖一个 chunk。函数最终以 `CONTINUE`、`NO_TOKEN` 或 `OTHER` 告诉 Scheduler 是否继续考察 `waiting_queue` 中的后续请求。

模型 context window 与 `chunk_tokens_limit` 属于两个层级：context window 表示单条序列可支持的最大总长度，`chunk_tokens_limit` 表示一次 prefill 调度迭代处理的 token 上限。生产环境通常使用更小的 chunk 将长 prompt 分多轮 prefill，以控制单批计算量和显存压力，并让其他请求在 chunk 之间获得调度机会。Full Attention 层的完整序列 KV 会随各 chunk 逐步累积；`chunk_tokens_limit=None` 时，Scheduler 在资源允许的情况下整段处理剩余 prompt。

#### 2.5.1 Chunked prefill 的跨轮循环

Chunked prefill 使用三个位置表达同一个 `Req` 在不同阶段的归属：

| 位置 | 含义 |
|---|---|
| `adder.new_chunked_req` | 本轮从 `waiting_queue` 新识别出的长请求 |
| `scheduler.chunked_req` | Scheduler 跨调度轮保存的活动 chunked 请求 |
| `adder.can_run_list` / `ScheduleBatch.reqs` | 本轮实际执行当前 chunk 的 batch 成员 |

首次遇到长请求时，`add_one_req()` 同时完成当前 chunk 的调度和跨轮状态的建立：

```text
waiting_queue 中的 req
    |
    | init_next_round_input(tree_cache)
    | add_one_req(req)
    v
input_tokens > chunk_tokens_limit
    |
    +-- req.extend_range = [prefix_len, prefix_len + chunk_len)
    +-- adder.can_run_list.append(req)       # 当前 chunk 进入本轮 batch
    +-- adder.new_chunked_req = req          # 标记后续仍有 prompt
    v
Scheduler 在 waiting_queue 循环结束后完成 handoff
    |
    +-- scheduler.chunked_req = adder.new_chunked_req
    +-- req.inflight_middle_chunks += 1
    +-- accepted req 从 waiting_queue 移出
    v
ScheduleBatch.prepare_for_extend() -> run_batch()
```

下一轮 `get_next_batch_to_run()` 先通过 `stash_chunked_request()` 把上一 chunk 产生的 KV 提交到 RadixTree，使 `req.prefix_indices` 推进到上一 chunk 的结尾。
随后 `_get_new_batch_prefill_raw()` 优先处理 `scheduler.chunked_req`：

```text
get_next_batch_to_run()
    |
    | stash_chunked_request(req)
    |   -> cache_unfinished_req()
    |   -> prefix_indices 推进
    v
_get_new_batch_prefill_raw()
    |
    | req.init_next_round_input()
    | scheduler.chunked_req = adder.add_chunked_req(req)
    v
add_chunked_req(req)
    |
    +-- 计算 remaining = full_len - len(prefix_indices)
    +-- 设置下一段 extend_range
    +-- can_run_list.append(req)
    |
    +-- remaining > current chunk budget
    |      -> return req
    |      -> scheduler.chunked_req 继续持有 req
    |      -> 下一轮再次进入本循环
    |
    +-- remaining <= current chunk budget
           -> 当前 range 覆盖 final chunk
           -> return None
           -> scheduler.chunked_req 清空
           -> req 仍由 can_run_list/ScheduleBatch 持有
           -> final prefill result 后进入 running_batch
```

完整循环可以压缩为：

```text
                    +--------------------------------------+
                    |                                      |
                    v                                      |
scheduler.chunked_req                                      |
    -> stash previous chunk                                |
    -> add_chunked_req()                                   |
    -> extend_range = next chunk                           |
    -> ScheduleBatch -> prefill forward                    |
    -> middle chunk: return req ---------------------------+
    -> final chunk:  return None -> running_batch -> decode
```

例如 `full_len=100`、初始 `prefix_len=20`、每轮 chunk budget 为 30：

| 调度轮 | 调用 | `extend_range` | 下一轮 stash 后 prefix | `scheduler.chunked_req` |
|---:|---|---|---:|---|
| 1 | `add_one_req()` | `[20, 50)` | 50 | `req` |
| 2 | `add_chunked_req()` | `[50, 80)` | 80 | `req` |
| 3 | `add_chunked_req()` | `[80, 100)` | 100 | `None`，转入 final prefill 处理 |

`inflight_middle_chunks` 记录已经启动、结果仍在处理链中的中间 chunk 数量，主要服务于 overlap event loop。启动 middle chunk 时计数增加，`process_batch_result_prefill()` 消费对应结果时计数减少；正计数表示请求仍处于 prompt 分块阶段，结果处理器据此累计 logprob/KV 状态并暂缓 token stream。final chunk 到达结果处理器时，中间计数已经归零，结果处理器追加首个生成 token，并让请求进入后续 decode 生命周期。

每个 Scheduler 同时维护一个活动 `chunked_req`，`assert self.chunked_req is None` 保证首次 handoff 的单一性；
其他普通等待请求仍可根据 mixed-chunk 配置和剩余预算加入同一个 prefill batch。

```text
waiting_queue
    |
    | match + budget + node lock
    v
PrefillAdder.can_run_list
    |
    | remove accepted requests from waiting_queue
    v
ScheduleBatch(forward_mode=EXTEND)
```

`ScheduleBatch.prepare_for_extend()` 位于 `python/sglang/srt/managers/schedule_batch.py:2222`。
它以 `prefix_indices` 为分界，构造本轮输入：

```python
input_ids = req.get_fill_ids()[len(req.prefix_indices):]
prefix_len = len(req.prefix_indices)
extend_len = req.extend_range.length
```

随后调用 `alloc_for_extend()`，入口位于 `python/sglang/srt/mem_cache/allocation.py:303`。
L1 索引分两步建立，第一步由 `ReqToTokenPool.alloc(reqs)` 分配请求行：

```text
req.req_pool_idx = row_id
```

`ReqToTokenPool` 是二维映射表：

```text
req_to_token[req_pool_idx, logical_token_position] = L1 slot_id
```

`free_slots` 保存可用的 request row IDs，因此可以直接使用 Python list 切片；
`TokenToKVPoolAllocator` 负责管理 L1 KV slots。

第二步由 `TokenToKVPoolAllocator` 为 miss suffix 分配 `out_cache_loc`，然后 `write_cache_indices()` 把命中前缀和新分配 suffix 写入同一请求行：

```text
request tokens       [A  B  C  D  E  F]
                      \-- hit --/ \miss/

prefix_indices       [32 33 40 41]
out_cache_loc                     [80 81]

req_to_token[row]    [32 33 40 41 80 81]
```

这一行就是 attention backend 的请求级 page table。
它把逻辑 token position 转换为 L1 物理 slot。三个相关索引各自承担固定职责：

```text
req_pool_idx     选择 req_to_token 的哪一行
prefix_indices   表示 Radix/L2 load-back 已提供的历史 L1 slots
out_cache_loc    表示本轮 forward 写入新 KV 的 L1 slots
```

`alloc_for_extend()` 同时创建或更新 `req.kv`，把 `kv_allocated_len` 设置为本轮分配后的序列长度。
`prepare_for_extend()` 把 `kv_committed_len` 设置为本轮将执行的 `seq_len`，并把 `req_pool_indices` 与 `out_cache_loc` 写入 `ScheduleBatch`，交给 worker。

### 2.6 Prefill 生产 KV 并提交 L1 前缀

Prefill forward 使用 `req_pool_indices` 读取每个请求的完整 slot row，使用 `out_cache_loc` 写入本轮 suffix。
`out_cache_loc` 按 token 分配一次，在当前 worker 的每个本地 attention layer 上复用：

```text
out_cache_loc = [80 81]

local layer 0: K/V[..., 80, 81, ...] <- layer 0 produced KV
local layer 1: K/V[..., 80, 81, ...] <- layer 1 produced KV
local layer 2: K/V[..., 80, 81, ...] <- layer 2 produced KV
```

因此一个 token slot 是 KV pool 的 token 维坐标。
layer 维由当前 attention layer 和 KV pool 的本地 layer range 给出。
forward 完成时，`req_to_token[row]` 指向的 suffix slots 已经包含有效 KV。

Prefill 结果由 `BatchResultProcessor.process_batch_result_prefill()` 处理，
入口位于 `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:187`。
普通生成请求会把采样得到的第一个 token 追加到 `req.output_ids`。
这个 token 是本轮 logits 的结果，并在下一轮作为输入执行 forward、生成对应 KV。

请求未结束时，result processor 调用：

```python
maybe_cache_unfinished_req(req, self.tree_cache)
```

实际进入 `RadixCache.cache_unfinished_req()`，源码位于 `python/sglang/srt/mem_cache/radix_cache.py:495`。
它处理截止到 `extend_range.end` 的 `req.get_fill_ids()`，因此提交范围正好是已经完成 prefill 的 tokens；
刚采样的 token 会在下一轮完成 forward 后进入提交范围。

提交过程是：

```text
token_ids = req.get_fill_ids()
kv_indices = req_to_token[req_pool_idx, :len(token_ids)]
radix_key = page_aligned(RadixKey(token_ids, extra_key))

RadixCache.insert(
    key   = radix_key,
    value = kv_indices,
)
```

这一步只提交 metadata 映射，KV bytes 继续保留在原 GPU slots：

```text
before commit:
    request row owns [80 81]
    physical KV already exists in slots [80 81]

after commit:
    TreeNode.key   = [E F]
    TreeNode.value = [80 81]
    RadixTree keeps these slots available for cross-request reuse
```

若相同 token prefix 已经由另一请求提交，`insert()` 返回已有 canonical prefix。
当前请求重复分配的 slots 被 allocator 释放，`cache_unfinished_req()` 再次 `match_prefix()`，
并把 request row 改写为 canonical slots：

```text
existing tree       [A B] -> [32 33]
current request     [A B] -> [80 81]

commit result:
    free [80 81]
    req_to_token[row, 0:2] = [32 33]
    req.prefix_indices     = [32 33]
```

`cache_protected_len` 随之推进，旧 `last_node` 的锁被释放，新 `last_node` 获得锁。请求继续运行期间，这段 tree-owned L1 prefix 保持受保护状态。

Chunked prefill 使用同一提交路径。一个 chunk 完成后，`Scheduler.stash_chunked_request()` 调用 `maybe_cache_unfinished_req(..., chunked=True)`，把已完成 chunk 提交到 RadixTree；`req_pool_idx` 和 request row 保留，下一 chunk 继续在同一生命周期中追加。

### 2.7 Prefill 转入 Decode 并持续追加 L1

请求在完整 prompt（含 final chunk）完成 prefill 并产生首个 output token 后获得 decode 资格。
该转换跨越两个 Scheduler 迭代：当前迭代执行 EXTEND forward 并把 batch 保存为 `last_batch`；
下一迭代把其中已完成 prompt 的请求合并到 `running_batch`。

```text
迭代 N：执行 prefill

get_next_batch_to_run()
  -> get_new_batch_prefill()
  -> ScheduleBatch.prepare_for_extend()
       forward_mode = EXTEND
  -> run_batch(prefill_batch)
  -> process_batch_result_prefill()
       -> req.output_ids 追加首个生成 token
       -> cache_unfinished_req(req)
  -> scheduler.last_batch = prefill_batch

迭代 N+1：建立 decode batch

get_next_batch_to_run(running_batch, last_batch)
  -> 构造 chunked_req_to_exclude
       middle chunk -> stash_chunked_request() -> 继续 chunk 循环
       final chunk  -> 保留在 last_batch
  -> last_batch.filter_batch(chunked_req_to_exclude)
  -> running_batch.merge_batch(last_batch)
  -> get_new_batch_prefill(running_batch)
       有新 prefill -> 优先返回新 EXTEND batch，running_batch 继续保存
       本轮 prefill 为空 -> update_running_batch(running_batch)
  -> ScheduleBatch.prepare_for_decode()
       forward_mode = DECODE
       alloc_for_decode()
  -> run_batch(decode_batch)
  -> process_batch_result_decode()
```

这里的关键状态转换是：

```text
prefill batch (EXTEND)
    -> process_batch_result_prefill()
    -> last_batch
    -> 过滤 middle chunk
    -> running_batch
    -> update_running_batch()
    -> prepare_for_decode()
    -> decode batch (DECODE)
```

`last_batch` 表示上一调度轮已经执行的 batch，`running_batch` 表示具备持续 decode 生命周期的请求集合。
middle chunk 保持在 `scheduler.chunked_req` 循环中；
final chunk 对应的请求进入 `running_batch`。
Scheduler 在普通模式下优先运行可形成的新 prefill batch，因此 decode-ready 请求可以在 `running_batch` 中等待下一次 decode 机会。
Mixed chunk 模式则调用 `running_batch.prepare_for_decode()` 和 `new_batch.mix_with_running(running_batch)`，让已有 decode 请求与当前 chunked prefill 共享一次 forward batch。

`ScheduleBatch.prepare_for_decode()` 位于 `schedule_batch.py:2890`。
普通非 speculative decode 每轮调用 `alloc_for_decode(batch, token_per_req=1)`：

```text
当前最后一个 output token
    |
    | 本轮作为模型输入
    v
allocator 为每个 request 分配 1 个 L1 slot
    |
    | req_to_token[req_pool_idx, current_seq_len] = slot
    v
decode forward 写入该 token 的 K/V
    |
    v
logits 产生下一个 output token
```

`alloc_for_decode()` 位于 `python/sglang/srt/mem_cache/allocation.py:539`。请求继续使用同一行，并在下一个逻辑位置追加 slot：

```text
before decode:
    req_pool_idx = 7
    req_to_token[7] = [32 33 40 41 80 81]

alloc_for_decode -> slot 95

after allocation:
    req_to_token[7] = [32 33 40 41 80 81 95]
```

`prepare_for_decode()` 同步递增 `req.kv_committed_len` 和 batch 的 `seq_lens`。forward 完成后，`BatchResultProcessor.process_batch_result_decode()` 把新 token 追加到 `output_ids`，更新 finish state。只要请求未结束，它保留 `req_pool_idx`、request row、Radix 节点锁和所有请求私有 L1 slots，下一轮继续追加。

普通 decode 使用运行中 request row 作为当前请求的完整执行索引；跨请求可见的 Radix prefix 在 prefill 提交点、chunk 提交点和请求结束点更新。

### 2.8 请求完成、释放与重新调度

请求达到 EOS、stop 条件或长度上限后，`process_batch_result_decode()` 调用 `_handle_finish_state_updated_req()`，随后进入 `release_kv_cache()`。该函数位于 `python/sglang/srt/mem_cache/common.py:132`。

完成路径先调用 `tree_cache.cache_finished_req()`：

```text
req.effective_kv_committed_len()
    |
    | 只选择已经执行过 forward 的 token 范围
    v
RadixCache.cache_finished_req()
    |
    +-- 将 page-aligned token -> L1 slots 插入树
    +-- 释放重复 prefix slots
    +-- 释放未对齐尾部
    +-- 释放 last_node lock
```

随后 `release_kv_cache()` 释放 allocator 预分配但未提交的尾部，调用 `ReqToTokenPool.free(req)` 归还请求行，并清空请求级 KV 状态：

```text
req.req_pool_idx -> None
req.kv           -> None
request row ID   -> ReqToTokenPool.free_slots
```

已经插入 RadixTree 的 slots 由 TreeNode 持有，并在以后由树淘汰；请求行可以独立释放：

```text
Req 生命周期结束
    |
    +-- request row 归还
    +-- private / overallocated slots 归还
    +-- tree-owned prefix slots 保留
```

Decode 内存不足时，`Scheduler.update_running_batch()` 先尝试从树中淘汰未锁定 L1 节点；容量仍不足时调用 `ScheduleBatch.retract_decode()`。被撤回请求通过 `release_req()` 执行：

```text
release_kv_cache(req, tree_cache, is_insert=False)
req.reset_for_retract()
scheduler._add_request_to_queue(req, is_retracted=True)
```

`is_insert=False` 选择直接回收当前请求 KV 的路径。`reset_for_retract()` 清空 `prefix_indices`、`last_node`、`req_pool_idx`、`kv` 和 committed length，并保留请求 token 历史。请求回到等待路径后重新执行 `init_next_round_input()`，从当时仍存在的 Radix prefix 开始匹配，再为 miss suffix 分配新的请求行与 L1 slots。

### 2.9 HiRadixTree 的 L1→L2 备份与 L1 淘汰

HiCache 对 request 生命周期的扩展发生在 Radix 节点提交之后。`HiRadixCache.insert()` 创建新的 L1 节点时，节点首先持有：

```text
TreeNode.key   = token segment
TreeNode.value = L1 device slots
```

在 write-through 策略达到 hit threshold 后，`HiRadixCache._inc_hit_count()` 调用 `write_backup()`；write-back 策略在 L1 淘汰时调用同一函数。源码入口分别位于 `python/sglang/srt/mem_cache/hiradix_cache.py:972` 和 `hiradix_cache.py:835`。

```text
TreeNode.value
    |
    | HiCacheController.write(device_indices)
    v
Host allocator 分配 L2 slots
    |
    | enqueue GPU -> Host layer-wise copy
    v
TreeNode.host_value = host_indices
```

DMA 完成后，节点成为 L1+L2。以后 L1 空间紧张且节点未被活跃请求锁定时，`HiRadixCache._evict_backuped()` 释放 `node.value` 对应的 GPU slots，并保留 `host_value`：

```text
L1+L2 node
    value=[32 33]
    host_value=[100 101]
        |
        | L1 eviction
        v
L2-only node
    value=None
    host_value=[100 101]
```

未来请求命中该节点时，生命周期回到第 2.4 节：`match_prefix()` 记录 `host_hit_length`，`PrefillAdder` 调用 `init_load_back()`，Host KV 被复制到新 L1 slots，`node.value` 与 `req.prefix_indices` 同时恢复。

这一闭环使用一套 token 前缀和两套物理地址：

```text
                     Radix token prefix
                            |
              +-------------+-------------+
              |                           |
              v                           v
      TreeNode.value              TreeNode.host_value
        L1 GPU slots                 L2 Host slots
              |                           |
              +------ backup/load --------+

Request forward 始终消费 L1 slots：

L1 hit  ------------------------------> req.prefix_indices
L2 hit -> allocate L1 -> Host→GPU ----> req.prefix_indices
miss   -> allocate L1 -> model write -> req_to_token row
```

### 2.10 主调用链

按请求生命周期阅读源码时，调用顺序如下：

```text
Scheduler.handle_generate_request
  -> Scheduler._add_request_to_queue
  -> Scheduler.get_next_batch_to_run
  -> Scheduler.get_new_batch_prefill
  -> Req.init_next_round_input
  -> RadixCache.match_prefix / HiRadixCache.match_prefix
  -> PrefillAdder.add_one_req
       -> HiRadixCache.init_load_back                 # L2 hit
       -> HiCacheController.load                      # L2 -> L1 enqueue
  -> ScheduleBatch.init_new
  -> ScheduleBatch.prepare_for_extend
       -> alloc_for_extend
       -> ReqToTokenPool.alloc
       -> TokenToKVPoolAllocator.alloc
       -> write_cache_indices
  -> model forward                                    # suffix KV -> L1
  -> BatchResultProcessor.process_batch_result_prefill
       -> RadixCache.cache_unfinished_req
  -> running_batch
  -> Scheduler.update_running_batch
  -> ScheduleBatch.prepare_for_decode
       -> alloc_for_decode
  -> model forward                                    # one-token KV -> L1
  -> BatchResultProcessor.process_batch_result_decode
  -> repeat or release_kv_cache
       -> RadixCache.cache_finished_req
       -> ReqToTokenPool.free

HiCache backup side:

RadixCache.cache_unfinished_req / cache_finished_req
  -> HiRadixCache.insert
  -> HiRadixCache.write_backup
  -> HiCacheController.write                          # L1 -> L2
  -> optional L1 eviction keeps host_value
```


## 3. Scheduler、RadixTree 与 HiCache 源码问答整理

本章整理围绕 `event_loop_normal()`、`create_tree_cache()`、
`match_prefix()` 和 HiCache load-back 的源码追踪结论。重点是区分：

1. 前缀查找是否已经完成；
2. 查找结果是 KV 数据还是地址索引；
3. L2 命中何时真正搬入 L1；
4. `HiRadixCache` 与 `HiCacheController` 各自负责什么。

### 3.1 读取 `plan.batch_to_run` 时，前缀查找完成了吗

基础串行调度循环的核心代码是：

```python
plan = self.get_next_batch_to_run(
    running_batch=self.running_batch,
    last_batch=self.last_batch,
)
self.running_batch = plan.running_batch
batch = plan.batch_to_run
```

普通聚合模式中，`get_next_batch_to_run()` 会进入
`get_new_batch_prefill()`，并在 `_get_new_batch_prefill_raw()` 遍历
`waiting_queue` 时执行：

```python
req.init_next_round_input(self.tree_cache)
```

`Req.init_next_round_input()` 同步调用 `tree_cache.match_prefix()`，把结果写入：

```python
req.prefix_indices
req.last_node
req.last_host_node
req.best_match_node
req.host_hit_length
req.swa_host_hit_length
req.mamba_host_hit_length
```

因此，对成功进入当前 prefill batch 的请求，读取 `plan.batch_to_run` 时，
本地前缀查询已经完成。但是“查询完成”不等于“L2 KV 已经全部搬进 L1”；
后者可以继续在独立 stream 上异步进行。

基础 `event_loop_normal()` 只用于 `DisaggregationMode.NULL`，其 batch 直接由
本地 `run_batch(batch)` 执行，并不会再发送到某个 prefill pod。PD Prefill 和
PD Decode 分别使用 `event_loop_normal_disagg_prefill()` 与
`event_loop_normal_disagg_decode()`。

PD Decode 的顺序有所不同：循环在构造 decode batch 之前先调用
`process_decode_queue()`。Decode-side Radix 查询、目标 KV slots 预分配和
`send_metadata()` 均发生在这里。新请求只有在 P→D KV transfer 成功，且所需
Decode HiCache restore 不再 pending 后，才会进入 decode batch。

### 3.2 `create_tree_cache()` 是否真的创建了树

`build_kv_cache()` 中的代码：

```python
tree_cache = create_tree_cache(TreeCacheBuildContext(...))
```

会调用 `mem_cache/registry.py` 中的 factory，并在 Scheduler 初始化时选择一个
具体实现。标准情形包括：

```text
普通模型                       -> RadixCache
普通模型 + hierarchical cache  -> HiRadixCache
Hybrid SWA/SSM/DSA             -> UnifiedRadixCache
关闭 Radix + chunked prefill   -> ChunkCache / SWAChunkCache
```

对 Radix 实现而言，这一步确实创建了实际树对象。以 `RadixCache` 为例，
构造函数调用 `reset()`，立即创建空 root：

```text
root_node
├── key = []
├── value = []
├── children = {}
└── lock_ref = 1
```

此时只有空根节点。模型产生 KV 后，`cache_unfinished_req()` 或
`cache_finished_req()` 再通过 `insert()` 填充树节点。

`ChunkCache` 是例外：它实现统一 `BasePrefixCache` 接口，但没有跨请求共享的
RadixTree，`match_prefix()` 始终返回空命中。

从类型职责看，`KVCacheBuildResult.tree_cache` 的稳定抽象类型应理解为
`BasePrefixCache`，而不是 `RadixCacheMetricsCollector`。Scheduler 最终只保存
一个选中的 cache 实例：

```python
self.tree_cache = result.tree_cache
```

不存在每次请求都在 `RadixCache` 和 `HiRadixCache` 之间路由的第二层 dispatcher。

### 3.3 Metrics Collector 不是树

实际所有关系是：

```text
Scheduler
└── tree_cache: RadixCache / HiRadixCache / UnifiedRadixCache
    └── metrics_collector: RadixCacheMetricsCollector
```

`RadixCacheMetricsCollector` 只记录 eviction、load-back token 数量、耗时等指标。
它不保存 token prefix，不执行树查询，也不决定缓存层级。

调试时可以分别检查：

```python
print(type(self.tree_cache).__name__)
print(type(self.tree_cache.metrics_collector).__name__)
self.tree_cache.pretty_print()
```

### 3.4 `match_prefix()` 前置代码与实际搜索

典型实现先做以下预处理：

```python
if self.disable:
    return self._empty_match_result

key = params.key
key, _ = key.maybe_to_bigram_view(self.is_eagle)
key = key.page_aligned(self.page_size)
if len(key) == 0:
    return self._empty_match_result
```

各步职责如下：

1. Radix disabled 时直接返回 miss；
2. 读取带 token IDs、`extra_key` 和可选长度上限的 `RadixKey`；
3. EAGLE 模式使用重叠 bigram 视图；
4. 将可查询长度截断到完整 page；
5. page 对齐后为空则直接返回 miss。

实际树遍历从这里开始：

```python
value_chunks, last_node = self._match_prefix_helper(self.root_node, key)
```

helper 从 root 开始，根据下一 token/page 查找 `children`，比较压缩节点的
token segment，收集命中节点的 `value`，直到 child 不存在或 token 不同。
若匹配在压缩节点内部结束，helper 可以分裂节点，建立精确的公共前缀边界。

### 3.5 `value` 是 KV slot index，不是 KV payload

虽然局部变量名叫 `value`，它的内容通常是 `torch.int64` KV slots：

```text
token prefix = [101, 42, 7, 8]
value        = [5000, 5001, 5002, 5003]
```

这些数字指向物理 KV pool 中对应 token 位置的各层 K/V。树只保存前缀到位置的
映射，不内嵌整份 K/V tensors。

在 `HiRadixCache` 中：

```text
node.value       -> L1 device slots
node.host_value  -> L2 host slots
node.hash_value  -> L3 page hashes
```

L2-only 节点的典型状态是：

```python
node.value = None
node.host_value = tensor([...])
```

`HiRadixCache._match_prefix_helper()` 可以继续沿 L2-only 节点匹配 token，但只将
未被 device eviction 的 `child.value` 加入 `device_indices`。随后
`match_prefix()` 计算 `host_hit_length`，区分 L1 连续命中与继续延伸的 L2 命中。

### 3.6 L2 命中何时搬入 L1

L2→L1 分为“为每个请求排队”和“batch 级启动”两步。

第一步发生在 `PrefillAdder.add_one_req()`：

```python
if req.needs_host_load_back():
    new_indices, req.last_node = self.tree_cache.init_load_back(...)
    req.prefix_indices = torch.cat([req.prefix_indices, new_indices])
```

`HiRadixCache.init_load_back()` 选择要恢复的 host-only nodes，拼接
`host_value`，再调用：

```python
device_indices = self.cache_controller.load(
    host_indices=host_indices,
    node_id=last_hit_node.id,
)
```

`HiCacheController.load()` 分配 L1 目标 slots，并将 raw copy operation 放入
`load_queue`。`HiRadixCache` 随即把返回的目标 indices 写入 `node.value` 并锁住
相关路径。此时 slots 已预留，但实际 KV bytes 可能仍未复制完成。

第二步发生在 waiting-request 循环结束并创建 `ScheduleBatch` 后：

```python
new_batch.hicache_consumer_index = (
    self.tree_cache.ready_to_load_host_cache()
)
```

`HiRadixCache.ready_to_load_host_cache()` 委托
`HiCacheController.start_loading()`。Controller 合并当前 batch 中各请求此前排队的
load operations，并在独立 load stream 上逐层执行 L2→L1：

```text
for req in waiting_queue:
    match prefix
    admission checks
    queue load-back if needed

after batch selection:
    merge load operations
    start asynchronous layer-wise copies
```

`hicache_consumer_index` 随 batch 传给 TP worker。Attention 读取第 `i` 层 KV
buffer 时，通过 `LayerDoneCounter.wait_until(i)` 等待该层恢复完成。因此 load 和
forward 可以逐层重叠，同时保证每层消费前 KV 已就绪。

最后，`HiRadixCache.check_hicache_events()` 消费 Controller ack，清理
`ongoing_load_back`、释放树锁并更新 metrics。

### 3.7 HiRadixCache 与 HiCacheController 的职责边界

最简洁的心智模型是：

```text
HiRadixCache      = prefix metadata / cache policy control plane
HiCacheController = physical allocation / data movement plane
```

| 职责 | `HiRadixCache` | `HiCacheController` |
| --- | ---: | ---: |
| Token-prefix RadixTree | 是 | 否 |
| Prefix match、insert、split、delete | 是 | 否 |
| 节点 L1/L2/L3 residency metadata | 是 | 否 |
| 节点 lock、LRU、priority、淘汰对象选择 | 是 | 否 |
| 决定恢复或备份哪些前缀节点 | 是 | 否 |
| 分配/释放 raw device 和 host slots | 发起请求 | 是 |
| 执行 H2D/D2H copy | 发起请求 | 是 |
| 管理 transfer stream、queue 和 event | 否 | 是 |
| 消费 ack 并提交 tree semantic state | 是 | 提供 ack |
| L3 操作的树锚点和请求跟踪 | 是 | 否 |
| L3 backend 查询、读写和后台线程 | 发起请求 | 是 |

`HiRadixCache` 维护 `ongoing_load_back`、`ongoing_write_through`、
`ongoing_prefetch` 和 `ongoing_backup`，把 Controller 的 operation ID 重新映射回
具体 TreeNode。Controller 只接收 raw indices、node ID、hash 和 token data，
并不知道这些数据在 RadixTree 中代表哪条共享前缀。

L2→L1 的完整边界如下：

```text
HiRadixCache.match_prefix()
    确定 L1/L2 命中边界
        |
HiRadixCache.init_load_back()
    选择 nodes，收集 host indices
        |
HiCacheController.load()
    分配 device slots，排队 copy
        |
HiRadixCache
    记录 provisional node.value，持有 locks
        |
HiCacheController.start_loading()
    启动逐层 H2D
        |
HiCacheController events/acks
        |
HiRadixCache.loading_check()
    确认语义完成，释放 locks
```

L1→L2 使用对称分工：`HiRadixCache.write_backup()` 选择 TreeNode，Controller
分配 Host slots 并执行 D2H；ack 到达后 HiRadixCache 发布 CPU residency，必要时
再要求 Controller 写入 L3。

结论是：`HiCacheController` 不是另一个 cache router，而是
`HiRadixCache` 用来落实层级 residency 决策的异步数据搬运执行器。
