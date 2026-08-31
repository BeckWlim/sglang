# SGLang HiCache 与 Mooncake L1–L4 KV Cache 架构调研报告

## 1. 核心结论

SGLang HiCache 与 Mooncake 共同构成一套由本地前缀索引、本地分层存储、外部对象存储和异构数据传输组成的 KV Cache 系统。
L1 位于 GPU VRAM，由设备 KV pool 保存 forward 可直接消费的 K/V；
L2 位于推理实例的 Host DRAM，由 Host KV pool 保存本地备份并承接外部存储读取；
SGLang 将 Mooncake 暴露为一个统一的外部 storage tier，Mooncake 内部再把该层细分为共享内存 L3 和 SSD、NVMe-oF、文件系统等更冷的 L4 介质。
因此，“L1/L2/external（通常称逻辑 L3）”描述当前 SGLang 接口，而“MEMORY L3/L4”描述 Mooncake 内部副本介质和恢复代价；本文在每处按上下文标明所用口径。

SGLang 掌握 token、请求逻辑位置、prefix 连续性、模型并行布局和 forward 依赖。
HiRadixCache 将 token prefix 映射到本地 L1/L2 pool indices，Scheduler 根据命中长度决定 prefill admission，HiCacheController 执行 L1/L2 之间的异步搬运。
Mooncake 掌握 object key、ObjectMetadata、Replica Descriptor 和存储资源。
Master 负责对象元数据、空间、租约和副本可见性，KV payload 沿 Client 数据面传输；Client 既是发起 Put/Get 的应用客户端，也可以作为托管已注册内存段的 Store server。
Client 的 `TransferSubmitter` 根据副本介质把字节操作分别分派给本地 memcpy、经典 Transfer Engine（或其 TENT 兼容执行路径）、文件后端或者 SPDK NVMe-oF worker。

当前典型的 external-storage 读取路径为：

```text
请求 token
  -> SGLang 本地 L1/L2 prefix match
  -> SGLang 生成剩余完整 page 的链式 hash
  -> Mooncake 精确查询 object 与 COMPLETE replica
  -> Mooncake 按 replica 介质执行 memory/file/SSD/NVMe-oF 数据路径
  -> 数据最终写入 SGLang L2 Host page
  -> HiCacheController 按 layer 回载到 L1 GPU slot
  -> forward stream 按 layer-ready event 消费 KV
```

对于 Mooncake `MEMORY` replica，backend 的 zero-copy 范围位于远端内存与 L2 之间：网络传输可以直接进入已经注册的 SGLang Host page，省去请求方独立结果缓冲区。
对于 `LOCAL_DISK` replica，磁盘数据当前先进入持有节点的临时 ClientBuffer，再由网络段直接写入请求方 L2；整条 L4 路径由 provider-side staging 和 requester-side direct placement 共同组成。
L2→L1 仍由 HiCacheController 完成，并通过逐层事件与模型计算重叠。

L1 prefix 命中采用原址复用。
已有 prefix slot 保持原物理位置，新 suffix 获得新的 L1 slot，`req_to_token` 把这些离散 slot 组织成一个连续的请求逻辑序列。
L2 prefix 命中采用显式 load-back：`host_indices` 指向 L2 源位置，`device_indices` 指向预留的 L1 目标位置。

一次 `start_loading()` 对应一轮合并后的 load wave，通常包含当前新 batch 内多个请求的 Host hit。
每套 layer event 服务于一个 load wave，每个 event 表示该轮操作在一个本地 layer 上的 K/V 已进入 L1。
该机制实现 layer-wise 数据搬运与 forward 重叠；
TP 负责同一次 forward 的权重与 KV shard 协作，PP 负责模型 layer 的 stage 划分，三者具有独立职责。

Mooncake 外部对象采用稳定 object key 进行跨实例共享。
SGLang 按 `page_size` 切分 token，计算带父页依赖的链式 page hash，再结合模型 namespace、K/V 类型和 TP/PP shard 规则展开成物理 object key。
Mooncake Master 对 `(tenant, object key)` 执行精确元数据查询，并返回可读副本。
各 SGLang 实例保留自己的本地 radix tree，跨实例发现通过稳定 key 完成。

当前架构在多个位置提供策略选择：
- L2 layout 决定 page I/O 与逐层 H2D 的组织方式；
- write policy 决定 KV 何时进入 L2/外部层；
- storage prefetch policy 决定请求等待远端命中的时间；
- Mooncake placement 与 replica policy 决定对象容量、局部性和读取带宽。

策略应以 prefix 重算成本、预期复用、SLO、L1/L2 空间和网络队列为共同依据。

从 SGLang 视角看，Mooncake 将共享 DRAM 进一步下沉到 SSD 时，L4 首先表现为 storage backend 内部的更冷 replica tier。
page key 和 SGLang storage 接口保持稳定，Mooncake 负责解析和读取所选介质；
当前高性能读取仍以 `MHATokenToKVPoolHost` 的 page buffer 为目标。
显式区分 L3/L4 的调度成本、将 L4 命中晋升到 L3，以及 L4→L1 direct restore，分别需要在 backend result、存储策略和目标内存接口上增加能力。

后续可在现有精确索引和状态管理基础上增加 L3 DRAM→L1 GPU direct restore。
Mooncake Transfer Engine 已具备 DRAM/VRAM 和 GPUDirect RDMA 传输能力。
SGLang 侧需要补充 L1 page reservation、GPU destination descriptor、逐层或逐对象完成事件、TP rank 一致性及失败回滚。
该路径适合已经准入并即将计算的前台请求；
现有 external→L2→L1 路径继续承担共享复用、可取消读取和 L1 空间不足时的 staging；其 `MEMORY` replica 分支即 L3→L2→L1。

## 2. 层级定义与职责边界

本文中的 L1、L2 和 external tier 描述 KV Cache 的软件存储层级，与 GPU 芯片内部的硬件 cache 层级分别讨论。
为兼容 SGLang HiCache 的常用说法，external tier 有时也被称为逻辑 L3；只有明确写出 Mooncake `MEMORY` L3 或 L4 时，才表示物理介质层次。

| 层级 | 典型介质 | 作用域 | 普通 MHA/GQA 实例 | 主要管理者 | 核心作用 |
|---|---|---|---|---|---|
| L1 | GPU VRAM/HBM | 当前 TP rank | `MHATokenToKVPool` | SGLang allocator、prefix cache、TP worker | 保存 forward 直接读取的 K/V |
| L2 | CPU Host DRAM | 当前推理实例/rank | `MHATokenToKVPoolHost` | HiRadixCache、HiCacheController | 本地容量、L1 备份、external-storage 接收和布局适配 |
| 外部层（SGLang 口径） | Mooncake MEMORY、SSD、NVMe-oF、文件后端 | 多实例共享或后端定义的范围 | Mooncake Store objects | Mooncake Master、Client、介质对应的数据面 | 跨实例共享、对象生命周期与外部恢复；本文按介质再称 L3/L4 |

普通 MHA/GQA 模型在启用 HiCache 后通常形成以下对象关系：

```text
Scheduler
  -> BasePrefixCache interface
       -> HiRadixCache / HiCache-enabled UnifiedRadixCache
            -> MHATokenToKVPool          L1
            -> MHATokenToKVPoolHost      L2
            -> HiCacheController
                 -> StorageBackend
                      -> MooncakeStore    external-storage adapter
```

SGLang 的 prefix cache 保存请求语义与本地物理位置之间的关系。
Mooncake Store 保存稳定对象名与分布式副本位置之间的关系。
Mooncake Client 再根据已经解析的副本介质、源、目标和长度选择相应数据路径；其中经典 Transfer Engine 主要承担已注册内存段之间的高速传输。
三部分分别处理 prefix、object 和 bytes，从而使本地 slot 的频繁变化与分布式对象的稳定命名相互独立。

## 3. SGLang KV Cache 的创建与本地索引结构

### 3.1 KV pool 与 `tree_cache` 的创建

`build_kv_cache()` 创建请求映射池、设备 KV pool allocator 和 prefix cache，随后返回 `KVCacheBuildResult`。其中：

- `req_to_token_pool` 保存请求逻辑位置到 L1 slot 的映射；
- `token_to_kv_pool_allocator` 管理设备 KV slot/page；
- `tree_cache` 保存 token prefix 到本地 L1/L2 cache state 的索引；
- hybrid、sliding-window 和缓存开关字段描述当前模型所需的附加布局。

`create_tree_cache(TreeCacheBuildContext(...))` 会构造实际 prefix-cache 对象。
factory 综合模型结构、radix backend、hierarchical cache、unified cache 和 streaming session 配置，
选择 RadixCache、HiRadixCache、UnifiedRadixCache、ChunkCache 或相应 wrapper。
在普通 Transformer + HiCache 场景中，实际对象提供 HiRadixCache 语义：radix node 可以关联设备 value、Host value、page hash、父子关系、引用锁和异步生命周期状态。

`RadixCacheMetricsCollector` 是 prefix cache 的观测组件，作为 `BasePrefixCache.metrics_collector` 成员记录 eviction、load-back 等指标。
实际 token tree、node 和 L1/L2 indices 由 prefix-cache implementation 保存。

每个 Scheduler/worker 持有自己的本地 prefix cache。
该本地结构负责当前 rank 的 prefix match 和 slot 生命周期；
Mooncake 外部层的跨实例共享通过 page hash 与 object key 完成。

### 3.2 四类索引与地址域

SGLang 与 Mooncake 的完整链路包含四类含义稳定的索引：

| 索引 | 输入与输出 | 生命周期 | 典型用途 |
|---|---|---|---|
| 请求逻辑映射 | `req_to_token[row, position] -> L1 slot` | 请求运行期间 | attention 查找逻辑 token 对应的 KV |
| L1 pool index | `device_indices` | 当前设备 pool 分配期间 | L1 hit、prefill 写入、decode 追加 |
| L2 pool index | `host_indices` | 当前 Host pool 分配期间 | L1 备份、L2 hit、external read 目标 |
| external object key | page hash 展开后的稳定字符串 | object 生命周期 | 跨实例 exact lookup 和 replica 解析 |

L1/L2 index 是当前进程 allocator 中的位置，随着分配和淘汰变化。
External object key 表达 token prefix 和 KV shard 身份，可以在实例重启或本地 slot 改变后重新计算。
Mooncake Replica Descriptor 再把 object key 解析到某个具体 endpoint、地址、大小或存储介质。

### 3.3 Radix node 中的 L1/L2 状态

HiRadixCache 使用 token prefix 组织节点。一个节点可以同时关联设备和 Host 状态：

```text
Radix node
  key/token prefix
  value       -> L1 device indices
  host_value  -> L2 host indices
  hash_value  -> 对应完整 page 的链式 hash
  parent/children/locks/lifecycle state
```

L1 与 L2 副本可以共存。
L1 eviction 释放 device indices 后，Host value 可以继续支撑本地恢复；
L2 eviction 释放 Host indices 后，external object 仍可由相同 token page 重新生成 key 并查询。
缓存层级变化体现为副本集合变化，prefix 的逻辑身份保持稳定。

## 4. L1 Prefix 命中与逻辑序列组合

### 4.1 `match_prefix()` 的输出

请求进入下一轮调度前，`Req.init_next_round_input()` 调用 `tree_cache.match_prefix()`。
HiRadixCache 沿 root node 和 token key 查找最长连续本地 prefix，并返回：

- `device_indices`：已经驻留 L1 的 prefix slots；
- `last_device_node`：设备命中的边界节点；
- `last_host_node`：Host 命中的边界节点；
- `host_hit_length`：普通 KV pool 中需要 L2→L1 的 token 数；
- SWA、Mamba 等附加 pool 的对应 Host hit length；
- `best_match_node` 与分支相关状态。

`value, last_node = _match_prefix_helper(root_node, key)` 完成 radix traversal。
这里的 `value` 是一组设备 pool indices，它把 token prefix 映射到已经存在的 L1 KV 数据。
实际 K/V payload 保存在各 layer 的设备 tensor 中。

### 4.2 Prefix 原址复用

L1 命中的 prefix slots 保持原位置。
prefill 为 cache-miss suffix 分配新的 slots，`write_cache_indices()` 将 prefix 与 suffix 写入同一条请求映射：

```text
logical position     0    1    2    3    4
req_to_token        81   82   19   44   45
                    L1 prefix  prefill suffix
```

逻辑序列连续性由 `req_to_token` 表达，物理 slot 可以离散分布。
attention backend 根据请求 mapping gather 对应 K/V。
这个结构使已有缓存 prefix、Host 恢复 prefix 和新生成 suffix 都可以使用各自的 allocator 位置，并在请求级别形成统一上下文。

prefill/decode 分离时，KV sender 同样读取请求 mapping，将目标 token range 转换成 page indices 后发送。
传输路径沿逻辑位置解析物理页，因此本地 pool 无需为网络发送重新整理成连续 prefix/suffix buffer。

### 4.3 L1 容量策略节点

L1 同时承载运行中的工作 KV 和等待后续复用的缓存 KV。
运行中 decode、即将执行的 prefill 以及在途 load-back 目标 page 具有直接计算依赖；
空闲 prefix page 体现未来复用价值。
容量策略应先满足活跃工作集和在途目标预留，再对可驱逐缓存执行 LRU、LFU、SLRU 或 priority policy。

| 对象状态 | 主要价值 | 适合的处理 |
|---|---|---|
| 运行中 decode KV | 每个 decode step 重复读取 | 引用锁保护，维持 L1 resident |
| 已准入 batch 的 load-back 目标 | 即将被当前 forward 消费 | 短期预留至完成事件和请求 mapping 建立 |
| 高频共享 prefix | 节省后续 prefill | 按 reuse frequency、recency 和重算成本保护 |
| L2/external 已有副本的低频 prefix | 恢复成本可控 | 作为较早的 L1 eviction 候选 |

## 5. L2 Host 存储与 load-back

### 5.1 L2 的三项作用

L2 Host pool 在当前架构中承担三个相互关联的角色。

- 实例私有的容量层。
L1 中暂时空闲的 prefix 可以备份到 L2，后续请求命中后通过 H2D 恢复，从而减少重复 prefill。

- L1 与外部 storage backend 之间的布局适配层。
Host pool 可以采用 page-first 类布局，使完整 token page 形成适合批量存储 I/O 的 buffer；
L1 仍以 attention backend 所需的逐层 K/V tensor 提供计算数据。

- external-storage 读取的接收层。
Mooncake adapter 注册 Host pool 的 page buffers，远端 object 直接写入目标 Host page。
读取完成后，这些 page 被插入本地 radix tree并转换为普通 Host hit。

### 5.2 Host hit demand 与 L1 目标分配

本地 match 将 Host 命中长度写入 Req。
`needs_host_load_back()` 汇总普通 KV、SWA 和 Mamba 等 pool 的 Host hit 状态，作为 admission 阶段建立 load-back 的统一条件。

`PrefillAdder.add_one_req()` 在接受该请求时调用 `tree_cache.init_load_back()`。
这一步完成两项准备：为 Host hit 分配 L1 目标 slots，并通过 HiCacheController 建立 cache operation。
operation 中的索引方向为：

```text
op.host_indices    -> L2 Host pool source
op.device_indices  -> L1 GPU pool destination
```

此时 copy 已进入 load queue，数据传输由 batch 构建结束后的 `start_loading()` 统一提交。

### 5.3 L2 layout 与 I/O backend 选择

Host layout 同时影响 external↔L2 的 page I/O 和 L2↔L1 的逐层恢复。

| Layout | 数据组织特征 | 适用方向 | 主要取舍 |
|---|---|---|---|
| `layer_first` | 同一 layer 的 token 数据集中 | 逐层 H2D 路径直观 | 外部 backend 需要处理 page 的跨 layer 分布 |
| `page_first` | 同一 token page 的多层数据集中 | 适合完整 page 的存储读写 | L2→L1 需要按 layer 提取 |
| `page_first_direct` | page-oriented，并适配 direct I/O | Mooncake/HF3FS 等外部 storage | 依赖 backend、attention layout 和硬件带宽验证 |

普通 MHA/GQA 使用 Mooncake backend 时，官方配置采用 `page_first` 或 `page_first_direct`。
选择依据包括 page bytes、H2D backend、远端 batch size、Host 内存带宽和目标 attention implementation。[SGLang HiCache Best Practices](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/hicache_best_practices.mdx)给出了 layout 与 I/O backend 的兼容配置。

## 6. Scheduler 中的本地缓存调度

### 6.1 从请求接收到 batch plan

`event_loop_normal()` 接收请求并调用 `process_input_requests()`，随后通过 `get_next_batch_to_run()` 建立 `NextBatchPlan`。
新 prefill 请求由 `get_new_batch_prefill()` 进入 `_get_new_batch_prefill_raw()`；
当前运行中的 decode batch、活动 chunked request、prefill token budget 和最大 batch size共同参与 admission。

`_get_new_batch_prefill_raw()` 遍历 Scheduler waiting queue。每个请求的核心顺序为：

```text
Req.init_next_round_input(tree_cache)
  -> 本地 L1/L2 match
  -> 将 device indices 与 Host hit length 写入 Req

PrefillAdder.add_one_req(req)
  -> 计算 prefix、extend 和 chunk budget
  -> 对 Host hit 调用 init_load_back()
  -> 将可运行请求加入 can_run_list
```

循环结束后，Scheduler 从 waiting queue 移除已接受请求，用 `ScheduleBatch.init_new()` 创建新 batch，
并通过 `prepare_for_extend()` 分配 suffix slots、写入请求 mapping 和准备 forward metadata。

### 6.2 本地搜索完成与数据就绪的时间边界

`match_prefix()` 返回后，本地索引搜索已经完成。此时不同层级具有不同的数据状态：

| 命中状态 | 已完成工作 | 后续工作 |
|---|---|---|
| L1 hit | prefix slot 已知且位于设备 pool | 写入 `req_to_token` 后直接复用 |
| L2 hit | Host source slot 与命中长度已知 | admission 时分配 L1 slot并提交 load-back |
| external candidate | 剩余 page hash 可计算 | 完成 object 查询、L2 分配、外部读取和本地重新匹配 |
| cache miss suffix | 未形成可复用 KV | prefill 计算并写入新 L1 slot |

因此 batch plan 同时携带“已经驻留的 L1 prefix”“正在恢复的 Host prefix”和“需要计算的 suffix”。
Scheduler 通过 mapping、consumer event index 和 extend metadata 把三类状态交给 forward。

### 6.3 外部存储异步预取

启用 storage backend 后，一个新请求通常在进入 waiting queue 前经历第一次本地 match。
`_add_request_to_queue()` 调用 `_prefetch_kvcache()`，后者先通过 `Req.init_next_round_input()` 获得当前 L1/L2 连续命中。
再从 `last_host_node` 对应的 page hash 继续计算本地未命中部分的候选 keys，并调用 `prefetch_from_storage()`。

第一次 `match_prefix()` 的结果来自本地 HiRadixCache。
外部存储命中由后续 storage worker 调用 backend existence 接口获得，二者的查询对象和执行线程分别为：

| 阶段 | 查询结构 | 结果 |
|---|---|---|
| 第一次 `match_prefix()` | 本地 radix nodes、`value`、`host_value` | 当前 L1/L2 连续命中边界 |
| `_storage_hit_query()` | chained page hashes 与 storage backend | external tier 最长连续存在 page 数 |
| `batch_get_v1()`/`batch_get()` | external objects 与 L2 目标 buffers | 已写入 PoolHost 的完整 KV pages |

`prefetch_from_storage()` 把 operation 加入 background prefetch queue。
storage query worker 计算候选 page hashes 并调用 `batch_exists()`；命中长度在相关 ranks 间取共同最小值。
Scheduler thread 收到 storage hit 后，按照实际命中 token 数分配 PoolHost indices，再由 storage I/O worker 执行 external→L2 读取。
这种延迟分配使 Host 容量与实际连续命中对齐。

当 `_get_new_batch_prefill_raw()` 再次遍历 waiting queue 时，会先调用 `check_prefetch_progress(req.rid)`：

- operation 仍需继续执行时，请求保留在 waiting queue，本轮跳过 admission；
- operation 达到当前 prefetch policy 的结束条件时，Scheduler 收割已经完成的连续 pages；
- 完成 pages 通过 `_insert_helper_host()` 写入本地 radix tree，使对应 nodes 获得 `host_value`；
- 请求随后再次调用 `Req.init_next_round_input()`，执行第二次本地 `match_prefix()`；
- 第二次 match 将刚从外部存储取得的数据识别为普通 L2 Host hit，并更新 `host_hit_length`。

因此，当前请求的典型搜索和发布顺序为：

```text
第一次本地 match_prefix
  -> 获得原有 L1/L2 prefix
  -> 生成剩余 page 的 external object keys

后台 storage 阶段
  -> batch_exists 获得 external tier 连续命中
  -> 按实际 hit 分配 PoolHost
  -> batch_get 将 external object 写入 L2
  -> _insert_helper_host 发布 host_value

第二次本地 match_prefix
  -> 将新 host_value 纳入 L2 命中
  -> PrefillAdder 执行 init_load_back
  -> 建立 host_indices 到 device_indices 的 load operation
```

第二次 match 继续使用本地 HiRadixCache；Mooncake object 查询已经在中间的 storage 阶段完成。
这个安排使 Scheduler 后续只处理统一的 L1 hit、L2 Host hit 和 prefill suffix 三种本地状态。
第二次 match 的保证范围是“外部数据已经发布为可恢复的 L2 Host hit”；对应 KV bytes 的 L1 计算可见性随后由 layer events 建立。
`PrefillAdder.add_one_req()` 随后通过 `init_load_back()` 分配 L1 slots、把 `device_indices` 纳入请求 prefix mapping，并将 H2D operation 加入 load queue；
Scheduler 可以据此完成 admission，而实际计算可见性由后续 layer events 建立。

prefetch policy 决定 `check_prefetch_progress()` 的结束时点。
`wait_complete` 等待目标读取完成；`timeout` 在 deadline 处收割已完成连续 pages；`best_effort` 在调度检查时使用当时已经完成的结果。
尚未形成可发布 Host page 的 operation 可以被撤销，已完成 page 进入第二次 match 和标准 load-back。

### 6.4 `start_loading()` 与 layer events

Scheduler 完成 waiting queue admission 并建立新 batch 后，调用 `ready_to_load_host_cache()`。
HiRadixCache 将该调用委托给 `HiCacheController.start_loading()`。

`start_loading()` 将本轮 load queue 中的 operations 合并，把各请求的 Host source indices 连接成一个向量，将对应 Device destination indices 连接成另一个向量。
随后它选择一套可复用的 `LayerDoneCounter` events，在独立 load stream 中按本地 layer 执行 K/V copy。

一次 event set 对应一次 load wave。假设本轮返回 consumer index 1：

```text
events[1][0] -> 本轮所有 layer-0 Host hit KV 已写入各自 L1 slots
events[1][1] -> 本轮所有 layer-1 Host hit KV 已写入各自 L1 slots
events[1][2] -> 本轮所有 layer-2 Host hit KV 已写入各自 L1 slots
```

load stream 完成 layer 0 后记录 `events[1][0]`，forward stream 在读取 layer 0 KV 前等待该事件。
随后 forward 计算 layer 0，load stream继续搬运 layer 1。这样形成一条逐层 producer-consumer pipeline。

这条路径包含四个不同的可见性边界：

| 边界 | SGLang 获得的保证 |
|---|---|
| 第二次 `match_prefix()` | 完整连续 prefix 已作为 `host_value` 存在于 L2，`host_hit_length` 可用于 admission 计算 |
| `init_load_back()` | L1 slots 已预留，`device_indices` 已进入请求/radix mapping，H2D 已排队 |
| `events[consumer][i]` | layer `i` 的 prefix K/V 已物理到达 L1，attention 可以消费该层 |
| whole-load `finish_event` | 本轮所有 layers 完成，可清理 `ongoing_load_back` 并释放保护引用 |

因此，某个 layer 的 suffix 计算完成时，同一 layer 的 prefix H2D 必然已经通过 layer-ready event；后续 layers 的 prefix H2D 仍可与该计算并行。
这里的 prefix restore 方向是 H2D。新生成 suffix 的 D2H backup 属于另一条异步持久化路径，其完成状态约束 L2 publication 和后续 L1 eviction，而非当前 forward 对 L1 suffix 的使用。

当 load queue 为空时，`start_loading()` 返回 `-1`，表示当前 batch 没有 L2→L1 load wave。
正整数 consumer index 表示 batch 应等待的 event set，而非请求编号或 page 编号。

### 6.5 Layer event 与 whole-load ack

逐层 event 服务于计算依赖。
模型访问某一层 K/V buffer 时，根据 batch 保存的 `hicache_consumer_index` 等待对应 layer-ready event，从而使计算在所需数据到达后立即推进。

`HiCacheAck` 服务于整轮操作的生命周期管理。
其 finish event 表示该 load wave 的全部 layers 已完成；
HiRadixCache 的 completion check据此清理 `ongoing_load_back`、释放 radix node 的保护引用并更新统计。
copy submission、forward readiness 和最终 bookkeeping 因而形成三个清晰阶段。

### 6.6 TP 与 PP 的作用

TP worker 持有普通 Transformer 权重和 KV 的 rank-local shard。
多个 TP ranks 接收相同请求 batch 的逻辑 token 范围，各自执行本 rank 的矩阵与 attention shard，并通过 collective 完成一次统一 forward。
每个 rank 的 HiCacheController 恢复本 rank 所需的 K/V object 和 L1 slots。

PP 将不同 layer 分配给不同 stage。
一个 PP worker 的 `layer_num` 和 layer events只覆盖本 stage 的 local layers。
逐层 load event体现的是本地数据传输与计算顺序；PP stage communication 则连接不同 worker 的模型计算。

外部存储命中需要相关 ranks 获得共同可用的连续 prefix。
SGLang 对 storage hit length 执行跨 attention groups 的最小值同步，使任一必要 shard 的缺失都会收缩最终可复用范围。

## 7. SGLang Storage Backend 接口：从本地缓存到外部存储

### 7.1 Backend attach 与资源注册

HiCacheController 通过 storage backend factory 创建 Mooncake、HF3FS、NIXL 等外部 backend。
attach 过程将当前 Host KV pool 注册给 backend，选择 page buffer 获取函数，并建立 storage query、I/O、backup 和 completion queues。

这一接口层连接两种数据模型：

- HiCache 使用 token page、Host indices、设备 indices 和 radix node；
- storage backend 使用 keys、buffer addresses、sizes 和异步操作结果。

adapter 将逻辑 page hash 展开为 backend keys，并把 Host pool page 转换为可传输 buffer metadata。
外部 backend完成 object I/O 后，HiCache 再更新本地 page 和 node 状态。

### 7.2 Storage 查询接口

对当前请求的本地 miss 部分，SGLang 根据 token pages 和最后一个本地 prefix hash 计算候选 page hashes，并通过 backend 的 batch existence/query 接口获取连续命中长度。

查询结果遵循完整 page 和最长连续 prefix 两项约束。
普通 MHA 的一个逻辑 page 包含当前 rank 所需的 K object 与 V object；
两个对象均处于可读状态时，该 page 进入可用命中。
查询到第一个不完整 page 时，本轮可复用 prefix 在该边界结束。

存在性查询只返回可用范围。
实际读取由 batch get 接口完成，目标 buffer 来自 HiCache 分配的 L2 Host pages。
query 与 data transfer 分离后，Scheduler 可以根据命中长度、等待策略和容量水位决定读取规模。

### 7.3 Storage 读取与本地发布

当前读取过程分成以下阶段：

1. storage query worker 计算候选 hashes 并查询 backend；
2. 相关 ranks 同步共同可用的 hit length；
3. HiRadixCache 按命中 page 数分配 L2 Host indices；
4. storage I/O worker 将 object 数据写入这些 Host pages；
5. completion path 将 page hashes 与 Host indices 插入本地 radix tree；
6. 请求重新执行 local match，刚读取的数据表现为 Host hit；
7. admission 和 `start_loading()` 执行标准 L2→L1 恢复。

这种发布顺序使本地 radix node 只关联已经完成的 Host page。
异步操作期间，anchor node、目标 page 和 operation state 由 HiCache 保持，完成或取消后统一释放。

### 7.4 Storage 写入接口

新生成的 KV 首先位于 L1。
HiCache 根据 write policy 将需要保留的 page 备份到 L2，并将 page hash、Host buffer 和 storage metadata提交给 backend。
Mooncake adapter 将每个逻辑 page 展开成物理 K/V objects，完成 Put 之后由 Mooncake Master 发布 COMPLETE replica。

写入策略决定共享副本形成的时间：

| 策略 | 写入时点 | 主要收益 | 主要代价 | 适用场景 |
|---|---|---|---|---|
| `write_through` | KV 生成后持续建立 L2/external 副本 | 后续请求较早获得共享命中，淘汰路径平滑 | 写带宽和容量消耗较高 | 稳定公共前缀、跨实例复用明确 |
| selective write-through | 达到热度、长度或价值条件后写 external tier | 降低低价值对象写放大 | 需要维护准入特征和阈值 | 多租户、热点集中、长尾 prompt |
| `write_back` | L1/L2 空间回收前建立下层副本 | 前台写流量较低 | 高水位时产生集中 backup 和回收等待 | 外部写带宽有限、L2 容量充足 |

策略判断宜组合 prefix 长度、独立会话复用次数、reuse interval、重算时间、对象 bytes、租户优先级和当前 storage queue。
单一访问次数可以作为起点，生产策略需要同时计入计算节省与写入代价。

### 7.5 Storage prefetch 等待策略

当前实现中的 prefetch 由正在处理的请求触发，目标是提前于 batch forward 完成 external-tier 查询和读取。
它属于 request-driven asynchronous fetch。Scheduler 对 operation 的等待方式由 policy 控制：

| Policy | 行为 | 适合条件 | 主要关注指标 |
|---|---|---|---|
| `best_effort` | 消费已经完成的结果，保持严格调度推进 | 短 prefix、交互式 TTFT、远端延迟波动较大 | 取消率、实际复用页、重复 prefill |
| `timeout` | 在给定 deadline 内等待，超时后按已完成范围继续 | 通用在线服务、SLO 可分解 | storage queue P99、SLO slack、重算时间 |
| `wait_complete` | 等待目标读取完成 | 超长稳定 prefix、离线任务、读取显著快于重算 | TTFT、GPU work saved、远端尾延迟 |

`timeout` 可以由 `min(配置上限, 当前 TTFT 余量, 预计重算时间 - 安全裕量)` 给出初始值，再根据实际 queue 和 hit value 调整。
策略效果应以有效复用 token、避免的 prefill GPU work 和 SLO goodput衡量。

## 8. SGLang external storage 在 Mooncake 中的实现

### 8.1 SGLang 与 Mooncake 的职责衔接

本章继续以 SGLang 的 page query、`batch_get`、`batch_put` 和本地 publication 为主线，Mooncake 逻辑架构用于说明这些 storage operations 在 backend 内部如何实现。
Mooncake Store 由 Master Service 和 Client 两类核心角色组成：

- Master 是控制面，管理逻辑存储池、对象 metadata、replica allocation/eviction、租约和 PROCESSING→COMPLETE 状态转换；对象 payload 由 Client 数据面承载。
- Client 具有双重角色。一侧接收 SGLang 的 Put/Get 请求，另一侧把本进程注册的 memory segment 暴露为 Store server，使其他 Client 可以直接读取或写入。
- 数据面在 Client 与 replica holder 或存储后端之间完成。Master 返回 descriptor 后，发起方选择可读副本并调用适合该介质的传输路径。

两套系统的职责因此是互补而非重叠：

| 能力 | SGLang 提供 | Mooncake 提供 |
|---|---|---|
| KV 语义 | token prefix、完整 page、模型/TP/PP 布局、K/V 完整性 | 保存带稳定 key 的 bytes，KV 语义由 SGLang namespace 表达 |
| 本地执行 | L1/L2 allocation、radix publication、forward readiness | 提供 object/replica 生命周期与数据恢复 |
| 共享发现 | 从 prefix 生成 page hash 和 object key | 由 Master 将 key 解析为 COMPLETE replicas |
| 数据目标 | 为当前接口提供最终 L2 Host page | 将所选 replica 的 bytes 搬到目标 buffer |
| 生命周期 | 保持本地 slot、异步 operation 和取消状态 | 管理 object/replica、lease、eviction 和持久介质 |

Put 时，SGLang 决定哪些完整 KV pages 值得共享并提供 key、源 buffer 和布局；Mooncake 分配 replica、执行相应介质写入，并仅在 PutEnd 后公开 COMPLETE 状态。
Get 时，SGLang 决定连续 prefix 和目标 L2 pages；Mooncake 解析副本并完成数据恢复。
Mooncake 因而扩展了 SGLang cache 的容量与作用域，而 SGLang 保留 KV 正确性和何时可被模型消费的最终决定权。

### 8.2 Page identity 与 Mooncake object index

SGLang 将 token 序列按 `page_size` 切分为 `P0、P1、P2...`，并计算链式 hash：

```text
H0 = Hash(P0)
H1 = Hash(H0, P1)
H2 = Hash(H1, P2)
```

hash 输入由当前 page 的 token IDs、前一页 hash 和固定 page boundary 构成。
由此得到的 `Hi` 表达“完整历史 prefix 下的第 i 个逻辑 page”。
相同 token prefix、tokenizer、page size 和 hash 规则会生成相同 page hash。

逻辑 page hash 随后根据物理 KV 布局展开。普通 MHA/GQA 通常形成：

```text
logical page Hi
  -> namespace + Hi + TP/PP shard + K suffix
  -> namespace + Hi + TP/PP shard + V suffix
```

模型版本、KV dtype、attention layout、page size、TP/PP mapping、tenant 或 cache salt 都可以进入 namespace 和 key 规则。
异构 TP 共享需要采用双方一致的 head shard 命名；
SGLang 官方 HiCache 配置通过 `tp_lcm_size` 支持特定 page-head layout 的兼容展开。

Mooncake Store 的逻辑管理单位是 Object。
Master 将 `(TenantId, Object Key)` 定位到 metadata shard，在租户 metadata map 中执行精确查找，并读取 ObjectMetadata 中的 Replica descriptors。
一个 object 可以拥有多个 replicas，每个 replica 记录状态、介质、provider endpoint、地址和大小等信息。

写入过程先创建 PROCESSING replica。
Client 完成数据传输并提交 PutEnd 后，Master 将匹配 replica 转换为 COMPLETE。
读取过程选择 COMPLETE replica，从而保证查询结果只引用已经完整提交的数据。

External-tier prefix query 的复杂度主要由候选逻辑 page 数、每 page 展开的物理 object 数和 metadata RPC batch 决定。
page size越小，部分 prefix 复用粒度越细，同时 object 数量和控制面 QPS 上升；
page size越大，批量 I/O 效率提高，同时尾部不完整 page 和局部 prefix mismatch 带来更大的复用损失。

### 8.3 `batch_get`/`batch_put` 的 Mooncake 数据路径

Mooncake 将对象解析和 payload 传输分开：

```text
控制面
Object Key
  -> Store Client
  -> Master RPC
  -> ObjectMetadata
  -> COMPLETE Replica Descriptor

数据面
Replica type + endpoint/address/path/size
  -> TransferSubmitter 按介质分派
       MEMORY: local memcpy 或 classic TE/TENT 的 RDMA/TCP 路径
       DISK/DFS: FileStorage/FileSystemAdapter 读取
       LOCAL_DISK: holder SSD -> ClientBuffer -> TE/TENT 网络传输
       NOF_SSD: SPDK NVMe-oF worker
  -> SGLang registered Host page
```

Master 管理 object visibility、replica、空间、租约和元数据高可用。
大块 KV payload 沿 Client 与 provider 之间的数据路径移动。
经典 Transfer Engine 基于已知内存地址和长度选择 transport，并利用内存注册、multi-NIC 和拓扑信息完成内存段传输。
TENT 可以作为下一代 runtime 接收同类请求。
当前 compatibility path 覆盖既有 memory transfer；文件和 SPDK 分支需要 Store 层增加显式 adapter 后，才能由 TENT transport selector 编排。

在 Get 方向，SGLang Mooncake adapter 的 `batch_get` 路径接收逻辑 page hashes 和 L2 Host indices。
adapter 为普通 MHA 展开 K/V keys，从 `MHATokenToKVPoolHost` 获取每个目标 page buffer 的 pointer 和 size，然后请求 Mooncake 将 object 数据写入对应位置。

对于 `MEMORY` replica，这一过程中的 zero-copy 表示 provider 数据直接进入最终 Host pool page。
对于 `LOCAL_DISK`，holder 端仍有 SSD→ClientBuffer 的 staging；
对于 legacy `DISK`/DFS 和 `NOF_SSD`，数据由相应 file/SPDK worker 写入 CPU 可寻址目标。
数据完成后，HiRadixCache 将该 Host page 与 hash 关联；
同一 page 后续可以被多个本地请求命中，也可以在 L1 slot 被回收后继续保留于 L2。

在 Put 方向，External backup 将 SGLang Host page 作为源 buffer。
Mooncake Master 先分配目标 replica 和地址，Client 再按 replica 介质选择 memory transfer、file backend、local-disk staging 或 SPDK 路径；
完成确认后 Master 才发布 COMPLETE 状态。
object key 在副本增加、迁移或介质变化时保持稳定，ObjectMetadata 更新实际 replica 集合。

该提交模型使写入失败可以清理 PROCESSING replica，同时保留上层 L1/L2 副本供请求继续运行或后续重算。
历史 KV Cache 的可恢复属性允许 backend 在故障时降级为本地 cache 与 prefill，服务正确性由完整 page publication 和 namespace compatibility 保障。

### 8.4 Placement、replica 与 SGLang storage policy

Mooncake 的 placement policy 决定新 replica 使用哪些 provider 和介质，replica selection 决定读取时采用哪一份完整副本。
策略空间主要包括容量均衡、数据局部性、故障域和热点读带宽。

| 策略目标 | 倾向选择 | 收益 | 代价 |
|---|---|---|---|
| 容量均衡 | free-ratio 类 placement | 减少单节点提前满载 | 读取距离可能增加 |
| 请求局部性 | local-first 或拓扑优先 | 降低网络 hop 和远端 latency | 热点可能集中于局部节点 |
| 高可用 | 跨故障域 replicas | 节点失效后继续读取 | 容量和写带宽按副本数增长 |
| 热点吞吐 | 为高扇出 object 增加副本 | 分散 provider/NIC 读压力 | 需要热度观测和动态回收 |

对象准入与副本数量是两个独立决策。
准入回答“该 prefix 是否值得占用外部存储”，副本策略回答“已准入对象需要多少读取带宽和故障域覆盖”。

## 9. 当前端到端读取与写入流程

### 9.1 请求读取流程

```text
R0  请求到达并生成 RadixKey
R1  HiRadixCache 匹配连续 L1/L2 prefix
R2  对剩余完整 pages 生成 chained hashes
R3  storage backend 查询 Mooncake 外部层的连续存在长度
R4  TP/PP attention groups 同步共同 hit length
R5  按命中长度分配 L2 Host pages
R6  Mooncake 选择 COMPLETE replicas并按介质路径恢复到 L2
R7  HiRadixCache 发布 Host pages，请求重新 match
R8  PrefillAdder 为 Host hit 分配 L1 destinations
R9  start_loading 按 layer 执行 L2→L1
R10 forward 复用 prefix，对剩余 suffix 执行 prefill
```

R1 完成本地 prefix index search，R3 完成分布式 object existence search，R6 执行外部介质读取及必要的远端 payload transfer，R9 执行本地 H2D。
这四个阶段可以分别记录 latency 和 bytes，形成清晰的控制面与数据面指标。

### 9.2 新 KV 的写入流程

```text
W0  Prefill/Decode 在 L1 生成新 KV
W1  请求 mapping 与 radix node 关联 L1 slots
W2  write policy 决定 L2 备份时点
W3  HiCacheController 执行 L1→L2
W4  storage admission 决定 Mooncake 外部层写入准入
W5  SGLang 生成 page hashes 与 physical keys
W6  Mooncake 创建 PROCESSING replicas
W7  Mooncake Client 按目标 replica 介质写入 object bytes
W8  PutEnd 发布 COMPLETE replicas
```

write-through 使 W3–W8 在对象生成后较早发生；
selective policy 在 W4 引入价值判断；write-back 将 W4–W8 推迟到 L2 回收压力出现前。
三种路径共享相同 page identity 和 object publication 规则。

### 9.3 驻留状态

用 D、H、S 分别表示 Device L1、Host L2 和 SGLang 可见的 Shared external storage。
S 在物理上可以对应 Mooncake MEMORY L3 或更冷的 L4 replica；该简化状态只表达 SGLang 能否从 backend 恢复对象，不表达恢复成本。
一个 prefix page 可以处于以下常见状态：

| 状态 | 含义 | 典型来源 | 后续动作 |
|---|---|---|---|
| D | 仅 L1 | 新生成 KV | 备份 L2、继续运行或直接淘汰 |
| DH | L1 + L2 | 本地 write-through | L1 原址复用或回收 device 副本 |
| H | 仅 L2 | L1 eviction 后保留 Host 副本 | 下次命中执行 load-back |
| DHS | 三个 SGLang 逻辑层均有副本 | external backup 完成且本地副本仍保留 | 按各层容量独立回收 |
| HS | L2 + shared external | L1 已回收 | 本地 Host hit或释放 L2 保留共享副本 |
| S | 仅 shared external | 本地 page 已淘汰 | 下次请求执行 external→L2→L1 |
| 空 | 所有层 miss | 首次请求或副本已淘汰 | 执行 prefill 并重新准入 |

Mooncake 的 PROCESSING/COMPLETE 描述 replica 的提交状态，与 D/H/S 层级驻留状态正交。

## 10. Page、slot、hash 与 object 的一致性边界

`page_size` 定义一个逻辑 page 覆盖的连续 token 数，也是当前本地 page allocation、external object hash generation、存在性查询、I/O 和 publication 的共同粒度。

slot 是本地 pool 中的地址单位。
page allocator 以满足布局的 base 和长度分配一组 slots；
请求的多个 pages可以位于不同物理区域。
page hash 表达逻辑身份，object key 表达后端物理分片身份，Replica Descriptor 表达某份副本的实时位置。

| 条件 | 对共享结果的影响 |
|---|---|
| tokenizer 或 token IDs变化 | chained hashes变化 |
| page size 或 page boundary变化 | page sequence 和 hashes变化 |
| 模型 revision、KV dtype 或 attention layout变化 | object namespace或物理字节布局变化 |
| TP/PP shard mapping变化 | physical object expansion变化 |
| tenant/cache salt变化 | object isolation domain变化 |
| replica 迁移或 DRAM/SSD介质变化 | key 保持稳定，descriptor 更新 |

跨实例共享要求所有影响 KV 语义和字节布局的字段进入统一 namespace 或兼容协议。
命中判断以完整 page 和完整必要 shard 为单位；请求尾部不足一个 page 的 token 留给正常 prefill。

## 11. 当前架构中的主要策略联动

分层缓存的策略旋钮共同作用于同一条请求路径。
page size影响本地碎片、部分命中、object 数和 RPC QPS；
write policy影响 L2/external 写量和共享副本可用时点；
prefetch policy影响 TTFT 与避免的 prefill；
placement 和 replica影响 external capacity、故障域和读带宽；
L1/L2 eviction影响复用窗口与恢复频率。

| 决策节点 | 核心输入 | 推荐判断目标 |
|---|---|---|
| L1 eviction | active bytes、reuse distance、L2/L3副本、restore cost | 保护计算工作集并最大化单位 HBM 的计算节省 |
| L2 capacity/layout | Host bytes、NUMA、H2D曲线、external page I/O | 平衡本地命中、staging容量和带宽 |
| external write admission | prefix长度、复用概率、写入bytes、重算成本 | 使预期计算节省覆盖写入和容量成本 |
| prefetch wait | SLO slack、远端queue、hit length、prefill时间 | 在 deadline内选择读取或重算 |
| replica placement | provider容量、拓扑、热点、故障域 | 满足读带宽与可用性并控制副本放大 |
| cache-aware routing | 本地命中、L3可取回长度、实例queue | 最小化排队、传输和剩余计算的总时间 |

一次 external physical hit 的业务价值由来源介质、读取完成时点和实际减少的 prefill work共同决定。
生产观测宜区分 object existence、完成读取的 page、最终复用 token 和避免的 GPU work，并分别记录 query、source medium、storage read、provider staging、network transfer、H2D 和 recompute 时间。

异步 queue 需要同时约束 operation count、inflight pages、inflight bytes、L2 reserved indices 和 deadline。
队列深度提高可以增加链路并发，也会增加 Host page占用和尾延迟；水位控制应同时覆盖 capacity 与 service time。

## 12. SGLang 视角下的 L4 接入

### 12.1 SGLang external tier 与 Mooncake L4 pipeline

本文将 Mooncake 共享 DRAM 记为 L3，将 SSD、NVMe、NVMe-oF、共享文件系统等更冷介质记为 L4。
该编号属于 SGLang 分层 KV Cache 的分析口径。
Mooncake Store 可以在同一个 object identity 下维护不同介质的 replicas，因此 L3 与 L4 可以共享 object key，只在 Replica Descriptor、读取代价和生命周期策略上体现差异。

SGLang 当前对外部存储采用 `HiCacheStorage` 抽象。
对 Scheduler 和 HiRadixCache 而言，storage backend 负责三类动作：查询连续 page 命中、把 page 读入本地目标、把本地 page 写出。
backend 内部使用分布式 DRAM、SSD 还是文件系统，由具体实现和 replica selection 决定。

这种抽象形成三种 L4 接入方式：

| 接入方式 | SGLang 可见状态 | Backend/Mooncake 责任 | 接入变化 |
|---|---|---|---|
| 透明 L4 | SGLang 仍看到统一 storage hit | Mooncake 管理 DRAM/SSD replicas、下沉、读取和晋升 | 配置、指标和超时策略变化，page key 与主接口保持稳定 |
| 显式 L4 | query result 标明 medium、预计代价或 tier | Backend 返回 replica tier 和能力 | Scheduler 可为 L3/L4 设置不同 deadline、准入和路径 |
| Direct L4 | SGLang 提供 L1 GPU destination | Backend 通过 GDS/NVMe-oF/GPU direct path 传输 | 增加 device destination、reservation、completion 和 fallback 接口 |

透明 L4 与当前 Mooncake SSD offload 的接口直接兼容。
Mooncake 将冷 object 从共享 DRAM 下沉到 SSD后，object key保持稳定；
后续查询解析到 SSD replica，数据经 Mooncake的数据面恢复。
SGLang 在 storage operation 完成后发布本地 Host page。

显式 L4 适合需要精细 SLO 控制的场景。
DRAM replica 与 SSD replica 的 service time和queue特征差异较大。
backend 若返回 `source_tier`、预计读取bytes、direct-destination capability和可用replica类型，
Scheduler可以将“等L4、从L3读、直接prefill”作为显式策略选择，并为SSD命中设置独立deadline和并发水位。

以下结论以本地 Mooncake checkout 的 `Replica::Descriptor`、`TransferSubmitter` 和 storage backend 为准。
当前 descriptor variant 由 `MemoryDescriptor`、`DiskDescriptor`、`LocalDiskDescriptor` 和 `NoFDescriptor` 组成。
实验性的分布式文件后端采用 `FileStorage` 和 `FileSystemAdapter`，按 object key 映射文件路径；descriptor-based shared DFS extent 可以作为另一种后续 metadata 设计讨论。

| Replica/后端 | 当前读路径 | classic TE/TENT 的职责 | 对 SGLang 的完成结果 |
|---|---|---|---|
| `MEMORY` | holder 注册内存 → local memcpy 或 RDMA/TCP → requester target | 承担非本地已注册内存段传输；启用兼容模式时 TENT 可接管该请求 | L2 Host page ready |
| `LOCAL_DISK` | requester 查到 holder → RPC 请求 holder → SSD 读入 holder `ClientBuffer` → 网络 pull 到 requester | 只承担 holder staging buffer 到 requester 的网络段；磁盘读由 holder 的 `FileStorage` 完成 | L2 Host page ready；可另行 promotion 为 `MEMORY` replica |
| requester 可见 file extent | object metadata 解析 file path、offset 与 length → TENT io_uring 写入 requester target | 将文件读取纳入 TENT file-segment 请求；满足 direct-I/O 条件时目标可直接使用 SGLang L2 pointer | L2 Host page ready；该路径属于后续 Store adapter integration |
| legacy `DISK` | `FilereadWorkerPool` 调用 storage backend，将文件读入 CPU-addressable slices | file worker 承担该分支，destination class 为 CPU-addressable slices | L2 Host page ready |
| experimental distributed FS backend | `FileSystemAdapter` 将 key 映射到文件/桶，并执行 vector read/write | 当前采用 file-backend path；shared-extent TENT mapping 属于后续 adapter 方案 | L2 Host page ready |
| `NOF_SSD` | `SpdkNofWorkerPool` 对连续、对齐的目标执行 SPDK NVMe-oF I/O | Store 当前采用 SPDK path；classic TE 的 NVMe-oF/cuFile Segment 是另一套可选执行能力 | CPU-addressable L2 target ready |

写方向也按介质分派：`MEMORY` replica 通过 local copy 或内存传输写入已分配 buffer；
legacy `DISK`/distributed FS 由 file worker/backend 持久化；`NOF_SSD` 由 SPDK worker 写入。
`LOCAL_DISK` offload 则由 holder 的 `StorageBackend::BatchOffload` 写 SSD，成功后通过 `NotifyOffloadSuccess` 让 Master 增加 `LOCAL_DISK` descriptor；
这是 holder-driven SSD offload pipeline，GDS write 可作为后续 direct pipeline 的执行方式。

`LOCAL_DISK` 的 staging buffer 是一次读取操作的临时中转资源；
可被其他请求查询的 L3 `MEMORY` replica 则具有独立的 allocation、publication 和生命周期。
热点冷对象可以通过 Mooncake promotion 流程另行创建并提交共享内存 replica。

TENT 已提供包含 `FileBufferDesc(path, length, offset)` 的 `FileSegmentDesc`，以及 GDS/io_uring transport；
上述 Store 分支通过显式 adapter integration 后即可使用这些 runtime 能力。
要形成 L4 file→L1 GDS，Store adapter 仍需把选中的 file replica 转为 TENT file segment、传入 GPU destination，并把 completion/fallback 接回 SGLang 的 L1 reservation 与 publication 状态机。

`FileSegmentDesc` 属于 transfer-plane address description，持久化格式继续由 Mooncake storage backend 管理。
它把已有文件范围表示为 `FileSegmentDesc.buffers[] -> FileBufferDesc {path, length, offset}`；与之对应，`MemorySegmentDesc` 通过 `BufferDesc` 表示 virtual address、length、registration 和 topology。
两种 segment 在同一请求框架中参与 transport selection，同时分别保留 file offset 与 memory address 的寻址语义。

### 12.2 SGLang 的 L4 恢复与适配点

当前实现的结论是：L4读取仍以PoolHost prefetch为标准路径。
高性能单pool storage接口使用 `batch_get_v1(page_hashes, host_indices, extra_info)`。
`host_indices`属于 `MHATokenToKVPoolHost`，backend由这些indices取得目标page的地址和长度。
多pool/hybrid接口通过 `PoolTransfer`描述多个Host pools，数据落点仍由相应Host pool管理。

因此，当Mooncake内部选中SSD/L4 replica时，从SGLang视角得到的标准路径为：

```text
SGLang page hashes
  -> Mooncake object/replica query
  -> Mooncake 选择 L4 SSD replica
  -> SSD read 与必要的 provider-side transfer
  -> requester 的 registered PoolHost pages
  -> HiRadixCache 发布 host_value
  -> init_load_back / start_loading
  -> L1 GPU slots
```

具体数据面由L4实现决定。
Mooncake本地SSD offload当前在provider侧使用Host `ClientBuffer`承接磁盘读取，再通过classic TE或TENT兼容网络路径发送到请求方；
legacy file/DFS和SPDK NoF分别由各自worker执行。
无论源端路径如何组织，当前SGLang HiCache的完成结果都表现为L2 Host page就绪，随后复用统一的Host hit和layer-wise load-back流程。

generic `batch_get()`路径也以L2为最终本地层级。
backend先返回flat page tensor，HiCache再将其写入Host pool；
v1 zero-copy路径直接写入最终Host page，减少一次中间tensor和Host memcpy。
两条路径在本地publication阶段汇合。

透明L4对现有prefix index的影响较小。HiRadixCache仍保存L1/L2位置和稳定page hash，Mooncake仍以object key解析replica。
主要变化集中在时延、队列、容量和promotion策略：

1. storage query需要区分“object存在”与“当前可用replica的介质和预计读取代价”；
2. SSD读取拥有独立的IOPS、吞吐和queue depth，需要与RDMA/TCP队列分别限流；
3. L4 operation占用PoolHost page的时间更长，L2 reservation和取消回收更重要；
4. 读取完成后可以保留L2副本，并由Mooncake执行热点object向共享DRAM L3的晋升策略；
5. timeout policy需要纳入SSD queue P95/P99与剩余prefill时间；
6. metrics需要把L3 DRAM bytes、L4 SSD bytes、provider staging、network bytes和H2D bytes分别记账。

显式L4还需要扩展storage result。一个适合策略层使用的结果可以包含：

```text
continuous_hit_pages
source_tier: shared_dram | local_ssd | remote_ssd | file
estimated_service_time
direct_to_host_supported
direct_to_device_supported
replica_count / selected_endpoint
```

这些字段服务调度决策，object key和KV page语义保持原有形式。Scheduler可以基于相同prefix分别比较远端恢复和prefill的预计完成时间。

### 12.3 Prefetch 在通信链路中的实际优化

HiCache prefetch由当前请求的本地storage miss触发，在请求正式进入prefill batch前启动page query和读取。
它将外部存储访问从“batch已准入后的同步等待”移动到“waiting/admission期间的异步准备”，并通过批量查询、目标buffer复用和完成策略缩短有效关键路径。

prefetch的链路优化可以分成六个部分：

| 优化位置 | 机制 | 解决的链路问题 | 直接收益 |
|---|---|---|---|
| 调度时序 | 在waiting/admission阶段发起query与I/O | storage latency集中暴露在forward之前 | 与其他请求调度、计算和queue等待重叠 |
| 控制面 | 对一组chained page keys执行batch exists | 每page独立RPC产生RTT和QPS放大 | 摊薄metadata RTT和Master开销 |
| 数据面 | 按连续hit pages执行batch get | 小传输和分散work request利用率较低 | 提高RDMA/TCP/storage batch效率 |
| 目标内存 | v1接口直接写registered PoolHost page | 临时CPU tensor、二次Host memcpy和重复注册 | 降低CPU开销、内存带宽和copy latency |
| 数据布局 | page-first/page-first-direct组织page buffers | 同一page跨layer分散导致scatter与小I/O | 形成稳定page地址和更大连续传输单元 |
| 完成策略 | best-effort、timeout、wait-complete与rank MIN | 远端尾延迟、部分完成和rank进度差异 | 在SLO内选择可用连续prefix并控制阻塞 |

prefetch完成L3/L4→L2后，`start_loading()`继续优化本地L2→L1链路。
它合并多个请求的indices，按layer执行H2D，并用layer-ready events让forward消费已完成层。
这两段优化分别覆盖外部storage到Host和Host到GPU：

```text
storage prefetch
  batch query + batch transfer + registered Host destination
  目标：让可复用page尽早成为本地Host hit

layer-wise load-back
  merged indices + per-layer H2D + compute event
  目标：让Host hit尽早成为当前layer可消费的GPU KV
```

prefetch带来的核心性能收益是降低storage restore在TTFT关键路径上的可见时间，并减少L3/L4→L2阶段的额外copy、RPC和小I/O。
原始介质带宽、网络带宽和H2D带宽仍构成物理上限；
优化通过并发、批量、zero-copy和deadline policy提高这些资源转化为有效缓存命中的比例。

prefetch lead time决定可隐藏的时延。
请求在waiting queue中停留较长、其他batch正在forward或prefix较长时，query与transfer拥有更大的重叠窗口；
请求立即被准入、L4 queue较长或命中page很大时，剩余读取时间仍会进入TTFT。
评测应分别记录prefetch start、query complete、first page complete、usable prefix publish、load-back start和first layer ready。

### 12.4 L4 读取策略选择

L4命中的最终选择可以基于恢复时间与重算时间比较。
设连续命中prefix的预计prefill时间为 `T_recompute`，L4读取与本地恢复时间为：

```text
T_restore = T_query
          + T_l4_queue
          + T_storage_read
          + T_provider_staging
          + T_network
          + T_publish
          + T_h2d_visible
```

`T_publish` 表示 SGLang requester 的本地 cache publication interval，其可测量边界为：

```text
t_data_ready
  = 选中的连续 prefix 已完成 external read，bytes 已进入预留的 L2 Host pages

t_prefix_usable
  = Scheduler 的第二次本地 match_prefix() 已观察到对应 host_value，
    并通过 host_hit_length 返回该 L2 hit

T_publish = t_prefix_usable - t_data_ready
```

这一阶段把“物理 bytes 已驻留”转换为“Scheduler 可以用于 admission 与 load-back planning 的 L2 cache hit”。
当前 pipeline 中，它包含 completion collection、prefetch policy 判断、下一次 Scheduler progress check、跨 attention ranks 的共同连续 prefix clamp、
`_insert_helper_host()` radix commit，以及第二次 `init_next_round_input()`/`match_prefix()` observation：

```text
external I/O completion
  -> operation.completed_tokens 可见
  -> check_prefetch_progress() 应用 wait_complete / timeout / best_effort
  -> terminate_prefetch() 收集 completed extent
  -> attention ranks 协商共同可用的连续 token count
  -> _insert_helper_host() 写入 host_value 与 page hashes
  -> 完成 ongoing_prefetch 与 Host-slot accounting
  -> 第二次 match_prefix() 返回新的 L2 hit
```

这里的 publication 对应 requester-local HiRadixCache state。
Mooncake `MEMORY` replica commit 或 `LOCAL_DISK` offload record 属于 Store control-plane publication，具有独立生命周期。
`T_network` 的结束边界是 external transfer 已交付选中的 bytes；
`T_publish` 的结束边界是 Scheduler 已能使用这些 bytes 规划请求；
`T_h2d_visible` 表示随后逐层 L2→L1 copy 在计算重叠后仍暴露于请求关键路径的部分。

对于 direct L4→L2 `io_uring` path，`t_data_ready` 对应所需 K/V slices 的 successful CQE processing。
对于 direct L3/L4→L1 path，publication 在 transfer completion 与 device-stream visibility 之后提交 `device_indices`，staged path 中的 `T_h2d_visible` 取值为零。
page- 或 prefix-granular completion 可以在 policy boundary 发布跨 ranks 的最大共同连续 prefix，从而缩短 publication interval。

`T_restore`低于`T_recompute`且满足TTFT余量时，等待L4读取具有正向计算收益；
远端queue或L2压力升高时，best-effort或短timeout可以消费已完成前缀并对剩余suffix执行prefill。

| 场景 | 建议路径 | 选择依据 |
|---|---|---|
| 超长稳定公共prefix、L4吞吐稳定 | wait-complete或较长timeout，经PoolHost恢复 | 重算代价显著高于完整读取 |
| 在线交互、SSD P99波动 | 有界timeout，经PoolHost使用已完成连续页 | 控制TTFT尾延迟 |
| 短prefix或低命中长度 | best-effort或直接prefill | metadata与IO成本接近重算收益 |
| 多请求共享同一冷prefix | L4→L2并保留Host副本，结合L3 promotion | 一次读取服务多个后续请求 |
| 请求已准入且GPU容量确定 | 未来L4→L1 direct path | 缩短Host staging与H2D路径 |
| L2 page水位或SSD queue达到上限 | 限制新prefetch并转向prefill | 维持queue稳定和Host容量安全 |

## 13. SGLang direct restore 的执行基础：Classic TE 与 TENT

### 13.1 SGLang storage adapter 下的 TE 与 TENT

[Mooncake Transfer Engine](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/transfer-engine/index.md)把 DRAM 和 VRAM 都表示为可注册 RAM Segment。
应用给出已经解析的 source、target、offset 和 length 后，classic TE 负责内存注册、请求切片、拓扑感知 NIC 选择、multi-NIC spraying、RDMA/TCP/local transport 和完成状态。
GPU buffer 可以注册到 RAM Segment，因此 classic TE 已经具备 GPUDirect RDMA 所需的 source/destination primitive，并可覆盖 DRAM→Host 与 DRAM→GPU 等目标组合。
TE 的 metadata service 发布和发现 segment、endpoint 与 transport metadata；Mooncake Master 则管理 object key、replica 状态、lease 和 eviction metadata，两者分别服务字节寻址与对象生命周期。

classic TE 还包含专门的 NVMe-oF Segment/cuFile 能力；Mooncake Store 当前 `NOF_SSD` replica 则采用 SPDK worker，形成两套独立接线。
是否实际形成 direct path，取决于上层是否把相应介质 descriptor 和 GPU destination 交给这套执行器。

TENT 在此基础上演进为下一代 transfer runtime。它保留“应用声明要移动哪些 bytes”的模型，用 `MemorySegmentDesc` 和 `FileSegmentDesc` 表达内存或文件端点，
并根据 segment type、memory kind、intent、priority、size 和 policy 在 runtime 内选择 transport 与 staging 路径。
执行层进一步组合 NUMA/telemetry-aware slice spraying、priority queue、全局 QoS 协调以及 completion-stage fallback、rail cooldown 和恢复；
file segment 可按策略选择 GDS 或 io_uring，从而覆盖 file↔GPU 与 file↔Host。

这里的 file segment 是现有文件范围的 runtime descriptor，KV bytes 继续采用 Mooncake storage backend 定义的持久化格式：

```text
SegmentDesc
  type = Memory -> MemorySegmentDesc -> BufferDesc
                 virtual address + length + registration/topology

  type = File   -> FileSegmentDesc.buffers[] -> FileBufferDesc
                 path + file offset + length
```

Mooncake Store 继续维护 object key、序列化布局、replica status、holder、lease、eviction 和 key-to-extent index；
adapter 在一次读取前把选中的 file replica 解析为 TENT segment 与 byte range。
因此，file segment 改变的是 transfer runtime 能够观察和调度的 source address domain，而磁盘上的 object bytes 和 storage layout 保持原有定义。

对 SGLang 最直接的增量是 file→L2 `io_uring` placement。
当前 `batch_get_v1()` 已经让 Scheduler/HiCache 先保留 `host_indices`，再由 `HostKVCache.get_page_buffer_meta()` 得到最终 L2 page 的 pointer 与 length。
若 Mooncake 同时把 object key 解析为 requester 可打开的 file segment 和 object offset，adapter 可以构造：

```text
Scheduler reserve L2 host_indices
  -> HostKVCache resolves final L2 pointers
  -> Mooncake resolves key to file segment + object extent
  -> TENT READ {
       source        = final L2 pointer,
       target_id     = file segment ID,
       target_offset = object file offset,
       length        = slice length
     }
  -> io_uring completion
  -> storage operation completes
  -> HiRadixCache publishes host_value
```

TENT `Request.source` 表示 local buffer；在 file-segment `READ` 中，该 pointer 是读取目的地。
当前 `IOUringTransport` 在文件系统支持时初始化 `O_DIRECT`，并以 buffered file I/O 作为 compatibility mode。
CPU destination address 满足 4 KiB 对齐时，request pointer 可直接交给 `io_uring`；
CUDA pointer 与其他 CPU address alignment 由 transport 分配 aligned buffer，读取完成后再复制到 request pointer。
`O_DIRECT` 路径还要求 file offset 与 length 满足文件系统对齐约束。
因此，严格的 L4→L2 immediate placement 需要同时验证 L2 pool base、per-page stride、object extent 和 request length。

SGLang KV page 可以由多个连续 slices 组成，adapter 可按 K/V、layer 或 page slice 提交多个 request，并在所需 slices 全部完成后发布该 page。
对 requester-visible file extent，这一数据面直接把 bytes 写入 L2，省去 holder `ClientBuffer` allocation/lease 和 holder-buffer-to-requester network stage，
同时保留 Scheduler 的 page reservation、异步完成、取消回收和 radix publication 状态机。

file visibility 决定执行位置。
TENT file transport 在执行 request 的进程中打开 path；同节点 local SSD 或该节点已挂载的文件系统可以形成直接 file→L2 path。
由远端 holder 管理的 `LOCAL_DISK` 采用 holder RPC、SSD→ClientBuffer 和网络 pull pipeline；
owner-side composed execution 或 requester-visible file mapping 对应 file-segment plan。

在 cluster 中，`io_uring` 是单个 Linux process/kernel instance 内的 submission 与 completion 机制。
ring、file descriptor 和 destination virtual address 属于执行 request 的本地地址空间；
cluster reachability 由该 file descriptor 对应的 local filesystem、DFS mount 或 NVMe-oF-backed filesystem 提供。
TENT 与 SGLang 位于同一进程时可直接使用 L2 pointer；独立 executor process 通过 shared mapping 在自身地址空间获得有效 destination address。

DFS 可以对上层提供统一 path 与 file-offset namespace，并在相同 `io_uring` request 之下选择多种 physical route：

```text
TENT FileSegment READ(path, offset, L2 pointer)
  -> local io_uring / VFS / DFS client
       +-- page-cache hit
       |     -> CPU memory copy -> SGLang L2
       +-- requester-local SSD 或 DFS client-cache hit
       |     -> PCIe/NVMe DMA -> SGLang L2
       +-- remote DFS extent
       |     -> storage-node SSD DMA
       |     -> DFS TCP/RDMA/custom network
       |     -> requester DFS client placement -> SGLang L2
       +-- requester-mounted NVMe-oF extent
             -> NVMe/TCP 或 NVMe/RDMA
             -> requester block/file layer -> SGLang L2
```

上述路径共享相同 logical operation 与 completion boundary：DFS 把所需 bytes 放入 local destination 后，`io_uring` CQE 报告完成。
physical transport 随数据位置与 storage protocol 变化：

| Physical source | 进入 SGLang worker 的典型 transport |
|---|---|
| Kernel page cache | CPU memory copy |
| Local SSD 或 local DFS cache | PCIe/NVMe DMA |
| Remote DFS over TCP | storage DMA + TCP + DFS-client placement |
| Remote DFS over RDMA | storage DMA + network RDMA + DFS-client placement |
| NVMe-oF/TCP | NVMe/TCP fabric |
| NVMe-oF/RDMA | NVMe/RDMA fabric |

DMA 与 RDMA 分别对应各自的 physical link。
DMA 表示同一机器内 device-to-memory transfer，例如 PCIe 上的 NVMe DMA；RDMA 表示机器间的 network memory transfer。
remote DFS/RDMA read 通常组合两者：storage-node NVMe DMA 先把 extent 从 SSD 读到 storage path，network RDMA 再把 bytes 送向 requester。
TCP-based DFS 与 NVMe-oF deployment 则采用相应 TCP data path。

TENT 当前为 file-segment selection context 设置 `same_machine = true`；该字段表达 file descriptor 在本地执行且 destination pointer 属于本地地址空间。
挂载的 DFS 或 NVMe-oF client 可以在这一 file descriptor 之下访问 remote storage。
因此，L4→L2 direct 表达 bytes 直接落入 SGLang 最终 L2 allocation；remote extent 的 physical pipeline 同时包含 DFS 或 storage-fabric network stage。

端到端 copy behavior 由 DFS implementation 决定。
`O_DIRECT` local extent 可以通过 storage DMA 写入 aligned L2 buffer；支持 registered-buffer placement 的 remote DFS 可以通过 RDMA 把 network data 写入该 destination；采用 internal receive buffer 的 DFS client 在 CQE 前完成 client-buffer placement。
面向 tier-aware scheduling，DFS 或 Mooncake adapter 可以返回 `source_location`、`transport`、`direct_to_host` 与 estimated service time，使 Scheduler 在统一 file key 下区分 local-cache 与 remote-extent cost。

TENT 的关键增量是把 RDMA、TCP、GDS、io_uring 和 staging 组织为同一个可观察、可选路、可降级的 runtime；
GPU direct primitive 则延续 classic TE 已有的 DRAM/VRAM segment 能力。
classic `TransferEngine` wrapper 可以把现有 memory transfer 请求翻译给 TENT，因此 MEMORY replica 可以在保持现有 `Replica` 数据结构的条件下启用 TENT。
Store 利用 TENT file/GDS 路径时，adapter 从现有 file/local-disk metadata 构造 `FileSegmentDesc`；
`Replica::Descriptor` 可以保持现有结构，也可以随 Store metadata 设计扩展。

当前 TENT failover 覆盖 completion-stage transport/rail failure，并提供 cooldown 与恢复；
submit-stage error 直接形成失败状态，transport switch 由完成状态驱动，tests 采用手工运行方式而 upstream CI 配置为 `USE_TENT=OFF`。
这些实现边界适合纳入 production readiness 与故障注入评测。
对 SGLang 而言，TENT 的直接价值是让 storage adapter 用统一请求描述表达 source medium、GPU/Host destination、priority 和 fallback intent；
KV page reservation、rank consistency 与 publication 仍由 HiCache 管理。

### 13.2 当前 SGLang 接口与 direct restore 路线

当前集成边界位于 SGLang–Mooncake adapter 的 destination 契约。
当前高性能 `batch_get_v1(keys, host_indices, ...)` 接收的是 L2 `host_indices`；Mooncake adapter 再调用 `MHATokenToKVPoolHost.get_page_buffer_meta()` 得到 CPU Host pointer。
现有参数集合描述 L2 Host destination；direct-to-L1 扩展需要增加 L1 `device_indices`、GPU pointer、CUDA stream/event 和 device-buffer registration lifetime。

| 路径 | 底层执行能力 | 当前 SGLang–Mooncake 端到端集成 |
|---|---|---|
| L3 `MEMORY`→L2 | classic TE/TENT RDMA/TCP 或 local copy | 已支持 |
| L3 `MEMORY`→L1 RDMA | classic TE 已有 GPUDirect RDMA primitive，TENT 可编排 | 后续集成项：GPU destination 与 L1 publication 状态机 |
| L4 `LOCAL_DISK`/file/NoF→L2 | Mooncake 的 holder staging、file backend 或 SPDK 路径 | 已支持为统一 external read，具体介质受配置与版本约束；当前 `LOCAL_DISK` 使用 holder staging |
| L4 visible file→L2 io_uring | TENT file segment 与 `IOUringTransport` 可直接使用 aligned CPU destination | 后续集成项：file extent mapping、L2 direct-I/O alignment 与 completion wiring |
| L4 file→L1 GDS | TENT file/GDS transport 或其他 GDS executor 可提供 primitive | 后续集成项：file replica mapping 与 GPU destination |

SGLang 当前通过统一 key/query/get 接口处理 L3/L4，因此两者具有相同的逻辑恢复语义；
Mooncake 在物理 pipeline、service time、queue、staging、promotion 和 direct capability 上保留介质差异。
Scheduler 当前消费统一 hit result，后续可以由 backend result 增加 source tier、estimated cost 与 destination capability 来利用这些差异。

基于现有能力，SGLang 可以增加 L3 DRAM→L1 GPU direct restore，并在后续为 file/GDS 增加同一目标接口。
direct restore 保留 chained hash、exact object lookup 和连续 prefix 规则，同时调整目标分配与发布过程：

1. storage query 确定共同可用的完整 pages；
2. Scheduler 为已准入请求保留准确数量的 L1 pages；
3. storage backend 接收 GPU destination descriptors；
4. Mooncake 将 K/V 或 layer slices 写入 rank-local GPU buffers；
5. completion events 按 page、object 或 layer 建立数据可见性；
6. HiRadixCache 在所有必要分量完成后发布 L1 indices；
7. 取消、超时和 rank failure 释放未发布 pages，并回到 staged path 或 prefill。

该接口需要表达 memory kind、device/rank、slot/page indices、K/V 与 layer scatter buffers、address/size、stream/event 和 registration lifetime。
Mooncake 负责执行 descriptor 指定的 transfer，SGLang 负责生成与 attention layout 一致的 destination descriptors，并决定何时允许 forward 消费。

direct 与 staged 两条路径适合不同状态：

| 路径 | 适用条件 | 主要收益 | 主要资源约束 |
|---|---|---|---|
| L3/L4→L1 direct | 请求已经准入、即将 forward、L1容量确定、远端命中价值高 | 省去 requester L2 staging 与 H2D；L4 provider staging是否消除取决于具体path | GPU page预留、GPUDirect/GDS拓扑、取消回滚 |
| external→L2→L1 staged | 请求可取消、prefix可被多请求共享、L1空间紧张、需要本地保留 | L2可复用、生命周期稳定、统一Host布局 | Host容量、PCIe/H2D带宽 |

实现顺序适合从 `MEMORY` replica 的整页 L3→L1 开始，先验证 K/V、layer、TP/PP layout 和失败处理，再评估 layer-wise scatter 和 L4 file→L1 GDS。
整页路径降低初始接口复杂度；layer-wise path可进一步提前首层 ready 时间，同时增加 transfer request数量和 metadata/completion开销。

## 14. 实现与评测建议

第一阶段建立分段观测。
分别记录 local match、external query、replica resolve、source medium、storage read、provider staging、network→L2、L2→L1、layer overlap、remaining prefill 和 request cancellation。
指标同时使用 logical tokens 和 physical bytes，TP ranks 的逻辑命中按请求归一，物理传输按 rank汇总。

第二阶段实现 page-level L3→L1 原型。
沿用现有 storage query和 rank MIN 结果，增加 GPU buffer registration、L1 reservation、direct batch get、completion和原子 publication。
故障注入覆盖 object部分缺失、单 rank超时、请求取消、GPU page压力和 transport fallback。

第三阶段加入路径策略。
决策输入包括请求 admission状态、L1/L2可用容量、prefix长度、预计L2再命中、远端queue、GPUDirect topology和SLO slack。
策略输出为 direct-to-L1、stage-in-L2或prefill。

第四阶段评估 layer-wise direct transfer和更冷介质。
比较整页object scatter、layer slices和物理object细分方案，测量控制面QPS、transfer batch效率、首层ready latency、TTFT和总吞吐。

评测矩阵至少包含：

- L1、L1+L2、L1+L2+MEMORY L3、L1+L2+L4 四种介质基线；
- `best_effort`、多档 `timeout` 和 `wait_complete`；
- write-through、selective 和 write-back；
- 长公共system prompt、多轮对话、低复用随机prompt和短prefix；
- page bytes、queue depth、TP size、NUMA/NIC topology和并发batch变化；
- TTFT、TPOT、goodput、有效复用token、避免的prefill GPU work、读写放大和P99 queue delay。

## 15. 正确性与生命周期检查表

- 本地与外部存储命中均保持最长连续完整page prefix；
- tokenization、page boundary、hash、model namespace和KV layout兼容；
- 普通MHA page的K/V及当前rank所需分量完整后发布；
- 相关TP/PP ranks对共同可用prefix长度达成一致；
- L1/L2 slots在异步I/O期间保持有效并受保护；
- radix node可见性晚于目标数据completion；
- event set与load wave的所有权明确，并在消费者完成后复用；
- layer-ready event与whole-load ack分别服务计算和bookkeeping；
- 取消与部分失败释放未发布page和PROCESSING replica；
- direct path具备staged path与prefill fallback；
- object key只包含稳定语义和布局字段；
- replica迁移和介质变化通过descriptor更新完成。

## 16. 资料与源码阅读入口

本文对当前实现的判断主要来自以下本地资料：

- [SGLang user-notes](guide/user-notes.md)：Scheduler、KV-cache search、Cache Reuse Workflow和Mooncake backend源码阅读记录；
- [Mooncake study-notes](</home/xffan2/code/Mooncake/.claude/notes/guide/study-notes.md>)：ObjectMetadata、Replica、Transfer Engine、HiCache page hash和L3/L4边界；
- [参考报告](</home/xffan2/work/tutorial/reports/SGLang-Mooncake-L1-L3-Architecture-Strategy.md>)：层级口径、策略节点、场景矩阵和客观论述风格。
- [SGLang HiCache KVCache Offload 方案调研](https://devpress.csdn.net/amd/6a38a74210ee7a33f280a27e.html)：用于补充核对HiCache层级、Host pool、storage backend和prefetch/load-back流程；机制结论以本地源码和官方资料为准。

Mooncake 的 descriptor 和 pipeline 结论绑定到本地 checkout。本文分别标注当前 Store integration 与底层 runtime capability，后续提交可以在两者之间增加新的 replica mapping 和 adapter path。

主要源码入口如下：

| 调研主题 | SGLang/Mooncake入口 |
|---|---|
| KV pools与tree cache构建 | `python/sglang/srt/mem_cache/kv_cache_builder.py`、`registry.py` |
| HiRadix本地match、L2/L3状态与publication | `python/sglang/srt/mem_cache/hiradix_cache.py` |
| waiting queue、batch plan与prefetch调度 | `python/sglang/srt/managers/scheduler.py`、`schedule_batch.py` |
| load queue、layer events、ack和storage queues | `python/sglang/srt/managers/cache_controller.py` |
| L1/L2 pool layout与indexed copy | `python/sglang/srt/mem_cache/memory_pool.py`、`pool_host/mha.py` |
| backend抽象与factory | `python/sglang/srt/mem_cache/hicache_storage.py`、`storage/backend_factory.py` |
| Mooncake key展开、buffer注册和batch I/O | `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` |
| Mooncake Object/Replica与Master索引 | `mooncake-store/include/replica.h`、`mooncake-store/src/master_service.cpp` |
| Mooncake Client与介质分派 | `mooncake-store/src/client_service.cpp`、`real_client.cpp`、`include/transfer_task.h`、`src/transfer_task.cpp` |
| local SSD staging与promotion | `mooncake-store/src/client_service.cpp`、`include/file_storage.h`、`src/file_storage.cpp` |
| distributed FS backend | `mooncake-store/include/storage/distributed/distributed_storage_backend.h`、`src/storage/distributed/distributed_storage_backend.cpp` |
| classic Transfer Engine | `mooncake-transfer-engine/` |
| TENT segments、selector与transports | `mooncake-transfer-engine/tent/`、`docs/source/design/tent/` |

官方资料用于核对当前能力与配置边界：

- [SGLang HiCache Best Practices](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/hicache_best_practices.mdx)；
- [SGLang Mooncake L3 Backend README](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/storage/mooncake_store/README.md)；
- [Mooncake Transfer Engine Design](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/transfer-engine/index.md)；
- [Mooncake Architecture](</home/xffan2/code/Mooncake/docs/source/design/architecture.md>) 与 [Mooncake Store Design](</home/xffan2/code/Mooncake/docs/source/design/mooncake-store.md>)；
- [Mooncake SSD Offload](</home/xffan2/code/Mooncake/docs/source/design/ssd-offload.md>)；
- [TENT Overview](</home/xffan2/code/Mooncake/docs/source/design/tent/overview.md>)、[Transport Selector](</home/xffan2/code/Mooncake/docs/source/design/tent/transport-selector.md>)、[QoS](</home/xffan2/code/Mooncake/docs/source/design/tent/qos.md>)、[Slice Spraying](</home/xffan2/code/Mooncake/docs/source/design/tent/slice-spraying.md>) 与 [Failover](</home/xffan2/code/Mooncake/docs/source/design/tent/failover.md>)；
- [NVIDIA GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)；
- [NVIDIA GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/overview-guide/)；
- [NIXL Backend Plugin Interface](https://github.com/ai-dynamo/nixl/blob/main/docs/BackendGuide.md)；
- [SGLang HiCache L3 KV Events RFC](https://github.com/sgl-project/sglang/issues/29709)，用于跟踪cache-aware routing与external-tier事件方向。

第1至11章描述 SGLang L1/L2 与统一 external tier 的当前实现和策略接口，第12章从 SGLang storage contract 展开 Mooncake L3/L4 介质 pipeline，第13章说明 classic TE、TENT 与后续 direct integration，第14章给出实现与评测建议。各部分共享相同的page identity、连续prefix、rank一致性和数据publication规则。
