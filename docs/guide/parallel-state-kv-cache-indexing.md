# SGLang `ParallelState`、KV 块切分与 HiCache L1/L2 索引

> 阅读基线：`131bd51b01d96de247fbd1966fb45bdbecbb80b5`（2026-08-01）
>
> 讨论范围：标准 MHA/GQA 与 MLA 主路径。SWA、Mamba、DSA indexer、HiSparse 等复合缓存会增加 sidecar pool，但不改变本文的核心索引链路。
>
> 本文中的 L1/L2 指 HiCache 的 GPU/Host 两级缓存，不是 CPU/GPU 芯片内部的硬件 cache。

## 1. 先给结论

Scheduler 中这段代码没有直接切分 KV：

```python
self.ps = ParallelState(...)
```

`ParallelState` 是一份不可变的并行拓扑描述。它先回答“当前进程属于哪些并行坐标”，后续组件再使用这些坐标决定：

```text
ParallelState
    |
    +-- attn_tp_size --------------------> 每个 rank 保存多少 KV heads
    |
    +-- pp_rank / pp_size --------------> 每个 rank 保存哪些 transformer layers
    |
    +-- attn_dp_rank --------------------> 当前请求落到哪套独立 KV cache
    |
    +-- attn_cp_rank --------------------> prefill token 计算如何分工
    |                                     默认最终仍物化完整序列 KV
    |
    +-- dcp_size + derived dcp_rank ----> decode 序列位置如何循环切分
    |
    +-- moe_ep / moe_dp ----------------> 不直接切 attention KV
                                          只可能通过权重占用间接影响容量
```

完整的索引链路是：

```text
token IDs / logical positions
          |
          v
  Radix/HiRadix prefix match
          |
          | prefix_indices: L1 slot IDs
          v
ReqToTokenPool[request_slot, token_position]
          |
          | packed as kv_indices + kv_indptr
          v
attention backend reads/writes
          |
          v
KVCache[layer][L1 slot/page][local KV head][head dim]

HiCache additionally maintains:

TreeNode.host_value ---> independent L2 Host slot IDs
```

最重要的三个判断是：

1. `attn_tp_size` 切 KV head，`pp_size` 切 KV layer。
2. 普通 `attn_cp_size` 主要切 prefill 计算，不等于持久 KV 序列切分；真正按序列位置切持久 KV 的是 `dcp_size`。
3. L2 索引不会直接进入 attention page table。L2 命中必须先分配新的 L1 slots 并回迁，attention 始终消费 L1 索引。

## 2. `tp_rank` 如何分解为 Attention DP/CP/TP

入口位于 [`scheduler.py:424-453`](../../../python/sglang/srt/managers/scheduler.py#L424)，计算函数位于 [`dp_attention.py:317-335`](../../../python/sglang/srt/layers/dp_attention.py#L317)。

定义：

```text
attn_dp_size = dp_size                         if enable_dp_attention
               1                              otherwise

attn_tp_size = tp_size / attn_dp_size / attn_cp_size

rank layout = (attn_dp, attn_cp, attn_tp)
              attn_tp 是变化最快的维度

tp_rank = (attn_dp_rank * attn_cp_size + attn_cp_rank)
          * attn_tp_size
          + attn_tp_rank
```

因此等价坐标为：

```text
attn_tp_rank = tp_rank % attn_tp_size
attn_cp_rank = (tp_rank // attn_tp_size) % attn_cp_size
attn_dp_rank = tp_rank // (attn_tp_size * attn_cp_size)
```

源码中的 `compute_dp_attention_world_info()` 直接返回 TP/DP 坐标；`attn_cp_rank` 在并行组初始化时已经计算并传入 Scheduler。

### 2.1 一个具体例子

假设：

```text
tp_size       = 8
dp_size       = 2
attn_cp_size  = 2
enable_dpa    = true
```

则：

```text
attn_dp_size = 2
attn_tp_size = 8 / 2 / 2 = 2

tp_rank    logical coordinate (attn_dp, attn_cp, attn_tp)
-------    ------------------------------------------------
   0                         (0, 0, 0)
   1                         (0, 0, 1)
   2                         (0, 1, 0)
   3                         (0, 1, 1)
   4                         (1, 0, 0)
   5                         (1, 0, 1)
   6                         (1, 1, 0)
   7                         (1, 1, 1)

                attn_dp lane 0                         attn_dp lane 1
        +-----------------------------+       +-----------------------------+
CP 0    | rank 0: TP-head shard 0     |       | rank 4: TP-head shard 0     |
        | rank 1: TP-head shard 1     |       | rank 5: TP-head shard 1     |
        +-----------------------------+       +-----------------------------+
CP 1    | rank 2: TP-head shard 0     |       | rank 6: TP-head shard 0     |
        | rank 3: TP-head shard 1     |       | rank 7: TP-head shard 1     |
        +-----------------------------+       +-----------------------------+
```

DP lane 0 和 lane 1 可以处理不同请求；同一 lane 内两个 `attn_tp_rank` 保存不同 KV head shard；两个 CP rank 在 prefill 时处理不同 token 分片，但标准路径会 gather 并各自在本地物化完整顺序的 KV。

## 3. 每个 `ParallelState` 字段对 KV 的影响

| 字段 | 是否直接改变 KV 切分 | 实际作用 |
|---|---:|---|
| `tp_rank`, `tp_size` | 间接 | 给出总 TP 拓扑；attention 实际使用由它分解出的 `attn_tp_*`、`attn_cp_*`、`attn_dp_*`。没有 DPA/CP 时 `attn_tp_size == tp_size`。 |
| `pp_rank`, `pp_size` | 是 | 按 layer 切分模型；每个 PP stage 只分配本 stage 的 KV layers。各 stage 可用 token 容量最终取全局最小值。 |
| `dp_rank`, `dp_size` | 普通 DP：不切单请求 KV | 普通 DP 是独立模型副本、独立请求和独立缓存。启用 DPA 后，`dp_size` 参与分解 `attn_tp_size`。 |
| `attn_tp_rank`, `attn_tp_size` | 是 | 标准 MHA/GQA 按 KV heads 切分；决定每 rank 的 `head_num`。 |
| `attn_cp_rank`, `attn_cp_size` | 默认不切最终序列 KV | 切 prefill token 计算；默认 gather 后写回完整 KV。它会缩小 `attn_tp_size`，所以会间接增加每 rank 的本地 KV heads。 |
| `attn_dp_rank`, `attn_dp_size` | 按请求切 | 每个 DPA lane 处理不同请求并维护私有 cache；不是把同一请求的一份 KV 横切给多个 DP lane。 |
| `moe_ep_rank`, `moe_ep_size` | 否 | 切 MoE expert 权重/计算，不改变 attention KV 的 head、layer、token 索引。减少权重占用后可能间接增加可分配 KV token 数。 |
| `moe_dp_rank`, `moe_dp_size` | 否 | MoE token dispatch/数据并行拓扑；同样只间接影响显存预算与调度。 |
| `dcp_size` | 是 | decode 时按 token position 循环切序列 KV。`dcp_rank` 没存进此 dataclass，而是在 ModelRunner 中由 `tp_rank % dcp_size` 得到。 |
| `gpu_id` | 否 | 只决定 pool 放在哪个设备，不改变索引数学。 |

`ParallelState` 本身定义于 [`parallel_state_wrapper.py:6-28`](../../../python/sglang/srt/distributed/parallel_state_wrapper.py#L6)。DCP rank 的派生见 [`model_runner.py:269-273`](../../../python/sglang/srt/model_executor/model_runner.py#L269)。

## 4. KV 物理块究竟沿哪些维度切分

可以把完整模型的 KV 看成下面的逻辑张量：

```text
                         token / sequence dimension
                    <---------------------------------->

layer 0    [ token0 ][ token1 ][ token2 ] ... [ tokenN ]
layer 1    [ token0 ][ token1 ][ token2 ] ... [ tokenN ]
  ...
layer L    [ token0 ][ token1 ][ token2 ] ... [ tokenN ]
              |
              +-- each cell contains local KV heads and head dimensions

PP       : horizontal cuts -- each rank owns a range of layers
attn TP  : depth cuts      -- each rank owns a subset of KV heads
DCP      : vertical cuts   -- each rank owns cyclic token positions
page_size: groups adjacent logical token positions into allocation units
```

### 4.1 MHA/GQA：`attn_tp_size` 决定本地 KV heads

[`model_config.py:1112-1120`](../../../python/sglang/srt/configs/model_config.py#L1112) 使用：

```text
H_local = max(1, H_kv_total // attn_tp_size)
```

当总 KV heads 少于 `attn_tp_size` 时，每个 GPU 至少保存一个 head，因此发生复制，而不是继续切到小于一个 head。

忽略量化 scale、padding 和特殊 sidecar，单 rank、单 token 的标准 MHA KV 字节数是：

```text
bytes_per_token_per_rank
    = local_layers
      * H_local
      * (K_head_dim + V_head_dim)
      * sizeof(kv_dtype)
```

对应计算见 [`pool_configurator.py:180-260`](../../../python/sglang/srt/model_executor/pool_configurator.py#L180)，pool 构造把 `attn_tp_size` 传给 `get_num_kv_heads()`，见 [`kv_cache_configurator.py:1383-1410`](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L1383)。

对一页 `P = page_size`：

```text
bytes_per_page_per_rank = P * bytes_per_token_per_rank
```

### 4.2 MLA：latent KV 通常不按 `attn_tp_size` 继续变小

MLA 的基础缓存维度为：

```text
kv_cache_dim = kv_lora_rank + qk_rope_head_dim

bytes_per_token_per_rank
    = local_layers * kv_cache_dim * sizeof(kv_dtype)
```

见 [`kv_cache_configurator.py:1892-1906`](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L1892)。基础公式没有除以 `attn_tp_size`，所以 vanilla TP 下 MLA KV 往往在 TP ranks 上复制。这正是 DeepSeek 等模型启用 DPA 的主要动机：例如 `tp=8, dp=8, enable_dp_attention=true, cp=1` 时，`attn_tp_size=1`，8 个 attention lane 各处理不同请求，不再让同一请求的 latent KV 在 8 个 TP ranks 上重复保存。

DSA indexer、FP4 scale buffer 等会在这个基础上追加缓存开销，不能只用上述简式估算总显存。

### 4.3 PP：按本地层范围建立 pool

`ModelLayerInfo(start_layer, end_layer, num_effective_layers)` 来自模型的 PP layer range，见 [`layer_setup.py:107-141`](../../../python/sglang/srt/model_executor/model_runner_components/layer_setup.py#L107)。

```text
full model layers:  0  1  2  3  4  5  6  7

PP rank 0 owns:    [0  1  2  3]
PP rank 1 owns:                [4  5  6  7]

L1 rank 0 pool: K/V buffers for layers 0..3
L1 rank 1 pool: K/V buffers for layers 4..7
```

不同 stage 因层数或其他内存占用不同，profile 得出的 token capacity 可能不同；SGLang 使用 `all_reduce(MIN)` 将最终容量同步为最小值，见 [`kv_cache_configurator.py:1646-1675`](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L1646)。

### 4.4 EP/MoE：只间接改变 KV 容量

EP 切的是 expert 权重，KV 属于 attention。因而：

```text
EP size increase
    -> each rank may hold fewer expert weights
    -> more free GPU memory may remain
    -> profiler may choose a larger max_total_num_tokens
    -> KV pool capacity increases

but

EP rank does not enter KV slot/page/head mapping formulas
```

## 5. 三层索引：请求位置、L1 slot、物理 K/V tensor

源码 [`memory_pool.py:14-19`](../../../python/sglang/srt/mem_cache/memory_pool.py#L14) 所说的 “two levels of memory pool” 不等于 HiCache L1/L2。它描述的是两个 GPU 内存管理层：

```text
Level A: ReqToTokenPool
         request + logical token position -> KV slot ID

Level B: TokenToKVPoolAllocator / KVCache
         allocate/free slot IDs; slot ID -> physical K/V tensor row/page
```

HiCache L2 Host pool 是另一个维度，后文单独说明。

### 5.1 `ReqToTokenPool`：请求自己的 page table

[`memory_pool.py:256-286`](../../../python/sglang/srt/mem_cache/memory_pool.py#L256) 创建：

```text
req_to_token.shape = [max_requests + 1, max_context_len]
req_to_token.dtype = int32
```

第 0 行是 CUDA Graph padding 的 dummy row。正常请求从 row 1 开始。

例子：

```text
                           logical token position
                        0     1     2     3     4     5
                      +-----+-----+-----+-----+-----+-----+
request row 3         |  32 |  33 |  34 |  35 |  80 |  81 |
                      +-----+-----+-----+-----+-----+-----+
request row 7         |  32 |  33 |  34 |  35 | 116 | 117 |
                      +-----+-----+-----+-----+-----+-----+
                         \________shared________/

values are L1 slot IDs, not token IDs and not byte addresses
```

两个请求共享前四个 slot，表示它们共享相同 prefix KV；后面的 suffix 使用各自新分配的 slots。

### 5.2 Paged allocator：`page_id * page_size + offset`

`page_size=P` 时，paged allocator 以整页管理空闲空间，返回展开后的 token slot IDs：

```text
page_id = slot_id // P
offset  = slot_id %  P

slot IDs of page p = [p*P, p*P+1, ..., p*P+P-1]
```

分配实现位于 [`allocator/paged.py:105-177`](../../../python/sglang/srt/mem_cache/allocator/paged.py#L105)。slot/page 0 被保留为 dummy padding 区域。

`P=4` 的示意：

```text
page 0 (dummy) : slots [ 0,  1,  2,  3]
page 1         : slots [ 4,  5,  6,  7]
page 2         : slots [ 8,  9, 10, 11]
...
```

extend/prefill 会先补齐 prefix 最后一页，再分配完整页和尾页；decode 在当前页还有空位时沿用下一 offset，跨页边界才分配新页。

### 5.3 slot 如何落到物理 tensor

标准 MHA pool 的常见 NHD layout 是：

```text
K[layer_id].shape = [slots + padding, local_kv_heads, K_head_dim]
V[layer_id].shape = [slots + padding, local_kv_heads, V_head_dim]

read/write address = K_or_V[local_layer][slot_id, :, :]
```

HND/paged layout 则是：

```text
K[layer_id].shape = [num_pages, local_kv_heads, page_size, K_head_dim]
V[layer_id].shape = [num_pages, local_kv_heads, page_size, V_head_dim]

read/write address = K_or_V[local_layer][page_id, :, offset, :]
```

具体 shape 见 [`memory_pool.py:1990-2001`](../../../python/sglang/srt/mem_cache/memory_pool.py#L1990)。因此 allocator 管理的是整数 slot/page ID，而不是大量 Python tensor 对象。

## 6. KV 索引如何被生产、写入和消费

### 6.1 Prefill/extend

[`allocation.py:303-381`](../../../python/sglang/srt/mem_cache/allocation.py#L303) 的主链路是：

```text
request token IDs
      |
      +-- Radix match ----------> prefix_indices (already cached L1 slots)
      |
      +-- allocator ------------> out_cache_loc (new suffix slots/pages)
                                      |
                                      v
          write_cache_indices(prefix_indices, out_cache_loc)
                                      |
                                      v
ReqToTokenPool[req_slot, 0:seq_len]
      = [prefix slots ........ | newly allocated suffix slots ........]
                                      |
                                      v
model forward, every attention layer calls set_kv_buffer(..., out_cache_loc, K, V)
```

同一批中所有请求的新 slots 会拼成 `out_cache_loc`，`prefix_lens`、`extend_lens` 决定每段写回哪个请求 row。

### 6.2 Decode

[`allocation.py:540-585`](../../../python/sglang/srt/mem_cache/allocation.py#L540) 对每个请求分配下一 token 的 slot，然后执行：

```text
ReqToTokenPool[req_slot, current_seq_len] = new_slot
```

此后每层把新 token 的 K/V 写到同一个逻辑 slot 在该层对应的 tensor 中。

### 6.3 Attention backend 消费 CSR 风格索引

一个 batch 的请求 row 长度不同。backend 会把二维 `req_to_token` 重排为：

```text
request A slots: [32, 33, 34, 35, 80, 81]     len=6
request B slots: [44, 45, 46]                 len=3

kv_indices = [32, 33, 34, 35, 80, 81, 44, 45, 46]
kv_indptr  = [0, 6, 9]

request A reads kv_indices[kv_indptr[0]:kv_indptr[1]]
request B reads kv_indices[kv_indptr[1]:kv_indptr[2]]
```

Triton 路径见 [`triton_backend.py:386-404`](../../../python/sglang/srt/layers/attention/triton_backend.py#L386)。所以 `ReqToTokenPool` 是长期的请求级映射，`kv_indices/kv_indptr` 是本轮 batch/kernel 的紧凑视图。

## 7. CP 与 DCP：最容易混淆的两种“上下文并行”

### 7.1 Prefill CP 默认切计算，不切最终持久 KV

以 zigzag CP 为例：

```text
full prompt positions:  0  1  2  3  4  5  6  7

CP rank 0 computes:    [local token shard ........]
CP rank 1 computes:    [local token shard ........]
                              |
                              v
                    all-gather + reorder K/V
                              |
                 +------------+------------+
                 v                         v
CP rank 0 L1 stores full ordered KV    CP rank 1 L1 stores full ordered KV
positions 0..7 for its head shard      positions 0..7 for its head shard
```

`materialize_full_kv()` 和 `materialize_full_mla_kv()` gather 后调用 pool 写入完整 cache，见 [`cp/zigzag.py:363-413`](../../../python/sglang/srt/layers/cp/zigzag.py#L363)。

所以增加 `attn_cp_size` 有两个效果：

```text
direct:   parallelize prefill token computation
indirect: attn_tp_size becomes smaller, so MHA local KV-head count becomes larger
```

例外是 `--enable-dsa-cache-layer-split`：DSA MLA 可把 KV/indexer layers 跨 CP ranks 切分，并为远端层增加 scratch buffer。该特殊路径见 [`cp/utils.py:40-104`](../../../python/sglang/srt/layers/cp/utils.py#L40)。

### 7.2 DCP 真正循环切分持久序列 KV

DCP 的 owner rule 是：

```text
owner(logical_position) = logical_position % dcp_size

if owner == dcp_rank:
    local_physical_index = virtual_slot // dcp_size
```

纯索引数学见 [`dcp/layout.py:23-51`](../../../python/sglang/srt/layers/dcp/layout.py#L23)。

`dcp_size=2` 时，忽略 dummy page：

```text
logical position       0       1       2       3       4       5
virtual slot           0       1       2       3       4       5
owner rank             0       1       0       1       0       1
                       |       |       |       |       |       |
                       v       v       v       v       v       v
DCP rank 0 physical    0               1               2
DCP rank 1 physical            0               1               2
```

写入时 Triton backend 使用 position mask，只把当前 rank 拥有的 token 写到 `out_cache_loc // dcp_size`，见 [`triton_backend.py:1195-1226`](../../../python/sglang/srt/layers/attention/triton_backend.py#L1195)。读取时也只产生当前 rank 的 local KV indices 和 local lengths，见 [`triton_backend.py:353-384`](../../../python/sglang/srt/layers/attention/triton_backend.py#L353)。各 rank 对自己的 KV shard 计算 partial attention，最后按 LSE 权重合并输出，见 [`dcp/comm.py:82-108`](../../../python/sglang/srt/layers/dcp/comm.py#L82)。

### 7.3 DCP 下 page size 的变化

配置 `page_size=P`、`dcp_size=N` 时：

```text
physical KV capacity per rank      = max_total_num_tokens
allocator virtual capacity         = max_total_num_tokens * N
allocator / Radix logical page     = P * N virtual slots
physical tokens per rank per page  = P
```

构造见 [`kv_cache_configurator.py:1509-1527`](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L1509)，RadixTree 使用 allocator 的放大 page size 以避免页淘汰冲突，见 [`kv_cache_builder.py:207-218`](../../../python/sglang/srt/mem_cache/kv_cache_builder.py#L207)。

换句话说，DCP 下 `ReqToTokenPool`/Radix value 保存的是可表达全局逻辑序列的 virtual slots，而真正访问 rank-local K/V tensor 前才按 owner rule 过滤并除以 `dcp_size`。

## 8. RadixTree 如何把 token prefix 组合成 L1 索引

Radix node 的核心字段见 [`radix_cache.py:217-251`](../../../python/sglang/srt/mem_cache/radix_cache.py#L217)：

```text
TreeNode
  key        : 这一压缩树边对应的 token ID 段
  value      : 同长度的 L1 GPU slot IDs
  host_value : 同长度的 L2 Host slot IDs
  hash_value : 可选的逐页 L3 内容 hash
```

注意一棵压缩树节点可以包含多个 page；节点边界和匹配长度在 `page_size>1` 时按页对齐。

### 8.1 插入与去重

请求完成或 chunk 暂停时，SGLang 从 `ReqToTokenPool` 取出：

```text
key   = token_ids[:aligned_len]
value = req_to_token[req_slot, :aligned_len]
```

然后插入 RadixTree，见 [`radix_cache.py:415-435`](../../../python/sglang/srt/mem_cache/radix_cache.py#L415) 和 [`radix_cache.py:437-528`](../../../python/sglang/srt/mem_cache/radix_cache.py#L437)。若相同 prefix 已存在，RadixTree 保留 canonical slots，释放本请求重复计算出来的 slots，并把请求 prefix 映射更新为 canonical indices。

```text
before insert

existing prefix "A B C D" -> L1 [32 33 34 35]
new request same prefix    -> L1 [80 81 82 83]

after insert/deduplicate

Radix canonical value      -> L1 [32 33 34 35]
new request mapping        -> L1 [32 33 34 35]
allocator frees                 [80 81 82 83]
```

### 8.2 下一请求命中

`match_prefix()` 返回最长连续 prefix 的 `device_indices`，见 [`radix_cache.py:355-413`](../../../python/sglang/srt/mem_cache/radix_cache.py#L355)。这些 indices 随后成为新请求的 `prefix_indices`，直接写入该请求的 `ReqToTokenPool` row；只有 suffix 需要新分配和模型计算。

## 9. HiCache L1/L2 中索引如何组合

HiCache 定义：

```text
L1 = rank-local GPU KV pool
L2 = rank-local Host KV pool
L3 = optional external/distributed storage
```

L1 与 L2 的索引属于两个独立地址空间：

```text
same token prefix node

TreeNode.key        = [token 10, token 11, token 12, token 13]
TreeNode.value      = [ 32,  33,  34,  35]   # L1 GPU slots
TreeNode.host_value = [128, 129, 130, 131]   # L2 Host slots

32 does not need to equal 128
```

### 9.1 L1 -> L2 backup

[`hiradix_cache.py:835-865`](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L835) 调用 Controller：

```text
TreeNode.value (L1 slots)
        |
        +-- L2 allocator.alloc(same number of token slots)
        |                  |
        |                  v
        |             host_indices
        |
        +-- async copy all local layers: GPU slots -> Host slots
                           |
                           v
               TreeNode.host_value = host_indices
```

Controller 的分配和异步 copy 入口见 [`cache_controller.py:673-742`](../../../python/sglang/srt/managers/cache_controller.py#L673)。标准 MHA Host pool 的 layout 可为：

```text
layer_first       [K/V, layer, host_slot, head, dim]
page_first        [K/V, host_slot, layer, head, dim]
page_first_direct [K/V, page, layer, offset, head, dim]
```

见 [`pool_host/mha.py:118-145`](../../../python/sglang/srt/mem_cache/pool_host/mha.py#L118)。布局改变拷贝方式，不改变 `host_value` 仍是 token slot/page indices 这一抽象。

### 9.2 L1 淘汰后的节点

```text
L1 + L2 resident              L2-only resident

value      = [32..35]         value      = None
host_value = [128..131]  -->  host_value = [128..131]

GPU slots [32..35] returned to L1 allocator
token-prefix node remains in HiRadixTree
```

此时 L2 prefix 不能直接交给 attention kernel，因为 kernel 只知道 rank-local GPU KV tensors。

### 9.3 L2 -> L1 load-back

[`hiradix_cache.py:1335-1406`](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1335) 执行：

```text
TreeNode.host_value = [128..131]
            |
            +-- allocate fresh L1 slots, e.g. [200..203]
            |
            +-- async copy all local layers Host -> GPU
            |
            v
TreeNode.value = [200..203]
            |
            v
new request's ReqToTokenPool row receives [200..203]
            |
            v
attention backend consumes [200..203]
```

因此一次 prefix 的 L1 slot 可以在不同时间变化：

```text
initial L1 slots     [32..35]
evicted              None
loaded-back L1 slots [200..203]

stable identity is token prefix (TreeNode.key), not a permanent GPU slot number
```

### 9.4 HiRadix 的连续 L1/L2 命中

`HiRadixCache.match_prefix()` 见 [`hiradix_cache.py:1701-1732`](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1701)，返回：

```text
device_indices  : 从 root 开始连续驻留在 L1 的 prefix slots
host_hit_length : 紧接其后、连续驻留在 L2 的 prefix 长度
last_device_node
last_host_node
```

示意：

```text
request tokens
[ page A ][ page B ][ page C ][ page D ][ page E ]
    L1        L1        L2        L2       miss
|<-- device_indices -->|
|<--------- local L1 + L2 hit ----------->|

page C/D must first load back to newly allocated L1 pages;
page E is then computed and written to new L1 pages.
```

它表达的是最长连续 prefix，而不是任意分散的 page hit bitmap。

## 10. 从请求到 L1/L2 的完整时序图

```text
Scheduler receives request tokens
              |
              v
      HiRadix.match_prefix()
              |
       +------+---------------------+
       |                            |
       v                            v
continuous L1 hit             following L2 hit
TreeNode.value                TreeNode.host_value
       |                            |
       |                            v
       |                    allocate fresh L1 slots
       |                            |
       |                    async Host -> GPU copy
       |                            |
       +--------------+-------------+
                      |
                      v
             prefix_indices (all L1)
                      |
                      +-- allocate suffix out_cache_loc
                      |
                      v
             ReqToTokenPool request row
                      |
                      v
              ForwardBatch metadata
                      |
              +-------+--------+
              |                |
              v                v
       write new K/V       build kv_indices /
       at out_cache_loc    kv_indptr for reads
              |                |
              +-------+--------+
                      |
                      v
          rank-local physical KV tensors
                      |
                      v
         cache unfinished/finished request
                      |
                      v
        Radix insert + duplicate slot cleanup
                      |
                      v
       optional L1 -> L2 asynchronous backup
```

DCP 打开时只需在 “L1 virtual slot -> physical KV tensor” 之间增加一层：

```text
virtual slot / logical position
        |
        +-- position % dcp_size selects owner rank
        |
        +-- virtual slot // dcp_size gives local physical slot
```

## 11. 如何读一组实际配置

建议按固定顺序计算，不要直接从 `tp_size` 猜 KV 分布。

```text
Step 1: attn_dp_size = enable_dp_attention ? dp_size : 1
Step 2: attn_tp_size = tp_size / attn_dp_size / attn_cp_size
Step 3: MHA local heads = max(1, total_kv_heads / attn_tp_size)
        MLA local latent dim usually unchanged
Step 4: local layers = current PP stage layer range
Step 5: per-rank bytes/token = local layers * local KV cell bytes
Step 6: apply page_size to get block bytes
Step 7: if DCP enabled, distinguish virtual slots from local physical slots
Step 8: distinguish independent DPA/DP cache instances
Step 9: distinguish L1 GPU slots from L2 Host slots
```

### 11.1 例：标准 GQA

```text
total KV heads = 8
layers         = 32
K/V head dim   = 128/128
kv dtype       = bf16 (2 bytes)
tp             = 8
dpa            = 2
cp             = 2
pp             = 2
page_size      = 16
```

```text
attn_tp_size = 8 / 2 / 2 = 2
local heads  = 8 / 2 = 4
local layers = 32 / 2 = 16             # 假设均匀 PP

bytes/token/rank
  = 16 * 4 * (128 + 128) * 2
  = 32 KiB

bytes/page/rank
  = 16 * 32 KiB
  = 512 KiB
```

这个 512 KiB 是当前 rank 上该 page 的本地 layer + 本地 head shard 数据，不是整个模型跨所有 ranks 的总 KV 大小。

### 11.2 例：MLA + full DPA

```text
tp=8, dp=8, enable_dp_attention=true, cp=1
attn_tp_size = 1
attn_dp_size = 8
```

```text
request A -> attention DP lane 0 -> lane 0 private L1/L2
request B -> attention DP lane 1 -> lane 1 private L1/L2
...

each request's latent KV is not copied to all eight attention ranks
```

这与普通 DP 的“8 套完整模型副本”不同：DPA 只把 attention 变成 DP，而 dense/MoE 其他部分仍可使用 TP/EP 通信。

## 12. 常见误读

### 误读 1：`tp_size` 就是 KV head 切分数

不总是。打开 DPA/CP 后应使用 `attn_tp_size`：

```text
attn_tp_size = tp_size / attn_dp_size / attn_cp_size
```

### 误读 2：`attn_cp_size` 把持久 KV sequence 平分了

标准 CP 路径不是。它把 prefill 计算分片，随后 gather 并物化完整 KV。DCP 才使用 position owner rule 切持久序列 KV。

### 误读 3：Radix node 的 value 是 KV 数据

不是。它是 slot IDs。真正的 K/V 数据在每个 local layer 的 tensor buffer 中。

### 误读 4：L1 slot ID 与 L2 slot ID 应该相同

不需要。它们来自两个 allocator、两个地址空间；TreeNode 用 `value` 与 `host_value` 分别记录映射。

### 误读 5：L2 hit 可以直接写进 `ReqToTokenPool`

不能。`ReqToTokenPool` 是 attention 消费的 L1 映射。L2 hit 必须先 load-back，得到新的 L1 indices。

### 误读 6：EP rank 决定 KV page 属于哪张卡

不决定。EP 切 experts；attention KV 归属由 attention TP/DP/CP、PP、DCP 和 allocator 决定。

## 13. 源码阅读顺序

```text
1. Scheduler 组装 ParallelState
   python/sglang/srt/managers/scheduler.py:424

2. Attention rank 数学
   python/sglang/srt/layers/dp_attention.py:317
   python/sglang/srt/distributed/parallel_state.py:2319

3. KV 容量和 pool 构造
   python/sglang/srt/model_executor/pool_configurator.py:180
   python/sglang/srt/mem_cache/kv_cache_configurator.py:1360

4. Request -> slot 与物理 KV tensors
   python/sglang/srt/mem_cache/memory_pool.py:256
   python/sglang/srt/mem_cache/allocator/paged.py:105
   python/sglang/srt/mem_cache/allocation.py:41

5. Attention batch page table
   python/sglang/srt/layers/attention/triton_backend.py:353
   python/sglang/srt/layers/attention/triton_backend.py:386

6. Radix prefix -> L1 slots
   python/sglang/srt/mem_cache/radix_cache.py:355
   python/sglang/srt/mem_cache/radix_cache.py:415

7. HiCache L1/L2
   python/sglang/srt/mem_cache/hiradix_cache.py:835
   python/sglang/srt/mem_cache/hiradix_cache.py:1335
   python/sglang/srt/managers/cache_controller.py:673

8. CP 与 DCP 特殊路径
   python/sglang/srt/layers/cp/zigzag.py:363
   python/sglang/srt/layers/dcp/layout.py:23
   python/sglang/srt/layers/dcp/comm.py:82
```

## 14. Rank、Shard 与 Slot 的严格区分

`rank` 不是切分后的最小 tensor 单元，而是一个分布式通信组中的执行参与者编号。SGLang 中它通常对应一个 worker 进程及其控制的一张 GPU，但 rank 的定义始终相对于某个 process group：

```text
rank  = 谁负责计算和存储
shard = 该 rank 负责的那部分权重、激活或 KV 数据
slot  = rank-local KV pool 中一个 token 的存储坐标
page  = allocator 一次分配、释放和淘汰的一组 slots
```

同一个 worker 可以同时拥有多个 rank 坐标：

```text
one worker / one GPU process
    |
    +-- global_rank
    +-- tp_rank
    +-- pp_rank
    +-- attn_tp_rank
    +-- attn_cp_rank
    +-- attn_dp_rank
    +-- dcp_rank
    +-- moe_ep_rank / moe_dp_rank
```

例如 TP 将完整权重切成四份时：

```text
full tensor W
    |
    +-- W shard 0 --> TP rank 0
    +-- W shard 1 --> TP rank 1
    +-- W shard 2 --> TP rank 2
    +-- W shard 3 --> TP rank 3
```

`rank 0` 是 owner/executor 的编号，`W shard 0` 才是切分后的 tensor。一个 rank 通常同时拥有大量 weight shards、KV layer buffers、token pages 和临时 workspace，因而不能把 rank 理解成最小存储单位。

### 14.1 Layer-wise KV 写入是当前主路径

Transformer forward 本来就是逐层执行；每层产生自己的 K/V 后立即写入 L1 KV pool。关键设计是：slot 只沿 token 维度分配一次，然后被所有本地 attention layers 复用。

```text
before forward
    allocator.alloc(batch tokens)
               |
               v
    out_cache_loc = [80, 81, 82]
               |
               +------------------------------------+
               |                                    |
               v                                    v
ReqToTokenPool[token positions]           ForwardBatch.out_cache_loc
       = [80, 81, 82]                              |
                                                   v
                                          every attention layer
                                          uses the same slots
```

假设当前 PP rank 负责全局 layers `[8, 16)`：

```text
pool.start_layer = 8

global layer 8  -> local layer 0 -> write slots [80,81,82]
global layer 9  -> local layer 1 -> write slots [80,81,82]
global layer 10 -> local layer 2 -> write slots [80,81,82]
...
```

最终形成：

```text
                  slot 80       slot 81       slot 82
               +-------------+-------------+-------------+
local layer 0  | layer 8 KV  | layer 8 KV  | layer 8 KV  |
               +-------------+-------------+-------------+
local layer 1  | layer 9 KV  | layer 9 KV  | layer 9 KV  |
               +-------------+-------------+-------------+
local layer 2  | layer 10 KV | layer 10 KV | layer 10 KV |
               +-------------+-------------+-------------+
```

因此 slot 可以理解为跨所有本地 KV layers 的一个 token 列坐标，但它通常不是覆盖所有 layers 的单一连续裸地址段。常见实现为每层各有一个 K tensor 和 V tensor，slot ID 在这些 tensors 中重复作为 token 维下标。

### 14.2 完整的 rank-local 写入坐标

标准 MHA/GQA 的完整本地定位可以写成：

```text
(
    local_layer_id,
    slot_id or (page_id, page_offset),
    local_kv_head,
    head_dimension,
    K_or_V
)
```

其中：

```text
local_layer_id = global_layer_id - pool.start_layer

page_id        = slot_id // page_size
page_offset    = slot_id %  page_size
```

rank 没有出现在本地 tensor 下标中，因为当前进程已经隐式选中了 rank-local pool：

```text
rank/topology
    |
    +-- selects local device and KV pool
    +-- selects local PP layer range
    +-- selects local attention TP head shard
    +-- selects DPA request lane
    +-- selects DCP token-position shard when enabled
```

如果从整个部署的全局视角描述，一个 KV 元素的身份则近似为：

```text
(
    cache instance / model namespace,
    parallel rank coordinates,
    global layer_id,
    logical token position,
    KV head shard,
    K_or_V,
    dimension
)
```

本地 pool 把前两组部署上下文隐含掉，最终只需要 `local_layer_id + slot/page` 即可落盘。

实际写入路径位于 [`triton_backend.py:1263-1285`](../../../python/sglang/srt/layers/attention/triton_backend.py#L1263)，它将 `forward_batch.out_cache_loc` 和当前 `RadixAttention.layer_id` 传给 pool。标准 MHA pool 在 [`memory_pool.py:2265-2341`](../../../python/sglang/srt/mem_cache/memory_pool.py#L2265) 中执行：

```text
buffer_index = layer.layer_id - start_layer

NHD:
    K[buffer_index][slot_ids, :, :] = produced_K
    V[buffer_index][slot_ids, :, :] = produced_V

HND:
    K[buffer_index][page_ids, :, offsets, :] = produced_K
    V[buffer_index][page_ids, :, offsets, :] = produced_V
```

DCP 再在这个过程前增加 owner 过滤和 virtual-to-physical 转换：

```text
owner(token_position) = token_position % dcp_size
physical_slot         = virtual_slot // dcp_size
```

### 14.3 Layer-wise L1/L2 传输使用同一组 slots

如果把 KV 从 L1 分层搬到 L2，不需要为每层建立另一套 Radix 索引。Controller 先为一个 token prefix 建立一次 slot 对：

```text
device_indices = [80, 81, 82]
host_indices   = [160,161,162]
```

然后 pool 根据自己的 layer/layout 信息解释这组映射：

```text
for each local layer:
    GPU_K[layer][device_indices] -> Host_K[layer][host_indices]
    GPU_V[layer][device_indices] -> Host_V[layer][host_indices]
```

当前 backup 入口以一次操作提交所有本地 layers，见 [`cache_controller.py:673-742`](../../../python/sglang/srt/managers/cache_controller.py#L673)；load-back 支持逐 layer copy 和 layer completion event，使第 N 层计算可以等待第 N 层 KV 就绪，而不必等待全部 layers 完成，见 [`cache_controller.py:790-817`](../../../python/sglang/srt/managers/cache_controller.py#L790) 和 [`memory_pool.py:2233-2260`](../../../python/sglang/srt/mem_cache/memory_pool.py#L2233)。

如果未来把一层 KV 直接流式发送给远端 consumer，传输描述符至少需要表达：

```text
model/cache namespace
token page key or token span
producer parallel-shard identity
global/local layer_id
source and destination slot vectors
K/V component
dtype, quantization and physical layout
```

这是传输层的复合地址；它不要求把 layer ID 加进本地 Radix token key。

## 15. HiRadixTree 对底层拓扑无感到什么程度

结论是：HiRadixTree 对 KV 的 head/layer/tensor layout 基本无感，但并非对所有物理约束完全无感。更准确的职责切分是：

```text
HiRadixTree = token prefix directory
KV pool     = topology-aware physical storage
allocator   = slot/page ownership manager
backend     = topology-aware reader/writer
controller  = L1/L2 slot mapping and data movement
```

### 15.1 Tree 不关心的内容

HiRadixTree 的核心映射是：

```text
TreeNode.key        = token prefix
TreeNode.value      = rank-local L1 slot vector
TreeNode.host_value = rank-local L2 slot vector
TreeNode.hash_value = optional L3 page hash chain
```

它不直接解析：

```text
how many local layers exist
how many local KV heads exist
whether payload is MHA K/V or MLA latent KV
whether tensors use NHD, HND or a quantized packed layout
which bytes belong to K, V or a sidecar scale buffer
```

同样的 token prefix 和同样的数值 slot 可以在不同 ranks 上表示不同物理 shard：

```text
TP rank 0                           TP rank 1

key   = [A,B,C]                     key   = [A,B,C]
value = [80,81,82]                  value = [80,81,82]
          |                                   |
          v                                   v
local KV head shard 0               local KV head shard 1
```

两个 `80` 是不同 GPU pool 中的本地坐标，不是一个跨 GPU 的全局物理地址。Tree 不把这些 shards 汇集成一个 global slot；它只在每个 rank 上维护相同 token 语义对应的本地 payload 位置。

不同并行方式被隐藏在 pool 后面：

```text
                         HiRadixTree
                    token prefix -> slots
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
       TP pool             PP pool             DCP pool
      local heads         local layers       local positions
```

- TP：相同 key 在不同 ranks 上定位不同 KV head shards。
- PP：相同 key 在不同 stages 上定位不同 local layer ranges。
- DPA：不同 lanes 拥有独立请求、独立 Tree 和独立 L1/L2 pools。
- CP：标准路径中 pool 物化完整序列；特殊 layer-split 由专用 pool 解释。
- DCP：Tree 保存 virtual slots，attention backend 转换为 rank-local physical slots。
- MLA：slot payload 是 latent KV，具体编码由 MLA pool 解释。

### 15.2 拓扑结构不是“编码进 slot”

应避免下面的表述：

```text
slot ID contains TP/PP/layer/head topology
```

更准确的是：

```text
slot ID + owning pool/runtime context
    |
    v
complete physical interpretation
```

概念上的完整地址为：

```text
(
    cache instance,
    rank coordinates,
    pool type/layout,
    local layer,
    slot/page,
    local head,
    K/V component
)
```

Tree 只持有最后几个维度中的 slot/page；其余信息由它关联的 allocator、device pool、host pool 和当前 rank 提供。

### 15.3 Tree 仍然必须关心的约束

虽然 Tree 不解析 tensor payload，它仍依赖以下不变量。

第一，page granularity 必须一致：

```text
match / insert / node split / eviction
                |
                v
          all page-aligned
```

DCP 下 Tree 必须使用 `configured_page_size * dcp_size` 的 virtual page size，否则可能在各 ranks 上形成不可安全淘汰的半页。

第二，allocator 语义必须一致。Tree 保存的 value 必须是当前 allocator 可以合法 `free`、`free_segment`、lock 和 evict 的 indices；Tree 虽不知道 HND/NHD，却要求 allocator 与 pool 对同一 slot 有一致解释。

第三，多 rank 的可用 prefix 必须一致。不同 ranks 保存不同 shards，但一个请求只能使用所有必要 shards 都有效的最长连续 prefix：

```text
rank 0 local usable prefix = 128 tokens
rank 1 local usable prefix = 128 tokens
rank 2 local usable prefix =  96 tokens
                              |
                              v
distributed usable prefix = MIN(...) = 96 tokens
```

HiRadixTree 不组合各 rank 的 KV bytes，但需要通过 attention TP/CP/PP groups 协调命中长度、容量和关键状态。相关同步入口见 [`hiradix_cache.py:210-240`](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L210)。

第四，cache semantic 必须兼容。Tree 不会读取 KV 内容来验证它是否来自相同模型，因此下列变化必须通过独立实例、`RadixKey.extra_key`、cache namespace 或 L3 backend tag 隔离：

```text
model weights or revision
LoRA / adapter
position encoding semantics
KV dtype and quantization format
MHA / MLA encoding
TP/PP/DCP shard layout
layer mapping or layer skipping policy
```

否则可能发生：

```text
same token prefix
       |
       v
Tree reports a hit
       |
       v
slot actually contains incompatible KV semantics
       |
       v
silent wrong output
```

所以所谓“Tree 不关心切分结构”的正确含义是：Tree 不在 node key/value 中显式编码和解释 layer/head/tensor layout；不是说任意模型、任意 topology 生成的 slot 都可以互换复用。

## 16. 一句话心智模型

```text
ParallelState 决定当前 rank 保存哪一部分 KV 语义；
allocator 把这部分语义放进哪些 L1 slot/page；
ReqToTokenPool 把请求 token 位置映射到这些 slots；
RadixTree 用 token prefix 跨请求复用这些 L1 slots；
HiRadixTree 再为同一 prefix 维护独立的 L2 Host slots，
但任何 L2 数据都必须先回迁为新的 L1 slots，才能进入 attention kernel。

rank 是 owner，不是 shard；slot 是 token 坐标，不是 topology encoding；
layer-wise KV 使用 local_layer_id + shared token slots 写入不同 layer buffers；
HiRadixTree 只管理 token prefix 到 rank-local slots 的目录关系，
具体 TP/PP/CP/DCP/MLA/MHA 布局由关联的 pool、allocator 和 backend 解释。
```
