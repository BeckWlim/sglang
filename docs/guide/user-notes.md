# SGLang

## Scheduler

### Request Admission and Prefill Scheduling

Analyze the Scheduler's execution chain in normal mode.
The main loop is called:

```python
def event_loop_normal(self):
    while True:
        # code...
```

Every request is received from `self.request_receiver.recv_requests()` and then processed by `process_input_requests`.

After that, the scheduling plan is built by `get_next_batch_to_run()`.
Its main steps include dLLM path selection and excluding an active `chunked_req` while completed-prefill requests are merged into `running_batch`.

Finally, this function returns an instance called `NextBatchPlan`.
At this point, we can review where the key values `ret` and `running_batch` come from.

The Scheduler passes `running_batch` to `self.get_new_batch_prefill()` while building a new prefill plan.
Just like:

```python
prefill_plan = self.get_new_batch_prefill(running_batch)
new_batch = prefill_plan.batch_to_run
running_batch = prefill_plan.running_batch
```

The method `get_new_batch_prefill()` is a wrapper.
The actual implementation is in the internal method `_get_new_batch_prefill_raw()`.

Because this raw function is also a `Scheduler` method, it can directly read `Scheduler.waiting_queue`.
For each `req`, the main processing happens in two lines:

```python
req.init_next_round_input(self.tree_cache)
res = adder.add_one_req(
    req,
    has_chunked_req=(self.chunked_req is not None),
    truncation_align_size=self.truncation_align_size,
)
```

The next step is to read deeper into `PrefillAdder.add_one_req()`.
I can return later to understand what this initialization block does:

```python
# Prefill policy
adder = PrefillAdder(
    self.page_size,
    self.tree_cache,
    self.token_to_kv_pool_allocator,
    running_batch,
    self.new_token_ratio_tracker.current,
    self.max_prefill_tokens,
    chunked_prefill_size,
    running_bs if self.is_mixed_chunk else 0,
    self.priority_scheduling_preemption_threshold,
    max_prefill_bs=self.max_prefill_bs,
    max_running_requests=self.max_running_requests,
    prefill_max_requests=get_schedule().prefill_max_requests,
    prefill_delayer_single_pass=prefill_delayer_single_pass,
    dllm_config=self.dllm_config,
    waiting_queue_len=len(self.waiting_queue),
)
```

OK, the `adder` stores information about `running_batch`,
together with Scheduler configuration such as `page_size`, `tree_cache`, `max_running_requests`, and `token_to_kv_pool_allocator`.

After some complex processing for `hybrid_swa`, `host_load_back`, and related paths, the core function distinguishes requests whose `input_tokens` fit within `chunk_tokens_limit` from requests that require chunking.

At this point, the reading context has moved into `PrefillAdder`. The normal path adds an accepted `req` to `PrefillAdder.can_run_list`.
For a `req` that requires chunking, `add_one_req()` changes its `extend_range`, adds it to `can_run_list`, and also assigns it to `new_chunked_req`.
One `Scheduler` maintains one active cross-iteration `chunked_req`.

The source code is like:

```python
self.can_run_list.append(req)
self.new_chunked_req = req

self._req_inc_lock_ref(req)
self._update_prefill_budget(
    prefix_len,
    trunc_len,
    0,
    req.retracted_stain,
    mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
)
```

This function returns `self.budget_state()`, which provides the scheduling signals `CONTINUE`, `NO_TOKEN`, or `OTHER`.
I should check where `_get_new_batch_prefill_raw()` uses `adder.can_run_list` and `adder.new_chunked_req`.

After finishing the `Scheduler.waiting_queue` loop, `can_run_list` is read from `adder`.
Accepted requests in `can_run_list` are removed from `self.waiting_queue`, 
while the remaining requests stay there for future scheduling iterations. 
This queue update occurs inside `_get_new_batch_prefill_raw()`.

`can_run_list` is then passed to `ScheduleBatch.init_new()`, and `prepare_for_extend()` turns it into the actual prefill-forward batch. 
A new `chunked_req` is transferred to `Scheduler.chunked_req` and repeatedly handled by `add_chunked_req()` until its final chunk. 
After final prefill, the request moves through `last_batch`, `running_batch`, and `prepare_for_decode()`.

`Scheduler.init_request_dispatcher()` is initialization-time setup, not a per-request call. 
During `Scheduler.__init__()`, it builds a `TypeBasedDispatcher` that maps each received object type to a bound Scheduler method. 
For example:

```python
self._request_dispatcher = TypeBasedDispatcher(
    [
        (TokenizedGenerateReqInput, self.handle_generate_request),
        (TokenizedEmbeddingReqInput, self.handle_embedding_request),
        (AbortReq, self.abort_request),
        # control requests omitted
    ]
)
```

Source: [`scheduler.py:1425-1534`](../../../python/sglang/srt/managers/scheduler.py#L1425).

The normal event loop uses that dispatcher indirectly. 
It first receives a list of decoded request objects, 
then `process_input_requests()` calls the dispatcher once for each object:

```text
event_loop_normal()
    +-- request_receiver.recv_requests()
    `-- process_input_requests(recv_reqs)
          `-- self._request_dispatcher(recv_req)
                `-- handle_generate_request(recv_req)  # generate request
```

Only after all currently received objects have been dispatched does the event loop call `get_next_batch_to_run()`. 
Source:
[`scheduler.py:1610-1635`](../../../python/sglang/srt/managers/scheduler.py#L1610) and [`scheduler.py:1766-1792`](../../../python/sglang/srt/managers/scheduler.py#L1766).

For an ordinary new generate request, `recv_req.session_params` is absent, 
so the first branch of `handle_generate_request()` constructs a new `Req` directly:

```python
session_id = (
    recv_req.session_params.id if recv_req.session_params is not None else None
)

if session_id is None or radix_native_session:
    req = Req(...)
```

The following `elif session_id in self.session_controller` is not the ordinary new-request path. 
It creates a continuation request from an existing legacy session with `session.create_req()`. 
A missing or closing requested session takes the final error branch. 
Source: [`scheduler.py:2220-2335`](../../../python/sglang/srt/managers/scheduler.py#L2220).

After common validation, multimodal preparation, logprob configuration, and grammar processing, 
the normal successful path reaches the final queue call:

```python
added_to_grammar_queue = self.grammar_manager.process_req_with_grammar(req)
if not added_to_grammar_queue:
    self._add_request_to_queue(req)
```

Most earlier `_add_request_to_queue()` calls in `handle_generate_request()` are early-return error paths that queue an aborted request for reporting and cleanup. 
If asynchronous grammar preparation is required, the request is held in the grammar queue; 
`_get_new_batch_prefill_raw()` calls `_add_request_to_queue()` after the grammar becomes ready.

Queue placement depends on disaggregation mode:

```text
DisaggregationMode.NULL
    _prefetch_kvcache(req) -> waiting_queue.append(req)

DisaggregationMode.PREFILL
    _prefetch_kvcache(req) -> disagg_prefill_bootstrap_queue.add(req)

DisaggregationMode.DECODE
    disagg_decode_prealloc_queue.add(req)
    # no _prefetch_kvcache() call on this path
```

Thus ordinary non-disaggregated serving starts the optional L3 prefetch before putting the request in `waiting_queue`. 
Disaggregated prefill also starts the prefetch, but its immediate destination is the bootstrap queue rather than the ordinary waiting queue. 
Source:
[`scheduler.py:2514-2563`](../../../python/sglang/srt/managers/scheduler.py#L2514).

### Local KV Cache Search

Before `PrefillAdder.add_one_req()` can calculate the required prefill tokens,
the request needs its cached-prefix information. 
`Req.init_next_round_input()` queries `Scheduler.tree_cache` and add related query and transfer info into `Req`'s member varaible:

```python
match_result = tree_cache.match_prefix(
    MatchPrefixParams(
        key=RadixKey(
            token_ids=token_ids_to_match,
            extra_key=self.extra_key,
            limit=key_limit,
        ),
        req=self,
        cow_mamba=cow_mamba,
    )
)
```

and then

```python
(
    self.prefix_indices,
    self.last_node,
    self.last_host_node,
    self.best_match_node,
    self.host_hit_length,
    self.swa_host_hit_length,
    self.mamba_host_hit_length,
    self.mamba_branching_seqlen,
) = (
    match_result.device_indices,
    match_result.last_device_node,
    match_result.last_host_node,
    match_result.best_match_node,
    match_result.host_hit_length,
    match_result.swa_host_hit_length,
    match_result.mamba_host_hit_length,
    match_result.mamba_branching_seqlen,
)
```

`tree_cache` is a `BasePrefixCache` implementation selected during KV-cache
construction:

```text
build_kv_cache()
    └── create_tree_cache(context)
          └── choose implementation
                ├── RadixCache(params)
                ├── HiRadixCache(params, server_args)
                ├── UnifiedRadixCache(params)
                └── ChunkCache(params)
```

This note follows the typical `HiRadixCache` implementation of the abstract matching interface. 
Its radix-tree traversal returns two categories of information in `MatchResult`:

1. `device_indices`, which identify a prefix already available in L1. Their
   exact meaning and their use in batch construction are covered in the next
   Scheduler subsection.
2. Host-hit lengths, which describe matched cache state that still requires
   L2-to-L1 restoration.

The host-load demand is separated by cache type:

```python
host_hit_length          # Full-KV tokens requiring L2 -> L1
swa_host_hit_length      # SWA tokens requiring L2 -> L1
mamba_host_hit_length    # Mamba slots requiring L2 -> L1
```

These lengths describe L2 hits but do not start a transfer. During admission,
`PrefillAdder.add_one_req()` uses `needs_host_load_back()` as the unified gate:

```python
if req.needs_host_load_back():
    new_indices, req.last_node = self.tree_cache.init_load_back(...)
```

The predicate is:

```python
return (
    self.host_hit_length > 0
    or self.swa_host_hit_length > 0
    or self.mamba_host_hit_length > 0
)
```

The waiting-queue path is therefore:

```text
_get_new_batch_prefill_raw()
    |
    `-- for req in self.waiting_queue:
          |
          +-- req.init_next_round_input()
          |     +-- tree_cache.match_prefix()
          |     `-- copy device indices and host-hit lengths onto Req
          |
          `-- adder.add_one_req()
                +-- req.needs_host_load_back()
                |     `-- optional tree_cache.init_load_back()
                `-- append an admitted request to can_run_list
```

### L2-to-L1 Load-back and L1 Reuse

For an admitted host-hit request, `tree_cache.init_load_back()` eventually calls `HiCacheController.load()`, 
which allocates destination L1 slots and appends a copy operation to `load_queue`. 
After the waiting-queue loop, the Scheduler builds `new_batch`, 
starts the accumulated transfers, and records their event-set index:

```python
new_batch.hicache_consumer_index = (
    self.tree_cache.ready_to_load_host_cache()
)
```

`ready_to_load_host_cache()` delegates to
`HiCacheController.start_loading()`. The controller merges and clears the
queue, selects one reusable event set, and submits layer-wise copies to its
asynchronous load stream:

```text
tree_cache.init_load_back()
    `-- HiCacheController.load()
          `-- allocate L1 slots and append to load_queue

tree_cache.ready_to_load_host_cache()
    `-- HiCacheController.start_loading()
          +-- merge and clear load_queue
          +-- submit layer-wise L2 -> L1 copies
          `-- return the selected event-set index
```

`LayerDoneCounter` maintains three event sets so overlapping load waves can use
different synchronization objects:

```text
events[0] = [layer-0-ready, layer-1-ready, ..., layer-N-ready]
events[1] = [layer-0-ready, layer-1-ready, ..., layer-N-ready]
events[2] = [layer-0-ready, layer-1-ready, ..., layer-N-ready]
```

Each event set corresponds to one `start_loading()` invocation: a batch-level
load wave containing the host-hit prefixes of all applicable admitted requests.
`CacheOperation.merge_ops()` concatenates their source and destination index
vectors:

```python
host_indices = torch.cat([op.host_indices for op in ops])
device_indices = torch.cat([op.device_indices for op in ops])
```

Source: [`cache_controller.py:140-154`](../../../python/sglang/srt/managers/cache_controller.py#L140).

For example:

```text
request A: host [7, 8]       -> L1 [81, 82]
request B: host [20, 21, 22] -> L1 [19, 44, 45]

merged host indices: [7, 8, 20, 21, 22]
merged L1 indices:   [81, 82, 19, 44, 45]
```

`[batch, seq_len, d]` is the logical attention view. 
The physical cache uses a flattened or paged pool addressed by the merged one-dimensional index vectors.
A common per-layer MHA layout is approximately:

```text
K[layer]: [num_slots, local_kv_heads, head_dim]
V[layer]: [num_slots, local_kv_heads, head_dim]
```

A page-oriented backend can instead use a layout such as:

```text
[num_pages, local_kv_heads, page_size, head_dim]
```

The concrete device layouts are defined in
[`memory_pool.py:1990-2054`](../../../python/sglang/srt/mem_cache/memory_pool.py#L1990).

For each local layer, `start_loading()` passes the complete merged index vectors
to `load_to_device_per_layer()`. Conceptually, it performs:

```text
host K[layer][merged_host_indices] -> L1 K[layer][merged_device_indices]
host V[layer][merged_host_indices] -> L1 V[layer][merged_device_indices]
```

The actual indexed K/V transfer is implemented in
[`pool_host/mha.py:212-241`](../../../python/sglang/srt/mem_cache/pool_host/mha.py#L212).
After every source slot in the merged wave has been copied for layer `i`, the controller records `events[event_set_index][i]`.

For example, if `start_loading()` returns `1`, the batch uses `events[1]`:

```python
new_batch.hicache_consumer_index = 1
```

```text
load stream:     load L0 --record E[1][0]-- load L1 --record E[1][1]--
forward stream:           wait E[1][0] -> L0       wait E[1][1] -> L1
```

The `HiCacheAck` appended at the end of `start_loading()` serves a different
purpose from the per-layer events:

```python
self.ack_load_queue.append(
    HiCacheAck(
        start_event=ack_start_event,
        finish_event=ack_finish_event,
        node_ids=op.node_ids,
        num_tokens=len(op.device_indices),
        timing_enabled=timing_enabled,
    )
)
```

At this point, the controller has already submitted the copies and event records to the asynchronous load stream. 
Appending the acknowledgment neither starts a new copy nor invokes model computation. 
The two completion paths are:

```text
per-layer event events[k][i]
    `-- compute stream may read layer i's L1 KV and execute that layer

whole-load ack.finish_event
    `-- HiCache may finalize bookkeeping after every layer has loaded
```

Model calculation continues through the ordinary Scheduler-to-TP-worker forward
path. The TP worker installs the batch's `hicache_consumer_index`; when attention
requests layer `i`'s K/V buffers, `get_key_buffer()` and `get_value_buffer()` call
`LayerDoneCounter.wait_until()`. This inserts a compute-stream dependency on the
matching layer-ready event. Source:
[`memory_pool.py:2233-2263`](../../../python/sglang/srt/mem_cache/memory_pool.py#L2233).

The whole-load acknowledgment is consumed separately by
`HiRadixCache.loading_check()`. Once `ack.finish_event` is ready, it removes the
operation from `ack_load_queue`, clears the associated nodes from
`ongoing_load_back`, releases their protection locks, and records load metrics.
Source:
[`hiradix_cache.py:1055-1084`](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1055).

Consequently, forward can start using layer 0 after its layer event even while
later layers are still loading. The request-level prefix/suffix composition is
already expressed by `req_to_token`, as described next.

For an ordinary Transformer using MHA or GQA, `HiRadixCache` obtains the L1 KV
pool from `token_to_kv_pool_allocator` and creates its matching L2 Host pool:

```text
token_to_kv_pool_allocator
    `-- MHATokenToKVPool                  L1/GPU
          `-- get_mha_host_pool_cls()
                `-- MHATokenToKVPoolHost  L2/Host
```

In the usual case, K and V have equal head dimensions, so the factory selects the `MHATokenToKVPoolHost` shown above. 
Models with unequal K/V head dimensions use `AsymmetricMHATokenToKVPoolHost`; 
MLA models use `MLATokenToKVPoolHost`. 
The ordinary selection is implemented in [`hiradix_cache.py:76-111`](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L76) and [`pool_host/mha.py:1299-1307`](../../../python/sglang/srt/mem_cache/pool_host/mha.py#L1299).

Without hierarchical cache, the model still has its L1 device KV pool, 
but this HiCache L2 Host pool is not created.

`MHATokenToKVPoolHost` also exposes page access and buffer metadata used by optional L3 storage adapters. 
The Mooncake-specific path is described in the final Scheduler subsection.

An L1 cache hit does not normally cause `HiCacheController.start_loading()` to move the prefix to a new location. 
The radix-cache match returns the slot IDs of KV data that is already resident in the device pool. 
`HiRadixCache.match_prefix()` collects the matched node values and returns them as `MatchResult.device_indices`:

```python
value, last_node = self._match_prefix_helper(self.root_node, key)
if value:
    value = torch.cat(value)

return MatchResult(
    device_indices=value,
    ...
)
```

Source: [`hiradix_cache.py:1701-1732`](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1701).

Here, a KV slot ID is an allocator-level address into the L1 KV pool, not a raw CUDA byte pointer. 
It identifies a row or page in the pool's K/V tensors. 
The request additionally has logical token positions, so the complete lookup has three conceptual levels:

```text
logical token position
        |
        v
req_to_token[request row, position]
        |
        v
L1 KV slot ID
        |
        v
row/page in the rank-local K/V tensors
```

For an L1-hit prefix followed by a cache-miss suffix, the existing prefix slots remain where they are. 
`alloc_for_extend()` takes each request's `prefix_indices` and allocates `out_cache_loc` only for the tokens that must be
computed by the extend/prefill forward:

```python
prefix_tensors = [r.prefix_indices for r in batch.reqs]

if alloc_page_size == 1:
    out_cache_loc = alloc_token_slots(batch.tree_cache, batch.extend_num_tokens)
else:
    out_cache_loc = alloc_paged_token_slots_extend(...)
```

Source: [`allocation.py:303-369`](../../../python/sglang/srt/mem_cache/allocation.py#L303).

The old prefix slots and new suffix slots need not be physically contiguous.
They are joined into one logical request sequence by `write_cache_indices()`:

```python
req_to_token_pool.write(
    (req_idx, slice(0, prefix_len)),
    prefix_tensors[i],
)
req_to_token_pool.write(
    (req_idx, slice(prefix_len, seq_len)),
    out_cache_loc[pt : pt + extend_len],
)
```

Source: [`allocation.py:55-101`](../../../python/sglang/srt/mem_cache/allocation.py#L55).

For example:

```text
logical position       0    1    2    3    4
                       |    |    |    |    |
req_to_token row      81   82   19   44   45
                      `-L1 hit-'  `--new suffix--'

physical interpretation:
  slots 81, 82 already contain the cached prefix KV
  slots 19, 44, 45 receive KV produced by prefill
```

The apparently scattered slot IDs do not require a tensor concatenation or a relocation step. 
Attention follows the request mapping to gather the KV for the logical sequence. 
The radix node protecting the hit prefix and the request's pool row keep those existing slots usable while the request is running.

In prefill/decode disaggregation, the sender also follows the logical mapping instead of requiring contiguous prefix and suffix storage:

```python
kv_indices = self.req_to_token_pool.req_to_token[
    req.req_pool_idx, start_idx:end_idx
]
page_indices = kv_to_page_indices(kv_indices, page_size)
req.disagg_kv_sender.send(page_indices, ...)
```

Source: [`prefill.py:1241-1251`](../../../python/sglang/srt/disaggregation/prefill.py#L1241).

Therefore the concise rule is:

```text
L1 hit  -> reuse existing prefix slot IDs
L2 hit  -> allocate L1 destination slots and queue load-back
miss    -> allocate suffix slots and let prefill write new KV

req_to_token joins all resulting slot IDs into one logical token sequence.
```

### Mooncake-backed External Storage

For the ordinary MHA/GQA path, Mooncake is an optional L3 store behind the target model's Host pool:

```text
MHATokenToKVPool                 L1/GPU
        ^  |
        |  | HiCacheController load/backup
        |  v
MHATokenToKVPoolHost             L2/Host
        ^  |
        |  | MooncakeStore adapter
        |  v
Mooncake distributed store       L3
```

The read path is one pipeline shared by three persistent loops.
`HiCacheController.attach_storage_backend()` calls `_start_storage_threads()`
once when the storage backend is attached.

```text
startup: _start_storage_threads() -> query loop -> starts transfer loop once

Scheduler loop                        Query loop                            Transfer loop
event_loop_normal()                   prefetch_thread_func()                prefetch_io_aux_func()
    |                                     |                                     |
    | process_input_requests()            |                                     |
    | _request_dispatcher()               |                                     |
    | handle_generate_request()           |                                     |
    | _add_request_to_queue()             |                                     |
    | _prefetch_kvcache()                 |                                     |
    | init_next_round_input()             |                                     |
    | prefetch_from_storage()             |                                     |
    |------ prefetch_queue.put(op) ------>|                                     |
    |                                     | prefetch_queue.get()                |
    |                                     | _storage_hit_query()                |
    |<------- hit/revoke result ----------|                                     |
    |                                     | return to queue.get()               |
    | later event-loop iteration          |                                     |
    | get_next_batch_to_run()             |                                     |
    | get_new_batch_prefill()             |                                     |
    | _get_new_batch_prefill_raw()        |                                     |
    | check_hicache_events()              |                                     |
    | _drain_and_alloc_storage_hit()      |                                     |
    | allocate L2 host_indices            |                                     |
    |------------------------ prefetch_buffer.put(op) ------------------------->|
    |                                     |                                     | prefetch_buffer.get()
    |                                     |                                     | _page_transfer()
    |                                     |                                     | batch_get_v1()
    |                                     |                                     | Client::BatchGet() -> L2
    |                                     |                                     | completed_tokens += bytes
    |<.......................... shared completion .............................|
    |                                     |                                     | return to buffer.get()
    | check_prefetch_progress()           |                                     |
    | commit Host radix nodes             |                                     |
    | second match; L2 -> L1              |                                     |
    |                                     |                                     |
    | repeat event loop                   |                                     |
    |                                     | repeat query loop                   |
    |                                     |                                     | repeat transfer loop
```

Each vertical line is one persistent execution loop. 
Horizontal arrows cross execution levels: a queue insertion publishes work but does not call the receiving function on the producer's stack. 
Each receiving loop resumes independently when its own scheduling condition is met.

`check_hicache_events()` is therefore not called by `process_input_requests()` and is not a callback from `prefetch_hit_queue`.
In the ordinary non-dLLM path, `event_loop_normal()` reaches it while planning the next batch through `get_next_batch_to_run()`, 
`get_new_batch_prefill()`, and `_get_new_batch_prefill_raw()`, as shown in the integral diagram.

This polling point appears before `_get_new_batch_prefill_raw()` scans `waiting_queue`, 
so newly published hit/revoke results can update cache state 
before the Scheduler decides whether a waiting request can enter the next prefill batch.

Request admission and the storage query form the first asynchronous cycle:

```text
Scheduler loop                              Query loop

_prefetch_kvcache()
  `-- prefetch_from_storage()
        `-- [prefetch_queue] ------------> prefetch_thread_func()
                                            wait for operation
                                            _storage_hit_query()
                                            publish hit or revoke
                                            return to prefetch_queue.get()
                                                    |
later Scheduler prefill-planning pass
  `-- check_hicache_events() <--- [hit/revoke queues]  Scheduler polls
        `-- _drain_and_alloc_storage_hit()
```

Allocation and byte transfer form the second asynchronous cycle:

```text
Scheduler loop                              Transfer loop

_drain_and_alloc_storage_hit()
  +-- allocate or evict L2 pages
  `-- [prefetch_buffer] ------------------> prefetch_io_aux_func()
                                            wait for allocation
                                            _page_transfer()
                                            batch_get_v1() -> L2
                                            advance completed_tokens
                                            return to prefetch_buffer.get()
                                                    |
later check_prefetch_progress() <--- [shared operation state]
  +-- commit Host-backed radix nodes
  +-- perform the second local match
  `-- continue through ordinary L2-to-L1 load
```

The source separates these responsibilities by execution level:

| Execution level | Persistent entry | Waits on or revisits | Produces |
|---|---|---|---|
| Scheduler | `event_loop_normal()` | Requests and later scheduling iterations | Operation registration, L2 allocation, radix commit, and L2-to-L1 scheduling |
| Storage query | `prefetch_thread_func()` | `prefetch_queue` | `prefetch_hit_queue` or `prefetch_revoke_queue` result |
| Storage transfer | `prefetch_io_aux_func()` | `prefetch_buffer` | KV bytes in L2 and updated `completed_tokens` |

`prefetch_hit_queue` and `prefetch_revoke_queue` are stored on `HiCacheController`. 
Their producer is the query worker, but their consumer is the Scheduler through `HiRadixCache.check_hicache_events()`; 
there is no dedicated thread blocked on either result queue. 
The Scheduler polls them with nonblocking reads. 
A revoke result releases the pending prefetch state. 
A hit result enters `_drain_and_alloc_storage_hit()`, where the Scheduler may allocate L2 pages and forward the operation to `prefetch_buffer`.

`prefetch_thread_func()` starts the I/O worker once, 
but the two workers then run as independent daemon threads. 
The query worker does not call the transfer worker for each operation. 
The Scheduler remains the only level that mutates the radix tree or allocates and evicts Host pages, 
and it bridges the workers only after assigning `operation.host_indices`. 
Diagram sources:
[`scheduler.py:1610-1619`](../../../python/sglang/srt/managers/scheduler.py#L1610),
[`scheduler.py:1766-1782`](../../../python/sglang/srt/managers/scheduler.py#L1766),
[`scheduler.py:2220`](../../../python/sglang/srt/managers/scheduler.py#L2220),
[`scheduler.py:2514-2550`](../../../python/sglang/srt/managers/scheduler.py#L2514),
[`scheduler.py:2835`](../../../python/sglang/srt/managers/scheduler.py#L2835),
[`scheduler.py:2979-3011`](../../../python/sglang/srt/managers/scheduler.py#L2979),
[`scheduler.py:3103-3137`](../../../python/sglang/srt/managers/scheduler.py#L3103),
[`hiradix_cache.py:1734-1773`](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1734),
[`cache_controller.py:360-383`](../../../python/sglang/srt/managers/cache_controller.py#L360),
[`cache_controller.py:496-510`](../../../python/sglang/srt/managers/cache_controller.py#L496), and
[`cache_controller.py:996-1088`](../../../python/sglang/srt/managers/cache_controller.py#L996).

The three address domains must remain distinct:

| Domain | Example | Meaning |
|---|---|---|
| Request mapping | `req_to_token[row, position]` | Logical token position to L1 slot |
| L1/L2 pool index | `device_indices` / `host_indices` | Location inside a local memory pool |
| L3 key | SHA256-based page-hash string | Distributed-storage identity for one logical KV page |

Each SGLang instance maintains its own RadixTree and derives external object keys deterministically from token pages. 
`HiRadixCache.insert()` calls `compute_node_hash_values()` to produce one prefix-dependent identity per page:

```text
h0 = H(page_tokens_0)
h1 = H(h0, page_tokens_1)
h2 = H(h1, page_tokens_2)
```

The preceding hash makes identical token pages under different prefixes distinct.
The local tree can recover the ordered hash chain without sharing radix nodes or pool indices with other SGLang instances. 
For ordinary non-split MHA, Mooncake then expands each logical page hash into rank-local K and V object keys:

```text
logical page hash:  h2
Mooncake K object:  [model/backend-tag_]h2_<tp-rank>[_<pp-rank>]_k
Mooncake V object:  [model/backend-tag_]h2_<tp-rank>[_<pp-rank>]_v
```

The optional prefix separates models or backend tags sharing one Store; 
the rank suffix selects the KV shard. 
Mooncake performs exact lookup on these keys.
Sources: [`hiradix_cache.py:1938-1953`](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1938),
[`utils.py:106-135`](../../../python/sglang/srt/mem_cache/utils.py#L106),
[`radix_cache.py:264-274`](../../../python/sglang/srt/mem_cache/radix_cache.py#L264), and
[`mooncake_store.py:964-997`](../../../python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L964).

At the Scheduler level, `event_loop_normal()` passes received objects to `process_input_requests()`. 
`_request_dispatcher` selects `handle_generate_request()`, and an accepted ordinary request reaches `_add_request_to_queue()`. 
The non-disaggregated path calls `_prefetch_kvcache()` before appending the request to `waiting_queue`. 
Storage prefetch is active only when `Scheduler.enable_hicache_storage` is true; 
at startup this means `server_args.hicache_storage_backend is not None`.

`_prefetch_kvcache()` first calls `req.init_next_round_input(self.tree_cache, cow_mamba=False)`.  This is the **first match**. 
It runs `tree_cache.match_prefix()` against local radix-tree metadata and reports:

```text
req.prefix_indices   = already-resident L1/GPU slots
req.host_hit_length  = matched L2/Host tokens requiring load-back
req.last_host_node   = local root/Host anchor for an optional L3 search
```

This first match covers local L1/L2 state; the following storage stage discovers fresh remote-only hits.
If an earlier request already fetched an L3 page and inserted it into this process's tree, 
that page is now ordinary L2 state and can appear in this local match.

The anchor condition:

```python
if tree_cache.is_backuped(req.last_host_node) or tree_cache.is_root(
    req.last_host_node
):
```

establishes a usable root or Host-backed anchor from which the chained external page hashes can be derived.
Suffix existence is evaluated by the subsequent Mooncake query.
The Scheduler then searches only the page-aligned tokens after the combined L1/L2 match and anchors the search with the last locally known page hash:

```text
local match:       [L1 prefix | L2 prefix]
L3 search range:                           [remaining page-aligned tokens]
anchor:            last hash of the local prefix
```

The Scheduler passes `new_input_tokens`, `last_hash`, and optional prefix keys to `tree_cache.prefetch_from_storage()`; 
see [`scheduler.py:2514-2537`](../../../python/sglang/srt/managers/scheduler.py#L2514).
`HiRadixCache.prefetch_from_storage()` page-aligns the candidate tokens, applies the prefetch threshold and rate limit, 
protects the local anchor node, and enqueues a `PrefetchOperation` without allocating Host slots yet. 
The enqueue and Scheduler-side registration occur in this order:

```python
operation = self.cache_controller.prefetch(...)
# Inside cache_controller.prefetch():
#   operation = PrefetchOperation(...)
#   self.prefetch_queue.put(operation)
#   return operation

self.ongoing_prefetch[req_id] = (
    last_host_node,
    prefetch_key,
    operation,
)
```

The queues in the diagrams carry work between execution levels; `ongoing_prefetch` is instead a Scheduler-owned lifecycle registry. 
Its entry keeps the protected radix anchor, requested token range, and shared operation available for commit, completion checks, or cleanup. 
Although `prefetch_queue.put(operation)` occurs immediately before this registration, the query worker never reads the registry or changes the tree. 
Its result waits in a control queue until a later Scheduler iteration. 
If the request has been removed by then, `_drain_and_alloc_storage_hit()` discards the stale result.

Sources:
[`cache_controller.py:900-914`](../../../python/sglang/srt/managers/cache_controller.py#L900),
[`cache_controller.py:1047-1085`](../../../python/sglang/srt/managers/cache_controller.py#L1047),
[`hiradix_cache.py:608-624`](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L608), and
[`hiradix_cache.py:1734-1773`](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1734).

The storage worker computes the candidate chained page hashes from `new_input_tokens` and `last_hash`, then asks Mooncake whether those pages exist:

```python
page_hashes = get_hash_str(tokens_to_fetch, last_hash, page_size=self.page_size)
hit_page_num = self.storage_backend.batch_exists(batch_hashes, extra_info)
```

Search stops at the first missing page, preserving longest-contiguous-prefix semantics. 
For MHA, `MooncakeStore.batch_exists()` requires both the rank-local K and V objects for each logical page. 
Source: [`cache_controller.py:1023-1045`](../../../python/sglang/srt/managers/cache_controller.py#L1023) and [`mooncake_store.py:1240-1267`](../../../python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L1240).

This is a control-plane query rather than a data read. 
Its result determines both whether prefetch is worthwhile and how much L2 memory is required. 
The hit length counts the consecutive prefix in tokens and is reduced to the minimum common value across participating attention ranks. 
If it is shorter than `prefetch_threshold`, the worker places the request ID in `prefetch_revoke_queue`:

```python
if storage_hit_count < self.prefetch_threshold:
    self.prefetch_revoke_queue.put(operation.request_id)
else:
    operation.storage_hit_count = storage_hit_count
    self.prefetch_hit_queue.put(operation)
```

Revocation cancels only this storage optimization. It does not abort the user request; 
the missing KV prefix is computed through ordinary prefill. 
Avoiding a short transfer also avoids paying storage submission and synchronization costs when recomputation is cheaper. 
The configured threshold is clamped to at least one storage page.

On the Mooncake side this is an exact control-plane lookup, conceptually:

```text
(tenant_id, physical object key)
    -> metadata shard
    -> ObjectMetadata
    -> readable COMPLETE replica descriptor
```

It is neither a similarity search nor a scan of KV payloads. 
Transfer Engine is not responsible for deciding whether the key exists; 
it moves the bytes only after the Store client has resolved a readable replica.

After the Scheduler receives a sufficiently long hit, it allocates exactly the required L2 Host slots. 
`_drain_and_alloc_storage_hit()` runs on the Scheduler thread when `check_hicache_events()` drains `prefetch_hit_queue`:

```python
alloc_len = operation.storage_hit_count
host_indices = cc.mem_pool_host.alloc(alloc_len)
if host_indices is None:
    self.evict_host(alloc_len)
    host_indices = cc.mem_pool_host.alloc(alloc_len)

operation.host_indices = host_indices
cc.prefetch_buffer.put(operation)  # transfer-scheduling handoff
```

The transfer boundary is therefore precise:

| Source event | Meaning | Does KV payload move? |
|---|---|---|
| `self.prefetch_hit_queue.put(operation)` | Query worker publishes a usable hit for later Scheduler polling | No |
| `cc.prefetch_buffer.put(operation)` | Scheduler has validated the request, allocated L2, and makes the I/O worker runnable | Not yet; this schedules the transfer stage |
| The `batch_get_v1()` call path reaches Mooncake `Client::BatchGet()` | Mooncake submits local-copy, TCP, or RDMA work into the supplied L2 buffers | Yes |

Thus `prefetch_hit_queue.put(operation)` has a necessary but indirect relation
to transfer. It does not wake `prefetch_io_aux_func()` and does not immediately
move data. The transfer can begin only on a later Scheduler iteration after
the hit is drained, the request is still valid, and L2 allocation succeeds.
If the request is stale or terminated, or allocation cannot provide an
acceptable prefix, the hit result is revoked and no transfer begins.

Allocation may fall back to a shorter page-aligned prefix under Host-memory
pressure. The drain function itself never moves KV bytes.

Source: [`cache_controller.py:1047-1085`](../../../python/sglang/srt/managers/cache_controller.py#L1047)
and [`hiradix_cache.py:608-653`](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L608).

The I/O worker consumes the allocated operation and begins the actual storage
read. `_page_transfer()` batches the hit prefix, and the selected
`_page_get_zero_copy()` boundary calls `MooncakeStore.batch_get_v1()` with the
page hashes and allocated Host indices.

`batch_get_v1()` expands the logical hashes into physical K/V keys and obtains the destination pointers and sizes from `MHATokenToKVPoolHost.get_page_buffer_meta()`. 
It then delegates to `_get_batch_zero_copy_impl()`:

```python
key_strs, buffer_ptrs, buffer_sizes = self._batch_preprocess(
    keys, host_indices
)
get_results = self._get_batch_zero_copy_impl(
    key_strs, buffer_ptrs, buffer_sizes
)
```

The helper selects the Mooncake API from the pointer shape:

```python
if self._uses_multi_buffer(buffer_ptrs):
    return self.store.batch_get_into_multi_buffers(
        key_strs, buffer_ptrs, buffer_sizes
    )
return self.store.batch_get_into(key_strs, buffer_ptrs, buffer_sizes)
```

The ordinary page-first MHA/GQA path produces one flat pointer per physical K or V object and therefore uses `batch_get_into()`. 
A nested pointer list means that one object is scattered into several non-contiguous destinations:

```text
key_strs[i]
    -> buffer_ptrs[i]  = [pointer_i0, pointer_i1, ...]
    -> buffer_sizes[i] = [size_i0,    size_i1,    ...]
```

This shape selects `batch_get_into_multi_buffers()`. 
The buffers are consecutive destination pieces of the same object; 
the storage lookup and transport model remain the same.

`self.store` is the pybind `MooncakeDistributedStore`. SGLang calls its public
Python methods; the binding converts integer addresses to `void*`, releases
the GIL, and dispatches to `RealClient`. The responsibility changes are:

```text
SGLang batch_get_v1(keys, host_indices)
    -> MooncakeDistributedStore flat or multi-buffer binding
    -> RealClient resolves a readable replica with BatchQuery
    -> RealClient builds slices over the supplied L2 destinations
    -> BatchGet writes through local copy, TCP, or RDMA into those destinations
```

The one-argument `Client::BatchQuery(keys)` is an overload forwarding to the
tenant-aware overload, not recursion. Flat destinations become contiguous
slices; multi-buffer destinations become several slices for one object. The
default path submits each object independently, while the optional same-node
preference may select the batch submitter.

`execute_timed_operation()` around the public multi-buffer method is only a
synchronous instrumentation wrapper. It times the internal lookup and transfer,
records positive byte counts, and returns the original per-key successes or
errors. It does not create a worker or transport data itself.

Sources:
[`mooncake_store.py:1030-1057`](../../../python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L1030),
[`mooncake_store.py:1306-1313`](../../../python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L1306),
[`store_py.cpp:2743`](</home/xffan2/code/Mooncake/mooncake-integration/store/store_py.cpp:2743>),
[`store_py.cpp:2911`](</home/xffan2/code/Mooncake/mooncake-integration/store/store_py.cpp:2911>),
[`real_client.cpp:4328`](</home/xffan2/code/Mooncake/mooncake-store/src/real_client.cpp:4328>),
[`real_client.cpp:4574`](</home/xffan2/code/Mooncake/mooncake-store/src/real_client.cpp:4574>),
[`real_client.cpp:5007`](</home/xffan2/code/Mooncake/mooncake-store/src/real_client.cpp:5007>),
[`real_client.cpp:5040`](</home/xffan2/code/Mooncake/mooncake-store/src/real_client.cpp:5040>),
[`client_service.cpp:1068`](</home/xffan2/code/Mooncake/mooncake-store/src/client_service.cpp:1068>), and
[`client_service.cpp:1339`](</home/xffan2/code/Mooncake/mooncake-store/src/client_service.cpp:1339>).

Zero-copy here means that Mooncake writes into the registered SGLang L2 pages instead of returning a separate payload buffer. 
`Client::BatchGet()` waits for the transfer futures, 
and `_page_get_zero_copy()` then advances the shared operation's `completed_tokens`.

Storage prefetch begins at request admission, before the request enters
`waiting_queue`:

```text
_add_request_to_queue(req)
    -> _prefetch_kvcache(req)
         -> first init_next_round_input(): observe existing L1/L2
         -> prefetch_from_storage(): enqueue the remaining L3 candidate
    -> waiting_queue.append(req)
```

At the beginning of a later `_get_new_batch_prefill_raw()` pass, `check_hicache_events()` drains synchronized storage-control results. 
In particular, it turns a `prefetch_hit_queue` result into an allocated L2 destination and a `prefetch_buffer` transfer task, or consumes a revoke result.
The per-request waiting-queue loop then applies the scheduling gate:

```python
prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)
if not prefetch_done:
    # Keep the request in waiting_queue while its L3 operation progresses.
    continue

loaded_tokens = self.tree_cache.pop_prefetch_loaded_tokens(req.rid)
req.init_next_round_input(self.tree_cache)
```

`check_prefetch_progress()` delegates the wait/stop decision to
`can_terminate_prefetch()`. Its result has two related but different meanings:

```text
can_terminate_prefetch(operation) == False
    -> the configured policy says to keep waiting
    -> check_prefetch_progress() returns False
    -> Scheduler skips this request for the current pass

can_terminate_prefetch(operation) == True
    -> finalize a completed operation, or stop an unfinished one
    -> publish only the completed page-aligned prefix, if any
    -> check_prefetch_progress() returns True
    -> Scheduler may run the second local prefix match and schedule prefill
```

The strategy is selected by `--hicache-storage-prefetch-policy`; the default is `timeout`:

| Policy | Waiting-queue behavior | Result used when the gate opens |
|---|---|---|
| `best_effort` | Never deliberately waits for L3. The first Scheduler check may stop work that has not finished. | The completed prefix already present in allocated L2 pages; otherwise the prefetch is revoked. |
| `wait_complete` | Returns `False` until every queried page has completed, so the request remains in `waiting_queue` while the background I/O continues. | The complete synchronized prefix. |
| `timeout` | Waits for completion until the operation-specific deadline, then stops the remaining work. | The completed synchronized prefix at completion or timeout. |

The timeout includes query and queueing time because `operation.start_time` is
set when the operation is created. It grows linearly with the known hit length:

```python
timeout = min(
    cfg.max,
    cfg.base + cfg.per_ki_token * num_tokens / 1024,
)
```

The defaults are a 2-second base, 0.1 seconds per 1024 tokens, and a 30-second maximum. 
Before the storage query has populated `operation.hash_value`, the base timeout applies. 
For `wait_complete` and `timeout`, attention ranks reduce the local completion/termination state so they agree on whether the gate can open.

When the gate opens before Host allocation, `check_prefetch_progress()` marks the operation terminated and removes its `ongoing_prefetch` entry. 
When Host allocation has occurred, it terminates or collects the operation, 
clamps to the cross-rank completed prefix, 
inserts that prefix as Host-backed radix nodes, and releases unused slots. 
The following `init_next_round_input()` is therefore the **second match** and can expose the newly published L2 pages through `host_hit_length`.

The storage-hit threshold is an earlier, separate policy. 
A query result shorter than `prefetch_threshold` enters `prefetch_revoke_queue`; 
after that result is drained, absence from `ongoing_prefetch` lets the request proceed without claiming an L3 hit. 
Similarly, stopping storage prefetch does not abort the user request: 
uncached tokens are recomputed by ordinary prefill. 
An actual request abort follows `release_aborted_request()` and additionally terminates and releases any in-flight prefetch resources.

After a completed hit, the standard path described in the preceding section allocates L1 slots and performs the layer-wise L2-to-L1 copy. 
Mooncake's read responsibility ends when the requested bytes are ready in L2; 
local radix publication, L1 loading, and model consumption remain SGLang responsibilities.
Sources:
[`scheduler.py:2514-2547`](../../../python/sglang/srt/managers/scheduler.py#L2514),
[`scheduler.py:3003-3011`](../../../python/sglang/srt/managers/scheduler.py#L3003),
[`scheduler.py:3123-3137`](../../../python/sglang/srt/managers/scheduler.py#L3123),
[`hiradix_cache.py:1481-1507`](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1481),
[`hiradix_cache.py:1540-1649`](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L1540),
[`hicache_storage.py:50-56`](../../../python/sglang/srt/mem_cache/hicache_storage.py#L50), and
[`server_args.py:2621-2629`](../../../python/sglang/srt/server_args.py#L2621).

The complementary write path starts only after L1 KV has been copied to L2.
`write_backup_storage()` passes the node's page hashes and Host slots to `HiCacheController.write_storage()`, 
which queues the Mooncake write. 
Sources:
[`hiradix_cache.py:911-935`](../../../python/sglang/srt/mem_cache/hiradix_cache.py#L911)
and [`cache_controller.py:1090-1117`](../../../python/sglang/srt/managers/cache_controller.py#L1090).

To enable this adapter at startup:

```bash
python -m sglang.launch_server \
    --model-path MODEL_PATH \
    --enable-hierarchical-cache \
    --hicache-storage-backend mooncake \
    --hicache-storage-backend-extra-config '{"master_server_address":"127.0.0.1:50051","local_hostname":"localhost","metadata_server":"http://127.0.0.1:8080/metadata","global_segment_size":"4gb","protocol":"rdma","device_name":""}'
```

`HiCacheController.attach_storage_backend()` uses `StorageBackendFactory` to
construct `MooncakeStore`, registers `MHATokenToKVPoolHost`, selects the
zero-copy page functions, and starts the background query/I/O queues. Source:
[`cache_controller.py:431-504`](../../../python/sglang/srt/managers/cache_controller.py#L431)
and [`backend_factory.py:196-211`](../../../python/sglang/srt/mem_cache/storage/backend_factory.py#L196).
Mooncake's other configuration mechanisms are documented in
[`storage/mooncake_store/README.md:185-240`](../../../python/sglang/srt/mem_cache/storage/mooncake_store/README.md#L185).
The complementary Mooncake-side explanation of object metadata, replica lookup,
and Transfer Engine responsibility is in
[`Mooncake study-notes.md:1358`](</home/xffan2/code/Mooncake/.claude/notes/guide/study-notes.md:1358>).

### Logical External Tier and Physical L4 Pipelines

The name “Mooncake L3” in the preceding discussion denotes SGLang's logical
external-storage boundary. Mooncake's physical replica hierarchy is described
separately by medium:

```text
SGLang-visible hierarchy

L1  GPU KV pool
 ^
 | explicit device_indices <-> host_indices load/backup
 v
L2  Host KV pool
 ^
 | HiCacheStorage: keys + host_indices
 v
external storage boundary
    +-- Mooncake MEMORY replica       logical L3 / shared DRAM
    +-- Mooncake LOCAL_DISK replica   logical L4 / local SSD
    +-- other file or block replicas  logical L4 when integrated
```

Current `HiCacheStorage` presents one external destination type.
Its zero-copy read interface is `batch_get_v1(keys, host_indices)`, and its v2
interface describes Host-to-storage work with `PoolTransfer.keys` and
`PoolTransfer.host_indices`. The same `PoolTransfer` describes local
device-to-Host work only when `device_indices` is also present. There is no
current storage operation shaped as `keys + device_indices`; that shape belongs
to a prospective direct-destination extension.
See
[`hicache_storage.py:68-80`](../../../python/sglang/srt/mem_cache/hicache_storage.py#L68)
and
[`hicache_storage.py:169-179`](../../../python/sglang/srt/mem_cache/hicache_storage.py#L169).

Mooncake's adapter follows that boundary literally. `batch_get_v1()` resolves
the supplied `host_indices` through
`mem_pool_host.get_page_buffer_meta()`, producing the pointer and size list
passed to Mooncake's zero-copy get API. The resulting destination is an L2 Host
page; a direct path would additionally carry SGLang L1 `device_indices`. See
[`mooncake_store.py:964-1022`](../../../python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L964)
and
[`mooncake_store.py:1030-1057`](../../../python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L1030).

Consequently, L3 and L4 share SGLang's logical storage semantics while retaining
medium-specific execution properties:

| Property | Relationship between L3 MEMORY and L4 storage |
|---|---|
| Logical page/object key | Shared identity rule |
| Longest-contiguous-prefix rule | Shared correctness rule |
| `batch_exists` / `batch_get` interface | Shared backend contract |
| Local read destination | Shared L2 Host destination |
| Local publication | Shared Host-backed radix-node state |
| Physical I/O path | Medium-specific pipeline |
| Queueing and service time | Medium-specific cost model |
| Replica placement, promotion, and eviction | Mooncake medium policy |
| Failure mode and resource consumption | Medium-specific operational behavior |

The current result primarily reports object/page existence and per-page I/O
success. A future tier-aware result can add `source_tier`, expected service time,
and direct-to-device capability so the Scheduler can compare L3 wait, L4 wait,
direct restore, and recomputation. Mooncake continues to choose and manage its
replicas behind the common object identity.

This is why SGLang's Mooncake documentation describes SSD offload as expanding
the L3 cache capacity: the extra physical level is transparent to the current
HiCache storage abstraction. The configuration is documented in
[`storage/mooncake_store/README.md:263-303`](../../../python/sglang/srt/mem_cache/storage/mooncake_store/README.md#L263).

The current support matrix separates transport capability from the destination
contract integrated by SGLang:

| Source and destination | Mooncake transport primitive | Current SGLang HiCache path |
|---|---|---|
| L3 MEMORY -> L2 Host | RDMA/TCP into registered memory | Supported |
| L3 MEMORY -> L1 GPU | GPUDirect RDMA is available in TE/TENT | Prospective direct-destination integration |
| L4 LOCAL_DISK -> L2 Host | Holder-side file read followed by TE/TENT | Supported by Mooncake SSD offload |
| L4 locally visible file -> L2 Host | TENT `io_uring` into the final Host pointer when the file extent and destination meet direct-I/O constraints | Prospective Mooncake file-segment integration |
| L4 shared file/DFS -> L2 Host | File/DFS-specific read path | Backend- and Mooncake-version-specific integration |
| L4 file -> L1 GPU | TENT GDS, or classic TE's narrower NVMe-oF/cuFile path, provides a primitive | Prospective file-replica and GPU-destination integration |

For an L3 MEMORY hit, the present end-to-end flow is:

```text
Mooncake MEMORY replica
    -> RDMA/TCP into requester MHATokenToKVPoolHost page
    -> publish host_value in the local radix tree
    -> init_load_back() reserves L1 slots
    -> layer-wise L2-to-L1 copy
    -> publish/use device_indices
```

For Mooncake `LOCAL_DISK` SSD offload, L4-to-L2 is already supported, but the
logical one-hop arrow contains physical staging:

```text
logical view:
    L4 LOCAL_DISK -----------------------------> requester L2

physical view:
    holder SSD
      -> holder ClientBuffer
      -> TE/TENT RDMA or TCP
      -> requester L2 Host page
```

The holder returns buffer pointers and a lease-like batch lifetime; the
requester pulls those bytes into the application destination and releases the
holder buffer after completion. Mooncake documents this path in
[`ssd-offload.md:105-128`](</home/xffan2/code/Mooncake/docs/source/design/ssd-offload.md:105>).
The temporary holder `ClientBuffer` is an operation-scoped physical staging
buffer. A published logical L3 MEMORY replica has a separate allocation and
publication lifecycle, and optional `LOCAL_DISK-only -> MEMORY` promotion runs
as an independent Mooncake action.

The current SGLang code therefore supports this common shape:

```text
L3 or supported L4 replica
    -> L2 Host page
    -> L1 GPU page
```

The corresponding direct-destination integrations are prospective paths:

```text
L3 MEMORY -> L1 GPU via direct RDMA
L4 file   -> L1 GPU via direct GDS
```

For restore-time analysis, the staged L4-to-L2-to-L1 path can be decomposed as:

```text
T_restore = T_query
          + T_l4_queue
          + T_storage_read
          + T_provider_staging
          + T_network
          + T_publish
          + T_h2d_visible
```

`T_publish` is SGLang's local cache-publication interval. Its measurable
boundaries are:

```text
t_data_ready
    = the selected contiguous prefix has completed its external read
      into reserved L2 Host pages

t_prefix_usable
    = the Scheduler's second local match_prefix() has observed those pages
      as host_value and returned them through host_hit_length

T_publish = t_prefix_usable - t_data_ready
```

The interval converts physically resident bytes into a Scheduler-visible L2 cache hit. 
In the current pipeline it comprises completion collection, prefetch-policy evaluation, 
the next Scheduler progress check, cross-rank clamping to a common contiguous prefix, 
`_insert_helper_host()` radix commit, and the second `init_next_round_input()`/`match_prefix()` observation:

```text
external I/O completion
    -> operation.completed_tokens becomes observable
    -> check_prefetch_progress() applies wait_complete/timeout/best_effort
    -> terminate_prefetch() collects the completed extent
    -> attention ranks agree on the usable contiguous token count
    -> _insert_helper_host() installs host_value and page hashes
    -> ongoing_prefetch entry and Host-slot accounting are finalized
    -> second match_prefix() returns the new L2 hit
```

This publication refers to the requester-local HiRadixCache state. 
Mooncake replica publication, such as committing a `MEMORY` replica or recording a `LOCAL_DISK` offload, 
is a separate Store control-plane lifecycle. 
`T_network` ends when the external transfer has delivered the selected bytes;
`T_publish` ends when the Scheduler can plan with them; 
`T_h2d_visible` covers the layer-wise L2-to-L1 work that remains visible on the request's critical path.

For a direct L4-to-L2 `io_uring` read, `t_data_ready` corresponds to successful CQE processing for all required K/V slices. 
For a direct L3/L4-to-L1 path, publication instead commits `device_indices` after transfer and device-stream visibility, 
and the staged-path `T_h2d_visible` term becomes zero. 
Page- or prefix-granular completion can reduce publication latency by committing the largest cross-rank contiguous prefix available at the policy boundary.

### TE and TENT within the SGLang Storage Path

Classic TE and TENT differ primarily in runtime planning rather than fundamental GPU-direct capability. 
Classic Mooncake Transfer Engine already models DRAM and VRAM as RAM Segments, registers GPU buffers for GPUDirect RDMA, and includes an NVMe-oF/cuFile transport capable of reading into DRAM or VRAM. 
Its documented matrix includes DRAM-to-VRAM and NVMe-oF-to-VRAM transfers. 
See [`transfer-engine/index.md:14-49`](</home/xffan2/code/Mooncake/docs/source/design/transfer-engine/index.md:14>).

TENT, Transfer Engine NEXT, changes the runtime model more than the fundamental DMA possibility:

| Concern | Classic TE | TENT |
|---|---|---|
| Application intent | Submit work to installed/configured transport machinery | Describe what to move; runtime resolves how |
| Segment model | Primarily RAM Segments plus a specialized NVMe-oF Segment | Uniform `MemorySegment` and `FileSegment` variants |
| Transport choice | More backend/deployment-specific and comparatively static | Per-request policy and dynamic transport selection |
| Heterogeneous paths | Supported by individual transports and topology logic | Unified selection across RDMA, NVLink, SHM, TCP, GDS, io_uring, and others |
| Multi-path scheduling | Backend-specific slicing/rail behavior | Runtime slice scheduling using queue/load telemetry |
| Fallback | Backend-specific | Runtime can select another transport or construct Host staging |
| Failure handling | Existing retry and endpoint failover in individual transports | Centralized per-attempt/per-slice failover and recovery |
| Scheduling metadata | Basic transfer request | Priority, intent, policy, hint, deadline, and cancellation state |

TENT's design goal is summarized in [`tent/overview.md:21-59`](</home/xffan2/code/Mooncake/docs/source/design/tent/overview.md:21>).
Its transport selector can apply different policies to memory and file segments and can prefer GDS with io_uring or another transport as fallback; 
see [`tent/transport-selector.md:25-81`](</home/xffan2/code/Mooncake/docs/source/design/tent/transport-selector.md:25>).

A TENT file segment is a transfer-plane description of an existing storage
format. Mooncake retains its object serialization, file contents, object index,
and replica lifecycle. The additional description makes an existing file
extent addressable to the transfer runtime:

```text
SegmentDesc
  type = Memory
    -> MemorySegmentDesc
         -> BufferDesc {virtual address, length, registration, topology}

  type = File
    -> FileSegmentDesc.buffers[]
         -> FileBufferDesc {path, length, offset}
```

The two variants create a common planning interface while preserving their
different address domains. A memory segment identifies bytes that are already
resident in DRAM or device memory. A file segment identifies persistent bytes
through a path and file range, allowing TENT to place file I/O inside the same
request-selection and completion framework as memory transfer. `FileBufferDesc`
can describe a local NVMe file or a mounted filesystem path; the storage medium
and deployment determine what the path represents.

This changes the transfer boundary relative to the ordinary Mooncake
`LOCAL_DISK` pipeline:

| Concern | Holder-staged `LOCAL_DISK` | TENT file-segment path |
|---|---|---|
| Object resolution | Key resolves to a holder and storage-backend entry | Key also resolves to an openable file segment and object extent |
| Storage read destination | Holder-owned aligned `ClientBuffer` | SGLang's final L2 pointer can be the `io_uring` destination |
| Transfer-engine input | Registered staging-memory pointer | File segment ID, file offset, local pointer, and length |
| Data movement | SSD -> holder Host -> network -> requester L2 | Visible file -> requester L2, subject to file visibility and I/O constraints |
| Persistent representation | Mooncake's existing storage layout | The same storage layout; the segment is descriptive metadata |

The direct file-to-L2 request is naturally expressed after the Scheduler has
reserved the destination Host pages:

```text
Scheduler reserves host_indices in the SGLang L2 pool
    -> HostKVCache resolves the final page pointers and slice lengths
    -> Mooncake resolves object keys to file segment + file offsets
    -> TENT READ requests use those L2 pointers as local destinations
    -> io_uring completes the reads
    -> storage completion publishes host_value in HiRadixCache
```

In TENT's current `Request` terminology, `source` is the local buffer pointer.
For a `READ` from a file segment, that local pointer is the destination:

```text
Request {
  opcode        = READ,
  source        = sglang_l2_pointer,
  target_id     = file_segment_id,
  target_offset = object_file_offset,
  length        = object_length
}
```

TENT's current `IOUringTransport` initializes `O_DIRECT` when the filesystem
supports it, with ordinary buffered file I/O as its compatibility mode. A CPU
pointer aligned to 4 KiB is submitted directly to `io_uring`. CUDA pointers and
CPU pointers with other alignments use a transport-private aligned buffer;
after a read completes, TENT copies from that buffer to the supplied pointer.
With `O_DIRECT`, the file offset and length also satisfy the filesystem's
alignment requirements. Consequently, an immediate L4-to-L2 path depends on
the Host pool base address, per-page stride, object extent, and request length
being compatible with direct I/O.
SGLang's page-first layouts provide stable per-page destinations, while the
actual base-and-stride alignment remains an allocator property that the adapter
must validate.

The destination is represented as several independently contiguous slices.
The adapter may submit one request for each K/V, layer, or page slice while
preserving SGLang's segmented KV layout.
HiCache publishes the page only after all required slices complete, so direct
placement removes an intermediate Host copy while retaining the Scheduler's
reservation and publication boundary.

File visibility determines where this plan can execute. TENT's file transports
open the path in the process executing the request. A same-node local SSD or a
filesystem mounted on that node can therefore supply a direct file-to-L2 path.
A `LOCAL_DISK` replica scoped to another Mooncake holder follows holder-side
I/O plus network transfer. An owner-side composed plan or a requester-visible
file mapping enables the corresponding file-segment plan.

In a cluster, `io_uring` remains a local Linux submission and completion
mechanism. Its ring, file descriptor, and destination virtual address belong to
the executing process and kernel instance. Cluster reachability comes from the
object represented by that local file descriptor: a local filesystem, a DFS
mount, or an NVMe-oF-backed filesystem. A TENT executor running in the SGLang
process can use the L2 pointer directly; a separate executor process uses a
shared mapping that gives the destination a valid address in its own address
space.

A DFS can expose one path and one file-offset namespace while selecting several
physical routes beneath the same `io_uring` request:

```text
TENT FileSegment READ(path, offset, L2 pointer)
    -> local io_uring / VFS / DFS client
         +-- page-cache hit
         |     -> CPU memory copy -> SGLang L2
         +-- requester-local SSD or DFS client-cache hit
         |     -> PCIe/NVMe DMA -> SGLang L2
         +-- remote DFS extent
         |     -> storage-node SSD DMA
         |     -> DFS TCP/RDMA/custom network
         |     -> requester DFS client placement -> SGLang L2
         +-- requester-mounted NVMe-oF extent
               -> NVMe/TCP or NVMe/RDMA
               -> requester block/file layer -> SGLang L2
```

The logical operation and completion boundary remain uniform: the `io_uring`
CQE reports completion after the DFS has placed the requested bytes in the
local destination. The physical transport reflects the selected data location:

| Physical source | Typical transport into the SGLang worker |
|---|---|
| Kernel page cache | CPU memory copy |
| Local SSD or local DFS cache | PCIe/NVMe DMA |
| Remote DFS over TCP | Storage DMA followed by TCP and DFS-client placement |
| Remote DFS over RDMA | Storage DMA followed by network RDMA and DFS-client placement |
| NVMe-oF/TCP | NVMe/TCP fabric |
| NVMe-oF/RDMA | NVMe/RDMA fabric |

DMA and RDMA describe different links. DMA covers a local device-to-memory
transfer, such as NVMe over PCIe. RDMA covers a network memory transfer between
machines. A remote DFS/RDMA read commonly combines both: storage-node NVMe DMA
first brings the extent from SSD, and network RDMA then carries it toward the
requester. TCP-based DFS and NVMe-oF deployments use their corresponding TCP
data paths.

TENT's current selector marks a file-segment request as `same_machine = true`.
This describes local execution of the file descriptor and local ownership of
the destination pointer. The mounted DFS or NVMe-oF client may still reach
remote storage underneath that descriptor. Consequently, “direct L4-to-L2”
describes placement into the final SGLang L2 allocation; a remote extent also
includes its DFS or storage-fabric network stage.

End-to-end copy behavior is a DFS property. An `O_DIRECT` local extent can use
storage DMA into an aligned L2 buffer. A remote DFS with registered-buffer
support can place network data into that destination through RDMA. A DFS client
with internal receive buffers performs its client-buffer placement before the
CQE. For tier-aware scheduling, the DFS or Mooncake adapter can expose
`source_location`, `transport`, `direct_to_host`, and estimated service time so
the Scheduler can distinguish local-cache and remote-extent cost behind the
same file key.

The current SGLang adapter already resolves `host_indices` into the final L2
pointers and passes them to Mooncake's `batch_get_into` interface. The
integration work centers on Mooncake's replica-to-file resolution, TENT request
construction, and validation of direct-I/O alignment. Scheduler-side
work continues to cover L2 reservation, asynchronous completion, cancellation,
and publication. See
[`mooncake_store.py:964-1057`](../../../python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L964),
[`segment.h:89-152`](</home/xffan2/code/Mooncake/mooncake-transfer-engine/tent/include/tent/runtime/segment.h:89>), and
[`io_uring_transport.cpp:186-270`](</home/xffan2/code/Mooncake/mooncake-transfer-engine/tent/src/transport/io_uring/io_uring_transport.cpp:186>).
The local file-selection context is constructed in
[`transfer_engine_impl.cpp:1045-1052`](</home/xffan2/code/Mooncake/mooncake-transfer-engine/tent/src/runtime/transfer_engine_impl.cpp:1045>).

The difference can be expressed as:

```text
classic TE emphasis:
    configured segment + configured transport -> byte transfer

TENT emphasis:
    source segment + destination memory + intent
        -> inspect memory/file types and topology
        -> select a transport plan
        -> direct transfer or internal staging
        -> retry/fail over and report completion
```

Classic TE can therefore provide an L3-to-L1 GPUDirect RDMA primitive if the application registers and supplies the L1 GPU destination. 
TENT can provide the same primitive with a more general selection, scheduling, and fallback model.
The actual destination follows the pointer and lifecycle supplied by SGLang; the current adapter supplies an L2 Host pointer.

Likewise, TENT makes the L4 design more general because it explicitly models a file location through a `FileSegmentDesc` containing `FileBufferDesc {path, length, offset}` entries
and has a GDS transport that resolves the file segment, 
registers the file with cuFile, and uses the request's local buffer as the I/O destination. 
See [`segment.h:89-152`](</home/xffan2/code/Mooncake/mooncake-transfer-engine/tent/include/tent/runtime/segment.h:89>) and [`gds_transport.cpp:233-285`](</home/xffan2/code/Mooncake/mooncake-transfer-engine/tent/src/transport/gds/gds_transport.cpp:233>).
Classic TE's NVMe-oF support already establishes file/storage-to-GPU DMA as an available primitive. 
TENT contributes a common file-segment and transport-selection framework across multiple storage transports.

There is already a compatibility wrapper: when TENT is enabled, the classic `TransferEngine::submitTransfer()` wrapper translates the existing opcode,
local pointer, remote segment, offset, length, and transport hint into a TENT request. 
See [`transfer_engine.cpp:613-635`](</home/xffan2/code/Mooncake/mooncake-transfer-engine/src/transfer_engine.cpp:613>).
This allows existing memory-transfer clients to adopt TENT while preserving Mooncake Store's object model. 
Richer TENT fields and the SGLang direct-restore state machine are separate adapter extensions.

Mooncake Store and TENT describe different phases and therefore use separate data structures:

```text
Mooncake Store control plane
    ObjectMetadata
      -> Replica::Descriptor
           identity, type/medium, status, owner/location, lifetime
                    |
                    | resolve/adapt for one I/O attempt
                    v
TENT data plane
    SegmentDesc
      -> MemorySegmentDesc or FileSegmentDesc
           addresses/path/offset, topology, available transports
                    |
                    v
    Request + attempt/completion state
```

Existing MEMORY transfers can enable TENT while retaining `Replica`.
The Store still decides which complete replica represents an object; 
TENT decides how to move bytes between the resolved source and target. 
The current Mooncake checkout's replica variant contains MEMORY, NOF, DISK, and LOCAL_DISK descriptors; 
see [`replica.h:167-207`](</home/xffan2/code/Mooncake/mooncake-store/include/replica.h:167>) and [`replica.h:471-477`](</home/xffan2/code/Mooncake/mooncake-store/include/replica.h:471>).

A Store descriptor evolves when a new medium requires additional resolution fields. 
Examples:

- A MEMORY replica already contains the registered-buffer location needed to
  resolve a memory segment, so the existing replica type maps directly.
- A simple file-per-object DISK descriptor can map `file_path + object_size`
  to a file segment with offset zero through an adapter.
- A LOCAL_DISK descriptor identifies a holder client and endpoint. 
  Requester-side GDS additionally requires a requester-visible, openable file; 
  holder-assisted staging remains the path for endpoint-only metadata.
- A shared-extent DFS design needs at least a stable path or shard, offset, logical length, and often alignment/version/lifetime information. 
  These fields can be represented by a new or extended Store descriptor before an  adapter constructs a TENT `FileSegment`.

Thus existing MEMORY replicas retain their structure under TENT, 
while a new persistent medium may introduce richer Store-to-medium metadata according to its addressing and lifetime requirements.

The current local Mooncake checkout defines MEMORY, NOF, DISK, and LOCAL_DISK variants in `Replica::Descriptor`. 
A `DistributedFSDescriptor` therefore belongs to a different source revision or a proposed design and should carry that version context explicitly.

### Direct-to-L1 Lifecycle and Policy

An end-to-end L3/L4-to-L1 cache hit combines GPUDirect RDMA or GDS with four
coordinated ownership domains:

```text
HiCache       owns token prefix and usable-hit publication
L1 allocator  owns GPU KV page reservation and reclamation
Mooncake      owns object/replica selection and source lifetime
TE/TENT       owns one byte-transfer attempt and its completion
```

For L3 MEMORY-to-L1, SGLang would need to:

1. Finish the continuous-page lookup and synchronize the usable hit across
   required TP/PP attention shards.
2. Admit the request or otherwise establish a safe lifetime for the target.
3. Reserve the exact L1 GPU pages before submitting I/O.
4. Expose layout-correct GPU pointers or scatter descriptors to the storage
   backend and register them for the selected transport.
5. Submit Mooncake MEMORY-to-GPU transfers through GPUDirect RDMA when the
   topology supports them.
6. Connect transfer completion to CUDA stream visibility and atomically
   publish `device_indices` only after all required K/V components are ready.
7. On cancellation, timeout, partial completion, or rank failure, prevent
   stale KV publication, release the reserved pages, and fall back to staged
   restore or recomputation.

For L4 file-to-L1, the same target lifecycle is required plus:

1. Convert the selected persistent replica and its lease/version into a TENT
   file segment.
2. Verify cuFile visibility, filesystem support, alignment, GPU-buffer pinning,
   and topology.
3. Submit and validate GDS completion, including exact byte count and any
   object integrity/version checks.
4. Fall back to file-to-Host L2 staging or recomputation when direct GDS is not
   available or is slower than the remaining prefill work.

Direct and staged restore should coexist rather than replace one another:

| Situation | Preferred path |
|---|---|
| Request admitted, L1 capacity fixed, long valuable prefix, direct topology healthy | L3/L4 -> L1 direct |
| Request still cancellable or consumption time uncertain | Prefetch into L2 |
| Multiple requests likely to reuse the same cold prefix | L4 -> L2 and optionally promote/retain a shared copy |
| L1 under pressure | Stage in L2 or recompute |
| Direct transport unavailable or completion fails | Staged fallback or recompute |

The essential conclusion is:

> TE/TENT supplies transport capabilities. Mooncake supplies object and
> replica semantics. Current SGLang HiCache supplies an L2 Host destination at
> its external-storage boundary. Direct L3/L4-to-L1 support adds an SGLang GPU
> destination, reservation, completion, publication, and rollback contract;
> TENT can execute the transfer plan described by that contract.
