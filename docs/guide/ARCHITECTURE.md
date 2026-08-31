# SGLang Architecture & Developer Guide

SGLang is a high-performance inference engine for large language models and
multimodal models. It serves trillions of tokens daily across 400,000+ GPUs
worldwide, powering everything from real-time chatbots to reinforcement-learning
rollouts for frontier model training.

This document is an **onboarding guide for engineers** who will read, modify, or
extend the SGLang codebase. It explains the big picture, walks through a request
from end to end, and provides a map of the source tree. For flag-level or
API-level details, consult [docs.sglang.io](https://docs.sglang.io).

---

## Table of Contents

- [1. Quick Start](#1-quick-start)
- [2. Architectural Overview](#2-architectural-overview)
- [3. Request Lifecycle](#3-request-lifecycle)
- [4. Key Concepts](#4-key-concepts)
- [5. Source Tree Map](#5-source-tree-map)
- [6. Coding Conventions](#6-coding-conventions)
- [7. Development Workflows](#7-development-workflows)
- [8. Further Reading](#8-further-reading)

---

## 1. Quick Start

### Install

```bash
pip install sglang
```

### Serve a model

```bash
# OpenAI-compatible HTTP API on port 30000
sglang serve --model-path meta-llama/Llama-3.1-8B-Instruct
```

```bash
# With tensor parallelism across 2 GPUs
sglang serve --model-path deepseek-ai/DeepSeek-V3 --tp 2
```

### Send a request

```bash
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "prompt": "The capital of France is", "max_tokens": 32}'
```

### Run tests

```bash
# Run a single test file (each file runs in its own process)
python -m pytest test/registered/unit/test_scheduler.py -xvs

# Run all unit tests
python -m pytest test/registered/unit/ -x
```

---

## 2. Architectural Overview

### 2.1 Process Model

SGLang uses a **three-process architecture** with ZMQ for inter-process
communication:

```
┌──────────────────────────────────────────────────┐
│                    MAIN PROCESS                   │
│                                                  │
│  ┌──────────┐   ┌──────────────┐   ┌─────────┐  │
│  │   HTTP   │   │   Engine     │   │  Token- │  │
│  │  Server  │──▶│  (orchestr-  │──▶│  izer   │  │
│  │ (FastAPI)│   │   ator)      │   │ Manager │  │
│  └──────────┘   └──────────────┘   └────┬────┘  │
│                                         │        │
└─────────────────────────────────────────┼────────┘
                                          │ ZMQ
┌─────────────────────────────────────────┼────────┐
│                    SUBPROCESS            │        │
│  ┌──────────────────────────────────────▼─────┐  │
│  │               Scheduler                    │  │
│  │  ┌──────────┐  ┌──────────┐  ┌─────────┐  │  │
│  │  │ Request  │  │ Schedule │  │ TpWorker │  │  │
│  │  │ Receiver │──▶│  Policy  │──▶│  (model  │  │  │
│  │  │          │  │          │  │  forward)│  │  │
│  │  └──────────┘  └──────────┘  └─────────┘  │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
          │ ZMQ
┌─────────▼────────────────────────────────────────┐
│                    SUBPROCESS                     │
│  ┌─────────────────────────────────────────────┐ │
│  │          DetokenizerManager                  │ │
│  │   (converts token IDs back into text)        │ │
│  └─────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
```

**Why three processes?**

1. **TokenizerManager** stays in the main process so the HTTP server never
   blocks on GPU work.
2. **Scheduler** runs in a dedicated subprocess — it owns the GPU, builds
   batches, and runs model forwards.
3. **DetokenizerManager** is a separate subprocess so detokenization (which can
   be CPU-heavy with complex grammars) doesn't stall the scheduler.

### 2.2 Key Source Files for the Process Boundary

| File | Role |
|---|---|
| `python/sglang/srt/entrypoints/http_server.py` | FastAPI HTTP server; runs in main process |
| `python/sglang/srt/entrypoints/engine.py` | `Engine` class — orchestrates all three components |
| `python/sglang/srt/managers/tokenizer_manager.py` | Tokenizes requests, sends to scheduler |
| `python/sglang/srt/managers/scheduler.py` | Core scheduler; runs in subprocess |
| `python/sglang/srt/managers/detokenizer_manager.py` | Detokenizes output tokens |
| `python/sglang/srt/managers/io_struct.py` | ZMQ message types exchanged between processes |
| `python/sglang/srt/managers/scheduler_components/ipc_channels.py` | ZMQ socket setup per process |

---

## 3. Request Lifecycle

Here is what happens when a user sends `{"prompt": "Hello", "max_tokens": 50}`
to the server.

### Step 1 — Receive & Tokenize (TokenizerManager)

```
HTTP POST /v1/completions
  │
  ▼
TokenizerManager.generate_request()
  │
  ├─ Apply chat template (if configured)
  ├─ Tokenize prompt text → token IDs  [1024, 20234, ...]
  ├─ Wrap in BatchTokenizedGenerateReqInput (msgspec.Struct)
  └─ Send via ZMQ to Scheduler
```

**File**: `python/sglang/srt/managers/tokenizer_manager.py`

### Step 2 — Schedule & Execute (Scheduler)

This is the hot path — the scheduler runs an infinite loop:

```
while True:
    recv_requests()          # 1. Pull new requests from ZMQ queue
    │
    process_batch()          # 2. Process results from previous forward
    │   ├─ Sample next tokens
    │   ├─ Check stop conditions (max_tokens, EOS, stop strings)
    │   ├─ Send finished sequences to DetokenizerManager
    │   └─ Recycle KV cache slots for finished requests
    │
    get_next_batch_to_run()  # 3. Build the next batch
    │   ├─ Apply scheduling policy (priority, FCFS, lark)
    │   ├─ Allocate KV cache slots (prefix-aware via RadixAttention)
    │   ├─ Merge requests into a ScheduleBatch
    │   └─ Handle prefill vs decode decisions
    │
    run_batch(batch)         # 4. Execute model forward pass
    │   ├─ TpWorker.forward_batch_generation()
    │   ├─ CUDA Graph replay (for decode) or eager forward (for prefill)
    │   └─ Return logits for each request
    │
    goto recv_requests
```

**Files**:
- `python/sglang/srt/managers/scheduler.py` — the main loop
- `python/sglang/srt/managers/schedule_batch.py` — `ScheduleBatch` data structure
- `python/sglang/srt/managers/schedule_policy.py` — which requests to run next
- `python/sglang/srt/managers/tp_worker.py` — dispatches to ModelRunner
- `python/sglang/srt/managers/scheduler_components/` — request receiver, output sender, metrics, etc.

### Step 3 — Model Forward (ModelRunner)

```
TpWorker.forward_batch_generation(batch)
  │
  ├─ Build ForwardBatch from ScheduleBatch
  ├─ Resolve attention backend (FlashInfer / FlashAttention / Triton / ...)
  ├─ Prepare input tensors (input_ids, positions, sequence lengths)
  │
  └─ Two code paths:
      │
      ├─ DECODE (1 token per request in batch):
      │   └─ CUDA Graph replay
      │       Zero Python overhead per token. The entire forward pass
      │       (embeddings → transformer layers → logits) runs as a
      │       captured CUDA graph.
      │
      └─ PREFILL / EXTEND (many tokens per request):
          └─ Eager forward (with optional torch.compile)
              Full forward pass with variable-length inputs.
```

**Files**:
- `python/sglang/srt/model_executor/model_runner.py` — orchestrates the forward
- `python/sglang/srt/model_executor/forward_batch_info.py` — `ForwardBatch` struct
- `python/sglang/srt/model_executor/cuda_graph_runner.py` — CUDA Graph capture & replay

### Step 4 — Detect & Detokenize

```
Scheduler sends finished token sequences via ZMQ
  │
  ▼
DetokenizerManager
  ├─ Convert token IDs → text
  ├─ Apply stop string detection
  ├─ Stream chunks back to TokenizerManager
  │
  ▼
TokenizerManager → HTTP response (streaming or complete)
```

---

## 4. Key Concepts

### 4.1 RadixAttention (Prefix Caching)

**The core innovation from the original SGLang paper.**

When two requests share a common prompt prefix, the KV cache for that prefix is
computed once and reused. A **radix tree** (compressed prefix tree) tracks all
KV cache entries across active requests. When a new request arrives, its prompt
is matched against the tree: the longest matching prefix gets its KV cache
reused, and only the suffix needs to be computed.

```
Request A: "The capital of France is Paris"     → full compute
Request B: "The capital of France is Lyon"      → reuse "The capital of France is", compute " Lyon"
Request C: "The capital of Germany is Berlin"    → reuse "The capital of ", compute "Germany is Berlin"
```

- **File**: `python/sglang/srt/mem_cache/radix_cache.py`
- **C++ backend**: `python/sglang/srt/mem_cache/cpp_radix_tree/`

### 4.2 CUDA Graph Replay

During the **decode phase**, each request produces exactly one token per forward.
The shape of the computation is fixed for a given batch size. SGLang captures
the entire decode forward pass as a CUDA graph, then replays it for each step.
This eliminates nearly all CPU ↔ GPU synchronization and Python interpreter
overhead per generated token.

The capture is managed by `python/sglang/srt/model_executor/cuda_graph_runner.py`
and `python/sglang/srt/model_executor/cuda_graph_config.py`.

### 4.3 Continuous Batching

Traditional serving systems wait for an entire request to finish before starting
the next. **Continuous batching** builds a new batch at every forward step,
adding new requests mid-generation and removing finished ones. This maximizes
GPU utilization — the batch stays full even as individual requests finish at
different times.

### 4.4 Prefill & Decode

LLM inference has two phases:

- **Prefill** (also called "extend"): Process the input prompt. Compute-bound —
  the model processes all prompt tokens in parallel. Many tokens per request.
- **Decode**: Generate output tokens one at a time. Memory-bound — each step
  produces one token per request, reading the entire KV cache.

SGLang handles both in the same scheduler loop, choosing the right execution
path (CUDA graph for decode, eager forward for prefill). For maximum throughput
at scale, prefill and decode can run on **separate GPU pools** — see §4.5.

### 4.5 Prefill/Decode Disaggregation (PD)

In large-scale deployments, prefill and decode phases can be separated across
different GPU servers. The prefill server computes the KV cache and transfers it
to the decode server, which handles token generation.

```
Client → Prefill Server (GPU Pool A) → KV Cache Transfer → Decode Server (GPU Pool B) → Client
```

- **Files**: `python/sglang/srt/disaggregation/prefill.py`, `decode.py`
- **Transfer backends**: `nixl/`, `mooncake/`, `mori/` under `disaggregation/`

### 4.6 Speculative Decoding

Instead of generating one token at a time, a lightweight **draft model** proposes
several tokens quickly, and the main model **verifies** them in parallel. If the
verification passes, multiple tokens are accepted in one forward pass.

SGLang supports multiple speculative methods:

| Method | Draft mechanism | Key file |
|---|---|---|
| **DFlash V2** | Next-generation tree-based drafting | `dflash_worker_v2.py` |
| **EAGLE** | Feature-level extrapolation | `eagle_worker_v2.py` |
| **Multi-Layer EAGLE** | Multiple draft layers | `multi_layer_eagle_worker_v2.py` |
| **N-gram** | N-gram pattern matching | `ngram_worker.py` |
| **Frozen KV MTP** | Multi-token prediction heads | `frozen_kv_mtp_worker_v2.py` |

All in `python/sglang/srt/speculative/`.

### 4.7 Distributed Parallelism

SGLang supports several parallelism strategies, often combined:

| Strategy | Flag | What it does |
|---|---|---|
| **Tensor Parallelism** | `--tp N` | Shards weight tensors across N GPUs |
| **Pipeline Parallelism** | `--pp N` | Splits layers across N GPU groups |
| **Expert Parallelism** | `--ep N` | Distributes MoE experts across N GPUs |
| **Data Parallelism** | `--dp N` | Replicates model across N GPUs, splits requests |
| **Context Parallelism** | `--cp N` | Splits long context across N GPUs |

- **Files**: `python/sglang/srt/distributed/`, `python/sglang/srt/eplb/`, `python/sglang/srt/elastic_ep/`

### 4.8 Structured Output (Constrained Decoding)

Force model output to conform to a JSON schema, regex, or grammar. The
constraint is applied at each sampling step by masking invalid tokens.

- **Files**: `python/sglang/srt/constrained/grammar_manager.py`, `xgrammar_backend.py`
- **Backends**: XGrammar (primary), LLGuidance, Outlines

### 4.9 Quantization

Support for FP8, FP4, INT4, AWQ, GPTQ, and other quantization formats.

- **Files**: `python/sglang/srt/layers/quantization/`

### 4.10 SGLang ↔ Mooncake Integration

[Mooncake](https://github.com/kvcache-ai/Mooncake) is a distributed KV cache
system that provides RDMA-accelerated memory pooling across cluster nodes.
SGLang integrates with Mooncake along **two independent paths**:

| Path | Mooncake component | What it does for SGLang |
|---|---|---|
| **Path 1: PD Transfer** | `mooncake.engine.TransferEngine` | Real-time GPU→GPU KV cache transfer between Prefill and Decode pods (RDMA P2P) |
| **Path 2: HiCache L3 Storage** | `mooncake.store.MooncakeDistributedStore` + Mooncake Master | Distributed persistent KV cache across requests — scales beyond single-node CPU RAM |

#### 4.10.1 Path 1: PD Disaggregation Transfer

When Prefill/Decode disaggregation is active (`--disaggregation-mode prefill/decode`,
`--disaggregation-transfer-backend mooncake`), every request's KV cache must be
moved from the Prefill pod to the Decode pod. SGLang orchestrates this in two
phases:

**Control plane (ZMQ):**
1. Decode pod sends its pre-allocated GPU KV buffer pointers (`KVArgsRegisterInfo`)
   + target KV indices (`TransferInfo`) to Prefill via ZMQ.
2. Prefill validates, transitions the room to `WaitingForInput`.

**Data plane (Mooncake RDMA):**
1. After Prefill forward, KV chunks are sharded across multiple `transfer_queues`
   and worker threads.
2. Transfer strategy is chosen per model architecture:

| Strategy | When used | Mechanism |
|---|---|---|
| **Direct bulk** | MLA model or same attn_tp_size | `engine.batch_transfer_sync()` — single RDMA call |
| **Staging** | Different TP sizes + staging enabled | GPU gather → CPU staging buffer → bulk RDMA → Decode scatter |
| **Slice** | Different TP sizes, MHA, no staging | Per-token-slot head slicing via RDMA |
| **DCP relayout** | DCP size > 1 (MoE context parallelism) | Token-level scatter/gather |

3. Auxiliary metadata is sent via RDMA or TCP fallback (`SGLANG_MOONCAKE_SEND_AUX_TCP`).
4. Mamba/SSM state, SWA, DSA, MiniMax sparse indices — each has TP-aware slice
   transfer in the last chunk.
5. When all Prefill ranks respond `Success`, the Decode request enters the
   waiting queue for generation.

**Key files:**
- `python/sglang/srt/disaggregation/mooncake/conn.py` (98 KB) — `MooncakeKVManager`, `MooncakeKVSender`, `MooncakeKVReceiver`, transfer strategies
- `python/sglang/srt/disaggregation/mooncake/utils.py` — NVLink/BAREX custom GPU memory pool init
- `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py` — singleton `MooncakeTransferEngine` wrapping `mooncake.engine.TransferEngine`
- `python/sglang/srt/disaggregation/common/staging_handler.py` — cross-TP staging buffer lifecycle
- `python/sglang/srt/disaggregation/decode_hicache_mixin.py` — HiCache integration on decode side
- `python/sglang/srt/disaggregation/prefill.py` + `decode.py` — lifecycle integration

#### 4.10.2 Path 2: HiCache L3 Storage (Mooncake Store)

While Path 1 handles real-time transfer for a single request, Path 2 caches KV
cache **across requests** so repeated prefixes hit a distributed memory pool
instead of recomputing.

**Cache hierarchy:**

```
┌──────────────────────────────────────────────────────┐
│ L1: GPU VRAM  (RadixAttention, radix_cache.py)       │  ← fastest, smallest
├──────────────────────────────────────────────────────┤
│ L2: CPU RAM   (Host KV Cache, hiradix_cache.py)      │  ← fast, medium
├──────────────────────────────────────────────────────┤
│ L3: Mooncake Distributed Pool (RDMA, cluster-wide)   │  ← slower, near-unlimited
│     + optional L4: SSD offload                        │
└──────────────────────────────────────────────────────┘
```

**How SGLang drives the cache:**

| Operation | SGLang action | Mooncake call |
|---|---|---|
| **Write** | `write_backup_storage()` — evict from L2 → L3 when `backuped and hit_count >= threshold` | `batch_put_from(ptrs, sizes)` — zero-copy RDMA write |
| **Read** | `prefetch_from_storage()` — query L3 on prefix match, issue async prefetch | `batch_get_into(ptrs, sizes)` — zero-copy RDMA read |
| **Query** | `query_storage_hit_length()` — probe how many suffix tokens exist in L3 | `batch_exists(keys)` — key existence check |
| **All-reduce** | Sync hit counts across TP/CP ranks with `all_reduce(MIN)` | — |
| **Evict** | `evict_storage()` — Mooncake Master handles LRU eviction when memory pressure hits | Managed by Mooncake Master (`eviction_high_watermark_ratio`) |

**Zero-copy I/O:** SGLang registers its L2 host memory buffers with Mooncake so
that `batch_put_from` / `batch_get_into` transfer data directly over RDMA
without intermediate copies.

**Configuration:** Mooncake store config is loaded from three sources (priority
order): `--hicache-storage-backend-extra-config` JSON, config file at
`SGLANG_HICACHE_MOONCAKE_CONFIG_PATH`, or environment variables
(`MOONCAKE_MASTER`, `MOONCAKE_PROTOCOL`, `MOONCAKE_DEVICE`,
`MOONCAKE_GLOBAL_SEGMENT_SIZE`, etc.).

**Key files:**
- `python/sglang/srt/mem_cache/hicache_storage.py` — abstract `HiCacheStorage` base class
- `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` — `MooncakeStore` (the `HiCacheStorage` impl)
- `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_embedding_store.py` — embedding cache
- `python/sglang/srt/mem_cache/storage/backend_factory.py` — registers `"mooncake"` backend
- `python/sglang/srt/mem_cache/hiradix_cache.py` — 3-tier radix cache (L1→L2→L3)
- `python/sglang/srt/disaggregation/decode_hicache_mixin.py` — `DecodePrefixMatch` with `l3_storage_hit_length`

#### 4.10.3 Additional Mooncake Integration Points

Beyond PD transfer and HiCache, Mooncake provides:

| Integration | SGLang file | Function |
|---|---|---|
| **MoE all-to-all** | `layers/moe/token_dispatcher/mooncake.py` | Mooncake-based token dispatch for expert parallelism (`--moe-a2a-backend mooncake`) |
| **Elastic EP** | `elastic_ep/expert_backup_client.py`, `expert_backup_manager.py` | Expert backup/restore transport via Mooncake (`--elastic-ep-backend mooncake`) |
| **Custom GPU allocator** | `disaggregation/mooncake/utils.py` | NVLink/BAREX allocators for per-layer parallel transfers |
| **Encoder disaggregation** | `disaggregation/encode_grpc_server.py` | Encoder transfer via Mooncake (`--encoder-transfer-backend mooncake`) |
| **Observability** | `observability/mooncake_trace.py` | Request-stage tracing for Mooncake operations |

#### 4.10.4 Deployment Topology

```
                         ┌─────────────────────┐
                         │     K8s Cluster      │
                         │                      │
  ┌──────────────────┐   │  ┌────────────────┐  │
  │  Mooncake Master │   │  │ Prefill Pod    │  │
  │  (memory pool    │◀──┼──│ (SGLang +      │  │
  │   orchestrator)  │   │  │  Mooncake TE)  │  │
  └──────────────────┘   │  └───────┬────────┘  │
                         │          │            │
  ┌──────────────────┐   │          │ RDMA       │  Path 1: PD transfer
  │ Mooncake Store   │   │          ▼            │  (TransferEngine P2P)
  │ Service (optional)│  │  ┌────────────────┐  │
  │ (dedicated memory │◀─┼──│ Decode Pod     │  │
  │  contributor)     │   │  │ (SGLang +      │  │
  └──────────────────┘   │  │  Mooncake TE)  │  │
                         │  └───────┬────────┘  │
                         │          │            │
                         │          │ RDMA       │  Path 2: HiCache L3
                         │          ▼            │  (MooncakeDistributedStore)
                         │  ┌────────────────┐  │
                         │  │ Mooncake Store │  │
                         │  │ Service        │  │
                         │  └────────────────┘  │
                         └─────────────────────┘
```

**SGLang's role in the partnership:** SGLang owns all scheduling intelligence —
when to transfer, what layers to transfer, which transfer strategy to use, how
cache hits affect prefill length, when to prefetch from L3. Mooncake provides
the infrastructure — RDMA transport channels (`TransferEngine`) and distributed
memory pooling (`MooncakeDistributedStore` + Mooncake Master).

---

## 5. Source Tree Map

### 5.1 Top-Level Directories

```
sglang/
├── python/sglang/srt/         ← CORE RUNTIME (the main codebase)
├── python/sglang/lang/        ← Frontend DSL (LLM call composition)
├── python/sglang/cli/         ← CLI entry points (sglang serve, ...)
├── python/sglang/kernels/     ← Custom CUDA/Triton GPU kernels
├── python/sglang/multimodal_gen/ ← Diffusion models (WAN, Qwen-Image)
├── rust/                      ← Rust components (server, gRPC, multimodal)
├── sgl-model-gateway/         ← Model gateway service
├── proto/                     ← Protobuf definitions
├── benchmark/                 ← 46 model-specific benchmark suites
├── test/                      ← CI tests
│   ├── registered/unit/       ← Unit tests (one file = one process)
│   └── registered/integration/← Integration tests
├── docker/                    ← Docker builds
├── docs_new/                  ← Mintlify documentation site
├── scripts/                   ← CI & ops scripts
├── 3rdparty/                  ← Vendored third-party code
└── experimental/              ← Experimental features
```

### 5.2 The `srt/` (SGLang Runtime) Subtree

```
python/sglang/srt/
│
├── entrypoints/               ← HTTP & gRPC servers, OpenAI API, Engine API
├── managers/                  ← Scheduler, TokenizerManager, DetokenizerManager
│   └── scheduler_components/  ← Scheduler sub-modules (output, metrics, ...)
│
├── model_executor/            ← Model loading, forward execution, CUDA graphs
│   └── model_runner_components/ ← Sub-modules (graph setup, MoE, attention)
│
├── models/                    ← 170+ model implementations (one file per arch)
├── configs/                   ← 56+ per-model default configurations
│
├── layers/                    ← Model layer implementations
│   ├── attention/             ← 30+ attention backends
│   ├── quantization/          ← FP8/FP4/INT4/AWQ/GPTQ
│   ├── moe/                   ← Mixture-of-Experts implementations
│   └── rotary_embedding/      ← RoPE variants
│
├── mem_cache/                 ← KV cache subsystem
│   ├── radix_cache.py         ← Core RadixAttention prefix cache
│   ├── hiradix_cache.py       ← Hierarchical (GPU+CPU+SSD) cache
│   ├── swa_radix_cache.py     ← Sliding window attention cache
│   ├── mamba_radix_cache.py   ← Mamba/SSM state cache
│   ├── hybrid_cache/          ← Attention + Mamba unified cache
│   ├── allocator/             ← KV cache slot allocators
│   ├── cpp_radix_tree/        ← C++ Radix Tree implementation
│   └── storage/mooncake_store/ ← Mooncake L3 distributed store (MooncakeDistributedStore)
│
├── speculative/               ← Speculative decoding (EAGLE, DFlash, N-gram, ...)
├── disaggregation/            ← Prefill/Decode separation
│   ├── mooncake/              ← Mooncake RDMA transfer backend (conn.py 98KB)
│   └── common/staging_handler.py ← Cross-TP staging buffer for Mooncake transfers
├── constrained/               ← Structured output (JSON, regex, grammar)
├── distributed/               ← TP/PP/EP/DP parallelism
├── lora/                      ← Multi-LoRA batching
├── multimodal/                ← Image/video/audio input processing
├── hardware_backend/          ← GPU/CPU/NPU/TPU hardware abstraction
├── connector/                 ← Model weight loading (S3, Azure, Redis)
├── session/                   ← Streaming chat session management
├── observability/             ← OpenTelemetry tracing & metrics
├── server_args.py             ← ALL CLI flags (413 KB — single source of truth)
├── environ.py                 ← ALL SGLANG_* environment variables
├── runtime_context.py         ← Process-level config singleton
└── configs/                   ← Per-model config overrides
```

### 5.3 Quick Reference: Where to Find Things

| I want to... | Look in |
|---|---|
| Add a new model | `python/sglang/srt/models/` + register in `registry.py` |
| Add a CLI flag | `python/sglang/srt/server_args.py` |
| Add an env var | `python/sglang/srt/environ.py` |
| Change scheduling policy | `python/sglang/srt/managers/schedule_policy.py` |
| Add an attention backend | `python/sglang/srt/layers/attention/` |
| Add a speculative method | `python/sglang/srt/speculative/` |
| Change KV cache behavior | `python/sglang/srt/mem_cache/` |
| Change the HTTP API | `python/sglang/srt/entrypoints/openai/` |
| Add a LoRA feature | `python/sglang/srt/lora/` |
| Change distributed setup | `python/sglang/srt/distributed/` |
| Debug CUDA crashes | `python/sglang/srt/debug_utils/` (see `debug-cuda-crash` skill) |
| Find a model config default | `python/sglang/srt/configs/<model_name>.py` |
| Configure PD KV transfer | `python/sglang/srt/disaggregation/mooncake/conn.py` |
| Configure HiCache L3 storage | `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` |
| Change MoE all-to-all backend | `python/sglang/srt/layers/moe/token_dispatcher/mooncake.py` |
| Configure Mooncake env vars | `python/sglang/srt/environ.py` (search `MOONCAKE`) + `mooncake_store/README.md` |

---

## 6. Coding Conventions

SGLang has a set of coding rules enforced in `.claude/rules/`. The highlights:

### Data Structures

- **Use `msgspec.Struct`**, not `@dataclass`. Faster, stricter, multi-language
  compatible (for the Rust migration).
  ```python
  class MyData(msgspec.Struct):
      field: int = 0
      items: list[str] = []
  ```

### Configuration Access

- **Never read `server_args.field` directly for business logic.** Config lives
  in namespace bags: `get_exec().moe.backend`, `get_schedule().max_requests`.
  - `python/sglang/srt/runtime_context.py` is the singleton container.
  - See the `sglang-runtime-context` skill for the full design.

### Immutability & Pure Functions

- Prefer **stateless functions** over mutating methods.
- **Never mutate a `ScheduleBatch` field in place** — build a new value and rebind.
- **`ForwardBatch.init_new`** must treat `ScheduleBatch` as read-only.
- Default to **protected** (`_name`) methods; only expose what callers use.
- **Pass specific values**, not god objects (`ModelRunner`, `Scheduler`).

### Defensive Coding

- **Don't use `getattr`/`hasattr` for fields that always exist.** Use `isinstance`
  for type narrowing or set fields to `None` and check for `None`.

### File Size

- Functions: keep under ~100 lines.
- Files: keep under ~2,000 lines. Split along cohesive boundaries.
- Core orchestration functions should read like pseudocode — push detail into
  well-named helpers.

### Naming

- New speculative decoding code follows the `speculative-naming` conventions.
- Large files (`scheduler.py`, `tokenizer_manager.py`, `model_runner.py`)
  follow the `large-class-style` guide.

### Tests

- **CI runs each test file in its own process.** A local monolithic pytest run
  can mask context-leak bugs.
- After publishing a `RuntimeContext`, call `reset_context()` in teardown.
- Use `get_context().override_server_args(**fields)` to control config in tests,
  never monkeypatch import bindings.

---

## 7. Development Workflows

### 7.1 Adding a New Model

1. Create `python/sglang/srt/models/my_model.py`
2. Implement the model class (follow a similar existing model for patterns)
3. Register in `python/sglang/srt/models/registry.py`
4. If the model needs special default config, add
   `python/sglang/srt/configs/my_model.py`
5. Add tests

### 7.2 Debugging CUDA Crashes

Enable kernel-level logging:
```bash
export SGLANG_KERNEL_API_LOG_LEVEL=1
```
See the `debug-cuda-crash` skill for the full workflow.

### 7.3 Running a Single Test Properly

```bash
# Per-file (the CI way — avoids context leaks between test files)
python -m pytest test/registered/unit/test_my_feature.py -xvs

# With CUDA assertions (catches silent KV-cache errors)
SGLANG_ENABLE_ASYNC_ASSERT=1 python -m pytest test/registered/unit/test_my_feature.py -xvs

# For launcher-path code (use unbuffered stdout to see child process crashes)
PYTHONUNBUFFERED=1 python -m pytest test/ -xvs -k "test_my_launcher_test"
```

### 7.4 Profiling

Use `SGLANG_PROFILE_TYPE=python` for Python-level profiling or
`SGLANG_PROFILE_TYPE=cuda` for CUDA kernel profiling. See the
`generate-profile` and `llm-torch-profiler-analysis` skills.

---

## 8. Further Reading

### Project Documentation

- [docs.sglang.io](https://docs.sglang.io) — Official documentation
- [lmsys.org/blog](https://lmsys.org/blog/) — Release blogs with architecture deep-dives
- `docs_new/` — Mintlify doc source (this repo)

### Claude Skills (in `.claude/skills/`)

These are reference guides loaded on demand. Key ones for development:

| Skill | When to load |
|---|---|
| `architecture-reference` | Get a subsystem map + file paths |
| `sglang-runtime-context` | Understand config/publish/bags design |
| `speculative-naming` | Work on speculative decoding code |
| `large-class-style` | Modify scheduler, tokenizer, or model_runner |
| `env-var-conventions` | Add or rename an `SGLANG_*` env var |
| `add-jit-kernel` | Add a JIT CUDA kernel |
| `add-sgl-kernel` | Add an AOT CUDA/C++ kernel |
| `write-sglang-test` | Write tests that follow project conventions |
| `debug-cuda-crash` | Debug CUDA crashes |
| `ci-workflow-guide` | Understand/modify CI pipelines |
| `mechanical-refactor-verify` | Do large-scale refactors safely |

### Coding Rules (in `.claude/rules/`)

Auto-applied on matching files. Read them before modifying affected code:

- `general-code-style.md` — stateless, immutable, small files
- `no-dataclasses.md` — use `msgspec.Struct`
- `no-getattr-defensive.md` — don't defensively `getattr` static fields
- `schedule-batch-out-of-place-mutation.md` — never mutate ScheduleBatch in place
- `forward-batch-init-new-purity.md` — ForwardBatch.init_new is read-only
- `modify-component-must-read.md` — prerequisites before modifying core components
- `unit-test-admission.md` — test quality bar

### Key Papers & Blog Posts

- [SGLang v0.1 (RadixAttention)](https://lmsys.org/blog/2024-01-17-sglang/) — The original architecture
- [v0.4: Zero-Overhead Scheduler](https://lmsys.org/blog/2024-12-04-sglang-v0-4/) — Current scheduler design
- [DFlash V2](https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/) — Next-gen speculative decoding
- [Large-Scale Expert Parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/) — EP at 96 GPUs
- [GB200 Rack-Scale Parallelism](https://lmsys.org/blog/2025-09-25-gb200-part-2/) — PD + EP on GB200

### Mooncake References

- [Mooncake Project](https://github.com/kvcache-ai/Mooncake) — Upstream Mooncake repo
- [SGLang HiCache with Mooncake Quick Start](https://kvcache-ai.github.io/Mooncake/getting_started/examples/sglang-integration/hicache-quick-start.html)
- [Mooncake x SGLang HiCache System Design](https://kvcache-ai.github.io/Mooncake/design/hicache-design.html)
- [SGLang HiCache Design](https://docs.sglang.io/advanced_features/hicache_design.html)
- [Mooncake Store README](python/sglang/srt/mem_cache/storage/mooncake_store/README.md) — In-repo deployment guide with code examples
