# HFC: HuggingFace Compiler — DRAM-Centric KV Offload via Graph Rewriting

## Problem

LLM inference on accelerators is HBM-bound. A single Qwen2-72B layer's KV cache
occupies ~128 MB per request at fp16; at 32 concurrent requests that's 4 GB per
layer, 96 GB across 12 layers — far exceeding a single Ascend 910's 64 GB HBM.
Weights also overflow: 72B params × 2 bytes = 144 GB.

The standard solution is offloading to host DRAM (2 TB on the target box), but
today this is done *imperatively* inside model code (vLLM's PagedAttention,
HuggingFace's `OffloadedCache`). That couples the offload logic to every model's
attention implementation, making it non-portable and hard to experiment with.

## Idea

**Treat offloading as a compiler pass.** Capture the model as an FX graph, profile
every intermediate tensor's size and lifetime, apply a pluggable policy to decide
what to tier, then rewrite the graph to inject `prefetch → compute → evict` around
the selected tensors. The model code never changes; the graph does.

This is analogous to register allocation in a traditional compiler — HBM is the
register file, DRAM is the stack, and the rewriter is the register allocator.

## Architecture

```
                    ┌──────────────┐
                    │  HF Model    │
                    │  (any)       │
                    └──────┬───────┘
                           │ torch.fx.symbolic_trace
                           ▼
                    ┌──────────────┐
                    │  FX Graph    │  hardware-agnostic IR
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Profiler │ │  Policy  │ │ Rewriter │
        └──────────┘ └──────────┘ └──────────┘
              │            │            │
              │ tensor     │ which      │ inject
              │ sizes &    │ tensors    │ offload/
              │ lifetimes  │ to tier    │ prefetch
              ▼            ▼            ▼
        ┌─────────────────────────────────────┐
        │  Rewritten FX Graph                 │
        │  (with offload_async / prefetch_sync)│
        └──────────────┬──────────────────────┘
                       │ .to("npu")
                       ▼
                ┌──────────────┐
                │  Ascend NPU  │
                └──────────────┘
```

### Pass 1: Profile

`ShapePropInterpreter` (subclass of `torch.fx.Interpreter`) runs the graph with
example inputs and records for every node:

| Field       | Description                                    |
|-------------|------------------------------------------------|
| `shape`     | output tensor shape (e.g. `(1, 32, 768)`)      |
| `dtype`     | element dtype                                   |
| `nbytes`    | total bytes (`nelement × element_size`)         |
| `topo_idx`  | position in topological execution order          |
| `consumers` | list of downstream nodes that read this tensor   |

This is the "liveness analysis" — it tells us when a tensor is born and when it
dies, which is the input to the allocation decision.

### Pass 2: Policy

A policy is any callable that takes the profile and returns a set of node names
to offload. Built-in policies:

| Policy              | Logic                                              |
|---------------------|----------------------------------------------------|
| `LargestNTensors`   | Offload the N biggest intermediates (greedy)       |
| `BudgetPolicy`      | Offload until remaining HBM fits under budget      |
| `AttentionKVOnly`   | Pattern-match k_proj/v_proj outputs only           |
| `ManualPolicy`      | Hand-pick nodes by name                            |
| `CallablePolicy`    | Wrap an arbitrary function                         |

The policy is the main knob for experimentation. A future `LearnedPolicy` could
train an eviction model on access patterns (à la PowerInfer).

### Pass 3: Rewrite

For each offloaded node, the rewriter inserts two FX nodes:

```
  ... = orig_node(...)            # producer
  handle = offload_async(...)     # ← inserted: move to DRAM
  ... = consumer(handle)          # consumer uses offloaded tensor
```

becomes:

```
  ... = orig_node(...)
  handle = offload_async(...)
  prefetched = prefetch_sync(handle)  # ← inserted: move back to HBM
  ... = consumer(prefetched)          # consumer uses prefetched copy
```

The rewriter also replaces the consumer's args to reference the prefetched tensor
instead of the original. On CPU the ops are identity functions; on NPU they become
real `torch_npu` H2D/D2H transfers.

## Current Status (April 2026)

Working end-to-end on the target box (8× Ascend 910, 2 TB DRAM, aarch64):

| Component          | Status | Notes                                           |
|--------------------|--------|-------------------------------------------------|
| FX graph capture   | ✓      | OPT-125m, Qwen2-0.5B traced and executed on NPU |
| Tensor profiler    | ✓      | Shapes, sizes, lifetimes recorded per node       |
| Policy framework   | ✓      | LargestN, Budget, KV-only, Manual all implemented|
| Graph rewriter     | ✓      | Injects offload/prefetch, CPU verification passes|
| NPU execution      | ✓      | Rewritten OPT-125m produces correct logits       |
| Real DRAM tiering  | ✗      | offload_async/prefetch_sync are identity stubs   |
| Double-buffering   | ✗      | No overlap of prefetch(N+1) with compute(N) yet  |
| KV-specific policy | partial| KV-only policy exists but not tested on large model|

Tested models:
- `facebook/opt-125m`: 837-node graph, 20 tensors offloaded, verified correct on `npu:0`
- `Qwen/Qwen2-0.5B`: 2903-node graph captured, attention fully inlined (visible Q·K^T/softmax/matmul)

## Suggested Path Forward

### Phase 1: Make the offload real (1-2 weeks)

Replace the identity stubs with actual HBM↔DRAM transfers:

```python
# hfc/backends/npu.py
import torch, torch_npu

DRAM_POOL = {}  # name → cpu pinned tensor

def offload_async(x: torch.Tensor) -> torch.Tensor:
    """Copy from HBM to pinned DRAM, return a CPU-side handle."""
    name = f"kv_{id(x)}"
    cpu_tensor = torch.empty_like(x, device="cpu").pin_memory()
    cpu_tensor.copy_(x, non_blocking=True)
    DRAM_POOL[name] = cpu_tensor
    return x  # original stays on device until eviction

def prefetch_sync(x: torch.Tensor) -> torch.Tensor:
    """Copy from pinned DRAM back to HBM, block until done."""
    name = f"kv_{id(x)}"
    cpu_tensor = DRAM_POOL.pop(name, None)
    if cpu_tensor is None:
        return x
    device_tensor = cpu_tensor.to(x.device, non_blocking=False)
    return device_tensor
```

Key implementation details:
- Use `torch.cuda.Stream` equivalent (`torch.npu.Stream`) for async overlap
- Pin DRAM allocations with `.pin_memory()` so DMA transfers bypass the CPU
- The rewriter needs to pass a stable *name* (not `id()`) through the graph so the
  offload/prefetch pair can find each other — modify the rewriter to pass the node
  name as a string arg

### Phase 2: Double-buffered pipeline (2-3 weeks)

The current rewriter inserts prefetch *right before* the consumer. For real
throughput, we need:

```
  compute(layer_N)     ||  prefetch(layer_N+1_KV)
  compute(layer_N+1)   ||  prefetch(layer_N+2_KV)
```

This requires:
1. The rewriter to hoist each prefetch to the *start of the previous layer's*
   computation rather than right before its consumer
2. Two NPU streams: one for compute, one for DMA
3. A sync point (event wait) right before the consumer actually reads the tensor

The FX graph supports this naturally — `inserting_before` lets us place the
prefetch anywhere in the topological order, as long as it's after the offload.

### Phase 3: KV-specific scheduling (3-4 weeks)

The generic "largest tensor" policy is useful but not optimal for KV. KV tensors
have specific properties the policy should exploit:

1. **Predictable access pattern**: layers are executed sequentially, so the prefetch
   schedule is static and deterministic
2. **Per-request isolation**: each request's KV is independent — can be placed
   anywhere in the DRAM pool without contention
3. **Growing size**: KV grows by `head_dim × seq_len_per_step` each decode step.
   The budget policy needs to account for this growth
4. **Compression opportunity**: quantize to int8 during offload (2× capacity,
   ~negligible quality loss for chat workloads)

The `AttentionKVOnly` policy already identifies the right nodes. What's missing:

- **Block-level paging**: instead of offloading the whole KV tensor, split into
  fixed-size blocks (like vLLM's PagedAttention) so we can partially prefetch
- **Hot/cold split**: keep the most recent N tokens' KV in HBM (always hit),
  evict older tokens to DRAM (prefetch on demand)
- **Prefix caching**: if multiple requests share a system prompt, deduplicate
  their KV blocks in DRAM

### Phase 4: Multi-model, multi-backend (later)

- Extend to encoder-decoder models (T5, Whisper) where cross-attention KV is
  shared across decoder steps and is the dominant offload target
- Add a CUDA backend (the same FX graph + policy works; only the `offload_async`
  / `prefetch_sync` implementations change)
- Add `torch.export` as an alternative capture frontend for models that don't
  trace cleanly under `torch.fx`

## Key Design Decisions

1. **FX graph as the IR, not torch.export** — `torch.fx` is more mature for HF
   models in torch 2.1; `torch.export` is the future but has worse HF coverage
   today. The policy/rewriter don't depend on which frontend produces the graph.

2. **Plain Python functions as offload ops** — not `torch.library` custom ops.
   This avoids torch version compatibility issues and lets the backend be
   monkey-patched at runtime. The tradeoff: no TorchScript support, but we
   don't need it.

3. **Policy is a callable, not a config** — this lets users write arbitrary
   selection logic (e.g., "offload only the KV from layers 8-23" or "use a
   neural net to predict which tensors to evict") without changing the compiler.

4. **Profile on CPU, execute on NPU** — profiling is hardware-agnostic (just
   shapes and sizes). This means the same profile works across devices, and the
   expensive NPU doesn't need to be reserved for profiling.

## Risks and Open Questions

- **FX tracing fragility**: many HF models don't trace cleanly. The fallback is
  `torch.export` (Phase 4) or hook-based capture (intercept at the `nn.Module`
  level instead of the FX level).

- **Causal mask shape baking**: Qwen2's `_update_causal_mask` bakes in the trace-time
  sequence length, causing shape mismatches when `past_len > 0`. Workaround: trace
  at `past_len=0` and handle the decode-path mask dynamically in the rewriter.

- **torch_npu API stability**: Ascend's PyTorch integration is evolving fast. The
  `torch.npu.Stream` API may differ from CUDA streams. Need to test on each
  torch_npu release.

- **Latency vs throughput**: double-buffering adds latency (the first layer waits
  for its prefetch). For single-stream serving this is acceptable; for batched
  serving the scheduler needs to interleave requests to hide the latency.
