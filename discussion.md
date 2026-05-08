# HFC Project — Discussion & Findings

## Project Overview

**HFC** is a DRAM-centric KV offload system for LLM inference on Ascend NPU. The core thesis: treat HBM as a cache over a DRAM-resident KV pool, implemented as a compiler pass via FX graph rewriting rather than a custom runtime.

Target hardware: `ssh -p 60008 kkrazy@47.103.62.251` — 16x Ascend 910 (9392), 64 GB HBM each, 2 TB DRAM, aarch64, CANN toolkit 8.1.RC2.

---

## 1. Architecture & Core Idea

HBM is a cache over a DRAM-resident KV pool. The graph-rewrite approach:

1. **Identify KV nodes automatically** — pattern-match the FX graph for `past_key_values` concat + attention flow
2. **Offload after write, prefetch before read** — insert DMA ops at those edges
3. **Double-buffer at layer granularity** — while layer N computes, prefetch layer N+1's KV (~2x recovery vs 3x overhead without)
4. **Eviction policy** — for long sequences, LRU or attention-weighted eviction with recomputation on miss

Key advantage: works as a compiler pass. No custom kernels, no custom runtime. Runs on any PyTorch backend — Ascend, CUDA, CPU.

### Why not vLLM

KV management is baked into custom CUDA kernels (PagedAttention), custom scheduler, custom serving framework. Can't extract just the KV offload. Doesn't run on Ascend.

---

## 2. FX Graph Capture Findings

### Models traced successfully

| Model | Nodes | Notes |
|-------|-------|-------|
| OPT-125m | 837 | attention hidden behind `call_module` |
| Qwen2-0.5B | 2903 | attention fully inlined (q/k/v_proj through rotary to o_proj visible) |
| Qwen3-30B-A3B (scale 0.1) | 1380 | 2 placeholder, 184 get_attr, 1193 call_function, 1 output |

GPT-2 FX trace fails (loops over heads). OPT traces but hides attention behind `call_module`.

### torch.fx / transformers compatibility

- `torch.fx.symbolic_trace` works on Python 3.9 + transformers 4.40.2
- transformers 4.51.3 breaks with "varnames too small" — pin to 4.40.2 (or use per-family venv)
- `offload_async` / `prefetch_sync` are plain Python functions (not `torch.library` custom ops — those broke on torch 2.1)
- Profile on CPU (fp32), execute on NPU (fp16) to avoid fp16 LayerNorm kernel issue on CPU

### Per-family venv pattern (Qwen3-MoE)

transformers 4.51 is needed for Qwen3 but incompatible with 4.40.2. Solution: separate venv per model family.

- `scripts/setup_remote.sh` accepts `HFC_VENV_NAME` + `HFC_REQ_NAME` env vars
- For Qwen3: `HFC_VENV_NAME=qwen3 HFC_REQ_NAME=requirements-qwen3.txt` → `~/hfc/.venv-qwen3`
- ~8 GB extra disk per venv

### Compat patch chain for transformers 4.51 + torch 2.1 + Python 3.9

Each is in `scripts/run_qwen3_moe.py`. Removing any one breaks capture:

1. **`torch.utils._pytree.register_pytree_node` polyfill** — torch 2.2+ API; back to `_register_pytree_node` on 2.1
2. **`torch.compiler.is_compiling` polyfill** — torch 2.3+ API. transformers' attention-mask helper calls it during tracing
3. **`einops`** — pulled in by `transformers/integrations/npu_flash_attention.py` when torch_npu is detected
4. **`AttentionInterface` → plain dict** — transformers 4.51 routes attention through a class with custom `__getitem__`; torch 2.1 dynamo can't inline it
5. **`cfg._attn_implementation = "eager"`** — avoids SDPA/FA2 dispatch paths
6. **Dense MoE forward replacement** — `Qwen3MoeSparseMoeBlock.forward` has data-dependent control flow. Replaced with dense (run all experts, weight by mask)
7. **`make_fx` instead of `torch.export`** — torch 2.1's `torch.export` blows up on `F.one_hot`. `make_fx(tracing_mode="real")` bypasses dynamo
8. **Rewrite `device='cpu'` → `device='npu:0'` on aten nodes** — `make_fx(tracing_mode="real")` bakes trace-time device into 7 nodes

---

## 3. Offload Benchmarks

### Qwen2-0.5B, seq_len=32, npu:0, fp16

| Policy | Tensors offloaded | Prefetch ops | Latency | Verify |
|--------|-------------------|--------------|---------|--------|
| none (baseline) | 0 | 0 | 23 ms | — |
| AttentionKVOnly | 96 | 96 | 41 ms (+75%) | OK |
| LargestNTensors n=20 | 20 | 20 | 68 ms (+190%) | OK |
| BudgetPolicy 50MB | 525 | 756 | works | OK |
| BudgetPolicy 0.5GB | 0 | 0 | — | — |

- **KV policy beats largest n=20**: KV tensors (16 KB each) are tiny; the 20 "largest" includes 32 MB rotary caches whose D2H stalls dominate. KV-targeted offload is cheaper than naive size-ranked.
- **BudgetPolicy with skip_params=True** treats rotary cos/sin (1.5 GB) as non-offloadable. Tighten to 50 MB to force offload.
- **Phase 2 (double-buffering)** would hoist prefetch(N+1) to overlap with compute(N), expected ~2x recovery.

### Cost model: structural, not just lifetime

The naive size x gap rule (LiveRangePolicy) is wrong for two reasons:

1. **Single-forward FX capture hides cross-step lifetime.** KV from layer L at token N is reread at token N+1 — that's all-other-layers + next compute. Single-forward graph shows KV with gap~10 nodes; real autoregressive lifetime is >> inter-layer compute time.
2. **aten flattening collapses parameter lifetime to gap=1.** `get_attr -> aten.t -> aten.addmm` in three adjacent nodes hides "weight idle until next forward."

The cost model classifies candidates into:
- **K/V projection weight**: always-selected (cross-step)
- **Other layer/shared param**: opt-in for weight streaming
- **Activation**: gap-based rule (only sound class for this metric)

### Pool semantics gotcha

`_offload_async_npu` writes one entry per producer node name to `_pool`. `_prefetch_sync_npu` pops on first call. For nodes with multiple consumers, only the first prefetch gets the DRAM-roundtripped tensor; later prefetches fall through to the still-resident NPU original. Works today because offload_async returns x unchanged (HBM not freed), but breaks once Phase 2 actually evicts. Will need refcount or peek-style pool.

---

## 4. Performance: NPUGraph, torchair, Batching

### NPUGraph (full Qwen3 forward, 1380 aten nodes)

| Mode | Eager | NPUGraph | Speedup |
|------|-------|----------|---------|
| BATCH (b=10) | 26.4 ms | 13.5 ms | 1.96x |
| STREAM (10x b=1) | 227 ms | 89 ms | 2.54x |

Combined NPUGraph+BATCH is **17x faster** than naive eager STREAM (744 vs 44 req/s).

Why much bigger than matmul-only NPUGraph (1.2x): FX GraphModule's `forward()` walks every node through the Python interpreter. 1380 nodes at ~5 us/node = ~7 ms pure Python overhead. NPUGraph replays captured kernels in one ACL submission.

### torchair (GE compile)

`torchair` is bundled inside `torch_npu` at `torch_npu.dynamo.torchair`, not a separate pip package. Requires `pkg_resources` (setuptools <70).

| Mode | Eager | torchair | Speedup |
|------|-------|----------|---------|
| b=1 single forward | 13.3 ms | 3.23 ms | 4.12x |
| b=10 BATCH | 13.3 ms | 5.58 ms | 2.38x |
| 10x b=1 STREAM | 123 ms | 30.6 ms | 4.01x |

**Combined torchair + BATCH = 1791 req/s** — 41x over original FX-wrapped eager STREAM baseline.

Implications:
- torchair becomes the default forward-runner backend
- Our FX-rewriter pipeline is orthogonal — composes BEFORE torchair as preprocessing
- NPUGraph drops to fallback for paths torchair can't compile
- Limitation: torchair refused to compile captured FX GraphModule (rejects `aten._to_copy(device=cpu)` ops)

### Batching dominates streaming

For Qwen3-30B-A3B (scale 0.1, seq=32, fp16), 10 requests:
- **BATCH** (b=10): 26.67 ms → 375 req/s
- **STREAM** (10x b=1): 230.19 ms → 43 req/s
- Speedup = **8.63x**

Single forward at b=10 takes only +5% over b=1. The 8.6x throughput gap is arithmetic intensity, not per-launch overhead.

**Takeaway**: Continuous batching is THE lever for serving (5-30x effect). Streaming submission is almost always wrong. Batched submission buys 5-30% on top but is not a substitute.

---

## 5. Multi-Stream Findings

### Multi-stream is null on Ascend at every granularity

Tested at three levels — all show no benefit:

| Test | Result |
|------|--------|
| Matmul micro-bench, round-robin | 0.22x (severe regression) |
| Matmul micro-bench, per-stream blocks | 0.70-1.00x |
| Qwen3 full-forward, per-request streams | 1.00x (noise) |

**Why**: Ascend serializes kernels at the device level regardless of which stream they were submitted to. No overlap between forwards on different streams.

**Implication**: Don't expect multi-stream to recover throughput. The only intra-device path to parallelism is bigger batches. Multiple streams remain useful for *separating concerns* (compute stream + DMA stream for offload) — not for parallelizing same-class work.

### 3-Stream matmul example (this session)

Built a 3-stream matmul demo (input/compute/output streams with event-based sync) as a skeleton for the offload pipeline pattern.

```
input_stream:    H2D(A), H2D(B) -> recordEvent(input_done)
compute_stream:  streamWaitEvent(input_done) -> aclnnMatmul -> recordEvent(compute_done)
output_stream:   streamWaitEvent(compute_done) -> D2H(C)
main thread:     synchronizeStream(output_stream)
```

**aclblas API (aclblasGemmEx, aclblasHgemm) fails** with error 100000 on all data type combos (FP32, FP16, mixed). Basic ACL ops work fine — issue is specific to the aclblas library.

**ACLNN API (aclnnMatmul) works correctly** using a 2-phase model:
- Phase 1: `aclnnMatmulGetWorkspaceSize(...)` — plans operation, returns workspace size
- Phase 2: `aclnnMatmul(workspace, wsSize, executor, stream)` — executes on given stream

256x256 FP16 matmul: expected=64.0, max_error=0.0, PASS.

Build notes:
- Link: `libascendcl` + `libaclnn_ops_infer` + `libnnopbase`
- `LD_LIBRARY_PATH` needs both `.../aarch64-linux/lib64` and `.../latest/lib64`

---

## 6. Similar Projects & Prior Art

| Project | Approach | Relevance |
|---------|----------|-----------|
| **vLLM (PagedAttention)** | KV as per-layer paged tensors, block table, batched DMA | Paging is the right abstraction; no DRAM offload today |
| **FlexGen** | 3-tier offload (HBM/DRAM/NVMe) with policy search | Proves 3-tier works; batch-mode, not serving-latency |
| **Mooncake** | Disaggregated prefill/decode, RDMA KV transfer | Disaggregated arch is orthogonal; prefix-caching applies |
| **LMCache** | Layer-level KV offload with DRAM/NVMe, hash-based | Closest to our approach; tied to HF model code |
| **HF OffloadedCache** | Layer-at-a-time async prefetch | Proves async prefetch at layer granularity |
| **Ascend NPU ecosystem** | No public KV offload; torch_npu provides `torch.npu.Stream` + `pin_memory()` | Building blocks exist |

---

## 7. Remote Environment Notes

- SSH: `ssh -p 60008 kkrazy@47.103.62.251`
- 16x Ascend 910 (9392), 64 GB HBM each, 2 TB DRAM, aarch64
- CANN toolkit 8.1.RC2 at `/usr/local/Ascend/ascend-toolkit/`
- Python 3.9.9, venv at `~/hfc/.venv`
- huggingface.co blocked; hf-mirror.com works (slow)
- `~/hfc/env.sh` sources CANN + activates venv
- CANN `set_env.sh` incompatible with `set -u` (uses unset `LD_LIBRARY_PATH`)
- CANN TBE deps needed in venv: decorator, attrs, cloudpickle, scipy, absl-py
- `aclFloat16` is `uint16_t` typedef; Ascend 910 Cube engine natively does FP16 matmul

---

## Open Questions

1. **aclblas deprecation** — Is it officially deprecated in 8.1? Worth checking with debug logs (`ASCEND_LOG_LEVEL=0`)
2. **FP32 matmul via ACLNN** — Does `cubeMathType` parameter support FP32 with internal FP16 compute?
3. **Multi-NPU distributed matmul** — Can we extend to HCCL all-reduce for distributed inference?
4. **torchair + FX-rewriter composition** — torchair rejects `aten._to_copy(device=cpu)` ops in captured GraphModule. Need custom op registration or different graph-rewrite shape.
5. **Phase 2 double-buffering** — Expected ~2x recovery for KV offload. Not yet implemented.
6. **Pool refcounting** — Current pop-on-prefetch semantics break once Phase 2 evicts from HBM. Need refcount or peek-style pool.
7. **AttentionKVOnly policy on aten graphs** — Pattern-matches `call_module` but `make_fx` produces `call_function` only. Need aten-level pattern matcher.

---

## Files

- `matmul_3stream.cpp` — Working 3-stream FP16 matmul using ACLNN API
