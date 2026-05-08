#!/usr/bin/env python3
"""Try existing fusion / kernel-batching paths on Qwen3 b=1 forward.

Approaches probed (in order of likely impact):
  A. torch.compile(backend="inductor")            ← Inductor codegen
  B. torch.compile(backend="aot_eager")           ← AOT autograd, no fusion codegen
  C. torch.compile(backend="cudagraphs")          ← stream-graph capture (CUDA-Graphs/NPU equiv)
  D. torch.npu.make_graphed_callables             ← high-level NPUGraph wrapper
  E. Manual NPUGraph capture + replay             ← what we already validated

For each that loads, run BATCH (b=10) and STREAM (10×b=1, default stream) bench
and compare against eager.
"""
from __future__ import annotations
import sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_qwen3_moe as _q3  # noqa: E402

import argparse
import gc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--scale", type=float, default=0.1)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--n-requests", type=int, default=10)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    return p.parse_args()


def _migrate_gm(gm, args):
    import torch
    target_dev = torch.device(args.device)
    cpu_dev = torch.device("cpu")
    rewritten = 0
    for node in gm.graph.nodes:
        if node.op != "call_function" or "device" not in node.kwargs:
            continue
        d = node.kwargs.get("device")
        if d == cpu_dev or d is None or (isinstance(d, str) and d == "cpu"):
            new_kw = dict(node.kwargs)
            new_kw["device"] = target_dev
            node.kwargs = new_kw
            rewritten += 1
    if rewritten:
        gm.graph.lint()
        gm.recompile()
    dtype = getattr(torch, args.dtype)
    gm = gm.to(args.device).to(dtype)
    for name in [n.target for n in gm.graph.nodes if n.op == "get_attr"]:
        attr = getattr(gm, name, None)
        if isinstance(attr, torch.Tensor) and attr.device != target_dev:
            new_t = attr.to(target_dev)
            if new_t.is_floating_point():
                new_t = new_t.to(dtype)
            try:
                setattr(gm, name, new_t)
            except Exception:
                object.__setattr__(gm, name, new_t)
    return gm


def capture(args, batch_size, model, cfg):
    import torch
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, args.seq_len))
    attention_mask = torch.ones(batch_size, args.seq_len, dtype=torch.long)
    example_inputs = dict(input_ids=input_ids, attention_mask=attention_mask)
    print(f"[capture] batch={batch_size} ...")
    gm, _ = _q3.capture_via_make_fx(model, example_inputs)
    ph_targets = [n.target for n in gm.graph.nodes if n.op == "placeholder"]
    return gm, ph_targets, example_inputs


def probe_fusion_apis():
    import torch, torch_npu  # noqa
    print("\n[probe] torch / torch_npu fusion APIs:")
    things = {
        "torch.compile": lambda: torch.compile,
        "torch._dynamo.list_backends": lambda: torch._dynamo.list_backends(),
        "torch.npu.make_graphed_callables": lambda: torch.npu.make_graphed_callables,
        "torch.npu.NPUGraph": lambda: torch.npu.NPUGraph,
        "torch_npu.contrib": lambda: __import__("torch_npu.contrib", fromlist=["*"]),
        "torch_npu.npu_fusion_attention": lambda: torch_npu.npu_fusion_attention,
        "torch_npu.npu_rms_norm": lambda: torch_npu.npu_rms_norm,
        "torch_npu.npu_swiglu": lambda: torch_npu.npu_swiglu,
        "torch_npu.npu.compile": lambda: torch_npu.npu.compile,
    }
    available = {}
    for name, get in things.items():
        try:
            obj = get()
            available[name] = obj
            extra = ""
            if name == "torch._dynamo.list_backends":
                extra = f" → {obj}"
            print(f"  {name:42s} OK{extra}")
        except Exception as e:
            print(f"  {name:42s} MISSING ({type(e).__name__})")
    return available


def bench_callable(fn, args, label, sync_fn):
    import time
    samples = []
    for _ in range(args.warmup):
        fn()
    sync_fn()
    for _ in range(args.iters):
        sync_fn()
        t0 = time.perf_counter()
        fn()
        sync_fn()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    med = samples[len(samples) // 2]
    print(f"  {label:32s} median = {med:8.2f} ms")
    return med


def main():
    args = parse_args()
    import torch
    import torch_npu  # noqa
    print(f"torch={torch.__version__}  torch_npu={torch_npu.__version__}")
    avail = probe_fusion_apis()

    cfg, model = _q3.build_model_from_config(args.model, dtype=torch.float32, scale=args.scale)
    _q3._patch_transformers_for_export()
    _q3._patch_moe_dense()

    gm10, ph10, ex10 = capture(args, args.n_requests, model, cfg)
    gm1, ph1, ex1 = capture(args, 1, model, cfg)
    gm10 = _migrate_gm(gm10, args)
    gm1 = _migrate_gm(gm1, args)

    keys = ["input_ids", "attention_mask"]
    npu_in10 = {pt: ex10[keys[i]].to(args.device) for i, pt in enumerate(ph10)}
    npu_in1 = {pt: ex1[keys[i]].to(args.device) for i, pt in enumerate(ph1)}

    sync_fn = torch.npu.synchronize

    print("\n=== Baseline (eager) ===")

    def _batch(): return gm10(*(npu_in10[t] for t in ph10))
    def _single(): return gm1(*(npu_in1[t] for t in ph1))
    def _stream():
        for _ in range(args.n_requests):
            _single()

    with torch.no_grad(), torch.device(args.device):
        eager_batch = bench_callable(_batch, args, "BATCH eager", sync_fn)
        eager_stream = bench_callable(_stream, args, "STREAM eager", sync_fn)

    # ── Approach: NPUGraph for b=1 (STREAM) and b=10 (BATCH) ───────────────
    print("\n=== NPUGraph capture ===")
    graph_stream = None
    graph_batch = None
    try:
        with torch.no_grad(), torch.device(args.device):
            # b=1 graph
            stream1 = torch.npu.Stream(device=args.device)
            with torch.npu.stream(stream1):
                for _ in range(3):
                    _single()
            torch.npu.synchronize()
            g1 = torch.npu.NPUGraph()
            with torch.npu.graph(g1, stream=stream1):
                _single()

            # b=10 graph
            stream10 = torch.npu.Stream(device=args.device)
            with torch.npu.stream(stream10):
                for _ in range(3):
                    _batch()
            torch.npu.synchronize()
            g10 = torch.npu.NPUGraph()
            with torch.npu.graph(g10, stream=stream10):
                _batch()
            print("  [graph] both captured")

            def _replay_stream():
                for _ in range(args.n_requests):
                    g1.replay()

            def _replay_batch():
                g10.replay()

            graph_stream = bench_callable(_replay_stream, args,
                                          "STREAM via NPUGraph replay", sync_fn)
            graph_batch = bench_callable(_replay_batch, args,
                                         "BATCH via NPUGraph replay", sync_fn)
            print(f"  STREAM speedup over eager: {eager_stream / graph_stream:.2f}×")
            print(f"  BATCH  speedup over eager: {eager_batch / graph_batch:.2f}×")
    except Exception:
        traceback.print_exc()

    # ── Approach: torch.compile via torchair (GE compile path) ─────────────
    print("\n=== torch.compile + torchair (GE compile path) ===")
    # Importing torchair registers `backend="npu"` and exposes get_npu_backend.
    try:
        from torch_npu.dynamo import torchair as ta
        config = ta.CompilerConfig()
        npu_backend = ta.get_npu_backend(compiler_config=config)
        print("  [torchair] backend obtained")

        compiled = torch.compile(gm1, backend=npu_backend, dynamic=False)
        with torch.no_grad(), torch.device(args.device):
            print("  [torchair] warming up (first call triggers GE compile — may take ~minutes)...")
            for i in range(args.warmup):
                compiled(*(npu_in1[t] for t in ph1))
                torch.npu.synchronize()
                print(f"    warmup iter {i+1}/{args.warmup} done")

            def _comp_stream():
                for _ in range(args.n_requests):
                    compiled(*(npu_in1[t] for t in ph1))

            comp_stream = bench_callable(_comp_stream, args,
                                         "STREAM compile(torchair-GE)", sync_fn)
            print(f"  speedup over eager STREAM: {eager_stream / comp_stream:.2f}×")
            if graph_stream:
                print(f"  speedup over NPUGraph STREAM: {graph_stream / comp_stream:.2f}×")
    except Exception as e:
        traceback.print_exc()
        print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}")

    # ── Approach: make_graphed_callables ───────────────────────────────────
    print("\n=== torch.npu.make_graphed_callables ===")
    try:
        sample = tuple(npu_in1[t] for t in ph1)
        graphed = torch.npu.make_graphed_callables(gm1, sample)
        with torch.no_grad(), torch.device(args.device):
            def _gc_stream():
                for _ in range(args.n_requests):
                    graphed(*sample)
            gc_stream = bench_callable(_gc_stream, args,
                                       "STREAM make_graphed_callables", sync_fn)
            print(f"  speedup over eager STREAM: {eager_stream / gc_stream:.2f}×")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")

    print("\n--- summary ---")
    print(f"  eager BATCH:                 {eager_batch:.2f} ms  (target — to beat)")
    print(f"  eager STREAM:                {eager_stream:.2f} ms  (10× b=1 forwards)")
    if graph_stream:
        print(f"  STREAM via NPUGraph replay:  {graph_stream:.2f} ms  "
              f"({eager_stream/graph_stream:.2f}× over eager STREAM)")


if __name__ == "__main__":
    main()
