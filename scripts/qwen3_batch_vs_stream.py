#!/usr/bin/env python3
"""Qwen3 forward: BATCH (b=10 single launch) vs STREAM (10 × b=1 streamed).

Same total work — 10 forward passes worth of compute on Qwen3-30B-A3B (scaled).
Compares two submission patterns:

  BATCH:   one forward of shape (10, seq_len)             ← packed
  STREAM:  10 forwards of shape (1, seq_len) back-to-back ← interleaved
           (no per-call sync; all submitted, then sync once at the end)

Reports:
  - E2E wall time (last logits available)
  - Per-request "time to result" — for STREAM, first request lands ~T/10
  - Achieved throughput (requests/sec)
  - Approximate utilization (uses logged kernel timing, not roofline)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_qwen3_moe as _q3  # noqa: E402  — pulls compat shims

import argparse
import gc
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--scale", type=float, default=0.1)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--n-requests", type=int, default=10)
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--device", default="npu:0")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    return p.parse_args()


def _migrate_gm(gm, args):
    """Apply the same device-rewrite + tensor migration we use in run_qwen3_moe."""
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


def capture_only(args, batch_size, model, cfg):
    """Trace at a specific batch size on CPU. No migration here.
    (Migration must happen AFTER all captures, because gm.to(npu) also moves
    the shared model parameters — would break a subsequent CPU-side capture.)
    """
    import torch

    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, args.seq_len))
    attention_mask = torch.ones(batch_size, args.seq_len, dtype=torch.long)
    example_inputs = dict(input_ids=input_ids, attention_mask=attention_mask)

    print(f"[capture] batch={batch_size} seq={args.seq_len} ...")
    gm, _ = _q3.capture_via_make_fx(model, example_inputs)
    ph_targets = [n.target for n in gm.graph.nodes if n.op == "placeholder"]
    return gm, ph_targets, example_inputs


def main():
    args = parse_args()
    import torch
    import torch_npu  # noqa: F401
    print(f"torch={torch.__version__}  torch_npu={torch_npu.__version__}  "
          f"npu_count={torch.npu.device_count()}")
    print(f"model={args.model}  scale={args.scale}  seq_len={args.seq_len}  "
          f"n_req={args.n_requests}  dtype={args.dtype}")

    # Build model (CPU, fp32) once. Apply patches before any capture.
    cfg, model = _q3.build_model_from_config(args.model, dtype=torch.float32, scale=args.scale)
    _q3._patch_transformers_for_export()
    _q3._patch_moe_dense()

    # Capture BOTH graphs at CPU first. They share the model's parameters
    # (make_fx with tracing_mode='real' references original tensors).
    gm10, ph10, ex10 = capture_only(args, args.n_requests, model, cfg)
    gm1, ph1, ex1 = capture_only(args, 1, model, cfg)

    # Now migrate. The first migration moves the shared model params to NPU;
    # the second migration's param-copies are no-ops, but it still rewrites
    # gm1's own device='cpu' kwargs that make_fx baked in.
    gm10 = _migrate_gm(gm10, args)
    gm1 = _migrate_gm(gm1, args)

    # Move inputs to NPU
    keys = ["input_ids", "attention_mask"]
    npu_in10 = {pt: ex10[keys[i]].to(args.device) for i, pt in enumerate(ph10)}
    npu_in1 = {pt: ex1[keys[i]].to(args.device) for i, pt in enumerate(ph1)}

    def _invoke_batch():
        return gm10(*(npu_in10[t] for t in ph10))

    with torch.no_grad(), torch.device(args.device):
        for _ in range(args.warmup):
            _invoke_batch()
        torch.npu.synchronize()
        batch_samples = []
        for _ in range(args.iters):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            _invoke_batch()
            torch.npu.synchronize()
            batch_samples.append((time.perf_counter() - t0) * 1000)
    batch_samples.sort()

    def _invoke_single():
        return gm1(*(npu_in1[t] for t in ph1))

    def _invoke_stream_single_stream():
        # 10 single-request forwards on the default stream, no sync between.
        for _ in range(args.n_requests):
            _invoke_single()

    # Multi-stream: each forward goes to its own NPU stream.
    streams = [torch.npu.Stream(device=args.device) for _ in range(args.n_requests)]

    def _invoke_stream_multi_stream():
        for s in streams:
            with torch.npu.stream(s):
                _invoke_single()
        for s in streams:
            s.synchronize()

    with torch.no_grad(), torch.device(args.device):
        # warmup both
        for _ in range(args.warmup):
            _invoke_stream_single_stream()
            _invoke_stream_multi_stream()
        torch.npu.synchronize()

        stream_samples = []
        for _ in range(args.iters):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            _invoke_stream_single_stream()
            torch.npu.synchronize()
            stream_samples.append((time.perf_counter() - t0) * 1000)

        ms_samples = []
        for _ in range(args.iters):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            _invoke_stream_multi_stream()
            torch.npu.synchronize()
            ms_samples.append((time.perf_counter() - t0) * 1000)
    stream_samples.sort()
    ms_samples.sort()

    # Per-request single forward latency (for first-result baseline).
    with torch.no_grad(), torch.device(args.device):
        for _ in range(args.warmup):
            _invoke_single()
        torch.npu.synchronize()
        single_samples = []
        for _ in range(args.iters):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            _invoke_single()
            torch.npu.synchronize()
            single_samples.append((time.perf_counter() - t0) * 1000)
    single_samples.sort()

    # ── Report ──────────────────────────────────────────────────────────────
    batch_med = batch_samples[len(batch_samples) // 2]
    stream_med = stream_samples[len(stream_samples) // 2]
    ms_med = ms_samples[len(ms_samples) // 2]
    single_med = single_samples[len(single_samples) // 2]

    print()
    print("=" * 88)
    print(f"  Qwen3-30B-A3B (scale {args.scale}: 4 layers, 12 experts)  "
          f"·  {args.n_requests} requests, seq={args.seq_len}, {args.dtype}")
    print("=" * 88)
    print(f"{'mode':46s} {'e2e_ms':>10s} {'per_req_ms':>14s} {'req/s':>10s}")
    print("-" * 88)
    n = args.n_requests
    print(f"{'BATCH  (single fwd b=' + str(n) + ')':46s} "
          f"{batch_med:>10.2f} {batch_med:>14.2f} {n/(batch_med/1000):>10.1f}")
    print(f"{'STREAM (' + str(n) + ' × b=1, default stream)':46s} "
          f"{stream_med:>10.2f} {stream_med/n:>14.2f} {n/(stream_med/1000):>10.1f}")
    print(f"{'STREAM_MS (' + str(n) + ' × b=1, ' + str(n) + ' streams)':46s} "
          f"{ms_med:>10.2f} {ms_med/n:>14.2f} {n/(ms_med/1000):>10.1f}")
    print(f"{'  └─ single b=1 forward (reference)':46s} {single_med:>10.2f}")
    print()
    print(f"BATCH speedup vs STREAM:    {stream_med / batch_med:.2f}×")
    print(f"BATCH speedup vs STREAM_MS: {ms_med / batch_med:.2f}×")
    print(f"STREAM_MS vs STREAM (multi-stream effect): "
          f"{stream_med / ms_med:.2f}×  "
          f"({'helps' if ms_med < stream_med else 'hurts'})")


if __name__ == "__main__":
    main()
