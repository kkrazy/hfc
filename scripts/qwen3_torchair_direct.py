#!/usr/bin/env python3
"""Compile the original Qwen3 module directly via torchair (no make_fx).

Skips our capture+rewrite pipeline; lets torchair handle device placement,
FakeTensor conversion, and GE compile from the eager module.

Compares:
  EAGER:    eager Qwen3 model.forward
  TORCHAIR: torch.compile(model, backend=npu_backend) — GE compile path
"""
from __future__ import annotations
import sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_qwen3_moe as _q3  # noqa: E402

import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--scale", type=float, default=0.1)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--n-requests", type=int, default=10)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=10)
    return p.parse_args()


def bench(fn, label, warmup, iters):
    import torch
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    med = samples[len(samples) // 2]
    print(f"  {label:38s} median = {med:8.2f} ms")
    return med


def main():
    args = parse_args()
    import torch
    import torch_npu  # noqa
    from torch_npu.dynamo import torchair as ta  # registers backend
    print(f"torch={torch.__version__}  torch_npu={torch_npu.__version__}")

    cfg, model = _q3.build_model_from_config(args.model, dtype=torch.float32, scale=args.scale)
    _q3._patch_transformers_for_export()
    _q3._patch_moe_dense()

    # Move whole module to NPU + cast to fp16 BEFORE compile
    dtype = getattr(torch, args.dtype)
    model = model.to(args.device).to(dtype).eval()

    n_req = args.n_requests
    input_ids_b1 = torch.randint(0, cfg.vocab_size, (1, args.seq_len), device=args.device)
    attention_mask_b1 = torch.ones(1, args.seq_len, dtype=torch.long, device=args.device)
    input_ids_b10 = torch.randint(0, cfg.vocab_size, (n_req, args.seq_len), device=args.device)
    attention_mask_b10 = torch.ones(n_req, args.seq_len, dtype=torch.long, device=args.device)

    def _eager_b1():
        with torch.no_grad():
            return model(input_ids=input_ids_b1, attention_mask=attention_mask_b1)
    def _eager_b10():
        with torch.no_grad():
            return model(input_ids=input_ids_b10, attention_mask=attention_mask_b10)
    def _eager_stream():
        for _ in range(n_req):
            _eager_b1()

    print("\n=== Baseline (eager direct on HF model) ===")
    eb1 = bench(_eager_b1, "eager b=1 single", args.warmup, args.iters)
    eb10 = bench(_eager_b10, "eager b=10 BATCH", args.warmup, args.iters)
    estream = bench(_eager_stream, "eager STREAM (10×b=1)", args.warmup, args.iters)

    # ── torchair compile ───────────────────────────────────────────────────
    print("\n=== torchair (GE compile of full HF model) ===")
    try:
        config = ta.CompilerConfig()
        npu_backend = ta.get_npu_backend(compiler_config=config)
        compiled = torch.compile(model, backend=npu_backend, dynamic=False)

        def _ta_b1():
            with torch.no_grad():
                return compiled(input_ids=input_ids_b1, attention_mask=attention_mask_b1)
        def _ta_b10():
            with torch.no_grad():
                return compiled(input_ids=input_ids_b10, attention_mask=attention_mask_b10)
        def _ta_stream():
            for _ in range(n_req):
                _ta_b1()

        print("  warmup b=1 (first call triggers GE compile — may take minutes)...")
        tb1 = bench(_ta_b1, "torchair b=1 single", args.warmup, args.iters)
        print("  warmup b=10 ...")
        tb10 = bench(_ta_b10, "torchair b=10 BATCH", args.warmup, args.iters)
        tstream = bench(_ta_stream, "torchair STREAM (10×b=1)", args.warmup, args.iters)

        print("\n--- summary ---")
        print(f"  eager b=1:          {eb1:.2f}  →  torchair: {tb1:.2f}  ({eb1/tb1:.2f}×)")
        print(f"  eager BATCH b=10:   {eb10:.2f}  →  torchair: {tb10:.2f}  ({eb10/tb10:.2f}×)")
        print(f"  eager STREAM:       {estream:.2f}  →  torchair: {tstream:.2f}  ({estream/tstream:.2f}×)")
    except Exception as e:
        traceback.print_exc()
        print(f"\n  FAILED: {type(e).__name__}: {str(e)[:400]}")


if __name__ == "__main__":
    main()
