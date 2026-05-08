#!/usr/bin/env python3
"""Find optimal K for serving N=100 requests via torchair on Qwen3.

Splits N requests into K batches of N/K, torchair-compiles a forward at
each batch size, runs K forwards back-to-back (streamed, no per-batch sync).

Reports per K:
  - e2e wall time (last batch completes)
  - per-request latency (e2e / N)
  - req/s throughput
  - TTFT (time-to-first-batch — when first N/K reqs complete)

Identifies the K that maximizes throughput.
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
    p.add_argument("--n-requests", type=int, default=100)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--ks", default="1,2,5,10,20,50,100",
                   help="K values; N/K must be an integer batch size")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    import torch_npu  # noqa
    from torch_npu.dynamo import torchair as ta
    print(f"torch={torch.__version__}  torch_npu={torch_npu.__version__}")
    print(f"model={args.model}  scale={args.scale}  N={args.n_requests}")

    cfg, model = _q3.build_model_from_config(args.model, dtype=torch.float32, scale=args.scale)
    _q3._patch_transformers_for_export()
    _q3._patch_moe_dense()

    dtype = getattr(torch, args.dtype)
    model = model.to(args.device).to(dtype).eval()

    config = ta.CompilerConfig()
    npu_backend = ta.get_npu_backend(compiler_config=config)
    compiled = torch.compile(model, backend=npu_backend, dynamic=False)

    Ks = [int(k) for k in args.ks.split(",")]
    Ks = sorted([k for k in Ks if args.n_requests % k == 0])

    print(f"\nSweep K values: {Ks}")
    print(f"\n{'K':>4} {'B':>5} {'one_fwd_ms':>11} {'e2e_ms':>10} "
          f"{'per_req_ms':>11} {'req/s':>10} {'ttft_ms':>10}")
    print("-" * 82)

    results = []
    for K in Ks:
        B = args.n_requests // K
        try:
            input_ids = torch.randint(0, cfg.vocab_size, (B, args.seq_len), device=args.device)
            attn_mask = torch.ones(B, args.seq_len, dtype=torch.long, device=args.device)

            def _one():
                with torch.no_grad():
                    return compiled(input_ids=input_ids, attention_mask=attn_mask)

            def _full_iter():
                for _ in range(K):
                    _one()

            print(f"  [K={K}, B={B}] warmup+compile ...", flush=True)
            for _ in range(args.warmup):
                _full_iter()
            torch.npu.synchronize()

            # Single-forward time (= TTFT for the first batch in STREAM-of-batches)
            ones = []
            for _ in range(args.iters):
                torch.npu.synchronize()
                t0 = time.perf_counter()
                _one()
                torch.npu.synchronize()
                ones.append((time.perf_counter() - t0) * 1000)
            ones.sort()
            one_med = ones[len(ones) // 2]

            # Full-K time
            samples = []
            for _ in range(args.iters):
                torch.npu.synchronize()
                t0 = time.perf_counter()
                _full_iter()
                torch.npu.synchronize()
                samples.append((time.perf_counter() - t0) * 1000)
            samples.sort()
            med = samples[len(samples) // 2]

            per_req = med / args.n_requests
            rps = args.n_requests / (med / 1000)

            print(f"  {K:>4} {B:>5} {one_med:>11.2f} {med:>10.2f} "
                  f"{per_req:>11.3f} {rps:>10.1f} {one_med:>10.2f}")
            results.append(dict(K=K, B=B, one_med=one_med, med=med,
                                per_req=per_req, rps=rps))
        except Exception as e:
            print(f"  [K={K}, B={B}] FAILED: {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc()

    if not results:
        return
    best = max(results, key=lambda r: r["rps"])
    print()
    print("=" * 82)
    print(f"  Best throughput: K={best['K']}  B={best['B']}")
    print(f"  E2E (100 reqs):     {best['med']:.2f} ms")
    print(f"  Throughput:         {best['rps']:.1f} req/s")
    print(f"  Per-request:        {best['per_req']:.3f} ms")
    print(f"  TTFT (first batch): {best['one_med']:.2f} ms")
    print("=" * 82)
    # Latency vs throughput frontier
    print("\nLatency / throughput frontier:")
    for r in sorted(results, key=lambda r: r["K"]):
        marker = " ★" if r is best else "  "
        print(f"  K={r['K']:>3}  B={r['B']:>3}  "
              f"req/s={r['rps']:>7.1f}  TTFT={r['one_med']:>7.2f}ms  "
              f"e2e={r['med']:>7.2f}ms{marker}")


if __name__ == "__main__":
    main()
