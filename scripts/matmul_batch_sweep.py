#!/usr/bin/env python3
"""Sweep matmul batch size to characterize launch overhead vs throughput.

Same total work — X rows of (N) @ (N, M) — split into different submission
batch sizes:

  batch=1:    X separate matmuls of (1, N) @ (N, M)         ← max launches
  batch=k:    X/k matmuls of (k, N) @ (N, M)
  batch=X:    1 matmul of (X, N) @ (N, M)                   ← min launches

Total FLOPs per iteration = 2 * X * N * M (constant across configs).
What changes is launches/iter and per-launch arithmetic intensity.

Reports per-config:
  launches/iter, mean_ms, p50_ms, achieved TFLOPS, utilization%

Utilization% = TFLOPS / peak_TFLOPS  (default peak = 256 for Ascend 910 fp16;
override with --peak-tflops for 910B/9392 etc.).
"""
from __future__ import annotations
import argparse
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--total", type=int, default=4096,
                   help="total rows X (must be a power of 2 for clean halving)")
    p.add_argument("--N", type=int, default=2048, help="K dim (input hidden)")
    p.add_argument("--M", type=int, default=2048, help="N dim (output hidden)")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    p.add_argument("--device", default="npu:0")
    p.add_argument("--peak-tflops", type=float, default=256.0,
                   help="device peak (fp16=256 on 910A; ~376 on 910B/9392)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--include-h2d", action="store_true",
                   help="also report a config that re-uploads input every launch (H2D-bound)")
    return p.parse_args()


def main():
    args = parse_args()
    import torch
    is_npu = args.device.startswith("npu")
    if is_npu:
        import torch_npu  # noqa
    print(f"torch={torch.__version__}")
    if is_npu:
        print(f"torch_npu={torch_npu.__version__}  npu_count={torch.npu.device_count()}")
    print(f"config: total={args.total}  N={args.N}  M={args.M}  "
          f"dtype={args.dtype}  device={args.device}")

    dtype = getattr(torch, args.dtype)

    # Persistent weight matrix on device (same across all configs).
    W = torch.randn(args.N, args.M, dtype=dtype, device=args.device)
    # Persistent maximal input on device — slice it for each config.
    X_full = torch.randn(args.total, args.N, dtype=dtype, device=args.device)

    # Build a sweep of batch sizes that cleanly divide total.
    batches = []
    bs = 1
    while bs <= args.total:
        if args.total % bs == 0:
            batches.append(bs)
        bs *= 2
    if args.total not in batches:
        batches.append(args.total)
    batches = sorted(set(batches))

    flops_per_iter = 2 * args.total * args.N * args.M
    bytes_per_iter_w = args.N * args.M * dtype_bytes(dtype)  # W is reused
    bytes_per_iter_x = args.total * args.N * dtype_bytes(dtype)
    bytes_per_iter_y = args.total * args.M * dtype_bytes(dtype)

    print(f"\nPer iteration: {flops_per_iter / 1e9:.2f} GFLOPs, "
          f"weights={bytes_per_iter_w/1024**2:.1f}MB (read), "
          f"act_in={bytes_per_iter_x/1024**2:.1f}MB, "
          f"act_out={bytes_per_iter_y/1024**2:.1f}MB")
    print(f"Sweep: {len(batches)} configs from batch=1 to batch={args.total}")
    print()
    print(f"{'batch':>6} {'launches':>10} {'mean_ms':>10} {'p50_ms':>10} "
          f"{'min_ms':>10} {'TFLOPS':>10} {'util%':>8} {'GB/s':>10}")
    print("-" * 78)

    results = []
    for batch in batches:
        n_launches = args.total // batch
        x = X_full[:batch].contiguous()  # take a slice; reused across launches in inner loop

        def _one_iter():
            for _ in range(n_launches):
                # @ creates a fresh output each call — measures full kernel cost
                _y = x @ W

        for _ in range(args.warmup):
            _one_iter()
        if is_npu:
            torch.npu.synchronize()

        samples = []
        for _ in range(args.iters):
            if is_npu:
                torch.npu.synchronize()
            t0 = time.perf_counter()
            _one_iter()
            if is_npu:
                torch.npu.synchronize()
            samples.append((time.perf_counter() - t0) * 1000)

        samples.sort()
        mean_ms = sum(samples) / len(samples)
        p50_ms = samples[len(samples) // 2]
        min_ms = samples[0]
        tflops = flops_per_iter / (mean_ms / 1000) / 1e12
        util_pct = 100.0 * tflops / args.peak_tflops
        # Effective HBM bandwidth: we read W once per launch (×n_launches),
        # read X once total, write Y once total → approximates per-iter bytes touched.
        bytes_per_iter = (bytes_per_iter_w * n_launches
                          + bytes_per_iter_x + bytes_per_iter_y)
        gbs = bytes_per_iter / (mean_ms / 1000) / 1e9

        print(f"{batch:>6} {n_launches:>10} {mean_ms:>10.2f} {p50_ms:>10.2f} "
              f"{min_ms:>10.2f} {tflops:>10.2f} {util_pct:>7.1f}% {gbs:>9.1f}")
        results.append({
            "batch": batch, "launches": n_launches,
            "mean_ms": mean_ms, "p50_ms": p50_ms, "min_ms": min_ms,
            "tflops": tflops, "util_pct": util_pct, "gbs": gbs,
        })

    # Summary: pick winner and the small-batch laggard.
    best = max(results, key=lambda r: r["tflops"])
    worst = min(results, key=lambda r: r["tflops"])
    speedup = best["tflops"] / worst["tflops"]
    launch_amp = worst["launches"] / max(best["launches"], 1)
    print()
    print(f"Best:  batch={best['batch']:>6}  {best['tflops']:.1f} TFLOPS  "
          f"({best['util_pct']:.1f}% util)  {best['mean_ms']:.2f} ms/iter")
    print(f"Worst: batch={worst['batch']:>6}  {worst['tflops']:.2f} TFLOPS  "
          f"({worst['util_pct']:.2f}% util)  {worst['mean_ms']:.2f} ms/iter")
    print(f"  → {launch_amp:.0f}× more launches in worst case, "
          f"{speedup:.1f}× throughput gap.")


def dtype_bytes(dt):
    import torch
    return {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}.get(dt, 2)


if __name__ == "__main__":
    main()
