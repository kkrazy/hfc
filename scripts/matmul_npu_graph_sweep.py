#!/usr/bin/env python3
"""NPUGraph vs eager — full batch-size sweep.

Same shape as matmul_batch_sweep.py (constant total work X*N*M, varying batch),
but each config runs both EAGER and NPUGraph and reports the comparison.

Per config (batch B):
  n_launches = X / B
  K = min(n_launches, --max-graph-size)            kernels captured per NPUGraph
  outer    = n_launches / K                        replays per iter

EAGER:        for _ in range(n_launches): y = x @ W
NPU_GRAPH:    g captured with K kernels; for _ in range(outer): g.replay()
"""
from __future__ import annotations
import argparse
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--total", type=int, default=4096)
    p.add_argument("--N", type=int, default=2048)
    p.add_argument("--M", type=int, default=2048)
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--device", default="npu:0")
    p.add_argument("--peak-tflops", type=float, default=376.0)
    p.add_argument("--max-graph-size", type=int, default=64)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--batches", type=str, default="1,4,16,64,256,1024,4096")
    return p.parse_args()


def bench_eager(x, W, n_launches, warmup, iters):
    """Per-iter timing: sync between iterations."""
    import torch

    def _iter():
        for _ in range(n_launches):
            _ = x @ W

    for _ in range(warmup):
        _iter()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        _iter()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return samples[len(samples) // 2]


def bench_streamed(x, W, n_launches, warmup, iters):
    """Streamed timing: submit iters * n_launches kernels back-to-back, sync once.
    Reports average per-iter time when the device never goes idle between iters.
    """
    import torch

    # warmup
    for _ in range(warmup):
        for _ in range(n_launches):
            _ = x @ W
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        for _ in range(n_launches):
            _ = x @ W
    torch.npu.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000
    return total_ms / iters


def bench_npu_graph(x, W, K, n_outer, warmup, iters, device):
    import torch
    NPUGraph = torch.npu.NPUGraph
    graph_ctx = torch.npu.graph

    stream = torch.npu.Stream(device=device)
    # Required warmup on capture stream BEFORE capture
    with torch.npu.stream(stream):
        for _ in range(3):
            _ = x @ W
    torch.npu.synchronize()

    g = NPUGraph()
    with graph_ctx(g, stream=stream):
        for _ in range(K):
            _ = x @ W

    def _iter():
        for _ in range(n_outer):
            g.replay()

    for _ in range(warmup):
        _iter()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        _iter()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return samples[len(samples) // 2]


def main():
    args = parse_args()
    import torch
    import torch_npu  # noqa
    print(f"torch={torch.__version__}  torch_npu={torch_npu.__version__}  "
          f"npu_count={torch.npu.device_count()}")
    print(f"config: total={args.total} N={args.N} M={args.M} dtype={args.dtype}  "
          f"max_graph_size={args.max_graph_size}")

    dtype = getattr(torch, args.dtype)
    W = torch.randn(args.N, args.M, dtype=dtype, device=args.device)
    flops_per_iter = 2 * args.total * args.N * args.M
    print(f"\nPer iteration: {flops_per_iter / 1e9:.2f} GFLOPs (constant across configs)")

    print()
    print(f"{'batch':>6} {'launches':>8} {'K':>5} {'outer':>5} "
          f"{'eager_ms':>10} {'strm_ms':>10} {'graph_ms':>10} "
          f"{'eag_us/k':>10} {'strm_us/k':>11} {'gr_us/k':>10} "
          f"{'eag_TF':>8} {'strm_TF':>8} {'gr_TF':>8} "
          f"{'strm_util':>10} {'gr_util':>9}")
    print("-" * 142)

    batches = [int(b) for b in args.batches.split(",")]
    rows = []
    for batch in batches:
        if args.total % batch != 0:
            print(f"  skip batch={batch}: total {args.total} not divisible")
            continue
        n_launches = args.total // batch

        # Choose K = greatest divisor of n_launches that is <= max_graph_size
        K = min(n_launches, args.max_graph_size)
        while n_launches % K != 0:
            K -= 1
        n_outer = n_launches // K

        x = torch.randn(batch, args.N, dtype=dtype, device=args.device)

        eager_med = bench_eager(x, W, n_launches, args.warmup, args.iters)
        streamed_med = bench_streamed(x, W, n_launches, args.warmup, args.iters)
        graph_med = bench_npu_graph(x, W, K, n_outer, args.warmup, args.iters, args.device)

        eager_us_per = eager_med * 1000 / n_launches
        streamed_us_per = streamed_med * 1000 / n_launches
        graph_us_per = graph_med * 1000 / n_launches
        eager_tf = flops_per_iter / (eager_med / 1000) / 1e12
        streamed_tf = flops_per_iter / (streamed_med / 1000) / 1e12
        graph_tf = flops_per_iter / (graph_med / 1000) / 1e12
        streamed_util = 100.0 * streamed_tf / args.peak_tflops
        graph_util = 100.0 * graph_tf / args.peak_tflops

        print(f"{batch:>6} {n_launches:>8} {K:>5} {n_outer:>5} "
              f"{eager_med:>10.2f} {streamed_med:>10.2f} {graph_med:>10.2f} "
              f"{eager_us_per:>10.2f} {streamed_us_per:>11.2f} {graph_us_per:>10.2f} "
              f"{eager_tf:>8.1f} {streamed_tf:>8.1f} {graph_tf:>8.1f} "
              f"{streamed_util:>9.1f}% {graph_util:>8.1f}%")
        rows.append({
            "batch": batch, "launches": n_launches, "K": K, "outer": n_outer,
            "eager_ms": eager_med, "streamed_ms": streamed_med, "graph_ms": graph_med,
        })

    print()
    print("Legend:")
    print("  K          = kernels captured per NPUGraph (= min(launches, max_graph_size))")
    print("  outer      = number of g.replay() calls per iter")
    print("  eager_ms   = per-iter time, sync between iters (default eager)")
    print("  strm_ms    = streamed: iters*launches kernels back-to-back, sync only at start/end;")
    print("               reports avg per-iter (proxy for steady-state under continuous load)")
    print("  graph_ms   = NPUGraph: K kernels captured, replayed `outer` times per iter")


if __name__ == "__main__":
    main()
