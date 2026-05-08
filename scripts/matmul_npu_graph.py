#!/usr/bin/env python3
"""Eager vs torch.npu.NPUGraph (kernel-level capture, single replay submission).

Different from make_fx — NPUGraph captures the actual *kernels* the device runs,
not the Python trace. Replay is one device-side dispatch, no per-kernel host
overhead. The CUDA Graphs analogue.

Pattern:
  warmup on stream → torch.npu.graph capture context → replay()

For 4096 total kernels, capture K per graph, replay (4096/K) times in eager loop.
Compare against:
  EAGER:        eager loop of 4096 `x @ W`
  CAPTURED_FX:  make_fx graph (Python forward calling aten.mm 64x) replayed 64x
  NPU_GRAPH:    NPUGraph of 64 kernels replayed 64x

The first two have host-side overhead per launch; NPU_GRAPH has overhead only
per replay() call.
"""
from __future__ import annotations
import argparse
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-launches", type=int, default=4096)
    p.add_argument("--N", type=int, default=2048)
    p.add_argument("--M", type=int, default=2048)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--graph-size", type=int, default=64,
                   help="K kernels captured per graph; outer loop replays (n_launches/K) times")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=15)
    return p.parse_args()


def probe_graph_api():
    """Find which graph-capture API torch_npu exposes."""
    import torch
    import torch_npu  # noqa
    cands = []
    for path in ["torch.npu.NPUGraph", "torch.npu.graph",
                 "torch_npu.npu.NPUGraph", "torch_npu.npu.graph",
                 "torch.npu.make_graphed_callables"]:
        parts = path.split(".")
        obj = None
        try:
            obj = __import__(parts[0])
            for p in parts[1:]:
                obj = getattr(obj, p)
            cands.append((path, "OK"))
        except (ImportError, AttributeError) as e:
            cands.append((path, f"missing ({type(e).__name__})"))
    return cands


def main():
    args = parse_args()
    import torch
    import torch_npu  # noqa: F401
    print(f"torch={torch.__version__}  torch_npu={torch_npu.__version__}")

    print("\n[probe] torch_npu graph APIs:")
    for path, status in probe_graph_api():
        print(f"  {path:40s} {status}")

    # Pick the right symbols. torch_npu mirrors torch.cuda's namespace.
    NPUGraph = getattr(torch.npu, "NPUGraph", None)
    graph_ctx = getattr(torch.npu, "graph", None)
    if NPUGraph is None or graph_ctx is None:
        print("\n[ERROR] torch.npu.NPUGraph / torch.npu.graph not found in this torch_npu.")
        print("        Try probing torch_npu.npu.NPUGraph or skip this test.")
        return 2

    if args.n_launches % args.graph_size != 0:
        raise SystemExit(f"n_launches must be divisible by graph_size")
    n_outer = args.n_launches // args.graph_size

    dtype = getattr(torch, args.dtype)
    W = torch.randn(args.N, args.M, dtype=dtype, device=args.device)
    x = torch.randn(args.batch, args.N, dtype=dtype, device=args.device)
    print(f"\nconfig: batch={args.batch} N={args.N} M={args.M} "
          f"n_launches={args.n_launches} graph_size={args.graph_size} dtype={args.dtype}")

    # ---- EAGER baseline ----
    def _eager():
        for _ in range(args.n_launches):
            _ = x @ W

    for _ in range(args.warmup):
        _eager()
    torch.npu.synchronize()
    eager_samples = []
    for _ in range(args.iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        _eager()
        torch.npu.synchronize()
        eager_samples.append((time.perf_counter() - t0) * 1000)
    eager_samples.sort()
    eager_med = eager_samples[len(eager_samples) // 2]

    # ---- NPU_GRAPH ----
    print(f"\n[capture] capturing {args.graph_size}-kernel NPUGraph ...")
    stream = torch.npu.Stream(device=args.device)

    # Warmup on the capture stream — required before capture (CUDA-Graphs convention).
    with torch.npu.stream(stream):
        for _ in range(3):
            _ = x @ W
    torch.npu.synchronize()

    g = NPUGraph()
    captured_out = None
    try:
        with graph_ctx(g, stream=stream):
            out = x @ W
            for _ in range(args.graph_size - 1):
                out = x @ W
            captured_out = out
    except TypeError:
        # Some torch_npu versions: graph_ctx(g) without stream kwarg
        with graph_ctx(g):
            out = x @ W
            for _ in range(args.graph_size - 1):
                out = x @ W
            captured_out = out
    print(f"[capture] graph captured")

    def _replay():
        for _ in range(n_outer):
            g.replay()

    for _ in range(args.warmup):
        _replay()
    torch.npu.synchronize()
    graph_samples = []
    for _ in range(args.iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        _replay()
        torch.npu.synchronize()
        graph_samples.append((time.perf_counter() - t0) * 1000)
    graph_samples.sort()
    graph_med = graph_samples[len(graph_samples) // 2]

    # ---- report ----
    eager_us_per = eager_med * 1000 / args.n_launches
    graph_us_per = graph_med * 1000 / args.n_launches
    speedup = eager_med / graph_med if graph_med > 0 else 0
    overhead_per_replay = (graph_med - eager_med * args.graph_size / args.n_launches)

    print()
    print("=" * 64)
    print(f"  Eager vs NPUGraph (batch={args.batch}, n_launches={args.n_launches})")
    print("=" * 64)
    print(f"{'mode':16s} {'total_ms':>10s} {'us/launch':>12s} {'launches':>10s}")
    print("-" * 64)
    print(f"{'EAGER':16s} {eager_med:>10.2f} {eager_us_per:>12.2f} {args.n_launches:>10d}")
    print(f"{'NPU_GRAPH':16s} {graph_med:>10.2f} {graph_us_per:>12.2f} "
          f"{n_outer} replay × {args.graph_size}")
    print(f"\nspeedup:                   {speedup:.2f}x")
    print(f"per-launch in graph:       {graph_us_per:.2f} us  "
          f"(eager: {eager_us_per:.2f} us)")
    print(f"per-replay() overhead est: {graph_med * 1000 / n_outer:.2f} us "
          f"(submitting {args.graph_size} kernels)")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
