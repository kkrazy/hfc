#!/usr/bin/env python3
"""Eager vs captured-graph matmul launch overhead.

For batch=1 (where launch+dispatch overhead dominates wall time), compare:

  EAGER:    Python loop of `x @ W` — every call walks the dispatcher
            (autograd mode, __torch_function__, type promotion, etc.).
  CAPTURED: a make_fx graph containing K aten.mm nodes; the generated forward
            calls torch.ops.aten.mm.default directly, skipping the dispatcher
            walk on every call.

To bound graph size (very large graphs hit Python local-var limits), we capture
a graph of K matmuls and call it `n_launches / K` times in an outer Python loop.
Outer-loop overhead is one dispatch per K kernels — meaningful reduction
without the codegen risk of capturing all 4096 in one graph.
"""
from __future__ import annotations
import argparse
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-launches", type=int, default=4096,
                   help="total matmul launches per iteration (must be divisible by --graph-size)")
    p.add_argument("--N", type=int, default=2048)
    p.add_argument("--M", type=int, default=2048)
    p.add_argument("--batch", type=int, default=1,
                   help="rows per matmul (use small to expose launch overhead)")
    p.add_argument("--graph-size", type=int, default=64,
                   help="K = matmuls captured per make_fx graph; outer-loop = n_launches/K")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=15)
    return p.parse_args()


def _move_gm_to_device(gm, target_dev, dtype):
    """Same fix as run_qwen3_moe.py: device='cpu' kwargs baked in by make_fx,
    plus bare tensor attrs that .to() missed."""
    import torch
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
    gm = gm.to(target_dev).to(dtype)
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


def main():
    args = parse_args()
    import torch
    import torch.nn as nn
    from torch.fx.experimental.proxy_tensor import make_fx

    is_npu = args.device.startswith("npu")
    if is_npu:
        import torch_npu  # noqa
    print(f"torch={torch.__version__}")
    if is_npu:
        print(f"torch_npu={torch_npu.__version__}")
    print(f"config: batch={args.batch} N={args.N} M={args.M} n_launches={args.n_launches} "
          f"graph_size={args.graph_size} dtype={args.dtype}")

    if args.n_launches % args.graph_size != 0:
        raise SystemExit(f"n_launches {args.n_launches} not divisible by graph_size {args.graph_size}")
    n_outer = args.n_launches // args.graph_size

    dtype = getattr(torch, args.dtype)
    sync = torch.npu.synchronize if is_npu else (lambda: None)

    W = torch.randn(args.N, args.M, dtype=dtype, device=args.device)
    x = torch.randn(args.batch, args.N, dtype=dtype, device=args.device)

    # ---- EAGER ----
    def _eager_iter():
        for _ in range(args.n_launches):
            _ = x @ W

    for _ in range(args.warmup):
        _eager_iter()
    sync()
    eager_samples = []
    for _ in range(args.iters):
        sync()
        t0 = time.perf_counter()
        _eager_iter()
        sync()
        eager_samples.append((time.perf_counter() - t0) * 1000)
    eager_samples.sort()
    eager_med = eager_samples[len(eager_samples) // 2]
    eager_us_per = eager_med * 1000 / args.n_launches

    # ---- CAPTURED via make_fx ----
    class KMatmul(nn.Module):
        def __init__(self, k, W):
            super().__init__()
            self.k = k
            self.W = nn.Parameter(W, requires_grad=False)

        def forward(self, x):
            out = x
            for _ in range(self.k):
                out = x @ self.W   # independent of `out`, but assigned for traceability
            return out

    # Trace in fp32 on CPU — CPU has no fp16 matmul kernel ("addmm_impl_cpu_
    # not implemented for 'Half'"). Cast back to fp16 after moving to NPU.
    print(f"\n[capture] tracing graph_size={args.graph_size} matmul nodes via make_fx (fp32 CPU) ...")
    W_trace = W.detach().cpu().float()
    x_trace = x.detach().cpu().float()
    m = KMatmul(args.graph_size, W_trace).cpu().eval()
    gm = make_fx(m, tracing_mode="real")(x_trace)
    n_aten_mm = sum(1 for n in gm.graph.nodes
                    if n.op == "call_function" and "mm" in str(n.target))
    print(f"[capture] graph: {len(list(gm.graph.nodes))} nodes ({n_aten_mm} mm-family ops)")

    target_dev = torch.device(args.device)
    gm = _move_gm_to_device(gm, target_dev, dtype)

    def _captured_iter():
        with torch.no_grad(), torch.device(args.device):
            for _ in range(n_outer):
                _ = gm(x)

    for _ in range(args.warmup):
        _captured_iter()
    sync()
    cap_samples = []
    for _ in range(args.iters):
        sync()
        t0 = time.perf_counter()
        _captured_iter()
        sync()
        cap_samples.append((time.perf_counter() - t0) * 1000)
    cap_samples.sort()
    cap_med = cap_samples[len(cap_samples) // 2]
    cap_us_per = cap_med * 1000 / args.n_launches

    speedup = eager_med / cap_med
    overhead_saved_us = eager_us_per - cap_us_per

    print()
    print("=" * 64)
    print(f"  Eager vs Captured-Graph (batch={args.batch}, n_launches={args.n_launches})")
    print("=" * 64)
    print(f"{'mode':12s} {'total_ms':>10s} {'us/launch':>12s} {'launches/iter':>16s}")
    print("-" * 64)
    print(f"{'EAGER':12s} {eager_med:>10.2f} {eager_us_per:>12.2f} {args.n_launches:>16d}")
    print(f"{'CAPTURED':12s} {cap_med:>10.2f} {cap_us_per:>12.2f} {n_outer} outer × {args.graph_size}")
    print(f"\nspeedup:                  {speedup:.2f}x")
    print(f"per-launch overhead saved: {overhead_saved_us:.2f} us")
    print(f"\nNotes:")
    print(f"  - eager calls {args.n_launches} dispatcher passes per iter (one per `@`)")
    print(f"  - captured calls {n_outer} dispatcher passes per iter (one per gm() invocation)")
    print(f"  - difference ≈ Python+dispatcher overhead between consecutive aten ops")


if __name__ == "__main__":
    main()
