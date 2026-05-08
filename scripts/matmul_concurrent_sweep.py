#!/usr/bin/env python3
"""Sequential vs N-stream concurrent matmul submission.

Same total work — X rows × (N, M) — done two ways:

  SEQ:  1 NPU stream, all launches issued back-to-back.
  PAR:  S NPU streams, single thread, round-robin submission. Kernels go to
        different device queues; the device can execute them concurrently if
        hardware (AICore count, DMA channels) permits.

(Earlier version used Python threads + a barrier. torch_npu's TBE subprocess
breaks under threading on this CANN version — "main process disappeared".
Single-thread multi-stream gives the same device-side concurrency without
that issue, just no host-side dispatch parallelism.)

What the comparison reveals:
  - small batch (launch-overhead-bound): PAR may help if device-side launch
    queue fills faster from S streams than from 1.
  - large batch (compute-bound): PAR ≈ SEQ — AICores already saturated, more
    streams just serialize on the same hardware.
"""
from __future__ import annotations
import argparse
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--total", type=int, default=4096,
                   help="X = total rows of work")
    p.add_argument("--N", type=int, default=2048)
    p.add_argument("--M", type=int, default=2048)
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--device", default="npu:0")
    p.add_argument("--peak-tflops", type=float, default=376.0)
    p.add_argument("--streams", type=int, default=8,
                   help="number of streams / threads in PAR mode")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--batches", type=str, default="1,8,64,256,1024,4096")
    return p.parse_args()


def run_seq(W, x, n_launches, sync_fn):
    sync_fn()
    t0 = time.perf_counter()
    for _ in range(n_launches):
        _ = x @ W
    sync_fn()
    return (time.perf_counter() - t0) * 1000


def run_par(W, x, n_streams, n_per_stream, make_stream, stream_ctx, sync_fn):
    """Single-thread, multi-stream — per-stream blocks.
    Submit all of stream 0's launches first, then stream 1's, etc. The device
    queues build up; once submission outpaces execution, the device can pipeline
    across streams. Avoids the per-launch context-switch overhead of round-robin.
    """
    streams = [make_stream() for _ in range(n_streams)]
    sync_fn()
    t0 = time.perf_counter()
    for s in streams:
        with stream_ctx(s):
            for _ in range(n_per_stream):
                _ = x @ W
    for s in streams:
        s.synchronize()
    return (time.perf_counter() - t0) * 1000


def main():
    args = parse_args()
    import torch
    is_npu = args.device.startswith("npu")
    if is_npu:
        import torch_npu  # noqa
    print(f"torch={torch.__version__}  is_npu={is_npu}")
    if is_npu:
        print(f"torch_npu={torch_npu.__version__}  npu_count={torch.npu.device_count()}")
    print(f"config: total={args.total}  N={args.N}  M={args.M}  dtype={args.dtype}  "
          f"device={args.device}  streams={args.streams}")

    dtype = getattr(torch, args.dtype)
    W = torch.randn(args.N, args.M, dtype=dtype, device=args.device)

    if is_npu:
        sync_fn = torch.npu.synchronize
        make_stream = lambda: torch.npu.Stream(device=args.device)
        stream_ctx = torch.npu.stream
    else:
        sync_fn = lambda: None
        make_stream = lambda: None
        # CPU fallback: stream_ctx returns a no-op contextmanager
        from contextlib import contextmanager
        @contextmanager
        def _noop(_):
            yield
        stream_ctx = _noop

    flops_per_iter = 2 * args.total * args.N * args.M
    print(f"\nPer iteration: {flops_per_iter / 1e9:.2f} GFLOPs")
    print(f"\n{'batch':>6} {'launches':>8} {'SEQ_ms':>10} {'PAR_ms':>10} {'speedup':>10} "
          f"{'SEQ_TF':>8} {'PAR_TF':>8} {'PAR_util%':>10}")
    print("-" * 84)

    batches = [int(b) for b in args.batches.split(",")]
    for batch in batches:
        if args.total % batch != 0:
            print(f"  skip batch={batch}: total {args.total} not divisible")
            continue
        n_total_launches = args.total // batch
        if n_total_launches < args.streams:
            print(f"  skip batch={batch}: only {n_total_launches} launches < {args.streams} streams")
            continue
        if n_total_launches % args.streams != 0:
            print(f"  skip batch={batch}: {n_total_launches} not divisible by {args.streams}")
            continue
        n_per_stream = n_total_launches // args.streams
        x = torch.randn(batch, args.N, dtype=dtype, device=args.device)

        for _ in range(args.warmup):
            run_seq(W, x, n_total_launches, sync_fn)
            run_par(W, x, args.streams, n_per_stream, make_stream, stream_ctx, sync_fn)

        seq_samples = [run_seq(W, x, n_total_launches, sync_fn) for _ in range(args.iters)]
        par_samples = [run_par(W, x, args.streams, n_per_stream,
                               make_stream, stream_ctx, sync_fn)
                       for _ in range(args.iters)]

        seq_samples.sort()
        par_samples.sort()
        seq_med = seq_samples[len(seq_samples) // 2]
        par_med = par_samples[len(par_samples) // 2]
        speedup = seq_med / par_med if par_med > 0 else 0
        seq_tflops = flops_per_iter / (seq_med / 1000) / 1e12
        par_tflops = flops_per_iter / (par_med / 1000) / 1e12
        par_util = 100.0 * par_tflops / args.peak_tflops

        print(f"{batch:>6} {n_total_launches:>8} {seq_med:>10.2f} {par_med:>10.2f} "
              f"{speedup:>9.2f}x {seq_tflops:>8.1f} {par_tflops:>8.1f} {par_util:>9.1f}%")

    print()
    print(f"SEQ:  1 stream, sequential issuance from 1 thread")
    print(f"PAR:  {args.streams} streams, {args.streams} threads, barrier-coordinated start")


if __name__ == "__main__":
    main()
