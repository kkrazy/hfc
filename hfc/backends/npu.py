"""NPU backend: real HBM↔DRAM transfers using pinned memory and NPU streams.

Replaces the identity stubs in hfc.rewriter with actual offloading:
  - offload_async: D2H copy to pinned CPU memory on a background stream
  - prefetch_sync: H2D copy from pinned CPU back to NPU, blocking until done

Usage:
    import hfc.backends.npu  # monkeypatches offload_async / prefetch_sync
    from hfc.rewriter import offload_async, prefetch_sync  # now does real I/O
"""
from __future__ import annotations

import atexit
import threading
from typing import Dict, Optional, Tuple

import torch

from hfc import rewriter as _rewriter_mod

# --------------------------------------------------------------------------- #
#  Pool: keyed by unique string, holds (cpu_tensor, device, dtype, stream)
# --------------------------------------------------------------------------- #

_pool: Dict[str, Tuple[torch.Tensor, torch.device, torch.dtype]] = {}
_pool_lock = threading.Lock()

# Cumulative byte counters (reset by reset_counters)
_bytes_offloaded: int = 0
_bytes_prefetched: int = 0
_offload_calls: int = 0
_prefetch_calls: int = 0

# One background stream per device for D2H copies
_streams: Dict[int, torch.npu.Stream] = {}


def _stream_for(device: torch.device) -> torch.npu.Stream:
    idx = device.index or 0
    if idx not in _streams:
        _streams[idx] = torch.npu.Stream(device=device)
    return _streams[idx]


# --------------------------------------------------------------------------- #
#  Real implementations
# --------------------------------------------------------------------------- #

def _offload_async_npu(x: torch.Tensor, key: str = "") -> torch.Tensor:
    """Copy x from NPU to pinned CPU DRAM on a background stream.

    Returns x unchanged (still on NPU). The CPU copy is stored in the pool
    under ``key``. Callers should not rely on x staying on NPU — in a later
    phase, x will be freed from HBM after the copy completes.
    """
    if not key:
        return x  # no key → no offload (safety fallback)

    stream = _stream_for(x.device)

    with torch.npu.stream(stream):
        cpu = torch.empty_like(x, device="cpu").pin_memory()
        cpu.copy_(x, non_blocking=True)

    nbytes = x.nelement() * x.element_size()
    with _pool_lock:
        _pool[key] = (cpu, x.device, x.dtype)
        global _bytes_offloaded, _offload_calls
        _bytes_offloaded += nbytes
        _offload_calls += 1

    return x


def _prefetch_sync_npu(x: torch.Tensor, key: str = "") -> torch.Tensor:
    """Copy tensor from pinned CPU DRAM back to NPU, blocking until done.

    ``x`` is the value passed through the FX graph (the original NPU tensor
    or a small sentinel). It is *not* used — the real data comes from the pool.
    """
    if not key:
        return x

    with _pool_lock:
        entry = _pool.pop(key, None)

    if entry is None:
        return x

    cpu_tensor, orig_device, orig_dtype = entry

    # Ensure the D2H copy is done before we read from CPU
    stream = _stream_for(orig_device)
    stream.synchronize()

    # H2D copy (blocking so the consumer sees the data immediately)
    device_tensor = cpu_tensor.to(device=orig_device, dtype=orig_dtype, non_blocking=False)
    nbytes = device_tensor.nelement() * device_tensor.element_size()
    with _pool_lock:
        global _bytes_prefetched, _prefetch_calls
        _bytes_prefetched += nbytes
        _prefetch_calls += 1
    return device_tensor


def _offload_async_cpu(x: torch.Tensor, key: str = "") -> torch.Tensor:
    """CPU fallback — identity (nothing to offload on CPU)."""
    return x


def _prefetch_sync_cpu(x: torch.Tensor, key: str = "") -> torch.Tensor:
    """CPU fallback — identity."""
    return x


# --------------------------------------------------------------------------- #
#  Install: monkeypatch the module-level functions in hfc.rewriter
# --------------------------------------------------------------------------- #

def install(device_type: str = "npu"):
    """Patch hfc.rewriter.offload_async / prefetch_sync with real implementations.

    Call once at startup, before any graph rewriting.
    """
    if device_type == "npu":
        _rewriter_mod.offload_async = _offload_async_npu
        _rewriter_mod.prefetch_sync = _prefetch_sync_npu
    else:
        _rewriter_mod.offload_async = _offload_async_cpu
        _rewriter_mod.prefetch_sync = _prefetch_sync_cpu


def pool_stats() -> Dict[str, int]:
    """Return {key: nbytes} for everything currently in the DRAM pool."""
    with _pool_lock:
        return {k: v[0].nelement() * v[0].element_size() for k, v in _pool.items()}


def pool_clear():
    """Drop everything in the pool (free CPU memory)."""
    with _pool_lock:
        _pool.clear()


def transfer_stats() -> Dict[str, int]:
    """Cumulative bytes moved since last reset_counters()."""
    with _pool_lock:
        return {
            "bytes_offloaded": _bytes_offloaded,
            "bytes_prefetched": _bytes_prefetched,
            "offload_calls": _offload_calls,
            "prefetch_calls": _prefetch_calls,
        }


def reset_counters():
    global _bytes_offloaded, _bytes_prefetched, _offload_calls, _prefetch_calls
    with _pool_lock:
        _bytes_offloaded = 0
        _bytes_prefetched = 0
        _offload_calls = 0
        _prefetch_calls = 0


@atexit.register
def _cleanup():
    pool_clear()
