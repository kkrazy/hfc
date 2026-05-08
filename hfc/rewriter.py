"""Rewrite an FX graph to inject tiered-memory prefetch/evict around selected nodes.

For each node whose output is marked for offloading:
  1. After the node (producer), insert: ``offload_async(output) → handle``
  2. Before each consumer, insert: ``prefetch_sync(handle) → tensor_on_device``
  3. Replace the consumer's input reference with the prefetched tensor.

Uses plain Python functions as FX call_function targets — no torch.library
registration needed. Backends can monkeypatch these to add real HBM↔DRAM
transfers (see hfc.backends.npu).
"""
from __future__ import annotations

import copy
from typing import Dict, Optional, Set

import torch
import torch.fx

from .profiler import TensorInfo

# --------------------------------------------------------------------------- #
#  Offload/prefetch functions — identity by default, backend can override
# --------------------------------------------------------------------------- #


def offload_async(x: torch.Tensor, key: str = "") -> torch.Tensor:
    """Move tensor to DRAM (async). Identity fallback for CPU testing."""
    return x


def prefetch_sync(x: torch.Tensor, key: str = "") -> torch.Tensor:
    """Move tensor back to device (sync). Identity fallback for CPU testing."""
    return x


# --------------------------------------------------------------------------- #
#  Graph rewriter
# --------------------------------------------------------------------------- #

def rewrite_with_offload(
    gm: torch.fx.GraphModule,
    offload_nodes: Set[str],
    profile: Optional[Dict[str, TensorInfo]] = None,
) -> torch.fx.GraphModule:
    """Rewrite ``gm`` in-place, wrapping each offloaded node's output with
    ``offload_async`` / ``prefetch_sync`` pairs.

    Args:
        gm: the FX GraphModule to rewrite.
        offload_nodes: set of node names whose outputs should be tiered.
        profile: optional TensorInfo dict (used for logging/decisions).

    Returns:
        The same GraphModule (mutated in-place), with recompiled graph.
    """
    name_to_node: Dict[str, torch.fx.Node] = {n.name: n for n in gm.graph.nodes}

    # We iterate in topological order. For each offloaded node we:
    #   1. Insert offload_async right after it.
    #   2. For every consumer of the original node, insert prefetch_sync
    #      right before the consumer, and replace the consumer's arg.
    #
    # Because we insert new nodes, we can't mutate during iteration over
    # gm.graph.nodes. Instead we build a worklist first.
    worklist = []
    for node in gm.graph.nodes:
        if node.name in offload_nodes:
            worklist.append(node)

    stats = {"offloaded": 0, "prefetch_inserted": 0}

    for orig_node in worklist:
        users = list(orig_node.users.keys())  # snapshot before we modify

        if not users:
            continue  # dead node, skip

        # Unique key so the offload/prefetch pair can find each other in the pool.
        pool_key = orig_node.name

        # 1) Insert offload_async right after orig_node
        with gm.graph.inserting_after(orig_node):
            offload_node = gm.graph.call_function(
                offload_async, args=(orig_node,), kwargs={"key": pool_key}
            )
            offload_node.name = f"offload_{orig_node.name}"
        stats["offloaded"] += 1

        # 2) For each consumer of orig_node, insert prefetch_sync before it
        #    and rewrite the consumer's args to use the prefetched tensor.
        for user in users:
            with gm.graph.inserting_before(user):
                prefetch_node = gm.graph.call_function(
                    prefetch_sync, args=(offload_node,), kwargs={"key": pool_key}
                )
                prefetch_node.name = f"prefetch_{orig_node_name_unique(orig_node.name, user.name)}"
            stats["prefetch_inserted"] += 1

            # Replace orig_node with prefetch_node in user's args/kwargs
            _replace_arg(user, orig_node, prefetch_node)

    gm.graph.lint()
    gm.recompile()

    print(f"[rewriter] offloaded {stats['offloaded']} tensors, "
          f"inserted {stats['prefetch_inserted']} prefetch ops")
    return gm


def _replace_arg(node: torch.fx.Node, old: torch.fx.Node, new: torch.fx.Node):
    """Replace all occurrences of ``old`` in ``node``'s args and kwargs with ``new``."""
    new_args = []
    for arg in node.args:
        new_args.append(new if arg is old else arg)
    node.args = tuple(new_args)

    new_kwargs = {}
    for k, v in node.kwargs.items():
        new_kwargs[k] = new if v is old else v
    node.kwargs = new_kwargs


_counter = 0

def orig_node_name_unique(base: str, consumer: str) -> str:
    global _counter
    _counter += 1
    return f"{base}_for_{consumer}_{_counter}"


# --------------------------------------------------------------------------- #
#  Verify: run original and rewritten model, compare outputs
# --------------------------------------------------------------------------- #

def verify_rewrite(
    original_gm: torch.fx.GraphModule,
    rewritten_gm: torch.fx.GraphModule,
    example_inputs: dict,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> bool:
    """Run both modules with the same inputs and compare logits."""
    with torch.no_grad():
        out_orig = original_gm(**example_inputs)
        out_rewr = rewritten_gm(**example_inputs)

    logits_orig = _extract_logits(out_orig)
    logits_rewr = _extract_logits(out_rewr)

    if logits_orig.shape != logits_rewr.shape:
        print(f"[verify] SHAPE MISMATCH: {logits_orig.shape} vs {logits_rewr.shape}")
        return False

    # Cast to float32 for comparison (fp16 comparisons are unreliable)
    match = torch.allclose(
        logits_orig.float().cpu(), logits_rewr.float().cpu(),
        atol=atol, rtol=rtol,
    )
    if match:
        print(f"[verify] OK — outputs match (atol={atol}, rtol={rtol})")
    else:
        diff = (logits_orig.float() - logits_rewr.float()).abs()
        print(f"[verify] MISMATCH — max diff = {diff.max().item():.6f}, "
              f"mean diff = {diff.mean().item():.6f}")
    return match


def _extract_logits(out):
    if isinstance(out, torch.Tensor):
        return out
    logits = getattr(out, "logits", None)
    if logits is not None:
        return logits
    if isinstance(out, (tuple, list)):
        return out[0]
    if isinstance(out, dict):
        v = out.get("logits")
        if v is not None:
            return v
        return next(iter(out.values()))
    raise ValueError(f"Can't extract logits from {type(out)}")
