"""Pluggable offload policies: given a tensor profile, decide what to tier.

Each policy takes a Dict[str, TensorInfo] and returns a set of node names
that should be offloaded to DRAM (or a lower tier).
"""
from __future__ import annotations

import abc
from typing import Callable, Dict, List, Optional, Set

from .profiler import TensorInfo


class OffloadPolicy(abc.ABC):
    """Base class for offload policies."""

    @abc.abstractmethod
    def select(self, profile: Dict[str, TensorInfo]) -> Set[str]:
        """Return the set of node names whose outputs should be offloaded."""
        ...

    def describe(self) -> str:
        return self.__class__.__name__


class LargestNTensors(OffloadPolicy):
    """Offload the N largest intermediate tensors.

    Useful for quick experiments — the biggest tensors in a transformer are
    almost always the KV cache and the weight matrices.
    """

    def __init__(self, n: int = 20, skip_params: bool = True):
        self.n = n
        self.skip_params = skip_params

    def select(self, profile: Dict[str, TensorInfo]) -> Set[str]:
        candidates = [
            (name, info)
            for name, info in profile.items()
            if info.nbytes > 0
            and info.node_op not in ("placeholder", "output")
            and not (self.skip_params and info.node_op == "get_attr")
        ]
        candidates.sort(key=lambda x: x[1].nbytes, reverse=True)
        return {name for name, _ in candidates[: self.n]}

    def describe(self) -> str:
        return f"LargestNTensors(n={self.n})"


class BudgetPolicy(OffloadPolicy):
    """Offload tensors greedily (largest first) until HBM fits under budget.

    Args:
        hbm_budget_bytes: target HBM usage for activations+intermediates.
            Tensors are offloaded until the remaining on-device tensors sum
            to <= this budget.
        skip_params: if True, don't offload model parameters (get_attr nodes).
    """

    def __init__(self, hbm_budget_bytes: int, skip_params: bool = True):
        self.hbm_budget_bytes = hbm_budget_bytes
        self.skip_params = skip_params

    def select(self, profile: Dict[str, TensorInfo]) -> Set[str]:
        candidates = [
            (name, info)
            for name, info in profile.items()
            if info.nbytes > 0
            and info.node_op not in ("placeholder", "output")
            and not (self.skip_params and info.node_op == "get_attr")
        ]
        candidates.sort(key=lambda x: x[1].nbytes, reverse=True)

        total = sum(info.nbytes for _, info in candidates)
        offloaded: Set[str] = set()

        for name, info in candidates:
            if total <= self.hbm_budget_bytes:
                break
            offloaded.add(name)
            total -= info.nbytes

        return offloaded

    def describe(self) -> str:
        return f"BudgetPolicy(hbm={self.hbm_budget_bytes/1024**3:.1f} GB)"


class AttentionKVOnly(OffloadPolicy):
    """Offload only the K and V tensors inside attention layers.

    Finds call_module nodes whose target matches k_proj/v_proj (or equivalent)
    and selects their immediate tensor output (the view/reshape after them).
    """

    # Naming conventions for K/V projection modules across model families.
    _KV_PATTERNS = (
        "k_proj", "v_proj",                    # Llama, Qwen2, Mistral, Gemma
        "self_attn.k_proj", "self_attn.v_proj", # wrapped in submodule
        "attention.k", "attention.v",          # Falcon-style
        "key_value",                           # fused KV (GPT-NeoX, some Falcon)
    )

    def select(self, profile: Dict[str, TensorInfo]) -> Set[str]:
        offloaded: Set[str] = set()
        for name, info in profile.items():
            if info.node_op != "call_module":
                continue
            target = info.node_target
            if any(pat in target for pat in self._KV_PATTERNS):
                # Offload the projection output AND the reshape/view after it.
                offloaded.add(name)
                # Also grab the first tensor consumer (usually a view/reshape).
                for consumer_name in info.consumers:
                    consumer = profile.get(consumer_name)
                    if consumer and consumer.node_op == "call_method" and consumer.node_target == "view":
                        offloaded.add(consumer_name)
        return offloaded

    def describe(self) -> str:
        return "AttentionKVOnly()"


class ManualPolicy(OffloadPolicy):
    """Offload specific nodes by name. Useful for debugging."""

    def __init__(self, node_names: Set[str]):
        self.node_names = node_names

    def select(self, profile: Dict[str, TensorInfo]) -> Set[str]:
        return {n for n in self.node_names if n in profile}


class CallablePolicy(OffloadPolicy):
    """Wrap an arbitrary callable ``(profile) -> Set[str]``."""

    def __init__(self, fn: Callable[[Dict[str, TensorInfo]], Set[str]], label: str = ""):
        self._fn = fn
        self._label = label or getattr(fn, "__name__", "custom")

    def select(self, profile: Dict[str, TensorInfo]) -> Set[str]:
        return self._fn(profile)

    def describe(self) -> str:
        return f"CallablePolicy({self._label})"
