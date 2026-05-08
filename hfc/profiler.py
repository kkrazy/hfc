"""Profile every tensor in an FX graph: shape, dtype, byte size, lifetime.

Run via ShapePropInterpreter — a torch.fx.Interpreter that records the output
metadata for every node after a single forward pass with example inputs.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.fx


@dataclasses.dataclass
class TensorInfo:
    """Metadata for one FX node's output tensor."""

    node_name: str
    node_op: str          # placeholder, call_function, call_method, call_module, get_attr, output
    node_target: str      # e.g. "model.layers.0.self_attn.k_proj"

    # --- filled by the interpreter pass ---
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[torch.dtype] = None
    nbytes: int = 0       # bytes (0 for non-tensor outputs like ints, None)

    # --- filled by lifetime analysis ---
    topo_idx: int = -1             # position in topological order
    producer: Optional[str] = None  # node that writes this tensor
    consumers: List[str] = dataclasses.field(default_factory=list)  # nodes that read it

    # --- derived ---
    def is_tensor(self) -> bool:
        return self.shape is not None

    def human_size(self) -> str:
        if self.nbytes == 0:
            return "0 B"
        for unit in ("B", "KB", "MB", "GB"):
            if self.nbytes < 1024:
                return f"{self.nbytes:.1f} {unit}"
            self_nbytes = self.nbytes  # avoid reassignment
            self_nbytes /= 1024
        return f"{self.nbytes:.1f} TB"

    def __repr__(self) -> str:
        sz = f"{self.nbytes:>12,} B" if self.nbytes else "     non-tensor"
        return f"TensorInfo({self.node_name:30s} {sz}  shape={self.shape}  dtype={self.dtype})"


class ShapePropInterpreter(torch.fx.Interpreter):
    """Run the FX graph on example inputs and record output metadata per node.

    Does NOT require GPU — runs on CPU with meta-tensors or real tensors.
    """

    def __init__(self, module: torch.fx.GraphModule, gpu_mem: bool = False):
        super().__init__(module)
        self.profile: Dict[str, TensorInfo] = {}
        self._gpu_mem = gpu_mem
        self._current_node: Optional[torch.fx.Node] = None

    def run_node(self, n):
        self._current_node = n
        return super().run_node(n)

    def placeholder(self, target, args, kwargs):
        out = super().placeholder(target, args, kwargs)
        self._record(self._current_node.name, "placeholder", target, out)
        return out

    def get_attr(self, target, args, kwargs):
        out = super().get_attr(target, args, kwargs)
        self._record(self._current_node.name, "get_attr", target, out)
        return out

    def call_function(self, target, args, kwargs):
        out = super().call_function(target, args, kwargs)
        self._record(self._current_node.name, "call_function", str(target), out)
        return out

    def call_method(self, target, args, kwargs):
        out = super().call_method(target, args, kwargs)
        self._record(self._current_node.name, "call_method", target, out)
        return out

    def call_module(self, target, args, kwargs):
        out = super().call_module(target, args, kwargs)
        self._record(self._current_node.name, "call_module", target, out)
        return out

    def output(self, target, args, kwargs):
        out = super().output(target, args, kwargs)
        self._record("output", "output", "output", out)
        return out

    # ------------------------------------------------------------------ #
    def _record(self, node_name: str, node_op: str, node_target: str, value: Any):
        shape = None
        dtype = None
        nbytes = 0

        if isinstance(value, torch.Tensor):
            shape = tuple(value.shape)
            dtype = value.dtype
            nbytes = value.nelement() * value.element_size()
        elif isinstance(value, (tuple, list)):
            # recurse into containers to find tensors and sum sizes
            for v in _flatten(value):
                if isinstance(v, torch.Tensor):
                    if shape is None:
                        shape = tuple(v.shape)
                        dtype = v.dtype
                    nbytes += v.nelement() * v.element_size()

        self.profile[node_name] = TensorInfo(
            node_name=node_name,
            node_op=node_op,
            node_target=node_target,
            shape=shape,
            dtype=dtype,
            nbytes=nbytes,
        )

    def run_and_analyze(self, *args, **kwargs) -> Dict[str, TensorInfo]:
        """Run the interpreter, then do lifetime analysis. Returns the profile."""
        self.run(*args, **kwargs)
        self._compute_lifetimes()
        return self.profile

    def _compute_lifetimes(self):
        """Fill topo_idx, producer, consumers for every profiled node."""
        gm = self.module
        name_to_idx = {}
        for idx, node in enumerate(gm.graph.nodes):
            name_to_idx[node.name] = idx

        for node_name, info in self.profile.items():
            info.topo_idx = name_to_idx.get(node_name, -1)

        # producer is the node itself; consumers are all users in the graph
        for node in gm.graph.nodes:
            if node.name not in self.profile:
                continue
            self.profile[node.name].producer = node.name
            for user in node.users:
                if user.name in self.profile:
                    self.profile[node.name].consumers.append(user.name)


def _flatten(container) -> list:
    """Flatten nested tuples/lists into a flat list of leaf values."""
    out = []
    if isinstance(container, (tuple, list)):
        for item in container:
            out.extend(_flatten(item))
    else:
        out.append(container)
    return out


# --------------------------------------------------------------------------- #
#  Convenience: run the profile and print a ranked summary
# --------------------------------------------------------------------------- #

def profile_graph(
    gm: torch.fx.GraphModule,
    example_inputs: dict[str, Any],
) -> Dict[str, TensorInfo]:
    """Profile an FX graph with example inputs. Returns {node_name: TensorInfo}."""
    interp = ShapePropInterpreter(gm)
    # Interpreter.run() takes positional args ordered by the graph's placeholders.
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    args = tuple(example_inputs[p.target] for p in placeholders)
    return interp.run_and_analyze(*args)


def print_profile_ranked(profile: Dict[str, TensorInfo], top_k: int = 30):
    """Print tensors ranked by size, largest first."""
    tensors = [t for t in profile.values() if t.nbytes > 0]
    tensors.sort(key=lambda t: t.nbytes, reverse=True)

    total = sum(t.nbytes for t in tensors)
    print(f"\n{'Rank':>4}  {'Node':30s}  {'Bytes':>14}  {'%Total':>7}  {'Shape':30s}  Target")
    print("-" * 120)
    for i, t in enumerate(tensors[:top_k]):
        pct = 100.0 * t.nbytes / total if total else 0
        shape_str = str(t.shape) if t.shape else ""
        print(f"{i+1:4d}  {t.node_name:30s}  {t.nbytes:>14,}  {pct:6.1f}%  {shape_str:30s}  {t.node_target}")
    print(f"\nTotal tensor memory: {total:,} bytes ({total/1024**3:.2f} GB)")
    print(f"Top {min(top_k, len(tensors))} account for "
          f"{100*sum(t.nbytes for t in tensors[:top_k])/total:.1f}% of total")
