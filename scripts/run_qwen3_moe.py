#!/usr/bin/env python3
"""Capture + offload Qwen3 MoE on NPU using torch.export instead of symbolic_trace.

symbolic_trace breaks under newer transformers on Python 3.9 ("varnames is too
small"). torch.export goes through dynamo and avoids that path. We also need a
compat shim because newer transformers calls torch.utils._pytree.register_pytree_node
which torch 2.1 only exposes as the private _register_pytree_node.

This script:
  1. Patches torch pytree API for transformers compat.
  2. Builds Qwen3 MoE from config (random weights — we test the offload pipeline,
     not generation correctness, so 60GB download is unnecessary).
  3. Captures a GraphModule via torch.export.
  4. Profiles, applies a policy, rewrites with offload/prefetch, runs on NPU.

Usage:
  python run_qwen3_moe.py --model Qwen/Qwen3-30B-A3B --policy kv --seq-len 32
"""
from __future__ import annotations

# --- compat shims: must run BEFORE `import transformers` ----------------------
# torch_npu 2.1.0.post17 forces torch==2.1, but transformers >=4.50 expects newer
# torch APIs. Two backports needed:
#
# 1. torch.utils._pytree.register_pytree_node   (added in torch 2.2)
# 2. torch.compiler.is_compiling                (added in torch 2.3)
#
# Without (2), torch.export → dynamo blows up with "module 'torch.compiler' has
# no attribute 'is_compiling'" the moment transformers' attention-mask helper
# tries to detect whether it's being traced.
import torch.utils._pytree as _p
if not hasattr(_p, "register_pytree_node"):
    _orig = _p._register_pytree_node

    def register_pytree_node(cls, flatten_fn, unflatten_fn, *, serialized_type_name=None,
                             to_dumpable_context=None, from_dumpable_context=None,
                             flatten_with_keys_fn=None):
        return _orig(cls, flatten_fn, unflatten_fn)

    _p.register_pytree_node = register_pytree_node

import torch.compiler as _tc
if not hasattr(_tc, "is_compiling"):
    try:
        from torch._dynamo import is_compiling as _is_compiling
    except ImportError:
        def _is_compiling():
            return False
    _tc.is_compiling = _is_compiling
# ------------------------------------------------------------------------------

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--device", default="npu:0")
    p.add_argument("--policy", default="kv", choices=["none", "largest", "budget", "kv"])
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--hbm-gb", type=float, default=4.0)
    p.add_argument("--no-run", action="store_true")
    p.add_argument("--scale", type=float, default=1.0,
                   help="shrink layers/experts to fit RAM. 1.0=full model. 0.1≈2 layers, 6 experts.")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out-dir", default=str(Path.home() / "hfc" / "out"))
    p.add_argument("--capture", default="make_fx", choices=["export", "make_fx"],
                   help="export = torch.export (dynamo+fake), make_fx = proxy_tensor (eager+real). "
                        "make_fx is the working path on torch 2.1 + transformers 4.51.")
    p.add_argument("--bench", action="store_true",
                   help="run warmup + timed iterations on NPU, report latency stats")
    p.add_argument("--bench-iters", type=int, default=10)
    p.add_argument("--bench-warmup", type=int, default=3)
    p.add_argument("--include-params", action="store_true",
                   help="for largest/budget policies: include get_attr (parameters) as offload candidates")
    return p.parse_args()


def _patch_moe_dense():
    """Replace Qwen3MoeSparseMoeBlock.forward with a dense-equivalent that has
    no data-dependent control flow.

    The shipped sparse implementation uses:
        idx, top_x = torch.where(expert_mask[e])
        if top_x.shape[0] == 0: continue
        ...
    where top_x's runtime length depends on routing decisions. make_fx and
    torch.export both reject this. The dense version runs every expert on every
    token and weights by a (N, num_experts) routing matrix — same mathematical
    output, full compute (fine for the offload-pipeline test, which doesn't
    care about throughput at this stage)."""
    import torch
    import torch.nn.functional as F
    from transformers.models.qwen3_moe import modeling_qwen3_moe as _qm

    def dense_forward(self, hidden_states):
        bsz, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if getattr(self, "norm_topk_prob", True):
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(flat.dtype)

        # Build (N, num_experts) dense weight matrix without one_hot/scatter
        # (both call _local_scalar_dense under proxy tracing on torch 2.1).
        e_range = torch.arange(self.num_experts, device=flat.device)
        # selected_experts: (N, K) ; e_range: (E,) ; broadcast → (N, K, E)
        match = (selected_experts.unsqueeze(-1) == e_range).to(routing_weights.dtype)
        # weighted match → (N, K, E) ; sum over K → (N, E)
        dense_w = (match * routing_weights.unsqueeze(-1)).sum(dim=1)

        out = torch.zeros_like(flat)
        for e in range(self.num_experts):
            expert_out = self.experts[e](flat)
            out = out + expert_out * dense_w[:, e].unsqueeze(-1)
        return out.view(bsz, seq_len, hidden_dim), router_logits

    _qm.Qwen3MoeSparseMoeBlock.forward = dense_forward


def _patch_transformers_for_export():
    """transformers >=4.50 routes attention through AttentionInterface, a
    user-defined class whose __getitem__ torch 2.1's dynamo can't inline. Swap
    the singleton for a plain dict — the values (callables) are unaffected.
    Must run AFTER `import transformers` and BEFORE model.forward is called."""
    import transformers.modeling_utils as _mu
    iface = getattr(_mu, "ALL_ATTENTION_FUNCTIONS", None)
    if iface is None or isinstance(iface, dict):
        return
    plain = dict(getattr(iface, "_global_mapping", {}))
    _mu.ALL_ATTENTION_FUNCTIONS = plain
    # Patch the symbol in any model module that imported it by reference.
    import sys
    for modname, mod in list(sys.modules.items()):
        if not modname.startswith("transformers.models."):
            continue
        if getattr(mod, "ALL_ATTENTION_FUNCTIONS", None) is iface:
            mod.ALL_ATTENTION_FUNCTIONS = plain


def build_model_from_config(model_id: str, dtype, scale: float):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    # eager attention avoids SDPA/FA2 dispatch — simpler graph for torch.export.
    cfg._attn_implementation = "eager"
    if scale < 1.0:
        # shrink layers + experts proportionally so we can hold a graph in CPU RAM
        old = (cfg.num_hidden_layers,
               getattr(cfg, "num_experts", None),
               getattr(cfg, "intermediate_size", None))
        cfg.num_hidden_layers = max(2, int(cfg.num_hidden_layers * scale))
        if hasattr(cfg, "num_experts"):
            cfg.num_experts = max(4, int(cfg.num_experts * scale))
        print(f"[build] scaled layers,experts,inter: {old} -> "
              f"({cfg.num_hidden_layers},{getattr(cfg,'num_experts',None)},"
              f"{getattr(cfg,'intermediate_size',None)})")
    print(f"[build] model_type={cfg.model_type} layers={cfg.num_hidden_layers} "
          f"hidden={cfg.hidden_size} experts={getattr(cfg,'num_experts',None)} "
          f"top_k={getattr(cfg,'num_experts_per_tok',None)}")
    m = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype)
    m.eval()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"[build] params = {n_params/1e9:.2f} B")
    return cfg, m


def capture_via_export(model, example_inputs):
    """torch.export → ExportedProgram → GraphModule."""
    import torch
    print("[capture] torch.export ...")
    # torch 2.1: no strict kwarg, no .module() — ExportedProgram has graph_module attribute
    ep = torch.export.export(model, args=(), kwargs=example_inputs)
    gm = ep.graph_module if hasattr(ep, "graph_module") else ep.module()
    nodes = list(gm.graph.nodes)
    by_op = {}
    for n in nodes:
        by_op[n.op] = by_op.get(n.op, 0) + 1
    print(f"[capture] graph: {len(nodes)} nodes  by_op={by_op}")
    return gm, ep


def capture_via_make_fx(model, example_inputs):
    """Fallback: torch.fx.experimental.proxy_tensor.make_fx — no dynamo, no
    fake tensors. Runs the model eagerly with ProxyTensors and records aten ops.
    Avoids torch 2.1's torch.export → dynamo → fake-tensor bugs (SymInt leaks
    into ops like one_hot). Cost: real activations are allocated, so the model
    must fit in CPU RAM (use --scale to shrink)."""
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx

    print("[capture] make_fx (real tensors, aten-level) ...")

    # Pin the keyword order so the traced wrapper signature is deterministic.
    keys = list(example_inputs.keys())
    args_tuple = tuple(example_inputs[k] for k in keys)

    def _forward(*args):
        kw = dict(zip(keys, args))
        out = model(**kw)
        # Strip dataclass/dict wrappers so make_fx can return a tensor tree.
        logits = getattr(out, "logits", None)
        if logits is not None:
            return logits
        if isinstance(out, (tuple, list)):
            return out[0]
        if isinstance(out, dict):
            return out.get("logits") or next(iter(out.values()))
        return out

    gm = make_fx(_forward, tracing_mode="real")(*args_tuple)
    nodes = list(gm.graph.nodes)
    by_op = {}
    for n in nodes:
        by_op[n.op] = by_op.get(n.op, 0) + 1
    print(f"[capture] graph: {len(nodes)} nodes  by_op={by_op}")
    return gm, None


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    print(f"[main] torch={torch.__version__}")
    if args.device.startswith("npu"):
        import torch_npu  # noqa
        print(f"[main] torch_npu={torch_npu.__version__}  npu_count={torch.npu.device_count()}")

    import transformers
    print(f"[main] transformers={transformers.__version__}")

    dtype = getattr(torch, args.dtype)

    # --- 1. build model on CPU (random weights) -----------------------------
    cfg, model = build_model_from_config(args.model, dtype=torch.float32, scale=args.scale)
    # Patch the dispatch dict AFTER from_config (which uses valid_keys) but
    # BEFORE export (where dynamo can't trace AttentionInterface.__getitem__).
    _patch_transformers_for_export()
    # Replace MoE block's forward with a tracer-friendly dense version.
    _patch_moe_dense()

    # --- 2. example inputs --------------------------------------------------
    bsz = 1
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, args.seq_len))
    attention_mask = torch.ones(bsz, args.seq_len, dtype=torch.long)
    example_inputs = dict(input_ids=input_ids, attention_mask=attention_mask)

    # --- 3. capture --------------------------------------------------------
    if args.capture == "export":
        gm, _ep = capture_via_export(model, example_inputs)
    else:
        gm, _ep = capture_via_make_fx(model, example_inputs)

    # --- 4. profile ---------------------------------------------------------
    from hfc.profiler import profile_graph, print_profile_ranked
    print("[main] profiling ...")
    # make_fx names placeholders positionally (arg0_1, arg1_1, ...) — remap so
    # profile_graph's placeholder-target lookup hits the right tensors.
    profile_inputs = example_inputs
    if args.capture == "make_fx":
        keys = list(example_inputs.keys())
        ph_targets = [n.target for n in gm.graph.nodes if n.op == "placeholder"]
        profile_inputs = {pt: example_inputs[keys[i]] for i, pt in enumerate(ph_targets)}
    profile = profile_graph(gm, profile_inputs)
    print_profile_ranked(profile, top_k=args.top_k)

    # --- 5. policy ----------------------------------------------------------
    from hfc.policy import LargestNTensors, BudgetPolicy, AttentionKVOnly

    if args.policy == "none":
        offload_set = set()
        policy_label = "none"
    elif args.policy == "largest":
        pol = LargestNTensors(n=args.n, skip_params=not args.include_params)
        offload_set = pol.select(profile); policy_label = pol.describe()
    elif args.policy == "budget":
        pol = BudgetPolicy(int(args.hbm_gb * 1024**3), skip_params=not args.include_params)
        offload_set = pol.select(profile); policy_label = pol.describe()
    elif args.policy == "kv":
        pol = AttentionKVOnly(); offload_set = pol.select(profile); policy_label = pol.describe()

    offload_bytes = sum(profile[n].nbytes for n in offload_set if n in profile)
    print(f"\n[main] policy={policy_label} selected {len(offload_set)} nodes "
          f"({offload_bytes/1024**2:.1f} MB total)")

    # --- 6. rewrite + verify ------------------------------------------------
    if offload_set:
        from hfc.rewriter import rewrite_with_offload, verify_rewrite
        gm_orig = copy.deepcopy(gm)
        gm = rewrite_with_offload(gm, offload_set, profile)
        print(f"[main] rewritten: {len(list(gm.graph.nodes))} nodes")
        # verify_rewrite uses **kwargs; for make_fx graphs the placeholder names
        # are arg0_1/arg1_1, not input_ids/attention_mask — pass the remapped
        # dict (same one we used for profiling).
        verify_inputs = profile_inputs if args.capture == "make_fx" else example_inputs
        ok = verify_rewrite(gm_orig, gm, verify_inputs, atol=1e-2, rtol=1e-2)
        if not ok:
            print("[main] WARN: rewrite changed outputs"); return 1

    if args.no_run:
        print("[main] --no-run done"); return 0

    # --- 7. NPU run ---------------------------------------------------------
    if args.device.startswith("npu"):
        from hfc.backends import npu as npu_backend
        npu_backend.install("npu")

    print(f"\n[main] -> {args.device} (dtype={args.dtype})")
    # make_fx with tracing_mode='real' bakes the trace-time device into every
    # tensor-creation op as a literal kwarg (device='cpu', because tracing
    # happens on CPU). Rewrite to the target device before lowering.
    target_dev = torch.device(args.device)
    cpu_dev = torch.device("cpu")
    rewritten = 0
    for node in gm.graph.nodes:
        if node.op != "call_function" or "device" not in node.kwargs:
            continue
        d = node.kwargs["device"]
        if d == cpu_dev or d is None or (isinstance(d, str) and d == "cpu"):
            new_kw = dict(node.kwargs)
            new_kw["device"] = target_dev
            node.kwargs = new_kw
            rewritten += 1
    if rewritten:
        gm.graph.lint()
        gm.recompile()
        print(f"[main] rewrote device='cpu' -> {args.device} on {rewritten} aten nodes")
    get_attr_targets = sorted({n.target for n in gm.graph.nodes if n.op == "get_attr"})
    print(f"[main] params={len(list(gm.parameters()))} buffers={len(list(gm.buffers()))} "
          f"get_attr_targets={len(get_attr_targets)}")
    if get_attr_targets:
        sample = get_attr_targets[0]
        sample_t = getattr(gm, sample, None)
        print(f"[main] sample {sample}: type={type(sample_t).__name__} "
              f"device={getattr(sample_t,'device',None)} dtype={getattr(sample_t,'dtype',None)}")

    gm = gm.to(args.device).to(dtype)

    # Force-migrate every get_attr target. Handles all of: nn.Parameter, buffer,
    # and plain tensor attributes that make_fx stashes on the module.
    moved = 0
    for name in get_attr_targets:
        attr = getattr(gm, name, None)
        if not isinstance(attr, torch.Tensor):
            continue
        if attr.device != target_dev or (attr.is_floating_point() and attr.dtype != dtype):
            new_t = attr.to(target_dev)
            if new_t.is_floating_point():
                new_t = new_t.to(dtype)
            # Use object.__setattr__ to bypass nn.Module's parameter/buffer
            # dispatch which would reject a plain Tensor for a known param name.
            try:
                setattr(gm, name, new_t)
            except (TypeError, AttributeError):
                object.__setattr__(gm, name, new_t)
            moved += 1
    print(f"[main] migrated {moved}/{len(get_attr_targets)} get_attr tensors to {args.device}")
    npu_in = {k: v.to(args.device) for k, v in example_inputs.items()}

    # make_fx graphs contain aten tensor-creation ops (scalar_tensor, full,
    # arange, etc.) that default to CPU. Wrap in a device context so they
    # materialize on the same NPU as the weights.
    is_npu = args.device.startswith("npu")
    keys = list(example_inputs.keys())

    def _invoke():
        if args.capture == "make_fx":
            return gm(*(npu_in[k] for k in keys))
        return gm(**npu_in)

    with torch.no_grad(), torch.device(args.device):
        out = _invoke()
        if args.bench:
            import time
            # Reset counters AFTER warmup so they reflect only timed iters.
            print(f"\n[bench] warming up ({args.bench_warmup} iters) ...")
            for _ in range(args.bench_warmup):
                _invoke()
            if is_npu:
                torch.npu.synchronize()
                from hfc.backends import npu as _npu_b
                _npu_b.reset_counters()
            samples = []
            for _ in range(args.bench_iters):
                if is_npu:
                    torch.npu.synchronize()
                t0 = time.perf_counter()
                _invoke()
                if is_npu:
                    torch.npu.synchronize()
                samples.append((time.perf_counter() - t0) * 1000)
            samples.sort()
            mean = sum(samples) / len(samples)
            median = samples[len(samples) // 2]
            mn, mx = samples[0], samples[-1]
            # Drop top/bottom 10% to denoise
            trim = max(1, args.bench_iters // 10)
            trimmed = samples[trim:-trim] if args.bench_iters > 2 * trim else samples
            tmean = sum(trimmed) / len(trimmed)
            label = f"policy={policy_label}"
            print(f"[bench] {label}  iters={args.bench_iters}  "
                  f"mean={mean:.2f}ms  trim_mean={tmean:.2f}ms  median={median:.2f}ms  "
                  f"min={mn:.2f}ms  max={mx:.2f}ms")
            if is_npu:
                stats = _npu_b.transfer_stats()
                if stats["offload_calls"] > 0:
                    per_iter_off = stats["bytes_offloaded"] / args.bench_iters / 1024**2
                    per_iter_pre = stats["bytes_prefetched"] / args.bench_iters / 1024**2
                    print(f"[bench] dram per-iter: "
                          f"{stats['offload_calls']//args.bench_iters} offloads "
                          f"({per_iter_off:.1f} MB/iter D2H), "
                          f"{stats['prefetch_calls']//args.bench_iters} prefetches "
                          f"({per_iter_pre:.1f} MB/iter H2D)")
    if hasattr(out, "logits"):
        logits = out.logits
    elif isinstance(out, torch.Tensor):
        logits = out
    elif isinstance(out, (tuple, list)):
        logits = out[0]
    else:
        logits = next(iter(out.values()))
    print(f"[main] OK  logits.shape={tuple(logits.shape)}  device={logits.device}")

    if args.device.startswith("npu"):
        from hfc.backends import npu as npu_backend
        stats = npu_backend.pool_stats()
        if stats:
            tot = sum(stats.values())
            print(f"[main] DRAM pool: {len(stats)} entries, {tot/1024**2:.1f} MB held")
        else:
            print("[main] DRAM pool empty (all prefetched back)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
