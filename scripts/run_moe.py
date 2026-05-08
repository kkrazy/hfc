#!/usr/bin/env python3
"""Validate the offload strategy on a Qwen2-MoE model (Qwen1.5-MoE-A2.7B).

Same pipeline as run_offload.py but uses random-initialized weights to skip
the 28 GB download — the offload policies depend on tensor shapes/sizes, not
weight values, and verify_rewrite compares the rewritten module to a deepcopy
of the original (same random weights on both sides).
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--device", default="npu:0")
    p.add_argument("--policy", default="kv", choices=["none", "largest", "budget", "kv"])
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--hbm-gb", type=float, default=2.0)
    p.add_argument("--no-run", action="store_true")
    p.add_argument("--scale", type=float, default=1.0,
                   help="shrink layers/experts. 1.0=full, 0.25 = quarter layers&experts")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--bench", action="store_true",
                   help="run baseline+rewritten, report mean latency over 10 iters")
    return p.parse_args()


def build_model(model_id: str, scale: float, dtype):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_id)
    if scale < 1.0:
        old = (cfg.num_hidden_layers,
               getattr(cfg, "num_experts", None),
               getattr(cfg, "moe_intermediate_size", None))
        cfg.num_hidden_layers = max(2, int(cfg.num_hidden_layers * scale))
        if hasattr(cfg, "num_experts"):
            cfg.num_experts = max(4, int(cfg.num_experts * scale))
        print(f"[build] scaled (layers,experts,moe_inter): {old} -> "
              f"({cfg.num_hidden_layers},{getattr(cfg,'num_experts',None)},"
              f"{getattr(cfg,'moe_intermediate_size',None)})")
    print(f"[build] model_type={cfg.model_type} layers={cfg.num_hidden_layers} "
          f"hidden={cfg.hidden_size}  experts={getattr(cfg,'num_experts',None)} "
          f"top_k={getattr(cfg,'num_experts_per_tok',None)}  "
          f"shared_experts={getattr(cfg,'shared_expert_intermediate_size',None)}")
    m = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype)
    m.eval()
    n = sum(p.numel() for p in m.parameters())
    print(f"[build] params = {n/1e9:.2f} B  ({n*2/1024**3:.1f} GB fp16)")
    return cfg, m


def main() -> int:
    args = parse_args()

    import torch
    print(f"[main] torch={torch.__version__}")
    if args.device.startswith("npu"):
        import torch_npu  # noqa
        print(f"[main] torch_npu={torch_npu.__version__}")

    import transformers
    print(f"[main] transformers={transformers.__version__}")

    from transformers.utils.fx import symbolic_trace, _SUPPORTED_MODELS

    cfg, model = build_model(args.model, args.scale, dtype=torch.float32)
    if cfg.model_type not in _SUPPORTED_MODELS:
        print(f"[main] WARN: {cfg.model_type} not in _SUPPORTED_MODELS")

    bsz = 1
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, args.seq_len))
    attention_mask = torch.ones(bsz, args.seq_len, dtype=torch.long)
    example_inputs = dict(input_ids=input_ids, attention_mask=attention_mask)

    print("[main] symbolic_trace ...")
    gm = symbolic_trace(model, input_names=["input_ids", "attention_mask"])
    nodes = list(gm.graph.nodes)
    by_op = {}
    for n in nodes:
        by_op[n.op] = by_op.get(n.op, 0) + 1
    print(f"[main] graph: {len(nodes)} nodes  by_op={by_op}")

    # MoE-specific node count
    moe_nodes = [n for n in nodes if "experts" in n.name or "gate" in n.name or "router" in n.name]
    print(f"[main] MoE-touching nodes: {len(moe_nodes)}")

    from hfc.profiler import profile_graph, print_profile_ranked
    print("[main] profiling (CPU fp32, Linear via einsum for aarch64 matmul bug) ...")
    # aarch64 PyTorch CPU matmul fails on (M,H)@(H,1) shapes (shared_expert_gate).
    # Replace Linear.forward with einsum, which uses a different kernel path.
    import torch.nn as _nn
    _orig_fwd = _nn.Linear.forward
    def _einsum_linear_forward(self, x):
        # x: (..., in_features), W: (out_features, in_features)
        out = torch.einsum("...i,oi->...o", x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out
    _nn.Linear.forward = _einsum_linear_forward
    profile = profile_graph(gm, example_inputs)
    print_profile_ranked(profile, top_k=args.top_k)

    from hfc.policy import LargestNTensors, BudgetPolicy, AttentionKVOnly

    if args.policy == "none":
        offload_set = set(); label = "none"
    elif args.policy == "largest":
        pol = LargestNTensors(n=args.n); offload_set = pol.select(profile); label = pol.describe()
    elif args.policy == "budget":
        pol = BudgetPolicy(int(args.hbm_gb * 1024**3)); offload_set = pol.select(profile); label = pol.describe()
    elif args.policy == "kv":
        pol = AttentionKVOnly(); offload_set = pol.select(profile); label = pol.describe()

    print(f"\n[main] policy={label} selected {len(offload_set)} nodes")

    if offload_set:
        from hfc.rewriter import rewrite_with_offload, verify_rewrite
        gm_orig = copy.deepcopy(gm)
        gm = rewrite_with_offload(gm, offload_set, profile)
        print(f"[main] rewritten: {len(list(gm.graph.nodes))} nodes")
        ok = verify_rewrite(gm_orig, gm, example_inputs, atol=1e-2, rtol=1e-2)
        if not ok:
            print("[main] WARN: rewrite changed outputs"); return 1

    # restore Linear forward before NPU run — NPU has working matmul
    _nn.Linear.forward = _orig_fwd

    if args.no_run:
        print("[main] --no-run done"); return 0

    dtype = getattr(torch, args.dtype)
    if args.device.startswith("npu"):
        from hfc.backends import npu as npu_backend
        npu_backend.install("npu")
        print("[main] NPU backend installed")

    print(f"[main] -> {args.device} dtype={args.dtype}")
    gm = gm.to(args.device).to(dtype)
    npu_in = {k: v.to(args.device) for k, v in example_inputs.items()}

    with torch.no_grad():
        for _ in range(3):  # warmup
            out = gm(**npu_in)
        if args.device.startswith("npu"):
            torch.npu.synchronize()

        if args.bench:
            import time
            t = time.perf_counter()
            for _ in range(10):
                out = gm(**npu_in)
            if args.device.startswith("npu"):
                torch.npu.synchronize()
            dt = (time.perf_counter() - t) / 10 * 1000
            print(f"[bench] mean latency = {dt:.2f} ms over 10 iters")

    logits = out.logits if hasattr(out, "logits") else (out[0] if isinstance(out, (tuple, list)) else next(iter(out.values())))
    print(f"[main] OK  logits.shape={tuple(logits.shape)}  device={logits.device}")

    if args.device.startswith("npu"):
        from hfc.backends import npu as npu_backend
        stats = npu_backend.pool_stats()
        if stats:
            tot = sum(stats.values())
            print(f"[main] DRAM pool: {len(stats)} entries, {tot/1024**2:.1f} MB held")
        else:
            print("[main] DRAM pool empty")
    return 0


if __name__ == "__main__":
    sys.exit(main())
