#!/usr/bin/env python3
"""End-to-end pipeline: capture FX graph → profile tensors → apply policy → rewrite → run on NPU.

Usage:
    python run_offload.py --model facebook/opt-125m --policy largest --n 20
    python run_offload.py --model Qwen/Qwen2-0.5B --policy budget --hbm-gb 0.5
    python run_offload.py --model Qwen/Qwen2-0.5B --policy kv
    python run_offload.py --model Qwen/Qwen2-0.5B --policy none   # baseline, no rewrite
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

# Make sure hfc package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Profile + offload FX graph for HF models on Ascend NPU")
    p.add_argument("--model", default="facebook/opt-125m")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--past-len", type=int, default=0,
                   help="past KV length (0 = prefill). Note: >0 may fail on some models due to FX shape baking.")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--device", default="npu:0")
    p.add_argument("--no-run", action="store_true", help="skip NPU execution (profile + rewrite only)")
    p.add_argument("--out-dir", default=str(Path.home() / "hfc" / "out"))
    # policy
    p.add_argument("--policy", default="largest", choices=["none", "largest", "budget", "kv"])
    p.add_argument("--n", type=int, default=20, help="for 'largest' policy: how many tensors to offload")
    p.add_argument("--hbm-gb", type=float, default=1.0, help="for 'budget' policy: HBM budget in GB")
    # display
    p.add_argument("--top-k", type=int, default=30, help="how many tensors to show in profile summary")
    p.add_argument("--dump-graph", action="store_true", help="write rewritten graph to out-dir")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- imports (after argparse so --help is fast) ---
    import torch

    print(f"[main] torch = {torch.__version__}")
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
        print(f"[main] torch_npu = {torch_npu.__version__}")
        if not torch.npu.is_available():
            print("[main] ERROR: NPU not available", file=sys.stderr)
            return 2
        print(f"[main] npu count = {torch.npu.device_count()}")

    from transformers import AutoModelForCausalLM, AutoConfig
    from transformers.utils.fx import symbolic_trace

    from hfc.profiler import profile_graph, print_profile_ranked
    from hfc.policy import LargestNTensors, BudgetPolicy, AttentionKVOnly, OffloadPolicy
    from hfc.rewriter import rewrite_with_offload, verify_rewrite

    dtype = getattr(torch, args.dtype)

    # ------------------------------------------------------------------ #
    #  1. Load model + FX trace
    # ------------------------------------------------------------------ #
    print(f"[main] loading {args.model} ...")
    config = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval()

    input_names = ["input_ids", "attention_mask"]
    print(f"[main] symbolic_trace(input_names={input_names})")
    gm = symbolic_trace(model, input_names=input_names)
    nodes = list(gm.graph.nodes)
    print(f"[main] graph: {len(nodes)} nodes")

    # ------------------------------------------------------------------ #
    #  2. Build example inputs for profiling (CPU, fp32)
    # ------------------------------------------------------------------ #
    # Profile in fp32 — CPU kernels (LayerNorm, etc.) don't support fp16.
    # Shapes are the same; byte sizes are 2× for fp32 but relative ranking
    # is unchanged, so the policy decisions are identical.
    vocab = config.vocab_size
    bsz = 1
    input_ids = torch.randint(0, vocab, (bsz, args.seq_len))
    attention_mask = torch.ones(bsz, args.seq_len, dtype=torch.long)
    example_inputs = dict(input_ids=input_ids, attention_mask=attention_mask)

    # Temporarily cast model to fp32 for profiling
    gm_fp32 = gm.float()

    # ------------------------------------------------------------------ #
    #  3. Profile: run interpreter, record tensor sizes
    # ------------------------------------------------------------------ #
    print("[main] profiling tensor sizes (fp32 on CPU) ...")
    profile = profile_graph(gm_fp32, example_inputs)
    print_profile_ranked(profile, top_k=args.top_k)

    # ------------------------------------------------------------------ #
    #  4. Select offload policy
    # ------------------------------------------------------------------ #
    policy: OffloadPolicy
    if args.policy == "none":
        print("[main] policy=none — no rewriting")
        policy = None
    elif args.policy == "largest":
        policy = LargestNTensors(n=args.n)
    elif args.policy == "budget":
        policy = BudgetPolicy(hbm_budget_bytes=int(args.hbm_gb * 1024**3))
    elif args.policy == "kv":
        policy = AttentionKVOnly()

    if policy is not None:
        offload_set = policy.select(profile)
        print(f"\n[main] policy={policy.describe()} selected {len(offload_set)} nodes to offload:")
        for name in sorted(offload_set):
            info = profile.get(name)
            sz = f"{info.nbytes:,} B" if info else "?"
            print(f"  {name:40s}  {sz}")
    else:
        offload_set = set()

    # ------------------------------------------------------------------ #
    #  5. Rewrite (deepcopy to preserve original for verification)
    # ------------------------------------------------------------------ #
    gm_original = copy.deepcopy(gm)

    if offload_set:
        print("\n[main] rewriting graph ...")
        gm = rewrite_with_offload(gm, offload_set, profile)
        print(f"[main] rewritten graph: {len(list(gm.graph.nodes))} nodes")

    if args.dump_graph:
        graph_path = out_dir / "rewritten_graph.txt"
        with graph_path.open("w") as f:
            f.write(f"# model={args.model}  policy={args.policy}\n")
            f.write(f"# original nodes={len(nodes)}  rewritten nodes={len(list(gm.graph.nodes))}\n\n")
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                gm.graph.print_tabular()
            f.write(buf.getvalue())
            f.write("\n\n# === gm.code ===\n")
            f.write(gm.code)
        print(f"[main] wrote {graph_path}")

    # ------------------------------------------------------------------ #
    #  6. (Optional) verify on CPU that rewrite doesn't change outputs
    # ------------------------------------------------------------------ #
    if offload_set:
        print("\n[main] verifying rewrite correctness on CPU ...")
        ok = verify_rewrite(gm_original, gm, example_inputs, atol=1e-2, rtol=1e-2)
        if not ok:
            print("[main] WARNING: rewrite changed outputs — not running on NPU")
            return 1

    if args.no_run:
        print("[main] --no-run, done")
        return 0

    # ------------------------------------------------------------------ #
    #  7. Run on NPU
    # ------------------------------------------------------------------ #
    # Activate real DRAM backend AFTER CPU verification, BEFORE NPU execution.
    # CPU verification uses the identity stubs; NPU uses real pinned-memory copies.
    # ------------------------------------------------------------------ #
    if args.device.startswith("npu"):
        from hfc.backends import npu as npu_backend
        npu_backend.install("npu")
        print(f"[main] NPU DRAM backend installed (real pinned-memory transfers)")
    else:
        print("[main] CPU device — using identity offload stubs")

    print(f"\n[main] moving to {args.device} ...")
    gm = gm.to(args.device).to(dtype)
    npu_inputs = {k: v.to(args.device) for k, v in example_inputs.items()}

    with torch.no_grad():
        out = gm(**npu_inputs)

    logits = getattr(out, "logits", None)
    if logits is None and isinstance(out, dict):
        logits = out.get("logits")
    if logits is None and isinstance(out, (tuple, list)):
        logits = out[0]

    print(f"[main] OK — logits.shape = {tuple(logits.shape)}  "
          f"device = {logits.device}  dtype = {logits.dtype}")
    print(f"[main] logits[0, 0, :5] = {logits[0, 0, :5].float().cpu().tolist()}")

    # Show DRAM pool stats
    if args.device.startswith("npu"):
        from hfc.backends import npu as npu_backend
        stats = npu_backend.pool_stats()
        if stats:
            total_dram = sum(stats.values())
            print(f"[main] DRAM pool after forward: {len(stats)} entries, "
                  f"{total_dram:,} bytes ({total_dram/1024**2:.1f} MB) held in pinned CPU memory")
        else:
            print("[main] DRAM pool empty (all tensors prefetched back)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
