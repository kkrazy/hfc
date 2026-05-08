#!/usr/bin/env python3
"""Capture an FX graph of a HuggingFace causal-LM and run it on Ascend NPU.

This is the foundation for the KV-cache graph rewriter: capture happens on CPU
(FX tracing is hardware-agnostic), then the GraphModule is moved to npu:0 and
executed there to prove the captured graph is faithful to the eager model.

Outputs:
  stdout — versions, NPU device info, graph stats, forward output summary
  out/graph.txt — FX print_tabular dump, generated forward source, KV-touching
                  node list (the future rewriter's pattern targets)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="facebook/opt-125m",
                   help="HF model id (default: facebook/opt-125m — "
                        "small, definitely in _SUPPORTED_MODELS)")
    p.add_argument("--seq-len", type=int, default=32,
                   help="dummy input sequence length for the NPU forward pass")
    p.add_argument("--past-len", type=int, default=0,
                   help="length of past_key_values to fake (0 = prefill, >0 = decode)")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    p.add_argument("--device", default="npu:0",
                   help="execution device for the captured graph (e.g. npu:0, cpu)")
    p.add_argument("--out-dir", default=str(Path.home() / "hfc" / "out"),
                   help="directory for graph.txt etc.")
    p.add_argument("--no-run", action="store_true",
                   help="skip the NPU forward pass (capture-only)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_txt = out_dir / "graph.txt"

    # --- Imports (after argparse so --help works without torch installed) ---
    import torch
    print(f"[capture] torch        = {torch.__version__}")

    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401  registers the npu device
        print(f"[capture] torch_npu    = {torch_npu.__version__}")
        if not torch.npu.is_available():
            print("[capture] ERROR: torch.npu.is_available() == False — "
                  "is CANN env sourced?", file=sys.stderr)
            return 2
        print(f"[capture] npu count    = {torch.npu.device_count()}")
        print(f"[capture] npu[0] name  = {torch.npu.get_device_name(0)}")

    import transformers
    from transformers import AutoModelForCausalLM, AutoConfig
    from transformers.utils.fx import symbolic_trace, _SUPPORTED_MODELS
    print(f"[capture] transformers = {transformers.__version__}")

    dtype = getattr(torch, args.dtype)

    # --- Load model on CPU ---
    print(f"[capture] loading {args.model} (dtype={args.dtype}) ...")
    config = AutoConfig.from_pretrained(args.model)
    if config.model_type not in _SUPPORTED_MODELS:
        print(f"[capture] WARNING: model_type={config.model_type!r} is not in "
              f"transformers.utils.fx._SUPPORTED_MODELS; tracing may fail.",
              file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval()

    # --- FX symbolic trace ---
    # Including past_key_values in input_names tells the HF tracer to take the
    # cached-decode branch — this is what produces the cat/getitem nodes the
    # KV rewriter will pattern-match against.
    input_names = ["input_ids", "attention_mask", "past_key_values"]
    print(f"[capture] symbolic_trace input_names={input_names}")
    try:
        gm = symbolic_trace(model, input_names=input_names)
    except Exception as e:
        print(f"[capture] tracing with past_key_values failed ({e!r}); "
              "retrying prefill-only", file=sys.stderr)
        input_names = ["input_ids", "attention_mask"]
        gm = symbolic_trace(model, input_names=input_names)

    # --- Graph stats + dump ---
    nodes = list(gm.graph.nodes)
    by_op: dict[str, int] = {}
    for n in nodes:
        by_op[n.op] = by_op.get(n.op, 0) + 1

    # Anything that smells like KV-cache plumbing — what the rewriter targets.
    def _is_kv_node(n) -> bool:
        s = f"{n.op}|{n.target}|{n.name}".lower()
        if "past_key" in s or "past_value" in s or "presents" in s:
            return True
        if n.op == "call_function" and "cat" in str(n.target).lower():
            # cat is the KV append op for most decoder models
            for a in n.args:
                if hasattr(a, "name") and ("past" in a.name or "key" in a.name or "value" in a.name):
                    return True
        if "scaled_dot_product_attention" in str(n.target):
            return True
        return False

    kv_nodes = [n for n in nodes if _is_kv_node(n)]

    print(f"[capture] FX nodes total = {len(nodes)}  by op = {by_op}")
    print(f"[capture] KV-touching nodes = {len(kv_nodes)}")

    with graph_txt.open("w") as f:
        f.write(f"# capture_graph.py output\n")
        f.write(f"# model={args.model}  dtype={args.dtype}  "
                f"input_names={input_names}\n")
        f.write(f"# torch={torch.__version__}  transformers={transformers.__version__}\n")
        f.write(f"# total_nodes={len(nodes)}  by_op={by_op}\n\n")

        f.write("=" * 78 + "\n=== gm.graph.print_tabular() ===\n" + "=" * 78 + "\n")
        # print_tabular writes to stdout — capture it
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            gm.graph.print_tabular()
        f.write(buf.getvalue())

        f.write("\n" + "=" * 78 + "\n=== gm.code (generated forward) ===\n"
                + "=" * 78 + "\n")
        f.write(gm.code)

        f.write("\n" + "=" * 78 + f"\n=== KV-touching nodes ({len(kv_nodes)}) ===\n"
                + "=" * 78 + "\n")
        for n in kv_nodes:
            f.write(f"  {n.op:14s} {str(n.target):40s} name={n.name}\n")

    print(f"[capture] wrote {graph_txt}  ({graph_txt.stat().st_size} bytes)")

    if args.no_run:
        print("[capture] --no-run set, skipping forward pass")
        return 0

    # --- Move to NPU and run a forward pass ---
    print(f"[capture] moving GraphModule to {args.device} ...")
    gm = gm.to(args.device).to(dtype)

    vocab = config.vocab_size
    bsz = 1
    seq_len = args.seq_len
    past_len = args.past_len

    input_ids = torch.randint(0, vocab, (bsz, seq_len), dtype=torch.long, device=args.device)
    attention_mask = torch.ones(bsz, seq_len + past_len, dtype=torch.long, device=args.device)

    fwd_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)

    if "past_key_values" in input_names:
        # Build per-layer (k, v) tuples with the right shapes for this config.
        # FX traced the decode-with-cache path, so the graph expects real tensors.
        # For prefill (past_len=0) we pass zero-length past — the graph's .size()
        # calls work fine on (batch, n_heads, 0, head_dim).
        n_layers = config.num_hidden_layers
        n_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        past = tuple(
            (
                torch.zeros(bsz, n_heads, past_len, head_dim, dtype=dtype, device=args.device),
                torch.zeros(bsz, n_heads, past_len, head_dim, dtype=dtype, device=args.device),
            )
            for _ in range(n_layers)
        )
        fwd_kwargs["past_key_values"] = past

    print(f"[capture] forward kwargs: "
          + ", ".join(f"{k}={tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__}"
                      for k, v in fwd_kwargs.items()))

    with torch.no_grad():
        out = gm(**fwd_kwargs)

    # gm output is a tuple/dict-ish thing; extract logits
    logits = getattr(out, "logits", None)
    if logits is None and isinstance(out, (tuple, list)):
        logits = out[0]
    if logits is None and isinstance(out, dict):
        logits = out.get("logits")
        if logits is None:
            logits = next(iter(out.values()))

    print(f"[capture] OK — logits.shape = {tuple(logits.shape)}  "
          f"device = {logits.device}  dtype = {logits.dtype}")
    print(f"[capture] logits[0, 0, :5] = {logits[0, 0, :5].float().cpu().tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
