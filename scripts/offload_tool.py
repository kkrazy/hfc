#!/usr/bin/env python3
"""Interactive offload-tuning tool for HFC (Qwen3 MoE for now).

Workflow:
  1. python offload_tool.py analyze            → writes ~/hfc/out/candidates.txt
  2. edit candidates.txt — toggle [x] (offload) / [ ] (skip)
  3. python offload_tool.py bench              → runs baseline + selection,
                                                 prints latency / DMA / HBM-saved diff
  4. goto 2 until satisfied

Reuses run_qwen3_moe.py's compat shims and capture pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure run_qwen3_moe is importable (pulls in pytree/torch.compiler shims
# before transformers is touched).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_qwen3_moe as _q3  # noqa: E402

import argparse
import copy
import dataclasses
import re
from typing import Dict, List, Set


DEFAULT_SIZE_MB = 1.0
DEFAULT_GAP_NODES = 50
DEFAULT_BANDWIDTH_GBS = 32.0  # PCIe Gen4 x16, practical for pinned memory


# --------------------------------------------------------------------------- #
#  Candidate list
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Candidate:
    name: str
    nbytes: int
    producer_idx: int
    last_use_idx: int
    gap_nodes: int
    is_param: bool
    consumers: int
    xfer_ms: float
    selected: bool
    reason: str
    # Layer awareness — populated for params via build_fqn_map / lookup.
    fqn: str = ""
    layer_idx: int = -1     # -1 = not part of model.layers.{N}
    role: str = ""          # e.g. "self_attn.k_proj.weight"
    is_kv: bool = False

    @property
    def size_mb(self) -> float:
        return self.nbytes / 1024**2


# --------------------------------------------------------------------------- #
#  FQN map: id(parameter_tensor) -> {fqn, layer_idx, role, is_kv}
# --------------------------------------------------------------------------- #


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(.+)$")
_KV_RE = re.compile(r"(?:^|\.)(k_proj|v_proj)(?:\.|$)")


def build_fqn_map(model):
    """Walk model.named_parameters() and build id->info plus data_ptr->info maps.

    make_fx stores the same Parameter objects as the original model (no copy),
    so id() match works. data_ptr() is a fallback in case of detach/clone.
    """
    by_id: Dict[int, dict] = {}
    by_dp: Dict[int, dict] = {}
    for name, p in model.named_parameters():
        layer_idx = -1
        role = name
        m = _LAYER_RE.search(name)
        if m:
            layer_idx = int(m.group(1))
            role = m.group(2)
        info = {
            "fqn": name,
            "layer_idx": layer_idx,
            "role": role,
            "is_kv": bool(_KV_RE.search(name)),
        }
        by_id[id(p)] = info
        try:
            by_dp[p.data_ptr()] = info
        except Exception:
            pass
    return by_id, by_dp


def _lookup_fqn(name: str, gm, by_id, by_dp) -> dict:
    import torch
    attr = getattr(gm, name, None)
    if attr is None:
        return {}
    info = by_id.get(id(attr))
    if info is None and isinstance(attr, torch.Tensor):
        try:
            info = by_dp.get(attr.data_ptr())
        except Exception:
            info = None
    return info or {}


def build_candidates(profile, fqn_map, gm, size_mb_thresh, gap_thresh, bw_gbs) -> List[Candidate]:
    by_id, by_dp = fqn_map
    cands: List[Candidate] = []
    for name, info in profile.items():
        if info.node_op in ("placeholder", "output"):
            continue
        if info.nbytes <= 0 or not info.consumers:
            continue
        last_use = max(profile[c].topo_idx for c in info.consumers if c in profile)
        gap = last_use - info.topo_idx
        is_param = info.node_op == "get_attr"
        xfer_ms = 2 * info.nbytes / (bw_gbs * 1e9) * 1000

        fqn = ""
        layer_idx = -1
        role = ""
        is_kv = False
        if is_param:
            f = _lookup_fqn(name, gm, by_id, by_dp)
            fqn = f.get("fqn", "")
            layer_idx = f.get("layer_idx", -1)
            role = f.get("role", "")
            is_kv = f.get("is_kv", False)

        big = info.nbytes >= size_mb_thresh * 1024**2

        # Selection rule — three classes:
        #   1. KV projection weight: ALWAYS selected (cross-step lifetime hidden
        #      from single-forward gap metric — gap=1 in graph but ≫gap_thresh
        #      in autoregressive workload).
        #   2. Other layer/shared param: not default-selected; user opts into
        #      weight-streaming policy by toggling [x] manually or via a
        #      future --stream-weights flag.
        #   3. Activation: gap-based rule (size AND gap thresholds).
        if is_param and is_kv:
            selected = True
            reason = "K/V proj — reused across decode steps; cross-step lifetime ≫ in-graph gap"
        elif is_param and layer_idx >= 0:
            selected = False
            reason = f"opt-in: layer {layer_idx} weight streaming"
        elif is_param:
            selected = False
            reason = "opt-in: shared param (embed/head/norm)"
        else:
            long_live = gap >= gap_thresh
            if big and long_live:
                selected, reason = True, "activation: large + long live range"
            elif not big and not long_live:
                selected, reason = False, "activation: too small AND gap too short"
            elif not big:
                selected, reason = False, f"activation too small (<{size_mb_thresh:.1f}MB)"
            else:
                selected, reason = False, f"activation gap too short ({gap}<{gap_thresh})"

        cands.append(Candidate(
            name=name, nbytes=info.nbytes,
            producer_idx=info.topo_idx, last_use_idx=last_use,
            gap_nodes=gap, is_param=is_param, consumers=len(info.consumers),
            xfer_ms=xfer_ms, selected=selected, reason=reason,
            fqn=fqn, layer_idx=layer_idx, role=role, is_kv=is_kv,
        ))
    cands.sort(key=lambda c: c.nbytes * max(c.gap_nodes, 1), reverse=True)
    return cands


# --------------------------------------------------------------------------- #
#  candidates.txt I/O — human-editable text format
# --------------------------------------------------------------------------- #


CANDS_HEADER = (
    "# HFC offload candidates — toggle [x] (offload) / [ ] (skip)\n"
    "# Lines starting with # are ignored. Re-run `bench` after editing.\n"
    "#\n"
    "# Cost model has three classes:\n"
    "#   - K/V projection weight:   always candidate (cross-step lifetime in\n"
    "#                              autoregressive decoding ≫ single-forward gap)\n"
    "#   - other layer/shared param: opt-in for weight-streaming policy\n"
    "#   - activation:              gap-based — select iff size>={size_mb:.1f}MB AND gap>={gap}n\n"
    "#\n"
    "# Model:  {model}  scale={scale}  seq_len={seq_len}\n"
    "# Graph:  {nodes} nodes  total tensor memory: {total_mb:.1f} MB\n"
    "# Layers: {n_layers}  KV-proj total: {kv_mb:.1f} MB\n"
    "# Bandwidth assumption: {bw_gbs:.0f} GB/s\n"
    "#\n"
    "# Per line:  [mark] node_name  size  KV?  gap_nodes  xfer_ms  # reason / fqn\n"
    "#\n"
)

CANDS_LINE_RE = re.compile(r"^\[([x ])\]\s+(\S+)")

# Activations smaller than this are dropped from the file to keep it editable.
# The user can still bench them by manually adding a [x] line — the bench reads
# every [x] regardless of whether it appeared here.
_ACT_DISPLAY_FLOOR_MB = 0.10


def _format_line(c: Candidate) -> str:
    mark = "[x]" if c.selected else "[ ]"
    kv = "KV" if c.is_kv else "  "
    fqn_short = (c.fqn.replace("model.", "") if c.fqn else "")
    note = c.reason if not fqn_short else f"{fqn_short} — {c.reason}"
    return (
        f"{mark} {c.name:30s} {c.size_mb:8.2f}MB  {kv}  "
        f"gap={c.gap_nodes:5d}  xfer={c.xfer_ms:7.2f}ms  # {note}\n"
    )


def write_candidates(path: Path, cands: List[Candidate], header_kwargs: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    # Group: per-layer params, shared params, activations.
    by_layer: Dict[int, List[Candidate]] = {}
    shared_params: List[Candidate] = []
    activations: List[Candidate] = []
    for c in cands:
        if c.is_param and c.layer_idx >= 0:
            by_layer.setdefault(c.layer_idx, []).append(c)
        elif c.is_param:
            shared_params.append(c)
        else:
            activations.append(c)

    header_kwargs = dict(header_kwargs)
    header_kwargs["n_layers"] = len(by_layer)
    header_kwargs["kv_mb"] = sum(c.size_mb for c in cands if c.is_kv)

    with path.open("w") as f:
        f.write(CANDS_HEADER.format(**header_kwargs))

        for L in sorted(by_layer.keys()):
            group = by_layer[L]
            tot = sum(c.size_mb for c in group)
            kv_tot = sum(c.size_mb for c in group if c.is_kv)
            f.write(f"\n# === LAYER {L:>2}  total={tot:.2f}MB  KV={kv_tot:.2f}MB ===\n")
            for c in sorted(group, key=lambda c: -c.nbytes):
                f.write(_format_line(c))

        if shared_params:
            f.write("\n# === SHARED / NON-LAYER PARAMS (embed_tokens, lm_head, final norm) ===\n")
            for c in sorted(shared_params, key=lambda c: -c.nbytes):
                f.write(_format_line(c))

        if activations:
            shown = [c for c in activations if c.size_mb >= _ACT_DISPLAY_FLOOR_MB or c.selected]
            shown.sort(key=lambda c: c.nbytes * max(c.gap_nodes, 1), reverse=True)
            f.write(f"\n# === ACTIVATIONS  ({len(shown)} of {len(activations)} shown; "
                    f">={_ACT_DISPLAY_FLOOR_MB}MB or default-selected) ===\n")
            for c in shown:
                f.write(_format_line(c))


def read_selection(path: Path) -> Set[str]:
    sel: Set[str] = set()
    with path.open() as f:
        for line in f:
            m = CANDS_LINE_RE.match(line.rstrip())
            if m and m.group(1) == "x":
                sel.add(m.group(2))
    return sel


# --------------------------------------------------------------------------- #
#  Capture / profile (shared by both subcommands)
# --------------------------------------------------------------------------- #


def _capture_and_profile(args):
    import torch
    cfg, model = _q3.build_model_from_config(args.model, dtype=torch.float32, scale=args.scale)
    _q3._patch_transformers_for_export()
    _q3._patch_moe_dense()
    # FQN map MUST be built from `model` BEFORE make_fx, while parameter
    # objects still have stable identity.
    fqn_map = build_fqn_map(model)
    bsz = 1
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, args.seq_len))
    attention_mask = torch.ones(bsz, args.seq_len, dtype=torch.long)
    example_inputs = dict(input_ids=input_ids, attention_mask=attention_mask)
    gm, _ = _q3.capture_via_make_fx(model, example_inputs)

    from hfc.profiler import profile_graph
    keys = list(example_inputs.keys())
    ph_targets = [n.target for n in gm.graph.nodes if n.op == "placeholder"]
    profile_inputs = {pt: example_inputs[keys[i]] for i, pt in enumerate(ph_targets)}
    profile = profile_graph(gm, profile_inputs)
    return gm, profile, profile_inputs, fqn_map


# --------------------------------------------------------------------------- #
#  analyze
# --------------------------------------------------------------------------- #


def cmd_analyze(args) -> int:
    gm, profile, _, fqn_map = _capture_and_profile(args)
    cands = build_candidates(profile, fqn_map, gm, args.size_mb, args.gap_nodes, args.bandwidth_gbs)
    total_bytes = sum(info.nbytes for info in profile.values())
    nodes = len(list(gm.graph.nodes))
    out_path = Path(args.cands)
    write_candidates(out_path, cands, dict(
        size_mb=args.size_mb, gap=args.gap_nodes, model=args.model,
        scale=args.scale, seq_len=args.seq_len, nodes=nodes,
        total_mb=total_bytes / 1024**2, bw_gbs=args.bandwidth_gbs,
    ))

    selected = [c for c in cands if c.selected]
    sel_bytes = sum(c.nbytes for c in selected)

    # Per-layer summary
    layers: Dict[int, List[Candidate]] = {}
    for c in cands:
        if c.is_param and c.layer_idx >= 0:
            layers.setdefault(c.layer_idx, []).append(c)
    kv_cands = [c for c in cands if c.is_kv]
    shared = [c for c in cands if c.is_param and c.layer_idx < 0]

    print(f"\n[analyze] wrote {out_path}")
    print(f"[analyze] {len(cands)} candidates total")
    print(f"[analyze]   {len(layers)} layers  ({sum(len(g) for g in layers.values())} per-layer params)")
    print(f"[analyze]   {len(shared)} shared/non-layer params "
          f"({sum(c.size_mb for c in shared):.0f} MB)")
    print(f"[analyze]   {len(kv_cands)} K/V proj weights "
          f"({sum(c.size_mb for c in kv_cands):.1f} MB) — default-selected")
    print(f"[analyze] {len(selected)} selected by default → "
          f"{sel_bytes/1024**2:.1f} MB HBM relief "
          f"({100*sel_bytes/max(total_bytes,1):.1f}% of {total_bytes/1024**2:.0f} MB total)")

    if layers:
        print(f"\nPer-layer breakdown:")
        print(f"  layer  total_MB  KV_MB   FFN_MB    n_params")
        for L in sorted(layers.keys()):
            grp = layers[L]
            tot = sum(c.size_mb for c in grp)
            kv = sum(c.size_mb for c in grp if c.is_kv)
            ffn = sum(c.size_mb for c in grp
                      if any(s in c.role for s in ("mlp", "experts", "ffn", "gate_proj", "up_proj", "down_proj")))
            print(f"   {L:>3d}    {tot:7.2f}  {kv:6.2f}  {ffn:7.2f}     {len(grp):>3d}")

    print(f"\nTop 20 by (size × gap) — note: aten gap=1 for all params is expected:")
    print(f"  sel  {'name':30s}  {'size':>10s}  KV  {'gap':>5s}  fqn / reason")
    for c in cands[:20]:
        mark = "[x]" if c.selected else "[ ]"
        kv = "KV" if c.is_kv else "  "
        note = (c.fqn.replace("model.", "") if c.fqn else c.reason)
        print(f"  {mark}  {c.name:30s}  {c.size_mb:7.2f} MB  {kv}  "
              f"{c.gap_nodes:5d}  {note}")
    print(f"\nEdit {out_path} to toggle [x]/[ ], then run: python offload_tool.py bench")
    return 0


# --------------------------------------------------------------------------- #
#  bench
# --------------------------------------------------------------------------- #


def _rewrite_device_kwargs(gm, target_dev):
    """Same fix as run_qwen3_moe.py — make_fx bakes device='cpu' into kwargs."""
    import torch
    cpu_dev = torch.device("cpu")
    n = 0
    for node in gm.graph.nodes:
        if node.op != "call_function" or "device" not in node.kwargs:
            continue
        d = node.kwargs["device"]
        if d == cpu_dev or d is None or (isinstance(d, str) and d == "cpu"):
            new_kw = dict(node.kwargs)
            new_kw["device"] = target_dev
            node.kwargs = new_kw
            n += 1
    if n:
        gm.graph.lint()
        gm.recompile()
    return n


def _bench_one(gm, profile_inputs, args, label, ph_targets):
    import torch
    import time
    target_dev = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    _rewrite_device_kwargs(gm, target_dev)
    gm = gm.to(args.device).to(dtype)
    npu_in = {k: v.to(args.device) for k, v in profile_inputs.items()}

    def _invoke():
        return gm(*(npu_in[t] for t in ph_targets))

    from hfc.backends import npu as npu_b
    with torch.no_grad(), torch.device(args.device):
        for _ in range(args.bench_warmup):
            _invoke()
        torch.npu.synchronize()
        npu_b.reset_counters()
        samples = []
        for _ in range(args.bench_iters):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            _invoke()
            torch.npu.synchronize()
            samples.append((time.perf_counter() - t0) * 1000)

    samples.sort()
    n = len(samples)
    trim = max(1, n // 10)
    trimmed = samples[trim:-trim] if n > 2 * trim else samples
    ts = npu_b.transfer_stats()
    stats = {
        "mean": sum(samples) / n,
        "trim_mean": sum(trimmed) / len(trimmed),
        "median": samples[n // 2],
        "min": samples[0],
        "max": samples[-1],
        "mb_offload": ts["bytes_offloaded"] / args.bench_iters / 1024**2,
        "mb_prefetch": ts["bytes_prefetched"] / args.bench_iters / 1024**2,
        "n_offload": ts["offload_calls"] // args.bench_iters,
        "n_prefetch": ts["prefetch_calls"] // args.bench_iters,
    }
    print(f"[{label:8s}] mean={stats['mean']:7.2f}ms  trim={stats['trim_mean']:7.2f}ms  "
          f"median={stats['median']:7.2f}ms  min={stats['min']:6.2f}ms  max={stats['max']:6.2f}ms")
    return stats


def cmd_bench(args) -> int:
    import torch
    cands_path = Path(args.cands)
    if not cands_path.exists():
        print(f"[bench] ERROR: {cands_path} not found. Run `analyze` first.")
        return 2
    selection = read_selection(cands_path)
    print(f"[bench] reading selection from {cands_path}: {len(selection)} nodes")

    gm_orig, profile, profile_inputs, _fqn_map = _capture_and_profile(args)
    ph_targets = [n.target for n in gm_orig.graph.nodes if n.op == "placeholder"]
    nodes_total = len(list(gm_orig.graph.nodes))

    valid = {n for n in selection if n in profile}
    invalid = selection - valid
    if invalid:
        print(f"[bench] WARN: {len(invalid)} selected nodes missing in fresh capture "
              f"(graph may have changed): {sorted(invalid)[:3]}...")
    sel_bytes = sum(profile[n].nbytes for n in valid)
    total_bytes = sum(info.nbytes for info in profile.values())
    print(f"[bench] selection: {len(valid)} nodes, {sel_bytes/1024**2:.1f} MB "
          f"of {total_bytes/1024**2:.1f} MB total tensor memory")

    # Make rewritten copy + verify on CPU before installing real backend.
    from hfc.rewriter import rewrite_with_offload, verify_rewrite
    gm_rewritten = copy.deepcopy(gm_orig)
    if valid:
        gm_rewritten = rewrite_with_offload(gm_rewritten, valid, profile)
        ok = verify_rewrite(gm_orig, gm_rewritten, profile_inputs, atol=1e-2, rtol=1e-2)
        if not ok:
            print("[bench] verify FAILED — rewrite changed outputs")
            return 1
        print(f"[bench] rewritten graph: {len(list(gm_rewritten.graph.nodes))} nodes "
              f"({len(list(gm_rewritten.graph.nodes)) - nodes_total:+d})")

    # Now install NPU backend and bench.
    if args.device.startswith("npu"):
        import torch_npu  # noqa
        from hfc.backends import npu as npu_b
        npu_b.install("npu")

    print(f"\n[bench] running on {args.device} dtype={args.dtype}  "
          f"warmup={args.bench_warmup}  iters={args.bench_iters}")
    base = _bench_one(gm_orig, profile_inputs, args, "baseline", ph_targets)
    if not valid:
        print("\n[bench] no nodes selected — skipping offload run")
        return 0
    # Free baseline gm's NPU memory before second run.
    del gm_orig
    if args.device.startswith("npu"):
        torch.npu.empty_cache()
    off = _bench_one(gm_rewritten, profile_inputs, args, "offload", ph_targets)

    print()
    print("=" * 84)
    print(f"  Comparison  (model={args.model}  scale={args.scale}  seq_len={args.seq_len})")
    print("=" * 84)
    fmt = "{:30s} {:>14s} {:>14s} {:>14s} {:>10s}"
    print(fmt.format("metric", "baseline", "offload", "delta", "delta%"))
    print("-" * 84)

    def row(label, b, o, fmtspec=":14.2f", suffix=""):
        d = o - b
        pct = (100 * d / b) if b else float("nan")
        print(("{:30s} {%s} {%s} {%s%s%s} {%s}" % (
            fmtspec, fmtspec, "+" if d >= 0 else "", fmtspec, "", ":+9.1f%%"
        )).format(label, b, o, d, pct))

    def fmtrow(label, b, o, kind="ms"):
        d = o - b
        pct = (100 * d / b) if b else 0.0
        print(f"{label:30s} {b:14.2f} {o:14.2f} {d:+14.2f} {pct:+9.1f}%")

    fmtrow("mean latency (ms)", base["mean"], off["mean"])
    fmtrow("trim_mean (ms)",    base["trim_mean"], off["trim_mean"])
    fmtrow("median (ms)",       base["median"], off["median"])
    fmtrow("min (ms)",          base["min"], off["min"])
    print("-" * 84)
    print(f"{'D2H per iter (MB)':30s} {0.0:14.1f} {off['mb_offload']:14.1f} "
          f"{off['mb_offload']:+14.1f}")
    print(f"{'H2D per iter (MB)':30s} {0.0:14.1f} {off['mb_prefetch']:14.1f} "
          f"{off['mb_prefetch']:+14.1f}")
    print(f"{'offload calls / iter':30s} {0:14d} {off['n_offload']:14d}")
    print(f"{'est HBM relief (MB)':30s} {0.0:14.1f} {sel_bytes/1024**2:14.1f}")
    print()
    delta_ms = off["trim_mean"] - base["trim_mean"]
    delta_pct = 100 * delta_ms / base["trim_mean"] if base["trim_mean"] else 0
    verdict = "WORSE" if delta_ms > 0 else "BETTER"
    print(f"[bench] verdict: offload is {verdict} by {abs(delta_ms):.2f} ms "
          f"({delta_pct:+.1f}%) — bought {sel_bytes/1024**2:.1f} MB HBM "
          f"for {off['mb_offload']+off['mb_prefetch']:.1f} MB DMA / iter")
    return 0


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    common.add_argument("--scale", type=float, default=0.1)
    common.add_argument("--seq-len", type=int, default=32)
    common.add_argument("--cands", default=str(Path.home() / "hfc" / "out" / "candidates.txt"))

    pa = sub.add_parser("analyze", parents=[common],
                        help="capture+profile, write candidates.txt")
    pa.add_argument("--size-mb", type=float, default=DEFAULT_SIZE_MB)
    pa.add_argument("--gap-nodes", type=int, default=DEFAULT_GAP_NODES)
    pa.add_argument("--bandwidth-gbs", type=float, default=DEFAULT_BANDWIDTH_GBS)
    pa.set_defaults(func=cmd_analyze)

    pb = sub.add_parser("bench", parents=[common],
                        help="bench baseline+selection on NPU")
    pb.add_argument("--device", default="npu:0")
    pb.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    pb.add_argument("--bench-iters", type=int, default=20)
    pb.add_argument("--bench-warmup", type=int, default=3)
    pb.set_defaults(func=cmd_bench)

    args = p.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
