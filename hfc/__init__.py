"""Tensor-size-driven offload rewriter for HuggingFace models on tiered memory.

Pipeline:  Profile → Policy → Rewrite → Execute
"""
