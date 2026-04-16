#!/usr/bin/env python3
"""
example_qwen.py
===============
Demonstrate model analysis for Qwen2-7B (≈8B parameter model).

Run:
  python examples/example_qwen.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root or examples/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_parser import load_config
from src.calculators.base import DType
from src.calculators.model import model_stats
from src.visualizer import visualize
from src.report import print_report


def main() -> None:
    config_path = Path(__file__).parent.parent / "configs" / "qwen2_7b.json"
    cfg = load_config(config_path, name="Qwen2-7B")

    print("=" * 72)
    print("  QWEN2-7B  —  Architecture Visualisation & Computation Analysis")
    print("=" * 72)
    print()

    # --- ASCII diagram ---------------------------------------------------
    print(visualize(cfg, use_flash_attn=False))
    print()

    # --- Prefill analysis (seq_len=2048) ---------------------------------
    print("─" * 72)
    print("  PREFILL  (batch=1, seq_len=2048, bf16, Standard Attention)")
    print("─" * 72)
    ms_prefill = model_stats(cfg, seq_len=2048, batch_size=1, dtype=DType.BF16, use_flash_attn=False)
    print_report(ms_prefill)

    # --- Flash attention comparison (seq_len=8192) -----------------------
    print("─" * 72)
    print("  PREFILL  (batch=1, seq_len=8192, bf16, Flash Attention)")
    print("─" * 72)
    ms_flash = model_stats(cfg, seq_len=8192, batch_size=1, dtype=DType.BF16, use_flash_attn=True)
    print_report(ms_flash)

    # --- Decode analysis (seq_len=1) -------------------------------------
    print("─" * 72)
    print("  DECODE   (batch=1, seq_len=1, bf16, Standard Attention)")
    print("─" * 72)
    ms_decode = model_stats(cfg, seq_len=1, batch_size=1, dtype=DType.BF16, use_flash_attn=False)
    print_report(ms_decode, plain=True)   # plain for brevity


if __name__ == "__main__":
    main()
