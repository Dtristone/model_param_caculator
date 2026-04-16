#!/usr/bin/env python3
"""
example_glm.py
==============
Demonstrate model analysis for GLM-4-9B.

Note: The original request references "GLM-5" which is not publicly available
at the time of writing. GLM-4-9B is the closest public GLM model and uses a
similar architecture (GQA, SwiGLU, RMSNorm).

Run:
  python examples/example_glm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_parser import load_config
from src.calculators.base import DType
from src.calculators.model import model_stats
from src.visualizer import visualize
from src.report import print_report


def main() -> None:
    config_path = Path(__file__).parent.parent / "configs" / "glm4_9b.json"
    cfg = load_config(config_path, name="GLM-4-9B")

    print("=" * 72)
    print("  GLM-4-9B  —  Architecture Visualisation & Computation Analysis")
    print("=" * 72)
    print()

    # --- ASCII diagram ---------------------------------------------------
    print(visualize(cfg, use_flash_attn=False))
    print()

    # --- Prefill analysis ------------------------------------------------
    print("─" * 72)
    print("  PREFILL  (batch=1, seq_len=2048, bf16, Standard Attention)")
    print("─" * 72)
    ms = model_stats(cfg, seq_len=2048, batch_size=1, dtype=DType.BF16, use_flash_attn=False)
    print_report(ms)

    # --- Flash attention comparison --------------------------------------
    print("─" * 72)
    print("  PREFILL  (batch=1, seq_len=8192, bf16, Flash Attention)")
    print("─" * 72)
    ms_flash = model_stats(cfg, seq_len=8192, batch_size=1, dtype=DType.BF16, use_flash_attn=True)
    print_report(ms_flash)


if __name__ == "__main__":
    main()
