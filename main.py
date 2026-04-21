#!/usr/bin/env python3
"""
main.py
=======
CLI entry point for the Model Parameter & Computation Calculator.

Usage
-----
  python main.py --config configs/qwen2_7b.json --seq-len 2048
  python main.py --config configs/glm4_9b.json  --seq-len 4096 --flash-attn
  python main.py --config configs/mixtral_8x7b.json --seq-len 1024 --batch-size 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

from src.config_parser import load_config
from src.calculators.base import DType
from src.calculators.model import model_stats
from src.visualizer import visualize
from src.report import print_report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute FLOPs, parameter count, and memory-access statistics "
                    "for a HuggingFace model architecture.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to a HuggingFace config.json file.",
    )
    parser.add_argument(
        "--name", "-n",
        default=None,
        help="Human-readable model name (defaults to config file stem).",
    )
    parser.add_argument(
        "--seq-len", "-s",
        type=int,
        default=2048,
        help="Default query/KV/cache length when explicit lengths are not provided (default: 2048).",
    )
    parser.add_argument(
        "--q-len",
        type=int,
        default=None,
        help="Query length / number of newly computed tokens (defaults to --seq-len).",
    )
    parser.add_argument(
        "--kv-len",
        type=int,
        default=None,
        help="Key/value length seen by attention (defaults to --seq-len).",
    )
    parser.add_argument(
        "--cache-len",
        type=int,
        default=None,
        help="Total KV cache length to size cache storage (defaults to --kv-len).",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Batch size (default: 1).",
    )
    parser.add_argument(
        "--dtype", "-d",
        choices=["fp32", "fp16", "bf16", "int8", "int4"],
        default="bf16",
        help="Element data type (default: bf16).",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        default=False,
        help="Model flash attention HBM access (default: standard attention).",
    )
    parser.add_argument(
        "--no-diagram",
        action="store_true",
        default=False,
        help="Skip printing the ASCII architecture diagram.",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        default=False,
        help="Use plain-text output instead of rich tables.",
    )
    parser.add_argument(
        "--kv-hit-ratio",
        type=float,
        default=0.0,
        help="KV cache hit ratio (0.0-1.0) for cache reuse analysis (default: 0.0).",
    )
    parser.add_argument(
        "--cache-layout",
        choices=["standard", "mla_compressed", "mla_expanded"],
        default=None,
        help="Override cache layout used for KV cache reporting.",
    )
    parser.add_argument(
        "--dsa-indexer-mode",
        choices=["fused_topk", "eager_dense_scores"],
        default=None,
        help="Override DSA indexer memory mode.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    # Load config --------------------------------------------------------
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(config_path, name=args.name)
    if args.cache_layout is not None:
        cfg.cache_layout = args.cache_layout
    if args.dsa_indexer_mode is not None:
        cfg.dsa_indexer_mode = args.dsa_indexer_mode
    dtype = DType(args.dtype)

    # Visualize ----------------------------------------------------------
    if not args.no_diagram:
        print(visualize(cfg, use_flash_attn=args.flash_attn))
        print()

    # Compute stats -------------------------------------------------------
    ms = model_stats(
        cfg=cfg,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        dtype=dtype,
        use_flash_attn=args.flash_attn,
        kv_hit_ratio=args.kv_hit_ratio,
        q_len=args.q_len,
        kv_len=args.kv_len,
        cache_len=args.cache_len,
    )

    # Report --------------------------------------------------------------
    print_report(ms, plain=args.plain)


if __name__ == "__main__":
    main()
