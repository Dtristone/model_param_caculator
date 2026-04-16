"""
report.py
=========
Rich-table report generation for model computation and memory statistics.
"""

from __future__ import annotations

from typing import List, Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box as rich_box
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

from .calculators.base import ComputeStats, fmt_num, fmt_bytes
from .calculators.model import ModelStats
from .calculators.kv_cache import KVCacheStats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pct(part: int, total: int) -> str:
    if total == 0:
        return "—"
    return f"{100 * part / total:.1f}%"


def _ai(stats: ComputeStats) -> str:
    ai = stats.arithmetic_intensity
    if ai == float("inf"):
        return "∞"
    return f"{ai:.2f}"


def _row(stats: ComputeStats, total_params: int, total_flops: int) -> list:
    return [
        stats.name,
        fmt_num(stats.num_params),
        _pct(stats.num_params, total_params),
        fmt_num(stats.flops),
        _pct(stats.flops, total_flops),
        fmt_bytes(stats.weight_bytes),
        fmt_bytes(stats.act_bytes),
        fmt_bytes(stats.hbm_total_bytes),
        _ai(stats),
    ]


# ---------------------------------------------------------------------------
# Rich report
# ---------------------------------------------------------------------------

def _rich_report(ms: ModelStats) -> None:
    """Print a rich-formatted report to the console."""
    console = Console()

    cfg = ms.config
    total_params = ms.total.num_params
    total_flops = ms.total.flops

    # ---- Header -----------------------------------------------------------
    attn_label = cfg.attention_type
    flash_label = "Flash Attention" if ms.use_flash_attn else "Standard Attention"
    header_lines = [
        f"[bold cyan]{cfg.name}[/bold cyan]",
        f"  model_type   : {cfg.model_type}",
        f"  hidden_size  : {cfg.hidden_size}",
        f"  layers       : {cfg.num_hidden_layers}",
        f"  attn heads   : {cfg.num_attention_heads} Q / {cfg.num_key_value_heads} KV ({attn_label})",
        f"  head_dim     : {cfg.head_dim}",
        f"  intermediate : {cfg.intermediate_size}",
        f"  ffn_type     : {cfg.ffn_type.upper()}",
        f"  vocab_size   : {cfg.vocab_size}",
        f"  dtype        : {ms.dtype.value}",
        f"  seq_len      : {ms.seq_len}",
        f"  batch_size   : {ms.batch_size}",
        f"  attn_mode    : {flash_label}",
    ]
    if cfg.is_moe:
        header_lines.append(
            f"  MoE          : {cfg.num_experts} experts, top-{cfg.num_experts_per_tok}"
        )
    console.print(Panel("\n".join(header_lines), title="Model Configuration", expand=False))

    # ---- Summary table ----------------------------------------------------
    table = Table(
        title=f"Model Summary  |  Total params: {fmt_num(total_params)}  |  FLOPs: {fmt_num(total_flops)}",
        box=rich_box.ROUNDED,
        show_lines=True,
    )
    cols = ["Component", "Params", "Param%", "FLOPs", "FLOPs%",
            "Weight Mem", "Peak Act Mem", "HBM Traffic", "Arith.Int."]
    styles = ["bold", "", "dim", "", "dim", "cyan", "magenta", "yellow", "green"]
    for col, sty in zip(cols, styles):
        table.add_column(col, style=sty, no_wrap=True)

    sections = [
        ("Embedding",            ms.embedding),
        ("All Transformer Layers", ms.all_layers),
        ("Final Norm + LM Head", ms.lm_head),
        ("─" * 8,                None),
        ("TOTAL",                ms.total),
    ]
    for name, stats in sections:
        if stats is None:
            table.add_row(*["─" * 8] * len(cols))
            continue
        row = _row(stats, total_params, total_flops)
        style = "bold" if name == "TOTAL" else ""
        table.add_row(*row, style=style)

    console.print(table)

    # ---- Per-layer breakdown ----------------------------------------------
    if ms.layer_breakdown:
        one = ms.layer_breakdown[0]
        layer_table = Table(
            title="Per-Layer Component Breakdown (one representative layer)",
            box=rich_box.SIMPLE_HEAVY,
            show_lines=True,
        )
        for col, sty in zip(cols, styles):
            layer_table.add_column(col, style=sty, no_wrap=True)

        for child in one.children:
            layer_table.add_row(*_row(child, one.num_params, one.flops))

        layer_table.add_row(*["─" * 8] * len(cols))
        layer_table.add_row(
            *_row(one, one.num_params, one.flops),
            style="bold",
        )
        console.print(layer_table)

    # ---- Attention comparison table (std vs flash) -------------------------
    _attn_comparison(console, ms)

    # ---- KV Cache analysis -------------------------------------------------
    if ms.kv_cache is not None:
        _kv_cache_report(console, ms)


def _attn_comparison(console, ms: ModelStats) -> None:
    """Print a flash vs standard attention HBM comparison."""
    from .calculators.attention import attention_stats

    cfg = ms.config
    B, s = ms.batch_size, ms.seq_len

    # Skip comparison for MLA/DSA models (different architecture)
    if cfg.use_mla:
        return

    std = attention_stats(
        hidden_size=cfg.hidden_size,
        num_q_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        seq_len=s,
        batch_size=B,
        use_flash_attn=False,
        dtype=ms.dtype,
    )
    flash = attention_stats(
        hidden_size=cfg.hidden_size,
        num_q_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        seq_len=s,
        batch_size=B,
        use_flash_attn=True,
        dtype=ms.dtype,
    )

    tbl = Table(
        title=f"Attention Kernel Comparison (seq_len={s}, per layer)",
        box=rich_box.SIMPLE_HEAD,
    )
    tbl.add_column("Metric")
    tbl.add_column("Standard Attention", style="yellow")
    tbl.add_column("Flash Attention", style="green")
    tbl.add_column("Reduction", style="cyan")

    def ratio(a, b):
        if b == 0:
            return "—"
        r = a / b
        if r > 1:
            return f"[red]{r:.1f}× more[/red]"
        return f"[green]{1/r:.1f}× less[/green]"

    tbl.add_row("FLOPs",
                fmt_num(std.flops), fmt_num(flash.flops),
                ratio(std.flops, flash.flops))
    tbl.add_row("HBM Traffic",
                fmt_bytes(std.hbm_total_bytes), fmt_bytes(flash.hbm_total_bytes),
                ratio(std.hbm_total_bytes, flash.hbm_total_bytes))
    tbl.add_row("Peak Act Mem",
                fmt_bytes(std.act_bytes), fmt_bytes(flash.act_bytes),
                ratio(std.act_bytes, flash.act_bytes))
    tbl.add_row("Arith. Intensity",
                _ai(std), _ai(flash), "")

    console.print(tbl)


def _kv_cache_report(console, ms: ModelStats) -> None:
    """Print KV cache analysis table."""
    kv = ms.kv_cache
    if kv is None:
        return

    tbl = Table(
        title=f"KV Cache Analysis ({kv.attention_type})",
        box=rich_box.SIMPLE_HEAD,
    )
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Value", style="cyan")

    tbl.add_row("Attention Type", kv.attention_type)

    if kv.k_cache_per_token > 0:
        tbl.add_row("K cache per token", f"{kv.k_cache_per_token} elements")
    if kv.v_cache_per_token > 0:
        tbl.add_row("V cache per token", f"{kv.v_cache_per_token} elements")
    if kv.index_cache_per_token > 0:
        tbl.add_row("Indexer K cache per token", f"{kv.index_cache_per_token} elements")

    tbl.add_row("Total per token per layer", f"{kv.total_per_token} elements")
    tbl.add_row("Bytes per token per layer", fmt_bytes(kv.per_token_per_layer_bytes))
    tbl.add_row("Bytes per token (all layers)", fmt_bytes(kv.per_token_all_layers_bytes))
    tbl.add_row("Total KV cache", fmt_bytes(kv.total_cache_bytes))
    tbl.add_row("Compression ratio vs MHA", f"{kv.compression_ratio:.1f}×")
    tbl.add_row("Layers", str(kv.num_layers))
    tbl.add_row("Seq len", str(kv.seq_len))
    tbl.add_row("Batch size", str(kv.batch_size))

    if kv.hit_ratio > 0:
        tbl.add_row("", "")
        tbl.add_row("[bold]Cache Hit Ratio[/bold]", f"{kv.hit_ratio:.1%}")
        tbl.add_row("Cached tokens", str(kv.cached_tokens))
        tbl.add_row("New tokens", str(kv.new_tokens))
        tbl.add_row("KV proj FLOPs saved", fmt_num(kv.kv_proj_flops_saved))
        tbl.add_row("HBM writes saved", fmt_bytes(kv.hbm_write_saved))

    console.print(tbl)


# ---------------------------------------------------------------------------
# Plain-text fallback
# ---------------------------------------------------------------------------

def _plain_report(ms: ModelStats) -> None:
    """Print a simple text report (no rich dependency)."""
    cfg = ms.config
    total_params = ms.total.num_params
    total_flops = ms.total.flops

    print(f"\n{'='*70}")
    print(f"  {cfg.name}")
    print(f"  hidden={cfg.hidden_size}  layers={cfg.num_hidden_layers}  "
          f"heads={cfg.num_attention_heads}/{cfg.num_key_value_heads}  "
          f"ffn={cfg.intermediate_size}  dtype={ms.dtype.value}")
    print(f"  seq_len={ms.seq_len}  batch={ms.batch_size}")
    print(f"{'='*70}")

    header = f"{'Component':<28} {'Params':>10} {'FLOPs':>10} {'Weight':>10} {'HBM':>10}"
    print(header)
    print("-" * len(header))

    for name, stats in [
        ("Embedding",       ms.embedding),
        ("All Layers",      ms.all_layers),
        ("LM Head",         ms.lm_head),
        ("TOTAL",           ms.total),
    ]:
        print(f"{name:<28} {fmt_num(stats.num_params):>10} {fmt_num(stats.flops):>10} "
              f"{fmt_bytes(stats.weight_bytes):>10} {fmt_bytes(stats.hbm_total_bytes):>10}")

    print(f"\nPer-layer breakdown:")
    if ms.layer_breakdown:
        one = ms.layer_breakdown[0]
        for child in one.children:
            print(f"  {child.name:<26} {fmt_num(child.num_params):>10} {fmt_num(child.flops):>10} "
                  f"{fmt_bytes(child.weight_bytes):>10} {fmt_bytes(child.hbm_total_bytes):>10}")

    # KV Cache
    if ms.kv_cache is not None:
        kv = ms.kv_cache
        print(f"\nKV Cache Analysis ({kv.attention_type}):")
        print(f"  Per token per layer: {kv.total_per_token} elements = {fmt_bytes(kv.per_token_per_layer_bytes)}")
        print(f"  Per token all layers: {fmt_bytes(kv.per_token_all_layers_bytes)}")
        print(f"  Total KV cache: {fmt_bytes(kv.total_cache_bytes)}")
        print(f"  Compression ratio vs MHA: {kv.compression_ratio:.1f}×")
        if kv.hit_ratio > 0:
            print(f"  Cache hit ratio: {kv.hit_ratio:.1%}")
            print(f"  Cached/New tokens: {kv.cached_tokens}/{kv.new_tokens}")
            print(f"  KV proj FLOPs saved: {fmt_num(kv.kv_proj_flops_saved)}")
            print(f"  HBM writes saved: {fmt_bytes(kv.hbm_write_saved)}")
    print()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def print_report(ms: ModelStats, plain: bool = False) -> None:
    """Print a model statistics report.

    Parameters
    ----------
    ms : ModelStats
        Statistics object from model_stats().
    plain : bool
        If True, use plain text even if rich is available.
    """
    if _RICH_AVAILABLE and not plain:
        _rich_report(ms)
    else:
        _plain_report(ms)
