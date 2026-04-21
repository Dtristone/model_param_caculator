"""
visualizer.py
=============
ASCII diagram generator for a transformer model architecture.

The diagram shows:
  - Model name + key dimensions
  - Embedding layer
  - Representative transformer layer (with per-component shape annotations)
  - "× N layers" indicator
  - LM head

Example output
--------------
  ╔══════════════════════════════════════════════════════════════════╗
  ║   Qwen2-7B  │  hidden=3584  layers=28  heads=28/4 (GQA)        ║
  ╚══════════════════════════════════════════════════════════════════╝
  ┌──────────────────────────────────────────────────────────────────┐
  │  Embedding   [152064 × 3584]                     Params: 544.9M │
  └───────────────────────────┬──────────────────────────────────────┘
                              │ × 28 layers
  ┌───────────────────────────▼──────────────────────────────────────┐
  │  TransformerLayer                                                │
  │  ┌────────────────────────────────────────────────────────────┐  │
  │  │  RMSNorm                                                   │  │
  │  ├────────────────────────────────────────────────────────────┤  │
  │  │  Q proj    [3584 → 3584]                    28 heads       │  │
  │  │  K proj    [3584 → 512]                      4 KV heads    │  │
  │  │  V proj    [3584 → 512]                      4 KV heads    │  │
  │  │  ── Attention (GQA, flash) ────────────────────────────── │  │
  │  │  O proj    [3584 → 3584]                                   │  │
  │  ├────────────────────────────────────────────────────────────┤  │
  │  │  RMSNorm                                                   │  │
  │  ├────────────────────────────────────────────────────────────┤  │
  │  │  Gate proj [3584 → 18944]                                  │  │
  │  │  Up   proj [3584 → 18944]   SwiGLU                        │  │
  │  │  Down proj [18944 → 3584]                                  │  │
  │  └────────────────────────────────────────────────────────────┘  │
  └───────────────────────────┬──────────────────────────────────────┘
                              │
  ┌───────────────────────────▼──────────────────────────────────────┐
  │  Final RMSNorm + LM Head  [3584 → 152064]        Params: 544.9M │
  └──────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import Optional

from .config_parser import ModelConfig
from .calculators.base import fmt_num


# Width of the inner content area (between box borders)
_BOX_W = 66


def _pad(text: str, width: int = _BOX_W, align: str = "left") -> str:
    if align == "right":
        return text.rjust(width)
    if align == "center":
        return text.center(width)
    return text.ljust(width)


def _line(content: str = "", width: int = _BOX_W) -> str:
    return f"│  {content:<{width}}  │"


def _hline(width: int = _BOX_W) -> str:
    return "├" + "─" * (width + 4) + "┤"


def _top(width: int = _BOX_W) -> str:
    return "┌" + "─" * (width + 4) + "┐"


def _bot(width: int = _BOX_W) -> str:
    return "└" + "─" * (width + 4) + "┘"


def _inner_top(width: int = _BOX_W) -> str:
    inner = width - 2
    return f"│  ┌{'─' * (inner + 2)}┐  │"


def _inner_bot(width: int = _BOX_W) -> str:
    inner = width - 2
    return f"│  └{'─' * (inner + 2)}┘  │"


def _inner_hline(width: int = _BOX_W) -> str:
    inner = width - 2
    return f"│  ├{'─' * (inner + 2)}┤  │"


def _inner_line(content: str, width: int = _BOX_W) -> str:
    inner = width - 2
    return f"│  │  {content:<{inner - 2}}│  │"


def _center_arrow(n_layers: int, width: int = _BOX_W) -> list[str]:
    total = width + 4
    label = f"  ×{n_layers} layers  "
    pad_l = (total - len(label)) // 2
    pad_r = total - len(label) - pad_l
    return [
        "└" + "─" * (total // 2 - 1) + "┬" + "─" * (total - total // 2 - 2) + "┘",
        " " * pad_l + label + " " * pad_r,
        " " * (total // 2) + "▼",
    ]


def _bottom_arrow(width: int = _BOX_W) -> list[str]:
    total = width + 4
    return [
        "└" + "─" * (total // 2 - 1) + "┬" + "─" * (total - total // 2 - 2) + "┘",
        " " * (total // 2 - 1) + "│",
        " " * (total // 2 - 1) + "▼",
    ]


def visualize(cfg: ModelConfig, use_flash_attn: bool = False) -> str:
    """Return a multi-line ASCII diagram string for the given model config."""
    lines: list[str] = []

    W = _BOX_W

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    attn_type = cfg.attention_type
    header = (
        f"  {cfg.name}  │  hidden={cfg.hidden_size}  "
        f"layers={cfg.num_hidden_layers}  "
        f"heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} ({attn_type})"
    )
    hdr_w = max(len(header) + 2, W + 4)
    lines.append("╔" + "═" * hdr_w + "╗")
    lines.append("║" + header.ljust(hdr_w) + "║")
    lines.append("╚" + "═" * hdr_w + "╝")
    lines.append("")

    # ------------------------------------------------------------------
    # Embedding block
    # ------------------------------------------------------------------
    emb_params = cfg.vocab_size * cfg.hidden_size
    emb_label = f"Embedding   [{cfg.vocab_size} × {cfg.hidden_size}]"
    emb_right = f"Params: {fmt_num(emb_params)}"
    emb_content = emb_label + "  " + emb_right.rjust(W - len(emb_label) - 2)

    lines.append(_top(W))
    lines.append(_line(emb_content, W))
    for arr in _center_arrow(cfg.num_hidden_layers, W):
        lines.append(arr)

    # ------------------------------------------------------------------
    # Transformer layer block
    # ------------------------------------------------------------------
    lines.append(_top(W))
    lines.append(_line("TransformerLayer", W))
    lines.append(_inner_top(W))

    # Input norm
    lines.append(_inner_line(f"{cfg.norm_type.upper()}", W))

    lines.append(_inner_hline(W))

    # Attention projections
    inner_w = W - 6   # inner box content width
    if cfg.use_mla:
        # MLA projections
        c_kv = cfg.kv_lora_rank
        c_q = cfg.q_lora_rank
        d_n = cfg.qk_nope_head_dim
        d_r = cfg.qk_rope_head_dim
        v_dim = cfg.v_head_dim
        a = cfg.num_attention_heads

        lines.append(_inner_line(f"KV down-proj [{cfg.hidden_size} → {c_kv + d_r}]  (c_kv+d_r)", W))
        kv_b_desc = f"KV up-proj   [{c_kv} → {a}×({d_n}+{v_dim})]  (absorbed)"
        lines.append(_inner_line(kv_b_desc, W))
        if c_q > 0:
            lines.append(_inner_line(f"Q down-proj  [{cfg.hidden_size} → {c_q}]", W))
            lines.append(_inner_line(f"Q up-proj    [{c_q} → {a}×{d_n + d_r}]", W))
        else:
            lines.append(_inner_line(f"Q proj       [{cfg.hidden_size} → {a}×{d_n + d_r}]", W))

        if cfg.use_dsa:
            attn_mode = f"DSA top-{cfg.index_topk}"
            prefix = f"── DSA+MLA Attention ({attn_mode}) "
            lines.append(_inner_line(prefix + "─" * max(0, inner_w - len(prefix)), W))
            idx_desc = f"Indexer: {cfg.index_n_heads}h×{cfg.index_head_dim}d"
            lines.append(_inner_line(idx_desc, W))
        else:
            attn_mode = "flash" if use_flash_attn else "std"
            prefix = f"── MLA Attention (absorbed, {attn_mode}) "
            lines.append(_inner_line(prefix + "─" * max(0, inner_w - len(prefix)), W))

        lines.append(_inner_line(f"O proj       [{a}×{v_dim} → {cfg.hidden_size}]", W))

        # KV cache info
        if cfg.cache_layout == "mla_expanded":
            kv_info = f"KV cache/token: {cfg.mla_kv_cache_per_token}"
            if cfg.use_dsa:
                kv_info += f" + {cfg.index_head_dim} (indexer)"
            kv_info += " elements [expanded]"
        else:
            kv_info = f"KV cache/token: {c_kv + d_r}"
            if cfg.use_dsa:
                kv_info += f" + {cfg.index_head_dim} (indexer)"
            kv_info += " elements [compressed]"
        lines.append(_inner_line(kv_info, W))
    else:
        q_out = cfg.num_attention_heads * cfg.head_dim
        kv_out = cfg.num_key_value_heads * cfg.head_dim
        q_row   = f"Q proj    [{cfg.hidden_size} → {q_out}]"
        q_right = f"{cfg.num_attention_heads} Q heads"
        k_row   = f"K proj    [{cfg.hidden_size} → {kv_out}]"
        k_right = f"{cfg.num_key_value_heads} KV heads"
        v_row   = f"V proj    [{cfg.hidden_size} → {kv_out}]"
        v_right = f"{cfg.num_key_value_heads} KV heads"

        for row, right in [(q_row, q_right), (k_row, k_right), (v_row, v_right)]:
            content = row + right.rjust(inner_w - len(row))
            lines.append(_inner_line(content, W))

        attn_mode = "flash attn" if use_flash_attn else "std attn"
        prefix = f"── Attention ({attn_type}, {attn_mode}) "
        attn_row = prefix + "─" * max(0, inner_w - len(prefix))
        lines.append(_inner_line(attn_row, W))

        o_row = f"O proj    [{q_out} → {cfg.hidden_size}]"
        lines.append(_inner_line(o_row, W))

        # KV cache info for standard attention
        kv_per_token = 2 * cfg.num_key_value_heads * cfg.head_dim
        lines.append(_inner_line(f"KV cache/token: {kv_per_token} elements", W))

    lines.append(_inner_hline(W))

    # Post-attn norm
    lines.append(_inner_line(f"{cfg.norm_type.upper()}", W))

    lines.append(_inner_hline(W))

    # FFN
    if cfg.is_moe and cfg.num_experts > 1:
        expert_inter = cfg.moe_intermediate_size if cfg.moe_intermediate_size > 0 else cfg.intermediate_size
        dense_prefix = f"Dense prefix: first {cfg.first_k_dense_replace} layers"
        lines.append(_inner_line(dense_prefix, W))
        lines.append(_inner_line(f"Router  [{cfg.hidden_size} → {cfg.num_experts} experts]", W))
        lines.append(_inner_line(
            f"Expert FFN × {cfg.num_experts_per_tok}/{cfg.num_experts} (top-K)  "
            f"[{cfg.hidden_size} → {expert_inter} → {cfg.hidden_size}]  {cfg.ffn_type.upper()}", W))
        if cfg.num_shared_experts > 0:
            lines.append(_inner_line(
                f"Shared Expert × {cfg.num_shared_experts}  "
                f"[{cfg.hidden_size} → {expert_inter * cfg.num_shared_experts} → {cfg.hidden_size}]",
                W,
            ))
    else:
        gate_row = f"Gate proj [{cfg.hidden_size} → {cfg.intermediate_size}]"
        up_row   = f"Up   proj [{cfg.hidden_size} → {cfg.intermediate_size}]"
        down_row = f"Down proj [{cfg.intermediate_size} → {cfg.hidden_size}]"
        ffn_tag  = cfg.ffn_type.upper()

        right = ffn_tag
        up_content = up_row + right.rjust(inner_w - len(up_row))
        lines.append(_inner_line(gate_row, W))
        lines.append(_inner_line(up_content, W))
        lines.append(_inner_line(down_row, W))

    lines.append(_inner_bot(W))

    for arr in _bottom_arrow(W):
        lines.append(arr)

    # ------------------------------------------------------------------
    # Final norm + LM head block
    # ------------------------------------------------------------------
    lm_params = cfg.vocab_size * cfg.hidden_size  # (tie_word_embeddings: weight shared)
    lm_label = (
        f"Final {cfg.norm_type.upper()} + LM Head  "
        f"[{cfg.hidden_size} → {cfg.vocab_size}]"
    )
    tie_note = " (tied)" if cfg.tie_word_embeddings else ""
    lm_right = f"Params: {fmt_num(lm_params)}{tie_note}"
    lm_content = lm_label + "  " + lm_right.rjust(W - len(lm_label) - 2)

    lines.append(_top(W))
    lines.append(_line(lm_content, W))
    lines.append(_bot(W))

    return "\n".join(lines)
