"""
ffn.py
======
FLOPs and memory-access statistics for Feed-Forward Network (FFN) / MLP blocks.

Three variants are supported:

  SwiGLU  (used by Qwen2, LLaMA, Mistral)
  ┌──────────────────────────────────────────────────────────┐
  │  gate  = Linear(h → ffn_h)   [2*B*s*h*ffn_h FLOPs]     │
  │  up    = Linear(h → ffn_h)   [2*B*s*h*ffn_h FLOPs]     │
  │  x     = SiLU(gate) * up     [2*B*s*ffn_h   FLOPs]     │
  │  out   = Linear(ffn_h → h)   [2*B*s*ffn_h*h FLOPs]     │
  └──────────────────────────────────────────────────────────┘
  Total FLOPs ≈ 6 * B * s * h * ffn_h
  Params      = 3 * h * ffn_h

  GeGLU  (similar to SwiGLU but uses GELU activation)
  Same parameter count and FLOPs as SwiGLU.

  Standard  (GELU / ReLU, used by GPT-2, BERT)
  ┌──────────────────────────────────────────────────────────┐
  │  up  = Linear(h → ffn_h)   [2*B*s*h*ffn_h FLOPs]       │
  │  act = GELU/ReLU(up)       [~B*s*ffn_h    FLOPs]        │
  │  out = Linear(ffn_h → h)   [2*B*s*ffn_h*h FLOPs]       │
  └──────────────────────────────────────────────────────────┘
  Total FLOPs ≈ 4 * B * s * h * ffn_h
  Params      = 2 * h * ffn_h
"""

from __future__ import annotations

from .base import ComputeStats, DType, elements_to_bytes
from .linear import LinearStats


def ffn_stats(
    hidden_size: int,
    intermediate_size: int,
    ffn_type: str = "swiglu",
    seq_len: int = 1,
    batch_size: int = 1,
    has_bias: bool = False,
    dtype: DType = DType.BF16,
) -> ComputeStats:
    """Compute FLOPs and memory-access statistics for a single FFN/MLP block.

    Parameters
    ----------
    hidden_size : int
        Input/output hidden dimension h.
    intermediate_size : int
        Intermediate (expanded) dimension ffn_h.
    ffn_type : str
        One of "swiglu", "geglu", "standard".
    seq_len : int
        Sequence length s.
    batch_size : int
        Batch size B.
    has_bias : bool
        Whether linear layers include a bias term.
    dtype : DType
        Element dtype for memory sizing.
    """
    B, s = batch_size, seq_len
    h, ffn_h = hidden_size, intermediate_size
    ffn_type = ffn_type.lower()

    stats = ComputeStats(name=f"FFN ({ffn_type.upper()})")

    if ffn_type in ("swiglu", "geglu"):
        # Three linear layers: gate, up, down
        gate = LinearStats("Gate proj", h, ffn_h, has_bias, s, B, dtype).compute()
        up   = LinearStats("Up proj",   h, ffn_h, has_bias, s, B, dtype).compute()
        down = LinearStats("Down proj", ffn_h, h, has_bias, s, B, dtype).compute()

        # Element-wise activation + multiply (gate ⊙ act(up))
        # FLOPs are an approximation: 1 op for act(gate) + 1 op for element-wise multiply.
        # Exact activation cost (SiLU ≈ 4–6 ops, GELU ≈ 8–14 ops) is platform-dependent.
        # Using 2 ops per element follows the same convention as most FLOPs estimation tools
        # (e.g. Megatron-LM, FLOPs counters in major ML papers) and is accepted for this use case.
        act_name = "SiLU" if ffn_type == "swiglu" else "GELU"
        ew_flops = 2 * B * s * ffn_h   # approximate: act(gate) + gate*up
        ew = ComputeStats(
            name=f"{act_name} + multiply",
            flops=ew_flops,
            hbm_read_bytes=elements_to_bytes(2 * B * s * ffn_h, dtype),   # read gate + up
            hbm_write_bytes=elements_to_bytes(B * s * ffn_h, dtype),
            act_bytes=elements_to_bytes(B * s * ffn_h, dtype),
        )

        stats.children = [gate, up, ew, down]

    else:  # standard FFN
        up   = LinearStats("Up proj",   h, ffn_h, has_bias, s, B, dtype).compute()
        down = LinearStats("Down proj", ffn_h, h, has_bias, s, B, dtype).compute()

        # FLOPs are an approximation: exact GELU cost (≈8–14 ops) is
        # platform/precision-dependent; using 1 op per element is accepted.
        act_flops = B * s * ffn_h
        act = ComputeStats(
            name="Activation",
            flops=act_flops,
            hbm_read_bytes=elements_to_bytes(B * s * ffn_h, dtype),
            hbm_write_bytes=elements_to_bytes(B * s * ffn_h, dtype),
            act_bytes=elements_to_bytes(B * s * ffn_h, dtype),
        )

        stats.children = [up, act, down]

    stats.aggregate_children()
    return stats
