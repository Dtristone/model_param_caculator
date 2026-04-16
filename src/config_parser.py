"""
config_parser.py
================
Parse a HuggingFace model config.json into a normalised ModelConfig dataclass.

Supported architectures
-----------------------
- Qwen / Qwen2 / Qwen2.5  (qwen2)
- ChatGLM / GLM-4           (chatglm)
- LLaMA / LLaMA-2 / LLaMA-3 (llama)
- Mistral / Mixtral          (mistral, mixtral)
- Generic dense transformer  (fallback)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Normalised model configuration used by the calculators."""

    name: str = "unknown"
    model_type: str = "unknown"

    # Architecture dimensions
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32      # k == num_attention_heads → MHA
    head_dim: int = 128                # d = hidden_size // num_attention_heads
    intermediate_size: int = 11008     # FFN intermediate dimension

    # Vocabulary
    vocab_size: int = 32000
    tie_word_embeddings: bool = True

    # FFN variant
    ffn_type: str = "swiglu"           # "swiglu" | "geglu" | "standard"

    # Normalisation
    norm_type: str = "rmsnorm"         # "rmsnorm" | "layernorm"

    # MoE (Mixture-of-Experts)
    is_moe: bool = False
    num_experts: int = 0               # Total number of experts E
    num_experts_per_tok: int = 0       # Top-K activated experts K
    moe_intermediate_size: int = 0     # Expert FFN intermediate size (if differs)

    # Positional encoding
    max_position_embeddings: int = 4096

    # Biases
    attention_bias: bool = False
    mlp_bias: bool = False

    def __post_init__(self) -> None:
        # Infer head_dim if not explicitly set
        if self.num_attention_heads > 0:
            inferred = self.hidden_size // self.num_attention_heads
            if inferred > 0:
                self.head_dim = inferred
        # Default KV heads to Q heads (MHA)
        if self.num_key_value_heads <= 0:
            self.num_key_value_heads = self.num_attention_heads

    @property
    def kv_dim(self) -> int:
        """Total KV hidden dimension (kv_heads × head_dim)."""
        return self.num_key_value_heads * self.head_dim

    @property
    def attention_type(self) -> str:
        """Return a human-readable attention type string."""
        if self.num_key_value_heads == 1:
            return "MQA"
        if self.num_key_value_heads < self.num_attention_heads:
            return "GQA"
        return "MHA"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _get(cfg: dict, *keys, default=None):
    """Return the first key found in cfg, or default."""
    for k in keys:
        if k in cfg:
            return cfg[k]
    return default


def _detect_ffn_type(cfg: dict) -> str:
    """Detect FFN activation type from config."""
    act = _get(cfg, "hidden_act", "activation_function", default="silu")
    if act in ("silu", "swish"):
        # Qwen2 / LLaMA / GLM use silu with a gated projection → SwiGLU.
        # Accept both "intermediate_size" and "ffn_hidden_size" (GLM alias).
        if "intermediate_size" in cfg or "ffn_hidden_size" in cfg:
            return "swiglu"
        return "standard"
    if act in ("gelu", "gelu_new", "gelu_fast", "gelu_approx"):
        # Check if gated (GeGLU)
        if _get(cfg, "is_gated_act", default=False):
            return "geglu"
        return "standard"
    if act in ("geglu",):
        return "geglu"
    # GLM-style (no hidden_act explicitly set but has ffn_hidden_size)
    if "ffn_hidden_size" in cfg:
        return "swiglu"
    return "standard"


def _parse_generic(cfg: dict, name: str) -> ModelConfig:
    """Build a ModelConfig from a raw HuggingFace config dict."""
    model_type = cfg.get("model_type", "unknown")

    # Hidden / layer dims —————————————————————————————————————————————————————
    hidden_size = int(_get(cfg, "hidden_size", "d_model", "n_embd", default=4096))
    num_layers = int(_get(cfg,
                          "num_hidden_layers", "num_layers", "n_layer", "n_layers",
                          default=32))
    n_heads = int(_get(cfg,
                       "num_attention_heads", "n_head", "num_heads",
                       default=32))
    kv_heads = int(_get(cfg,
                        "num_key_value_heads",
                        "multi_query_group_num",   # GLM
                        "num_kv_heads",
                        default=n_heads))
    head_dim_explicit = int(_get(cfg, "head_dim", default=0))
    head_dim = head_dim_explicit if head_dim_explicit > 0 else (hidden_size // n_heads if n_heads > 0 else 128)

    # FFN ——————————————————————————————————————————————————————————————————————
    intermediate = int(_get(cfg,
                            "intermediate_size",
                            "ffn_hidden_size",      # GLM
                            "n_inner",
                            default=hidden_size * 4))
    ffn_type = _detect_ffn_type(cfg)

    # Vocab ——————————————————————————————————————————————————————————————————
    vocab_size = int(_get(cfg, "vocab_size", "padded_vocab_size", default=32000))
    tie = bool(_get(cfg, "tie_word_embeddings", default=True))

    # Norm ——————————————————————————————————————————————————————————————————
    norm = "rmsnorm"
    if _get(cfg, "norm_type", default="") == "layernorm":
        norm = "layernorm"
    elif model_type in ("gpt2", "gpt_neo", "bloom", "chatglm"):
        norm = "layernorm"

    # MoE ——————————————————————————————————————————————————————————————————
    num_experts = int(_get(cfg, "num_experts", "num_local_experts", default=0))
    experts_per_tok = int(_get(cfg, "num_experts_per_tok", "num_selected_experts",
                               "top_k", default=2 if num_experts > 0 else 0))
    moe_inter = int(_get(cfg, "moe_intermediate_size", default=0))
    is_moe = num_experts > 1

    # Biases ——————————————————————————————————————————————————————————————
    attn_bias = bool(_get(cfg, "attention_bias", "add_bias_linear", default=False))
    mlp_bias = bool(_get(cfg, "mlp_bias", default=False))

    max_pos = int(_get(cfg, "max_position_embeddings", "seq_length",
                        "max_sequence_length", default=4096))

    return ModelConfig(
        name=name,
        model_type=model_type,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate,
        vocab_size=vocab_size,
        tie_word_embeddings=tie,
        ffn_type=ffn_type,
        norm_type=norm,
        is_moe=is_moe,
        num_experts=num_experts,
        num_experts_per_tok=experts_per_tok,
        moe_intermediate_size=moe_inter,
        max_position_embeddings=max_pos,
        attention_bias=attn_bias,
        mlp_bias=mlp_bias,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: str | Path, name: Optional[str] = None) -> ModelConfig:
    """Load a HuggingFace config.json and return a ModelConfig."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_name = name or path.stem
    return _parse_generic(cfg, model_name)


def from_dict(cfg: dict, name: str = "custom") -> ModelConfig:
    """Build a ModelConfig from a raw config dict."""
    return _parse_generic(cfg, name)
