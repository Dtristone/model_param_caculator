"""
config_parser.py
================
Parse a HuggingFace model config.json into a normalised ModelConfig dataclass.

Supported architectures
-----------------------
- Qwen / Qwen2 / Qwen2.5  (qwen2)
- ChatGLM / GLM-4 / GLM-5  (chatglm, glm5)
- LLaMA / LLaMA-2 / LLaMA-3 (llama)
- Mistral / Mixtral          (mistral, mixtral)
- DeepSeek-V2 / V3 (MLA)    (deepseek_v2)
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
    num_shared_experts: int = 0        # Always-on shared experts
    first_k_dense_replace: int = 0     # First K layers use dense MLP instead of MoE
    mlp_layer_types: Optional[list[str]] = None

    # Positional encoding
    max_position_embeddings: int = 4096

    # Biases
    attention_bias: bool = False
    mlp_bias: bool = False
    use_qk_norm: bool = False
    num_nextn_predict_layers: int = 0

    # --- MLA (Multi-Latent Attention) fields --------------------------------
    use_mla: bool = False              # True for DeepSeek-V2/V3, GLM-5
    kv_lora_rank: int = 0              # KV latent compression rank (c_kv)
    q_lora_rank: int = 0               # Q latent compression rank  (c_q)
    qk_head_dim: int = 0               # total Q/K head dim
    qk_nope_head_dim: int = 0          # dimension of Q/K not using RoPE
    qk_rope_head_dim: int = 0          # dimension of Q/K using RoPE
    v_head_dim: int = 0                # value head dimension

    # --- DSA (Differential Sparse Attention) fields -------------------------
    use_dsa: bool = False              # True for GLM-5 with DSA
    index_head_dim: int = 0            # indexer head dimension
    index_n_heads: int = 0             # number of indexer attention heads
    index_topk: int = 2048             # top-k tokens selected by indexer
    cache_layout: str = "standard"     # standard | mla_compressed | mla_expanded
    dsa_indexer_mode: str = "fused_topk"  # fused_topk | eager_dense_scores

    def __post_init__(self) -> None:
        # Infer head_dim if not explicitly set
        if self.num_attention_heads > 0:
            inferred = self.hidden_size // self.num_attention_heads
            if inferred > 0 and self.head_dim == 128:
                self.head_dim = inferred
        # Default KV heads to Q heads (MHA)
        if self.num_key_value_heads <= 0:
            self.num_key_value_heads = self.num_attention_heads
        if self.qk_head_dim <= 0:
            inferred_qk = self.qk_nope_head_dim + self.qk_rope_head_dim
            self.qk_head_dim = inferred_qk if inferred_qk > 0 else self.head_dim
        if self.first_k_dense_replace < 0:
            self.first_k_dense_replace = 0
        if self.cache_layout == "standard" and self.use_mla:
            # MLA models need an explicit non-standard cache layout even when the
            # raw config does not provide one.
            self.cache_layout = "mla_compressed"

    @property
    def kv_dim(self) -> int:
        """Total KV hidden dimension (kv_heads × head_dim)."""
        return self.num_key_value_heads * self.head_dim

    @property
    def mla_q_head_dim(self) -> int:
        """Total Q/K head dim for MLA = qk_nope + qk_rope."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def q_proj_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_proj_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def attn_out_width(self) -> int:
        if self.use_mla and self.v_head_dim > 0:
            return self.num_attention_heads * self.v_head_dim
        return self.q_proj_width

    @property
    def mla_kv_cache_per_token(self) -> int:
        """Elements stored in KV cache per token for MLA.

        MLA caches the compressed KV latent (kv_lora_rank) plus the
        RoPE-applied key portion (qk_rope_head_dim), instead of full KV.
        """
        if self.cache_layout == "mla_expanded":
            return self.num_attention_heads * (self.qk_head_dim + self.v_head_dim)
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def attention_type(self) -> str:
        """Return a human-readable attention type string."""
        if self.use_mla:
            if self.use_dsa:
                return "DSA+MLA"
            return "MLA"
        if self.num_key_value_heads == 1:
            return "MQA"
        if self.num_key_value_heads < self.num_attention_heads:
            return "GQA"
        return "MHA"

    def is_sparse_layer(self, layer_idx: int) -> bool:
        if self.mlp_layer_types is not None and 0 <= layer_idx < len(self.mlp_layer_types):
            return self.mlp_layer_types[layer_idx].lower() == "sparse"
        if not self.is_moe or self.num_experts <= 1:
            return False
        return layer_idx >= self.first_k_dense_replace


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
    elif model_type in ("gpt2", "gpt_neo", "bloom"):
        norm = "layernorm"
    elif model_type == "chatglm":
        # GLM-4 has an explicit "rmsnorm" boolean field.  When it is True
        # the model uses RMSNorm; otherwise fall back to LayerNorm (older GLM).
        if _get(cfg, "rmsnorm", default=False):
            norm = "rmsnorm"
        else:
            norm = "layernorm"

    # MoE ——————————————————————————————————————————————————————————————————
    num_experts = int(_get(cfg, "num_experts", "num_local_experts",
                          "n_routed_experts", default=0))
    experts_per_tok = int(_get(cfg, "num_experts_per_tok", "num_selected_experts",
                               "top_k", default=2 if num_experts > 0 else 0))
    moe_inter = int(_get(cfg, "moe_intermediate_size", default=0))
    is_moe = num_experts > 1
    num_shared_experts = int(_get(cfg, "n_shared_experts", "num_shared_experts", default=0))
    raw_mlp_layer_types = _get(cfg, "mlp_layer_types", default=None)
    mlp_layer_types = list(raw_mlp_layer_types) if isinstance(raw_mlp_layer_types, list) else None
    first_k_dense_replace = int(_get(
        cfg,
        "first_k_dense_replace",
        default=0 if is_moe else num_layers,
    ))

    # Biases ——————————————————————————————————————————————————————————————
    attn_bias = bool(_get(cfg, "attention_bias", "add_bias_linear", default=False))
    mlp_bias = bool(_get(cfg, "mlp_bias", default=False))
    use_qk_norm = bool(_get(cfg, "use_qk_norm", default=False))
    num_nextn_predict_layers = int(_get(cfg, "num_nextn_predict_layers", default=0))

    max_pos = int(_get(cfg, "max_position_embeddings", "seq_length",
                        "max_sequence_length", default=4096))

    # MLA (Multi-Latent Attention) ——————————————————————————————————————
    kv_lora_rank = int(_get(cfg, "kv_lora_rank", default=0))
    q_lora_rank = int(_get(cfg, "q_lora_rank", default=0))
    qk_head_dim = int(_get(cfg, "qk_head_dim", default=0))
    qk_nope_head_dim = int(_get(cfg, "qk_nope_head_dim", "qk_head_dim", default=0))
    qk_rope_head_dim = int(_get(cfg, "qk_rope_head_dim", "qk_pos_emb_head_dim", default=0))
    v_head_dim = int(_get(cfg, "v_head_dim", default=0))
    use_mla = kv_lora_rank > 0

    # DSA (Differential Sparse Attention) ———————————————————————————————
    index_head_dim = int(_get(cfg, "index_head_dim", default=0))
    index_n_heads = int(_get(cfg, "index_n_heads", "index_num_attention_heads", default=0))
    index_topk = int(_get(cfg, "index_topk", default=2048))
    use_dsa = index_n_heads > 0
    default_cache_layout = "standard"
    if use_mla:
        default_cache_layout = "mla_expanded" if model_type.startswith("glm") else "mla_compressed"
    cache_layout = str(_get(cfg, "cache_layout", default=default_cache_layout))
    dsa_indexer_mode = str(_get(cfg, "dsa_indexer_mode", default="fused_topk"))

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
        num_shared_experts=num_shared_experts,
        first_k_dense_replace=first_k_dense_replace,
        mlp_layer_types=mlp_layer_types,
        max_position_embeddings=max_pos,
        attention_bias=attn_bias,
        mlp_bias=mlp_bias,
        use_qk_norm=use_qk_norm,
        num_nextn_predict_layers=num_nextn_predict_layers,
        use_mla=use_mla,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=q_lora_rank,
        qk_head_dim=qk_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        use_dsa=use_dsa,
        index_head_dim=index_head_dim,
        index_n_heads=index_n_heads,
        index_topk=index_topk,
        cache_layout=cache_layout,
        dsa_indexer_mode=dsa_indexer_mode,
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
