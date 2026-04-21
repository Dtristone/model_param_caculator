"""
model.py
========
Aggregates per-component statistics into a full model breakdown.

Component hierarchy
-------------------
Model
├── Embedding
├── TransformerLayer × N
│   ├── Input Norm
│   ├── QKV Projection (or MLA Projection for MLA models)
│   ├── Attention (Standard / Flash / MLA / DSA+MLA)
│   ├── Output Projection (folded into MLA Projection for MLA models)
│   ├── Post-Attention Norm
│   └── FFN / MoE
└── Final Norm + LM Head
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..config_parser import ModelConfig
from .base import ComputeStats, DType, elements_to_bytes
from .linear import qkv_proj_stats, output_proj_stats, LinearStats
from .attention import attention_stats
from .ffn import ffn_stats
from .moe import moe_stats
from .mla import mla_proj_stats, mla_attention_stats
from .dsa import dsa_indexer_stats, dsa_sparse_attention_stats
from .kv_cache import kv_cache_stats, KVCacheStats

TOKEN_INDEX_BYTES = 4


# ---------------------------------------------------------------------------
# Norm helper
# ---------------------------------------------------------------------------

def _norm_stats(
    hidden_size: int,
    norm_type: str,
    seq_len: int,
    batch_size: int,
    dtype: DType,
) -> ComputeStats:
    """FLOPs/params for a single normalisation layer."""
    B, s, h = batch_size, seq_len, hidden_size
    if norm_type == "rmsnorm":
        # mean-sq (h ops) + rsqrt (1) + norm (h) + scale (h) ≈ 3h+1 ≈ 4h per token
        flops = 4 * B * s * h
        n_params = h          # scale vector only (no bias)
    else:  # layernorm
        # mean (h) + var (2h) + norm (2h) + scale+bias (2h) ≈ 7h per token
        flops = 7 * B * s * h
        n_params = 2 * h      # scale + bias

    w_bytes = elements_to_bytes(n_params, dtype)
    hbm_read = elements_to_bytes(B * s * h, dtype) + w_bytes
    hbm_write = elements_to_bytes(B * s * h, dtype)
    act = elements_to_bytes(B * s * h, dtype)

    return ComputeStats(
        name=norm_type.upper(),
        num_params=n_params,
        flops=flops,
        weight_bytes=w_bytes,
        act_bytes=act,
        hbm_read_bytes=hbm_read,
        hbm_write_bytes=hbm_write,
    )


def _qk_norm_stats(
    name: str,
    head_dim: int,
    num_heads: int,
    seq_len: int,
    batch_size: int,
    dtype: DType,
) -> ComputeStats:
    tokens = batch_size * seq_len
    elems = tokens * num_heads * head_dim
    return ComputeStats(
        name=name,
        num_params=head_dim,
        flops=4 * elems,
        weight_bytes=elements_to_bytes(head_dim, dtype),
        act_bytes=elements_to_bytes(elems, dtype),
        hbm_read_bytes=elements_to_bytes(elems + head_dim, dtype),
        hbm_write_bytes=elements_to_bytes(elems, dtype),
    )


def _layer_family_name(layer: ComputeStats) -> str:
    return layer.name.split("[", 1)[0]


# ---------------------------------------------------------------------------
# Single transformer layer
# ---------------------------------------------------------------------------

def _transformer_layer_stats(
    cfg: ModelConfig,
    q_len: int,
    kv_len: int,
    batch_size: int,
    use_flash_attn: bool,
    dtype: DType,
    layer_idx: int = 0,
    is_sparse: bool = False,
) -> ComputeStats:
    """Build ComputeStats for one transformer layer."""
    B, q_tokens = batch_size, q_len
    h = cfg.hidden_size

    layer_kind = "Sparse" if is_sparse else "Dense"
    layer = ComputeStats(name=f"{layer_kind} TransformerLayer[{layer_idx}]")

    # --- Input norm ---------------------------------------------------------
    in_norm = _norm_stats(h, cfg.norm_type, q_tokens, B, dtype)
    in_norm.name = f"Input {cfg.norm_type.upper()}"
    layer.children.append(in_norm)

    # --- QKV projections / MLA projections / Attention --------------------
    if cfg.use_mla:
        # MLA projection: kv_a, kv_b (absorbed), q_a/q_b, o_proj
        mla_proj = mla_proj_stats(
            hidden_size=h,
            num_q_heads=cfg.num_attention_heads,
            kv_lora_rank=cfg.kv_lora_rank,
            q_lora_rank=cfg.q_lora_rank,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            v_head_dim=cfg.v_head_dim,
            seq_len=q_tokens,
            batch_size=B,
            has_bias=cfg.attention_bias,
            dtype=dtype,
        )
        layer.children.append(mla_proj)

        if cfg.use_dsa:
            # DSA: indexer branch + sparse MLA attention
            indexer = dsa_indexer_stats(
                hidden_size=h,
                q_lora_rank=cfg.q_lora_rank,
                index_n_heads=cfg.index_n_heads,
                index_head_dim=cfg.index_head_dim,
                index_topk=cfg.index_topk,
                seq_len=q_tokens,
                batch_size=B,
                dtype=dtype,
                q_len=q_tokens,
                kv_len=kv_len,
                indexer_mode=cfg.dsa_indexer_mode,
            )
            layer.children.append(indexer)

            sparse_attn = dsa_sparse_attention_stats(
                num_q_heads=cfg.num_attention_heads,
                kv_lora_rank=cfg.kv_lora_rank,
                qk_rope_head_dim=cfg.qk_rope_head_dim,
                v_head_dim=cfg.v_head_dim,
                index_topk=cfg.index_topk,
                seq_len=q_tokens,
                batch_size=B,
                dtype=dtype,
                q_len=q_tokens,
                kv_len=kv_len,
            )
            layer.children.append(sparse_attn)
        else:
            # Standard MLA attention (absorbed, no sparsity)
            mla_attn = mla_attention_stats(
                num_q_heads=cfg.num_attention_heads,
                kv_lora_rank=cfg.kv_lora_rank,
                qk_nope_head_dim=cfg.qk_nope_head_dim,
                qk_rope_head_dim=cfg.qk_rope_head_dim,
                v_head_dim=cfg.v_head_dim,
                seq_len=q_tokens,
                batch_size=B,
                use_flash_attn=use_flash_attn,
                dtype=dtype,
                q_len=q_tokens,
                kv_len=kv_len,
            )
            layer.children.append(mla_attn)
    else:
        # Standard QKV + Attention + O projection
        qkv = qkv_proj_stats(
            hidden_size=h,
            num_q_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            seq_len=q_tokens,
            batch_size=B,
            has_bias=cfg.attention_bias,
            dtype=dtype,
        )
        layer.children.append(qkv)

        if cfg.use_qk_norm:
            layer.children.append(
                _qk_norm_stats(
                    name="Q RMSNorm",
                    head_dim=cfg.head_dim,
                    num_heads=cfg.num_attention_heads,
                    seq_len=q_tokens,
                    batch_size=B,
                    dtype=dtype,
                )
            )
            layer.children.append(
                _qk_norm_stats(
                    name="K RMSNorm",
                    head_dim=cfg.head_dim,
                    num_heads=cfg.num_key_value_heads,
                    seq_len=q_tokens,
                    batch_size=B,
                    dtype=dtype,
                )
            )

        attn = attention_stats(
            hidden_size=h,
            num_q_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            seq_len=q_tokens,
            batch_size=B,
            use_flash_attn=use_flash_attn,
            dtype=dtype,
            q_len=q_tokens,
            kv_len=kv_len,
            q_width=cfg.q_proj_width,
            kv_width=cfg.kv_proj_width,
            attn_out_width=cfg.q_proj_width,
        )
        layer.children.append(attn)

        o_proj = output_proj_stats(
            in_features=cfg.q_proj_width,
            out_features=h,
            seq_len=q_tokens,
            batch_size=B,
            has_bias=False,
            dtype=dtype,
        )
        layer.children.append(o_proj)

    # --- Post-attention norm -------------------------------------------------
    post_attn_norm = _norm_stats(h, cfg.norm_type, q_tokens, B, dtype)
    post_attn_norm.name = f"Post-Attn {cfg.norm_type.upper()}"
    layer.children.append(post_attn_norm)

    # --- FFN / MoE ----------------------------------------------------------
    if is_sparse and cfg.is_moe and cfg.num_experts > 1:
        expert_inter = cfg.moe_intermediate_size if cfg.moe_intermediate_size > 0 else cfg.intermediate_size
        ffn = moe_stats(
            hidden_size=h,
            num_experts=cfg.num_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            expert_intermediate_size=expert_inter,
            num_shared_experts=cfg.num_shared_experts,
            ffn_type=cfg.ffn_type,
            seq_len=q_tokens,
            batch_size=B,
            has_bias=cfg.mlp_bias,
            dtype=dtype,
        )
    else:
        ffn = ffn_stats(
            hidden_size=h,
            intermediate_size=cfg.intermediate_size,
            ffn_type=cfg.ffn_type,
            seq_len=q_tokens,
            batch_size=B,
            has_bias=cfg.mlp_bias,
            dtype=dtype,
        )
    layer.children.append(ffn)

    layer.aggregate_children()
    return layer


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

@dataclass
class ModelStats:
    """Statistics for the full model, broken down by component."""

    config: ModelConfig
    seq_len: int
    q_len: int
    kv_len: int
    cache_len: int
    batch_size: int
    dtype: DType
    use_flash_attn: bool

    total: ComputeStats = field(default_factory=lambda: ComputeStats("Total"))
    embedding: ComputeStats = field(default_factory=lambda: ComputeStats("Embedding"))
    per_layer: ComputeStats = field(default_factory=lambda: ComputeStats("Per Layer (avg)"))
    all_layers: ComputeStats = field(default_factory=lambda: ComputeStats("All Layers"))
    lm_head: ComputeStats = field(default_factory=lambda: ComputeStats("LM Head"))
    layer_breakdown: List[ComputeStats] = field(default_factory=list)
    kv_cache: Optional[KVCacheStats] = None


def model_stats(
    cfg: ModelConfig,
    seq_len: int = 2048,
    batch_size: int = 1,
    dtype: DType = DType.BF16,
    use_flash_attn: bool = False,
    kv_hit_ratio: float = 0.0,
    q_len: int | None = None,
    kv_len: int | None = None,
    cache_len: int | None = None,
) -> ModelStats:
    """Compute full-model statistics.

    Dense and sparse layer patterns are aggregated explicitly when the config
    includes heterogeneous MLP layers.
    """
    B = batch_size
    q_tokens = q_len if q_len is not None else seq_len
    kv_tokens = kv_len if kv_len is not None else seq_len
    cache_tokens = cache_len if cache_len is not None else kv_tokens
    h, V, N = cfg.hidden_size, cfg.vocab_size, cfg.num_hidden_layers

    result = ModelStats(
        config=cfg,
        seq_len=seq_len,
        q_len=q_tokens,
        kv_len=kv_tokens,
        cache_len=cache_tokens,
        batch_size=batch_size,
        dtype=dtype,
        use_flash_attn=use_flash_attn,
    )

    # --- Embedding ----------------------------------------------------------
    emb_params = V * h
    emb_w = elements_to_bytes(emb_params, dtype)
    result.embedding = ComputeStats(
        name="Embedding",
        num_params=emb_params,
        flops=0,          # look-up, not multiply-add
        weight_bytes=emb_w,
        act_bytes=elements_to_bytes(B * q_tokens * h, dtype),
        hbm_read_bytes=B * q_tokens * TOKEN_INDEX_BYTES + elements_to_bytes(B * q_tokens * h, dtype),
        hbm_write_bytes=elements_to_bytes(B * q_tokens * h, dtype),
    )

    # --- Transformer layers --------------------------------------------------
    all_layers = ComputeStats(name=f"Transformer Layers ×{N}")
    representative_layers: list[ComputeStats] = []

    for layer_idx in range(N):
        is_sparse = cfg.is_sparse_layer(layer_idx)
        layer = _transformer_layer_stats(
            cfg=cfg,
            q_len=q_tokens,
            kv_len=kv_tokens,
            batch_size=B,
            use_flash_attn=use_flash_attn,
            dtype=dtype,
            layer_idx=layer_idx,
            is_sparse=is_sparse,
        )
        all_layers += layer
        if not representative_layers or _layer_family_name(representative_layers[-1]) != _layer_family_name(layer):
            representative_layers.append(layer)

    result.per_layer = representative_layers[0] if representative_layers else ComputeStats("Per Layer")
    result.all_layers = all_layers
    result.layer_breakdown = representative_layers

    # --- Final norm ----------------------------------------------------------
    final_norm = _norm_stats(h, cfg.norm_type, q_tokens, B, dtype)
    final_norm.name = f"Final {cfg.norm_type.upper()}"

    # --- LM Head ------------------------------------------------------------
    lm_head_params = 0 if cfg.tie_word_embeddings else V * h
    lm_head_w = elements_to_bytes(lm_head_params, dtype)
    lm_head_flops = 2 * B * q_tokens * h * V
    lm_head = ComputeStats(
        name="LM Head",
        num_params=lm_head_params,
        flops=lm_head_flops,
        weight_bytes=lm_head_w,
        act_bytes=elements_to_bytes(B * q_tokens * V, dtype),
        hbm_read_bytes=elements_to_bytes(B * q_tokens * h, dtype) + (emb_w if cfg.tie_word_embeddings else lm_head_w),
        hbm_write_bytes=elements_to_bytes(B * q_tokens * V, dtype),
    )
    result.lm_head = ComputeStats(
        name="Final Norm + LM Head",
        num_params=lm_head.num_params + final_norm.num_params,
        flops=lm_head.flops + final_norm.flops,
        weight_bytes=lm_head.weight_bytes + final_norm.weight_bytes,
        act_bytes=max(lm_head.act_bytes, final_norm.act_bytes),
        hbm_read_bytes=lm_head.hbm_read_bytes + final_norm.hbm_read_bytes,
        hbm_write_bytes=lm_head.hbm_write_bytes + final_norm.hbm_write_bytes,
    )

    # --- Total --------------------------------------------------------------
    total_params = result.embedding.num_params + all_layers.num_params + result.lm_head.num_params
    total_flops  = result.embedding.flops + all_layers.flops + result.lm_head.flops
    total_w      = result.embedding.weight_bytes + all_layers.weight_bytes + result.lm_head.weight_bytes
    total_hbm_r  = result.embedding.hbm_read_bytes + all_layers.hbm_read_bytes + result.lm_head.hbm_read_bytes
    total_hbm_w  = result.embedding.hbm_write_bytes + all_layers.hbm_write_bytes + result.lm_head.hbm_write_bytes

    result.total = ComputeStats(
        name=f"Total ({cfg.name})",
        num_params=total_params,
        flops=total_flops,
        weight_bytes=total_w,
        act_bytes=max(
            [result.embedding.act_bytes, result.lm_head.act_bytes]
            + [layer.act_bytes for layer in representative_layers]
        ),
        hbm_read_bytes=total_hbm_r,
        hbm_write_bytes=total_hbm_w,
    )

    # --- KV Cache analysis --------------------------------------------------
    result.kv_cache = kv_cache_stats(
        hidden_size=h,
        num_q_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        num_layers=N,
        seq_len=seq_len,
        batch_size=B,
        dtype=dtype,
        q_len=q_tokens,
        kv_len=kv_tokens,
        cache_len=cache_tokens,
        use_mla=cfg.use_mla,
        kv_lora_rank=cfg.kv_lora_rank,
        qk_head_dim=cfg.qk_head_dim,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        v_head_dim=cfg.v_head_dim,
        use_dsa=cfg.use_dsa,
        index_head_dim=cfg.index_head_dim,
        cache_layout=cfg.cache_layout,
        hit_ratio=kv_hit_ratio,
    )

    return result
