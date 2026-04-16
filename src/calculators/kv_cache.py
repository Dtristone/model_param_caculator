"""
kv_cache.py
===========
KV cache size and memory access analysis considering cache hit ratios.

This module models the per-token KV cache for different attention types and
computes memory access patterns considering KV cache reuse (hit ratio).

KV cache per token per layer
-----------------------------
  Standard GQA/MHA:
    K cache: kv_heads × head_dim
    V cache: kv_heads × head_dim
    Total per token: 2 × kv_heads × head_dim

  MLA (Multi-Latent Attention):
    Compressed KV latent: kv_lora_rank
    RoPE key portion:     qk_rope_head_dim
    Total per token: kv_lora_rank + qk_rope_head_dim
    (Massive reduction vs standard: e.g. 576 vs 2×128×128 = 32768 for DeepSeek-V2)

  DSA + MLA (GLM-5):
    Main KV cache: kv_lora_rank + qk_rope_head_dim  (same as MLA)
    Indexer K cache: index_head_dim                  (for sparse selection)
    Total per token: kv_lora_rank + qk_rope_head_dim + index_head_dim

KV cache hit ratio
------------------
When serving requests with shared prefixes (system prompts, few-shot examples),
a portion of the KV cache can be reused across requests.  The hit_ratio
parameter models this:
  - hit_ratio = 0.0: no reuse, all KV cache entries are freshly computed
  - hit_ratio = 0.5: 50% of tokens have cached KVs (only need to read, not compute)
  - hit_ratio = 1.0: entire KV cache is pre-filled (decode-only scenario)

Effect on memory access:
  Cached tokens: only HBM reads (no projection FLOPs, no HBM writes for KV)
  New tokens: full projection FLOPs + HBM writes for KV

Effect on computation:
  KV projection FLOPs scale as (1 - hit_ratio) × original_flops
  Attention FLOPs remain the same (still attend to all s tokens)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .base import ComputeStats, DType, dtype_bytes


@dataclass
class KVCacheStats:
    """Per-token and total KV cache statistics for a model."""

    # Identification
    attention_type: str = "GQA"  # GQA/MHA/MQA/MLA/DSA+MLA

    # Per-token per-layer cache sizes (in elements, not bytes)
    k_cache_per_token: int = 0       # K elements per token per layer
    v_cache_per_token: int = 0       # V elements per token per layer
    total_per_token: int = 0         # total elements per token per layer

    # DSA-specific
    index_cache_per_token: int = 0   # indexer K cache per token per layer

    # Aggregated
    per_token_per_layer_bytes: float = 0.0   # bytes per token per layer
    per_token_all_layers_bytes: float = 0.0  # bytes per token, all layers
    total_cache_bytes: float = 0.0           # total cache for given seq_len

    # Model params used
    num_layers: int = 0
    seq_len: int = 0
    batch_size: int = 1

    # Cache efficiency
    compression_ratio: float = 1.0   # vs standard MHA cache size

    # Hit ratio analysis
    hit_ratio: float = 0.0
    cached_tokens: int = 0
    new_tokens: int = 0
    kv_proj_flops_saved: int = 0     # FLOPs saved by cache hits
    hbm_write_saved: int = 0         # HBM writes saved by cache hits


def kv_cache_stats(
    hidden_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_layers: int,
    seq_len: int = 2048,
    batch_size: int = 1,
    dtype: DType = DType.BF16,
    # MLA params
    use_mla: bool = False,
    kv_lora_rank: int = 0,
    qk_rope_head_dim: int = 0,
    v_head_dim: int = 0,
    # DSA params
    use_dsa: bool = False,
    index_head_dim: int = 0,
    # Hit ratio
    hit_ratio: float = 0.0,
) -> KVCacheStats:
    """Compute KV cache statistics.

    Parameters
    ----------
    hit_ratio : float
        Fraction of tokens whose KV cache entries are already present
        (0.0 = all new, 1.0 = all cached / decode step).
    """
    eb = dtype_bytes(dtype)
    B = batch_size

    if use_mla:
        # MLA: cache compressed latent + RoPE key
        k_per_token = kv_lora_rank + qk_rope_head_dim  # compressed KV
        v_per_token = 0    # V is part of the compressed latent
        total_per_token = k_per_token
        idx_per_token = index_head_dim if use_dsa else 0
        total_per_token += idx_per_token
        attn_type = "DSA+MLA" if use_dsa else "MLA"

        # Reference: standard MHA cache for same model
        # MHA equivalent: 2 * num_q_heads * head_dim_equivalent
        mha_equiv = 2 * num_kv_heads * (head_dim if head_dim > 0 else 128)
    else:
        # Standard GQA/MHA/MQA
        k_per_token = num_kv_heads * head_dim
        v_per_token = num_kv_heads * head_dim
        total_per_token = k_per_token + v_per_token
        idx_per_token = 0
        mha_equiv = 2 * num_q_heads * head_dim  # MHA equivalent for ratio

        if num_kv_heads == 1:
            attn_type = "MQA"
        elif num_kv_heads < num_q_heads:
            attn_type = "GQA"
        else:
            attn_type = "MHA"

    compression_ratio = mha_equiv / total_per_token if total_per_token > 0 else 1.0

    per_token_per_layer_bytes = total_per_token * eb
    per_token_all_layers = per_token_per_layer_bytes * num_layers
    total_cache = per_token_all_layers * seq_len * B

    # Hit ratio analysis
    cached_tokens = int(seq_len * hit_ratio)
    new_tokens = seq_len - cached_tokens

    # FLOPs saved: KV projection for cached tokens is skipped
    # For MLA: kv_a_proj FLOPs = 2 * h * (c_kv + d_r) per token
    # For GQA: K+V proj FLOPs = 2 * 2 * h * kv_heads * head_dim per token
    if use_mla:
        kv_proj_flops_per_token = 2 * hidden_size * (kv_lora_rank + qk_rope_head_dim)
    else:
        kv_proj_flops_per_token = 2 * 2 * hidden_size * num_kv_heads * head_dim

    kv_proj_flops_saved = B * cached_tokens * kv_proj_flops_per_token * num_layers

    # HBM writes saved: don't need to write KV cache for cached tokens
    hbm_write_saved = int(B * cached_tokens * total_per_token * eb * num_layers)

    return KVCacheStats(
        attention_type=attn_type,
        k_cache_per_token=k_per_token,
        v_cache_per_token=v_per_token,
        total_per_token=total_per_token,
        index_cache_per_token=idx_per_token,
        per_token_per_layer_bytes=per_token_per_layer_bytes,
        per_token_all_layers_bytes=per_token_all_layers,
        total_cache_bytes=total_cache,
        num_layers=num_layers,
        seq_len=seq_len,
        batch_size=batch_size,
        compression_ratio=compression_ratio,
        hit_ratio=hit_ratio,
        cached_tokens=cached_tokens,
        new_tokens=new_tokens,
        kv_proj_flops_saved=kv_proj_flops_saved,
        hbm_write_saved=hbm_write_saved,
    )
