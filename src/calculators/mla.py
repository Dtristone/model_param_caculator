"""
mla.py
======
FLOPs, parameters, and memory-access statistics for Multi-Latent Attention
(MLA) in the absorbed inference path.

This matches the optimized path visible in the official GLM-5 code, where the
KV up-projection is not materialized into full K/V activations. Instead:

  * kv_a_proj produces [c_kv + d_r] per token for cache storage
  * q_b_proj produces Q heads in [d_n + d_r]
  * q_nope is contracted with W_kc to enter the compressed latent space
  * attention runs on [c_kv + d_r] keys
  * the latent output is expanded by W_vc to value heads of width v_dim

Notation
--------
  h       = hidden_size
  a       = num_attention_heads
  c_kv    = kv_lora_rank
  c_q     = q_lora_rank
  d_n     = qk_nope_head_dim
  d_r     = qk_rope_head_dim
  v_dim   = v_head_dim

KV cache per token = c_kv + d_r elements.
"""

from __future__ import annotations

from .base import ComputeStats, DType, elements_to_bytes
from .linear import LinearStats


def _rmsnorm_stats(name: str, hidden_size: int, seq_len: int, batch_size: int, dtype: DType) -> ComputeStats:
    tokens = batch_size * seq_len
    return ComputeStats(
        name=name,
        num_params=hidden_size,
        flops=4 * tokens * hidden_size,
        weight_bytes=elements_to_bytes(hidden_size, dtype),
        act_bytes=elements_to_bytes(tokens * hidden_size, dtype),
        hbm_read_bytes=elements_to_bytes(tokens * hidden_size + hidden_size, dtype),
        hbm_write_bytes=elements_to_bytes(tokens * hidden_size, dtype),
    )


def mla_proj_stats(
    hidden_size: int,
    num_q_heads: int,
    kv_lora_rank: int,
    q_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    cache_layout: str = "mla_compressed",
    seq_len: int = 1,
    batch_size: int = 1,
    has_bias: bool = False,
    dtype: DType = DType.BF16,
) -> ComputeStats:
    """Compute FLOPs/params for absorbed MLA projection layers.

    Included runtime operations:
      - kv_a_proj
      - q_a_proj / q_b_proj (or direct q proj)
      - q_absorb: q_nope × W_kc
      - v_expand: latent_out × W_vc
      - o_proj

    The full kv_b_proj parameter tensor is counted once as model weight storage,
    while its runtime cost is split across q_absorb and v_expand.
    """
    a = num_q_heads
    c_kv = kv_lora_rank
    c_q = q_lora_rank
    d_n = qk_nope_head_dim
    d_r = qk_rope_head_dim
    v_dim = v_head_dim
    q_head_dim = d_n + d_r
    tokens = batch_size * seq_len
    stats = ComputeStats(name="MLA Projection")

    kv_a = LinearStats(
        name="KV down-proj (kv_a)",
        in_features=hidden_size,
        out_features=c_kv + d_r,
        has_bias=has_bias,
        seq_len=seq_len,
        batch_size=batch_size,
        dtype=dtype,
    ).compute()
    stats.children.append(kv_a)
    stats.children.append(
        _rmsnorm_stats(
            name="KV latent RMSNorm (kv_a_layernorm)",
            hidden_size=c_kv,
            seq_len=seq_len,
            batch_size=batch_size,
            dtype=dtype,
        )
    )

    kv_b_params = c_kv * a * (d_n + v_dim)

    if c_q > 0:
        q_a = LinearStats(
            name="Q down-proj (q_a)",
            in_features=hidden_size,
            out_features=c_q,
            has_bias=has_bias,
            seq_len=seq_len,
            batch_size=batch_size,
            dtype=dtype,
        ).compute()
        stats.children.append(q_a)
        stats.children.append(
            _rmsnorm_stats(
                name="Q latent RMSNorm (q_a_layernorm)",
                hidden_size=c_q,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
            )
        )
        q_source_dim = c_q
    else:
        q_source_dim = hidden_size

    q_b = LinearStats(
        name="Q up-proj (q_b)" if c_q > 0 else "Q proj (direct)",
        in_features=q_source_dim,
        out_features=a * q_head_dim,
        has_bias=False,
        seq_len=seq_len,
        batch_size=batch_size,
        dtype=dtype,
    ).compute()
    stats.children.append(q_b)

    if cache_layout == "mla_expanded":
        expanded_cache_out = tokens * a * (q_head_dim + v_dim)
        stats.children.append(
            ComputeStats(
                name="KV up-proj + expanded cache",
                num_params=kv_b_params,
                flops=2 * tokens * a * c_kv * (d_n + v_dim),
                weight_bytes=elements_to_bytes(kv_b_params, dtype),
                act_bytes=elements_to_bytes(tokens * (c_kv + d_r) + expanded_cache_out, dtype),
                hbm_read_bytes=elements_to_bytes(tokens * (c_kv + d_r) + kv_b_params, dtype),
                hbm_write_bytes=elements_to_bytes(expanded_cache_out, dtype),
            )
        )
    else:
        stats.children.append(
            ComputeStats(
                name="KV up-proj weights (absorbed)",
                num_params=kv_b_params,
                weight_bytes=elements_to_bytes(kv_b_params, dtype),
            )
        )

        q_absorb_in = tokens * a * d_n
        q_absorb_out = tokens * a * c_kv
        q_absorb_weight = a * d_n * c_kv
        stats.children.append(
            ComputeStats(
                name="Q absorb (q_nope × W_kc)",
                flops=2 * tokens * a * d_n * c_kv,
                act_bytes=elements_to_bytes(q_absorb_in + q_absorb_out, dtype),
                hbm_read_bytes=elements_to_bytes(q_absorb_in + q_absorb_weight, dtype),
                hbm_write_bytes=elements_to_bytes(q_absorb_out, dtype),
            )
        )

        v_expand_in = tokens * a * c_kv
        v_expand_out = tokens * a * v_dim
        v_expand_weight = a * c_kv * v_dim
        stats.children.append(
            ComputeStats(
                name="V expand (latent × W_vc)",
                flops=2 * tokens * a * c_kv * v_dim,
                act_bytes=elements_to_bytes(v_expand_in + v_expand_out, dtype),
                hbm_read_bytes=elements_to_bytes(v_expand_in + v_expand_weight, dtype),
                hbm_write_bytes=elements_to_bytes(v_expand_out, dtype),
            )
        )

    o_proj = LinearStats(
        name="O Projection",
        in_features=a * v_dim,
        out_features=hidden_size,
        has_bias=has_bias,
        seq_len=seq_len,
        batch_size=batch_size,
        dtype=dtype,
    ).compute()
    stats.children.append(o_proj)

    stats.aggregate_children()
    return stats



def mla_attention_stats(
    num_q_heads: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    cache_layout: str = "mla_compressed",
    seq_len: int = 1,
    batch_size: int = 1,
    use_flash_attn: bool = False,
    dtype: DType = DType.BF16,
    q_len: int | None = None,
    kv_len: int | None = None,
) -> ComputeStats:
    """Compute stats for the absorbed MLA attention kernel.

    Compressed mode runs in absorbed latent space and accounts for latent->value
    expansion in `mla_proj_stats`. Expanded mode reads full per-head K/V cache.
    """
    B = batch_size
    a = num_q_heads
    c_kv = kv_lora_rank
    d_n = qk_nope_head_dim
    d_r = qk_rope_head_dim
    v_dim = v_head_dim
    Q = q_len if q_len is not None else seq_len
    T = kv_len if kv_len is not None else seq_len

    q_head_dim = d_n + d_r

    if cache_layout == "mla_expanded":
        flops_qkt = 2 * B * a * Q * T * q_head_dim
        flops_av = 2 * B * a * Q * T * v_dim
        q_elems = B * Q * a * q_head_dim
        k_cache_elems = B * T * a * q_head_dim
        v_cache_elems = B * T * a * v_dim
        attn_out_elems = B * Q * a * v_dim
        hbm_base_read = elements_to_bytes(q_elems + k_cache_elems + v_cache_elems, dtype)
        hbm_base_write = elements_to_bytes(attn_out_elems, dtype)
        mode_label = "expanded"
    else:
        absorbed_dim = c_kv + d_r
        flops_qkt = 2 * B * a * Q * T * absorbed_dim
        flops_av = 2 * B * a * Q * T * c_kv
        q_elems = B * Q * a * q_head_dim
        kv_cache_elems = B * T * absorbed_dim
        attn_out_elems = B * Q * a * c_kv
        hbm_base_read = elements_to_bytes(q_elems + kv_cache_elems, dtype)
        hbm_base_write = elements_to_bytes(attn_out_elems, dtype)
        mode_label = "absorbed"

    flops_softmax = 5 * B * a * Q * T
    total_flops = flops_qkt + flops_softmax + flops_av

    if use_flash_attn:
        hbm_read = hbm_base_read
        hbm_write = hbm_base_write
        act_bytes = elements_to_bytes(2 * B * a * Q, dtype)
        label = f"MLA Flash Attention ({mode_label})"
    else:
        attn_matrix = elements_to_bytes(B * a * Q * T, dtype)
        hbm_read = hbm_base_read + attn_matrix
        hbm_write = attn_matrix + hbm_base_write
        act_bytes = attn_matrix
        label = f"MLA Standard Attention ({mode_label})"

    return ComputeStats(
        name=label,
        flops=total_flops,
        act_bytes=act_bytes,
        hbm_read_bytes=hbm_read,
        hbm_write_bytes=hbm_write,
    )
