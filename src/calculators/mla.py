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

from .base import ComputeStats, DType, dtype_bytes
from .linear import LinearStats


def mla_proj_stats(
    hidden_size: int,
    num_q_heads: int,
    kv_lora_rank: int,
    q_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
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
    eb = dtype_bytes(dtype)

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

    kv_b_params = c_kv * a * (d_n + v_dim)
    stats.children.append(
        ComputeStats(
            name="KV up-proj weights (absorbed)",
            num_params=kv_b_params,
            weight_bytes=int(kv_b_params * eb),
        )
    )

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
        q_source_dim = c_q
    else:
        q_source_dim = hidden_size

    q_b = LinearStats(
        name="Q up-proj (q_b)" if c_q > 0 else "Q proj (direct)",
        in_features=q_source_dim,
        out_features=a * q_head_dim,
        has_bias=has_bias,
        seq_len=seq_len,
        batch_size=batch_size,
        dtype=dtype,
    ).compute()
    stats.children.append(q_b)

    q_absorb_in = tokens * a * d_n
    q_absorb_out = tokens * a * c_kv
    q_absorb_weight = a * d_n * c_kv
    stats.children.append(
        ComputeStats(
            name="Q absorb (q_nope × W_kc)",
            flops=2 * tokens * a * d_n * c_kv,
            act_bytes=int((q_absorb_in + q_absorb_out) * eb),
            hbm_read_bytes=int((q_absorb_in + q_absorb_weight) * eb),
            hbm_write_bytes=int(q_absorb_out * eb),
        )
    )

    v_expand_in = tokens * a * c_kv
    v_expand_out = tokens * a * v_dim
    v_expand_weight = a * c_kv * v_dim
    stats.children.append(
        ComputeStats(
            name="V expand (latent × W_vc)",
            flops=2 * tokens * a * c_kv * v_dim,
            act_bytes=int((v_expand_in + v_expand_out) * eb),
            hbm_read_bytes=int((v_expand_in + v_expand_weight) * eb),
            hbm_write_bytes=int(v_expand_out * eb),
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
    seq_len: int = 1,
    batch_size: int = 1,
    use_flash_attn: bool = False,
    dtype: DType = DType.BF16,
) -> ComputeStats:
    """Compute stats for the absorbed MLA attention kernel.

    Attention runs in compressed latent space with:
      QK^T dim = c_kv + d_r
      AV dim   = c_kv

    The latent -> value expansion via W_vc is accounted in `mla_proj_stats`.
    """
    B, s = batch_size, seq_len
    a = num_q_heads
    c_kv = kv_lora_rank
    d_r = qk_rope_head_dim
    eb = dtype_bytes(dtype)

    absorbed_dim = c_kv + d_r

    flops_qkt = 2 * B * a * s * s * absorbed_dim
    flops_av = 2 * B * a * s * s * c_kv
    flops_softmax = 5 * B * a * s * s
    total_flops = flops_qkt + flops_softmax + flops_av

    q_elems = B * s * a * absorbed_dim
    kv_cache_elems = B * s * absorbed_dim
    latent_out_elems = B * s * a * c_kv

    if use_flash_attn:
        hbm_read = int((q_elems + kv_cache_elems) * eb)
        hbm_write = int(latent_out_elems * eb)
        act_bytes = int(2 * B * a * s * eb)
        label = "MLA Flash Attention (absorbed)"
    else:
        attn_matrix = int(B * a * s * s * eb)
        hbm_read = int((q_elems + kv_cache_elems) * eb) + attn_matrix
        hbm_write = attn_matrix + int(latent_out_elems * eb)
        act_bytes = attn_matrix
        label = "MLA Standard Attention (absorbed)"

    return ComputeStats(
        name=label,
        flops=total_flops,
        act_bytes=act_bytes,
        hbm_read_bytes=hbm_read,
        hbm_write_bytes=hbm_write,
    )
