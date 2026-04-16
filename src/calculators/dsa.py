"""
dsa.py
======
FLOPs, parameters, and memory-access statistics for Differential Sparse
Attention (DSA) as used in GLM-5.

The official GLM-5 implementation builds a lightweight indexer branch that
computes top-k token indices, then runs sparse MLA against those selected KV
entries. This file models the *runtime* path rather than a dense score-matrix
materialization.
"""

from __future__ import annotations

from .base import ComputeStats, DType, dtype_bytes
from .linear import LinearStats

INDEX_DTYPE_BYTES = 4


def dsa_indexer_stats(
    hidden_size: int,
    q_lora_rank: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    seq_len: int = 1,
    batch_size: int = 1,
    dtype: DType = DType.BF16,
) -> ComputeStats:
    """Compute stats for the DSA indexer branch.

    We model the indexer as blockwise top-k selection:
      * dot-product cost is still O(s²)
      * HBM/activation storage is O(s·k), not O(s²), because only top-k scores
        and indices are retained.
    """
    B, s = batch_size, seq_len
    idx_a = index_n_heads
    idx_h = index_head_dim
    c_q = q_lora_rank
    k_sel = min(index_topk, seq_len)
    eb = dtype_bytes(dtype)

    stats = ComputeStats(name="DSA Indexer")

    stats.children.append(
        LinearStats(
            name="Indexer Q proj (wq_b)",
            in_features=c_q,
            out_features=idx_a * idx_h,
            has_bias=False,
            seq_len=seq_len,
            batch_size=batch_size,
            dtype=dtype,
        ).compute()
    )
    stats.children.append(
        LinearStats(
            name="Indexer K proj (wk)",
            in_features=hidden_size,
            out_features=idx_h,
            has_bias=False,
            seq_len=seq_len,
            batch_size=batch_size,
            dtype=dtype,
        ).compute()
    )
    stats.children.append(
        LinearStats(
            name="Indexer head weights",
            in_features=hidden_size,
            out_features=idx_a,
            has_bias=False,
            seq_len=seq_len,
            batch_size=batch_size,
            dtype=dtype,
        ).compute()
    )

    k_norm_params = 2 * idx_h
    stats.children.append(
        ComputeStats(
            name="Indexer K norm",
            num_params=k_norm_params,
            flops=7 * B * s * idx_h,
            weight_bytes=int(k_norm_params * eb),
            hbm_read_bytes=int(B * s * idx_h * eb + k_norm_params * eb),
            hbm_write_bytes=int(B * s * idx_h * eb),
            act_bytes=int(B * s * idx_h * eb),
        )
    )

    q_elems = B * s * idx_a * idx_h
    k_elems = B * s * idx_h
    score_elems = B * s * k_sel
    index_bytes = B * s * k_sel * INDEX_DTYPE_BYTES
    flops_qkt = 2 * B * idx_a * s * s * idx_h
    flops_softmax = 5 * B * s * k_sel

    stats.children.append(
        ComputeStats(
            name=f"Index top-k scoring (top-{k_sel})",
            flops=flops_qkt + flops_softmax,
            act_bytes=int(score_elems * eb),
            hbm_read_bytes=int((q_elems + k_elems) * eb),
            hbm_write_bytes=int(score_elems * eb + index_bytes),
        )
    )

    stats.aggregate_children()
    return stats



def dsa_sparse_attention_stats(
    num_q_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    index_topk: int,
    seq_len: int = 1,
    batch_size: int = 1,
    dtype: DType = DType.BF16,
) -> ComputeStats:
    """Compute stats for the DSA sparse MLA main branch.

    Runtime assumption:
      * queries are read once
      * per query, top-k KV entries are gathered from the compressed KV cache
      * only the selected score buffer is materialized
    """
    B, s = batch_size, seq_len
    a = num_q_heads
    c_kv = kv_lora_rank
    d_r = qk_rope_head_dim
    eb = dtype_bytes(dtype)

    k_sel = min(index_topk, seq_len)
    absorbed_dim = c_kv + d_r

    flops_qkt = 2 * B * a * s * k_sel * absorbed_dim
    flops_av = 2 * B * a * s * k_sel * c_kv
    flops_softmax = 5 * B * a * s * k_sel
    total_flops = flops_qkt + flops_softmax + flops_av

    q_elems = B * s * a * absorbed_dim
    gathered_kv_elems = B * s * k_sel * absorbed_dim
    latent_out_elems = B * s * a * c_kv
    index_read_bytes = B * s * k_sel * INDEX_DTYPE_BYTES

    return ComputeStats(
        name=f"DSA Sparse MLA (top-{k_sel})",
        flops=total_flops,
        act_bytes=int(B * a * s * k_sel * eb),
        hbm_read_bytes=int((q_elems + gathered_kv_elems) * eb + index_read_bytes),
        hbm_write_bytes=int(latent_out_elems * eb),
    )
