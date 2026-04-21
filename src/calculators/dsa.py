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

from .base import ComputeStats, DType, elements_to_bytes
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
    q_len: int | None = None,
    kv_len: int | None = None,
    indexer_mode: str = "fused_topk",
) -> ComputeStats:
    """Compute stats for the DSA indexer branch.

    We model the indexer as blockwise top-k selection:
      * dot-product cost is still O(s²)
      * HBM/activation storage is O(s·k), not O(s²), because only top-k scores
        and indices are retained.
    """
    B = batch_size
    idx_a = index_n_heads
    idx_h = index_head_dim
    c_q = q_lora_rank
    Q = q_len if q_len is not None else seq_len
    T = kv_len if kv_len is not None else seq_len
    k_sel = min(index_topk, T)

    stats = ComputeStats(name="DSA Indexer")

    stats.children.append(
        LinearStats(
            name="Indexer Q proj (wq_b)",
            in_features=c_q,
            out_features=idx_a * idx_h,
            has_bias=False,
            seq_len=Q,
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
            seq_len=T,
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
            seq_len=Q,
            batch_size=batch_size,
            dtype=dtype,
        ).compute()
    )

    k_norm_params = idx_h  # RMSNorm stores only a scale vector
    stats.children.append(
        ComputeStats(
            name="Indexer K norm",
            num_params=k_norm_params,
            flops=4 * B * T * idx_h,  # RMSNorm: mean-sq + rsqrt + normalize + scale
            weight_bytes=elements_to_bytes(k_norm_params, dtype),
            hbm_read_bytes=elements_to_bytes(B * T * idx_h + k_norm_params, dtype),
            hbm_write_bytes=elements_to_bytes(B * T * idx_h, dtype),
            act_bytes=elements_to_bytes(B * T * idx_h, dtype),
        )
    )

    q_elems = B * Q * idx_a * idx_h
    k_elems = B * T * idx_h
    score_elems = B * Q * k_sel
    dense_score_elems = B * Q * idx_a * T
    index_bytes = B * Q * k_sel * INDEX_DTYPE_BYTES
    flops_qkt = 2 * B * Q * T * idx_a * idx_h
    flops_relu = B * Q * T * idx_a  # ReLU activation over per-head dense scores
    flops_weighted_sum = 2 * B * Q * T * idx_a  # Weighted reduction across index heads
    flops_topk = B * Q * k_sel  # lightweight backend-agnostic approximation

    stats.children.append(
        ComputeStats(
            name=f"Index top-k scoring (top-{k_sel})",
            flops=flops_qkt + flops_relu + flops_weighted_sum + flops_topk,
            act_bytes=(
                elements_to_bytes(score_elems, dtype)
                if indexer_mode == "fused_topk"
                else elements_to_bytes(dense_score_elems, dtype)
            ),
            hbm_read_bytes=elements_to_bytes(q_elems + k_elems + B * Q * idx_a, dtype),
            hbm_write_bytes=(
                elements_to_bytes(score_elems, dtype) + index_bytes
                if indexer_mode == "fused_topk"
                else elements_to_bytes(dense_score_elems + score_elems, dtype) + index_bytes
            ),
        )
    )

    stats.aggregate_children()
    return stats



def dsa_sparse_attention_stats(
    num_q_heads: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    index_topk: int,
    cache_layout: str = "mla_compressed",
    seq_len: int = 1,
    batch_size: int = 1,
    dtype: DType = DType.BF16,
    q_len: int | None = None,
    kv_len: int | None = None,
) -> ComputeStats:
    """Compute stats for the DSA sparse MLA main branch.

    Runtime assumption:
      * queries are read once
      * per query, top-k KV entries are gathered from the compressed KV cache
      * only the selected score buffer is materialized
    """
    B = batch_size
    a = num_q_heads
    c_kv = kv_lora_rank
    d_r = qk_rope_head_dim
    v_dim = v_head_dim
    Q = q_len if q_len is not None else seq_len
    T = kv_len if kv_len is not None else seq_len

    k_sel = min(index_topk, T)
    q_head_dim = qk_nope_head_dim + d_r

    if cache_layout == "mla_expanded":
        flops_qkt = 2 * B * a * Q * k_sel * q_head_dim
        flops_av = 2 * B * a * Q * k_sel * v_dim
        q_elems = B * Q * a * q_head_dim
        gathered_k_elems = B * Q * k_sel * a * q_head_dim
        gathered_v_elems = B * Q * k_sel * a * v_dim
        out_elems = B * Q * a * v_dim
    else:
        absorbed_dim = c_kv + d_r
        flops_qkt = 2 * B * a * Q * k_sel * absorbed_dim
        flops_av = 2 * B * a * Q * k_sel * c_kv
        q_elems = B * Q * a * q_head_dim
        gathered_k_elems = B * Q * k_sel * absorbed_dim
        gathered_v_elems = 0
        out_elems = B * Q * a * c_kv

    flops_softmax = 5 * B * a * Q * k_sel
    total_flops = flops_qkt + flops_softmax + flops_av

    index_read_bytes = B * Q * k_sel * INDEX_DTYPE_BYTES

    return ComputeStats(
        name=f"DSA Sparse MLA (top-{k_sel})",
        flops=total_flops,
        act_bytes=elements_to_bytes(B * a * Q * k_sel, dtype),
        hbm_read_bytes=elements_to_bytes(q_elems + gathered_k_elems + gathered_v_elems, dtype) + index_read_bytes,
        hbm_write_bytes=elements_to_bytes(out_elems, dtype),
    )
