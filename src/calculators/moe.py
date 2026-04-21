"""
moe.py
======
FLOPs and memory-access statistics for a Mixture-of-Experts (MoE) layer.

Architecture
------------
Each MoE layer consists of:

  1. Router  — a linear projection h → num_experts, followed by top-K selection.
     FLOPs  = 2 * B * s * h * num_experts  (gating linear)
     Params = h * num_experts

  2. Expert FFNs — E independent FFN sub-networks; only K are active per token.
     Active FLOPs  = K × FFN_FLOPs(h, expert_intermediate)
     Total Params  = E × FFN_Params(h, expert_intermediate)

Notes
-----
* Parameter count uses ALL experts (they all live in HBM).
* FLOPs count only the K *activated* experts per token.
* HBM bandwidth is typically dominated by loading expert weights even for
  inactive experts when using naive dispatch (but expert parallelism can
  reduce this); here we model the *activated* weight traffic only.
"""

from __future__ import annotations

from .base import ComputeStats, DType
from .ffn import ffn_stats
from .linear import LinearStats


def moe_stats(
    hidden_size: int,
    num_experts: int,
    num_experts_per_tok: int,
    expert_intermediate_size: int,
    num_shared_experts: int = 0,
    ffn_type: str = "swiglu",
    seq_len: int = 1,
    batch_size: int = 1,
    has_bias: bool = False,
    dtype: DType = DType.BF16,
) -> ComputeStats:
    """Compute FLOPs and memory-access statistics for a MoE layer.

    Parameters
    ----------
    hidden_size : int
        Hidden dimension h.
    num_experts : int
        Total number of experts E.
    num_experts_per_tok : int
        Number of experts activated per token K (top-K).
    expert_intermediate_size : int
        Intermediate size of each expert FFN.
    ffn_type : str
        FFN variant used by each expert.
    seq_len, batch_size : int
        Sequence length and batch size.
    has_bias : bool
        Whether expert linear layers have bias.
    dtype : DType
        Element dtype.
    """
    stats = ComputeStats(
        name=f"MoE (routed={num_experts}, shared={num_shared_experts}, K={num_experts_per_tok})"
    )

    # ------------------------------------------------------------------
    # 1. Router
    # ------------------------------------------------------------------
    router = LinearStats(
        name="Router",
        in_features=hidden_size,
        out_features=num_experts,
        has_bias=False,
        seq_len=seq_len,
        batch_size=batch_size,
        dtype=dtype,
    ).compute()
    stats.children.append(router)

    # ------------------------------------------------------------------
    # 2. Expert FFNs
    # ------------------------------------------------------------------
    # FLOPs: only K active experts per token
    one_expert = ffn_stats(
        hidden_size=hidden_size,
        intermediate_size=expert_intermediate_size,
        ffn_type=ffn_type,
        seq_len=seq_len,
        batch_size=batch_size,
        has_bias=has_bias,
        dtype=dtype,
    )
    one_expert.name = f"Routed Expert FFN ×{num_experts_per_tok} (active)"

    # Scale FLOPs to K active experts (keep params as E × one_expert.params)
    active_expert = ComputeStats(
        name=one_expert.name,
        num_params=one_expert.num_params * num_experts,       # ALL experts in memory
        flops=one_expert.flops * num_experts_per_tok,         # only K active
        weight_bytes=one_expert.weight_bytes * num_experts,   # all weights in HBM
        act_bytes=one_expert.act_bytes * num_experts_per_tok,
        hbm_read_bytes=one_expert.hbm_read_bytes * num_experts_per_tok,
        hbm_write_bytes=one_expert.hbm_write_bytes * num_experts_per_tok,
        children=[],
    )
    stats.children.append(active_expert)

    if num_shared_experts > 0:
        shared_expert = ffn_stats(
            hidden_size=hidden_size,
            intermediate_size=expert_intermediate_size * num_shared_experts,
            ffn_type=ffn_type,
            seq_len=seq_len,
            batch_size=batch_size,
            has_bias=has_bias,
            dtype=dtype,
        )
        shared_expert.name = f"Shared Expert FFN ×{num_shared_experts}"
        stats.children.append(shared_expert)

    stats.aggregate_children()
    return stats
