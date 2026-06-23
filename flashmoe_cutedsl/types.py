from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import ceil
from typing import Any, Sequence


class Topology(IntEnum):
    NVLINK_ONLY = 0
    MIXED = 1


class MLPType(IntEnum):
    GATED = 0
    VANILLA = 1


class DataType(IntEnum):
    BF16 = 0
    FP16 = 1
    FP32 = 2
    FP64 = 3


class ActivationType(IntEnum):
    IDENTITY = 0
    SILU = 1
    GELU = 2
    RELU = 3


@dataclass(slots=True)
class InitArgs:
    data_type: DataType
    tokens_per_rank: int
    token_dim: int
    ffn_size: int
    num_experts: int
    top_k: int
    mlp_type: MLPType = MLPType.GATED
    act_type: ActivationType = ActivationType.SILU
    device_id: int | None = None
    ep_world: int | None = None
    ep_rank: int | None = None
    my_pe: int | None = None
    num_local_experts: int | None = None
    expert_map: Sequence[int] | None = None
    rank_map: Sequence[int] | None = None
    expert_peer_capacity: int | None = None
    stream: Any | None = None

    def __post_init__(self) -> None:
        if self.tokens_per_rank <= 0:
            raise ValueError("tokens_per_rank must be positive")
        if self.token_dim <= 0 or self.ffn_size <= 0:
            raise ValueError("token_dim and ffn_size must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.top_k <= 0 or self.top_k > self.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if self.expert_peer_capacity is None:
            self.expert_peer_capacity = ceil(self.tokens_per_rank / self.num_experts) * self.top_k
        if self.expert_peer_capacity <= 0:
            raise ValueError("expert_peer_capacity must be positive")


@dataclass(slots=True)
class RoutingResult:
    token_ids: Any
    token_probs: Any
    expert_counts: Any
    top_experts: Any
    top_probs: Any
    logits: Any | None = None


@dataclass(slots=True)
class ExpertWeights:
    local_expert_up: Any
    local_bias_up: Any
    local_expert_down: Any
    local_bias_down: Any
    local_expert_up_v: Any | None = None
    local_bias_up_v: Any | None = None


@dataclass(slots=True)
class ForwardArgs:
    tokens: Any
    weights: ExpertWeights
    router_weights: Any | None = None
    routing: RoutingResult | None = None
    moe_out: Any | None = None
    swish_alpha: float = 1.0
    swish_beta: float = 1.0
    return_logits: bool = False
    stream: Any | None = None
