from __future__ import annotations

from .types import ActivationType, MLPType


def apply_activation(x, act_type: ActivationType):
    import torch.nn.functional as F

    if act_type == ActivationType.IDENTITY:
        return x
    if act_type == ActivationType.SILU:
        return F.silu(x)
    if act_type == ActivationType.GELU:
        return F.gelu(x)
    if act_type == ActivationType.RELU:
        return F.relu(x)
    raise ValueError(f"unsupported activation: {act_type}")


def expert_mlp(
    tokens,
    *,
    up,
    bias_up,
    down,
    bias_down,
    mlp_type: MLPType,
    act_type: ActivationType,
    up_v=None,
    bias_up_v=None,
    swish_alpha: float = 1.0,
    swish_beta: float = 1.0,
):
    """Run one expert MLP using the layout from FlashMoE quickstart.py.

    ``up`` and ``up_v`` are ``[I, H]``; ``down`` is ``[H, I]``.
    """
    hidden = tokens.matmul(up.t())
    if bias_up is not None:
        hidden = hidden + bias_up

    if mlp_type == MLPType.GATED:
        if up_v is None or bias_up_v is None:
            raise ValueError("gated MLP requires up_v and bias_up_v")
        gate = swish_alpha * apply_activation(hidden * swish_beta, act_type)
        value = tokens.matmul(up_v.t())
        if bias_up_v is not None:
            value = value + bias_up_v
        hidden = gate * value
    else:
        hidden = apply_activation(hidden, act_type)

    out = hidden.matmul(down.t())
    if bias_down is not None:
        out = out + bias_down
    return out


def local_reference_forward(
    tokens,
    routing,
    weights,
    *,
    mlp_type: MLPType,
    act_type: ActivationType,
    top_k: int,
    swish_alpha: float = 1.0,
    swish_beta: float = 1.0,
    output=None,
):
    """Single-process reference for correctness and CPU smoke tests."""
    import torch

    result = output if output is not None else torch.empty_like(tokens)
    result.zero_()

    num_experts = int(routing.token_ids.shape[0])
    capacity = int(routing.token_ids.shape[1])
    for expert_idx in range(num_experts):
        count = min(int(routing.expert_counts[expert_idx].item()), capacity)
        if count == 0:
            continue
        ids = routing.token_ids[expert_idx, :count].long()
        routed = tokens.index_select(0, ids)
        expert_out = expert_mlp(
            routed,
            up=weights.local_expert_up[expert_idx],
            bias_up=weights.local_bias_up[expert_idx],
            down=weights.local_expert_down[expert_idx],
            bias_down=weights.local_bias_down[expert_idx],
            mlp_type=mlp_type,
            act_type=act_type,
            up_v=None if weights.local_expert_up_v is None else weights.local_expert_up_v[expert_idx],
            bias_up_v=None if weights.local_bias_up_v is None else weights.local_bias_up_v[expert_idx],
            swish_alpha=swish_alpha,
            swish_beta=swish_beta,
        )
        if top_k == 1:
            result.index_copy_(0, ids, expert_out)
        else:
            probs = routing.token_probs[expert_idx, :count].to(expert_out.dtype).unsqueeze(-1)
            result.index_add_(0, ids, expert_out * probs)
    return result
