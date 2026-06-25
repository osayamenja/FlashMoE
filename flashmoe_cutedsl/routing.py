from __future__ import annotations

from .types import RoutingResult


def route_topk(tokens, router_weights, *, top_k: int, capacity: int, return_logits: bool = False) -> RoutingResult:
    """Build FlashMoE-style expert-major routing tables.

    The returned ``token_ids`` and ``token_probs`` tensors have shape
    ``[num_experts, capacity]``.  ``expert_counts`` keeps the pre-cap count,
    matching the CUDA implementation's drop-on-capacity behavior.
    """
    import torch

    if tokens.ndim != 2 or router_weights.ndim != 2:
        raise ValueError("tokens and router_weights must be rank-2 tensors")
    if tokens.shape[1] != router_weights.shape[0]:
        raise ValueError("router_weights must have shape [hidden_dim, num_experts]")

    num_experts = int(router_weights.shape[1])
    logits = tokens.float().matmul(router_weights.float())
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_experts = torch.topk(probs, k=top_k, dim=-1)
    denom = top_probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(top_probs.dtype).tiny)
    top_probs = top_probs / denom

    flat_experts = top_experts.reshape(-1).to(torch.int64)
    flat_tokens = (
        torch.arange(tokens.shape[0], device=tokens.device, dtype=torch.int64)
        .unsqueeze(1)
        .expand(-1, top_k)
        .reshape(-1)
    )
    flat_probs = top_probs.reshape(-1).float()

    order = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts.index_select(0, order)
    sorted_tokens = flat_tokens.index_select(0, order)
    sorted_probs = flat_probs.index_select(0, order)

    expert_counts = torch.bincount(sorted_experts, minlength=num_experts).to(torch.int32)
    starts = torch.cumsum(
        torch.cat(
            [
                torch.zeros(1, device=tokens.device, dtype=torch.int64),
                expert_counts.to(torch.int64)[:-1],
            ]
        ),
        dim=0,
    )
    slot = torch.arange(sorted_experts.numel(), device=tokens.device, dtype=torch.int64) - starts.index_select(0, sorted_experts)
    keep = slot < capacity

    token_ids = torch.full((num_experts, capacity), -1, device=tokens.device, dtype=torch.int64)
    token_probs = torch.zeros((num_experts, capacity), device=tokens.device, dtype=torch.float32)
    kept_experts = sorted_experts[keep]
    kept_slots = slot[keep]
    token_ids[kept_experts, kept_slots] = sorted_tokens[keep]
    token_probs[kept_experts, kept_slots] = sorted_probs[keep]

    return RoutingResult(
        token_ids=token_ids,
        token_probs=token_probs,
        expert_counts=expert_counts,
        top_experts=top_experts,
        top_probs=top_probs,
        logits=logits if return_logits else None,
    )
