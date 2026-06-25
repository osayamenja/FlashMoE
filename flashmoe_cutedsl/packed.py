from __future__ import annotations

import os

from .reference import apply_activation
from .types import ActivationType, ExpertWeights, MLPType


def _route_mask(routing):
    flat_ids = routing.token_ids.reshape(-1).long()
    return flat_ids, flat_ids >= 0


def gather_tokens_torch(tokens, routing, *, output=None):
    import torch

    flat_ids, valid = _route_mask(routing)
    if output is None:
        gathered = torch.zeros(
            flat_ids.numel(),
            tokens.shape[1],
            device=tokens.device,
            dtype=tokens.dtype,
        )
    else:
        if output.shape != (*routing.token_ids.shape, tokens.shape[1]):
            raise ValueError("gather output has the wrong shape")
        gathered = output.reshape(flat_ids.numel(), tokens.shape[1])
        gathered.zero_()
    gathered[valid] = tokens.index_select(0, flat_ids[valid])
    return gathered.reshape(*routing.token_ids.shape, tokens.shape[1])


def combine_tokens_torch(expert_out, routing, *, top_k: int, output):
    import torch

    flat_ids, valid = _route_mask(routing)
    values = expert_out.reshape(flat_ids.numel(), expert_out.shape[-1])
    ids = flat_ids[valid]
    vals = values[valid]
    result = output if output is not None else torch.empty(
        int(routing.top_experts.shape[0]),
        expert_out.shape[-1],
        device=expert_out.device,
        dtype=expert_out.dtype,
    )
    result.zero_()
    if top_k == 1:
        result.index_copy_(0, ids, vals)
    else:
        probs = routing.token_probs.reshape(-1)[valid].to(vals.dtype).unsqueeze(-1)
        result.index_add_(0, ids, vals * probs)
    return result


def expert_mlp_batched(
    expert_in,
    weights: ExpertWeights,
    *,
    mlp_type: MLPType,
    act_type: ActivationType,
    swish_alpha: float = 1.0,
    swish_beta: float = 1.0,
    use_cutedsl_mlp: bool = False,
):
    if use_cutedsl_mlp and expert_in.is_cuda:
        if mlp_type != MLPType.GATED:
            raise NotImplementedError("CuTe DSL MLP currently implements the gated FlashMoE path")
        if act_type != ActivationType.SILU:
            raise NotImplementedError("CuTe DSL gated MLP currently implements SiLU activation")
        import torch
        import cutlass
        from .ampere_tensorop_gemm import batched_tensorop_gemm
        from .kernels import gated_up, linear_down, silu_product

        if os.environ.get("FLASHMOE_CUTEDSL_SCALAR_MLP", "0") == "1":
            hidden = torch.empty(
                (*expert_in.shape[:2], weights.local_expert_up.shape[1]),
                device=expert_in.device,
                dtype=expert_in.dtype,
            )
            out = torch.empty_like(expert_in)
            gated_up(
                expert_in,
                weights,
                hidden,
                act_type=int(act_type),
                swish_alpha=swish_alpha,
                swish_beta=swish_beta,
            )
            linear_down(hidden, weights, out)
            return out

        if expert_in.dtype == torch.bfloat16:
            cutlass_dtype = cutlass.BFloat16
        elif expert_in.dtype == torch.float16:
            cutlass_dtype = cutlass.Float16
        else:
            raise NotImplementedError("CuTe tensor-op MLP currently supports bf16/fp16")

        experts, capacity, _ = expert_in.shape
        ffn = int(weights.local_expert_up.shape[1])
        token_dim = int(expert_in.shape[2])
        up = torch.empty((experts, capacity, ffn), device=expert_in.device, dtype=expert_in.dtype)
        up_v = torch.empty_like(up)
        hidden = torch.empty_like(up)
        out = torch.empty((experts, capacity, token_dim), device=expert_in.device, dtype=expert_in.dtype)

        a = expert_in.permute(1, 2, 0)
        batched_tensorop_gemm(
            a,
            weights.local_expert_up.permute(1, 2, 0),
            up.permute(1, 2, 0),
            ab_dtype=cutlass_dtype,
            c_dtype=cutlass_dtype,
        )
        batched_tensorop_gemm(
            a,
            weights.local_expert_up_v.permute(1, 2, 0),
            up_v.permute(1, 2, 0),
            ab_dtype=cutlass_dtype,
            c_dtype=cutlass_dtype,
        )
        up.add_(weights.local_bias_up[:, None, :])
        up_v.add_(weights.local_bias_up_v[:, None, :])
        silu_product(up, up_v, hidden, swish_alpha=swish_alpha, swish_beta=swish_beta)
        batched_tensorop_gemm(
            hidden.permute(1, 2, 0),
            weights.local_expert_down.permute(1, 2, 0),
            out.permute(1, 2, 0),
            ab_dtype=cutlass_dtype,
            c_dtype=cutlass_dtype,
        )
        out.add_(weights.local_bias_down[:, None, :])
        return out

    hidden = expert_in.bmm(weights.local_expert_up.transpose(1, 2))
    hidden = hidden + weights.local_bias_up[:, None, :]

    if mlp_type == MLPType.GATED:
        if weights.local_expert_up_v is None or weights.local_bias_up_v is None:
            raise ValueError("gated MLP requires up_v and bias_up_v")
        gate = swish_alpha * apply_activation(hidden * swish_beta, act_type)
        value = expert_in.bmm(weights.local_expert_up_v.transpose(1, 2))
        value = value + weights.local_bias_up_v[:, None, :]
        hidden = gate * value
    else:
        hidden = apply_activation(hidden, act_type)

    out = hidden.bmm(weights.local_expert_down.transpose(1, 2))
    return out + weights.local_bias_down[:, None, :]


def local_packed_forward(
    tokens,
    routing,
    weights: ExpertWeights,
    *,
    mlp_type: MLPType,
    act_type: ActivationType,
    top_k: int,
    swish_alpha: float = 1.0,
    swish_beta: float = 1.0,
    output=None,
    expert_in=None,
    use_cutedsl_gather: bool = False,
    use_cutedsl_combine: bool = False,
    use_cutedsl_mlp: bool = False,
):
    if use_cutedsl_gather and tokens.is_cuda:
        try:
            import torch
            from .kernels import gather_tokens

            if expert_in is None:
                expert_in = torch.empty(
                    (*routing.token_ids.shape, tokens.shape[1]),
                    device=tokens.device,
                    dtype=tokens.dtype,
                )
            gather_tokens(tokens, routing.token_ids, expert_in)
        except Exception:
            if os.environ.get("FLASHMOE_CUTEDSL_STRICT", "0") == "1":
                raise
            expert_in = None

    if expert_in is None:
        expert_in = gather_tokens_torch(tokens, routing)
    elif not use_cutedsl_gather:
        gather_tokens_torch(tokens, routing, output=expert_in)

    try:
        expert_out = expert_mlp_batched(
            expert_in,
            weights,
            mlp_type=mlp_type,
            act_type=act_type,
            swish_alpha=swish_alpha,
            swish_beta=swish_beta,
            use_cutedsl_mlp=use_cutedsl_mlp,
        )
    except Exception:
        if os.environ.get("FLASHMOE_CUTEDSL_STRICT", "0") == "1":
            raise
        expert_out = expert_mlp_batched(
            expert_in,
            weights,
            mlp_type=mlp_type,
            act_type=act_type,
            swish_alpha=swish_alpha,
            swish_beta=swish_beta,
        )
    if use_cutedsl_combine and top_k == 1 and expert_out.is_cuda:
        try:
            from .kernels import combine_top1

            result = output
            if result is None:
                import torch

                result = torch.empty(
                    int(routing.top_experts.shape[0]),
                    expert_out.shape[-1],
                    device=expert_out.device,
                    dtype=expert_out.dtype,
                )
            result.zero_()
            values = expert_out.reshape(-1, expert_out.shape[-1])
            token_ids = routing.token_ids.reshape(-1)
            combine_top1(values, token_ids, result)
            return result
        except Exception:
            if os.environ.get("FLASHMOE_CUTEDSL_STRICT", "0") == "1":
                raise
    return combine_tokens_torch(expert_out, routing, top_k=top_k, output=output)


def local_packed_reference_forward(*args, **kwargs):
    return local_packed_forward(*args, **kwargs)
