from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from . import nvshmem_backend as backend
from .packed import expert_mlp_batched, local_packed_forward
from .reference import expert_mlp, local_reference_forward
from .routing import route_topk
from .types import DataType, ExpertWeights, ForwardArgs, InitArgs, MLPType, RoutingResult


def _torch_dtype(data_type: DataType):
    import torch

    if data_type == DataType.BF16:
        return torch.bfloat16
    if data_type == DataType.FP16:
        return torch.float16
    if data_type == DataType.FP32:
        return torch.float32
    if data_type == DataType.FP64:
        return torch.float64
    raise ValueError(f"unsupported dtype: {data_type}")


def _uniform_expert_map(num_experts: int, world: int) -> tuple[list[int], int]:
    if num_experts % world != 0:
        raise ValueError("num_experts must divide ep_world for the default blocked expert map")
    n_local = num_experts // world
    return [idx // n_local for idx in range(num_experts)], n_local


@dataclass(slots=True)
class _SymmetricState:
    dispatch_tokens: Any
    dispatch_ids: Any
    dispatch_probs: Any
    dispatch_counts: Any
    return_tokens: Any
    return_ids: Any
    return_probs: Any
    return_counts: Any


class FlashDMoEPort:
    """Host-orchestrated FlashMoE forward path in Python.

    The CUDA implementation fuses dispatch, compute and combine into one
    persistent kernel.  This port keeps the same logical buffers and routing
    tables, but uses NVSHMEM4Py stream puts between Python stages.
    """

    def __init__(
        self,
        args: InitArgs,
        *,
        use_nvshmem: bool = True,
        use_cutedsl_combine: bool | None = None,
        use_packed: bool = True,
        use_cutedsl_gather: bool | None = None,
    ):
        self.args = args
        self.use_nvshmem = use_nvshmem
        self.use_packed = use_packed
        self.use_cutedsl_gather = (
            os.environ.get("FLASHMOE_CUTEDSL_GATHER", "0") == "1"
            if use_cutedsl_gather is None
            else use_cutedsl_gather
        )
        self.use_cutedsl_combine = (
            os.environ.get("FLASHMOE_CUTEDSL_COMBINE", "0") == "1"
            if use_cutedsl_combine is None
            else use_cutedsl_combine
        )
        self._state: _SymmetricState | None = None
        self._local_expert_in: Any | None = None
        self._local_output: Any | None = None

        if use_nvshmem:
            backend.initialize(args.device_id)
            ep_world = backend.world_size()
            ep_rank = backend.rank()
            my_pe = ep_rank
        else:
            ep_world = args.ep_world or 1
            ep_rank = args.ep_rank or 0
            my_pe = args.my_pe or ep_rank

        self.ep_world = args.ep_world if args.ep_world is not None else ep_world
        self.ep_rank = args.ep_rank if args.ep_rank is not None else ep_rank
        self.my_pe = args.my_pe if args.my_pe is not None else my_pe

        if args.expert_map is None:
            expert_map, num_local = _uniform_expert_map(args.num_experts, self.ep_world)
        else:
            expert_map = [int(x) for x in args.expert_map]
            if len(expert_map) != args.num_experts:
                raise ValueError("expert_map length must equal num_experts")
            if args.num_local_experts is None:
                counts = [expert_map.count(rank) for rank in range(self.ep_world)]
                if len(set(counts)) != 1:
                    raise ValueError("non-uniform expert maps need explicit num_local_experts")
                num_local = counts[0]
            else:
                num_local = int(args.num_local_experts)

        self.expert_map = expert_map
        self.num_local_experts = args.num_local_experts or num_local
        self.rank_map = list(args.rank_map) if args.rank_map is not None else list(range(self.ep_world))
        if len(self.rank_map) != self.ep_world:
            raise ValueError("rank_map length must equal ep_world")

        self.global_to_local = self._build_global_to_local()
        self.local_to_global = self._build_local_to_global()

    def _local_workspace(self, tokens, routing: RoutingResult, output):
        if output is not None:
            local_output = output
        else:
            if (
                self._local_output is None
                or self._local_output.shape != tokens.shape
                or self._local_output.dtype != tokens.dtype
                or self._local_output.device != tokens.device
            ):
                import torch

                self._local_output = torch.empty_like(tokens)
            local_output = self._local_output

        expected = (*routing.token_ids.shape, tokens.shape[1])
        if (
            self._local_expert_in is None
            or self._local_expert_in.shape != expected
            or self._local_expert_in.dtype != tokens.dtype
            or self._local_expert_in.device != tokens.device
        ):
            import torch

            self._local_expert_in = torch.empty(expected, device=tokens.device, dtype=tokens.dtype)
        return self._local_expert_in, local_output

    def _build_global_to_local(self) -> list[int]:
        seen = [0 for _ in range(self.ep_world)]
        mapping: list[int] = []
        for owner in self.expert_map:
            mapping.append(seen[owner])
            seen[owner] += 1
        if any(count != self.num_local_experts for count in seen):
            raise ValueError("all EP ranks must own num_local_experts experts")
        return mapping

    def _build_local_to_global(self) -> list[int]:
        local = [-1 for _ in range(self.num_local_experts)]
        for expert_idx, owner in enumerate(self.expert_map):
            if owner == self.ep_rank:
                local[self.global_to_local[expert_idx]] = expert_idx
        if any(idx < 0 for idx in local):
            raise ValueError("local expert mapping is incomplete")
        return local

    def _ensure_state(self) -> _SymmetricState:
        if self._state is not None:
            return self._state
        if not self.use_nvshmem:
            raise RuntimeError("symmetric state is only available when use_nvshmem=True")

        import torch

        dtype = _torch_dtype(self.args.data_type)
        shape_tokens = (
            self.ep_world,
            self.num_local_experts,
            self.args.expert_peer_capacity,
            self.args.token_dim,
        )
        shape_meta = (
            self.ep_world,
            self.num_local_experts,
            self.args.expert_peer_capacity,
        )
        shape_counts = (self.ep_world, self.num_local_experts)
        self._state = _SymmetricState(
            dispatch_tokens=backend.symmetric_tensor(shape_tokens, dtype),
            dispatch_ids=backend.symmetric_tensor(shape_meta, torch.int64),
            dispatch_probs=backend.symmetric_tensor(shape_meta, torch.float32),
            dispatch_counts=backend.symmetric_tensor(shape_counts, torch.int32),
            return_tokens=backend.symmetric_tensor(shape_tokens, dtype),
            return_ids=backend.symmetric_tensor(shape_meta, torch.int64),
            return_probs=backend.symmetric_tensor(shape_meta, torch.float32),
            return_counts=backend.symmetric_tensor(shape_counts, torch.int32),
        )
        backend.sync_all(self.args.stream)
        return self._state

    def route(self, tokens, router_weights, *, return_logits: bool = False) -> RoutingResult:
        return route_topk(
            tokens,
            router_weights,
            top_k=self.args.top_k,
            capacity=self.args.expert_peer_capacity,
            return_logits=return_logits,
        )

    def forward(self, args: ForwardArgs):
        if args.routing is None:
            if args.router_weights is None:
                raise ValueError("forward needs either routing or router_weights")
            routing = self.route(args.tokens, args.router_weights, return_logits=args.return_logits)
        else:
            routing = args.routing

        if not self.use_nvshmem or self.ep_world == 1:
            if self.use_packed:
                expert_in, output = self._local_workspace(args.tokens, routing, args.moe_out)
                return local_packed_forward(
                    args.tokens,
                    routing,
                    args.weights,
                    mlp_type=self.args.mlp_type,
                    act_type=self.args.act_type,
                    top_k=self.args.top_k,
                    swish_alpha=args.swish_alpha,
                    swish_beta=args.swish_beta,
                    output=output,
                    expert_in=expert_in,
                    use_cutedsl_gather=self.use_cutedsl_gather,
                    use_cutedsl_combine=self.use_cutedsl_combine,
                )
            return local_reference_forward(
                args.tokens,
                routing,
                args.weights,
                mlp_type=self.args.mlp_type,
                act_type=self.args.act_type,
                top_k=self.args.top_k,
                swish_alpha=args.swish_alpha,
                swish_beta=args.swish_beta,
                output=args.moe_out,
            )

        state = self._ensure_state()
        self._zero_state(state)
        backend.sync_all(args.stream)
        self._dispatch(args.tokens, routing, state, args.stream)
        if self.use_packed:
            self._compute_and_return_packed(args.weights, state, args.swish_alpha, args.swish_beta, args.stream)
        else:
            self._compute_and_return(args.weights, state, args.swish_alpha, args.swish_beta, args.stream)
        return self._combine(args.tokens, state, args.moe_out)

    def _zero_state(self, state: _SymmetricState) -> None:
        state.dispatch_tokens.zero_()
        state.dispatch_ids.fill_(-1)
        state.dispatch_probs.zero_()
        state.dispatch_counts.zero_()
        state.return_tokens.zero_()
        state.return_ids.fill_(-1)
        state.return_probs.zero_()
        state.return_counts.zero_()

    def _put_or_copy(self, dst, src, *, remote_ep_rank: int, stream: Any | None) -> None:
        if remote_ep_rank == self.ep_rank:
            dst.copy_(src)
            return
        import nvshmem.core as nvshmem

        nvshmem.put(dst, src.contiguous(), remote_pe=self.rank_map[remote_ep_rank], stream=backend.torch_stream(stream))

    def _dispatch(self, tokens, routing: RoutingResult, state: _SymmetricState, stream: Any | None) -> None:
        import torch

        cap = self.args.expert_peer_capacity
        for expert_idx, owner_rank in enumerate(self.expert_map):
            count = min(int(routing.expert_counts[expert_idx].item()), cap)
            local_idx = self.global_to_local[expert_idx]
            count_tensor = torch.tensor([count], device=tokens.device, dtype=torch.int32)

            dst_count = state.dispatch_counts[self.ep_rank, local_idx : local_idx + 1]
            self._put_or_copy(dst_count, count_tensor, remote_ep_rank=owner_rank, stream=stream)
            if count == 0:
                continue

            ids = routing.token_ids[expert_idx, :count].long().contiguous()
            probs = routing.token_probs[expert_idx, :count].float().contiguous()
            payload = tokens.index_select(0, ids).contiguous()
            self._put_or_copy(
                state.dispatch_tokens[self.ep_rank, local_idx, :count, :],
                payload,
                remote_ep_rank=owner_rank,
                stream=stream,
            )
            self._put_or_copy(
                state.dispatch_ids[self.ep_rank, local_idx, :count],
                ids,
                remote_ep_rank=owner_rank,
                stream=stream,
            )
            self._put_or_copy(
                state.dispatch_probs[self.ep_rank, local_idx, :count],
                probs,
                remote_ep_rank=owner_rank,
                stream=stream,
            )
        backend.sync_all(stream)

    def _compute_and_return(
        self,
        weights: ExpertWeights,
        state: _SymmetricState,
        swish_alpha: float,
        swish_beta: float,
        stream: Any | None,
    ) -> None:
        import torch

        cap = self.args.expert_peer_capacity
        device = state.dispatch_tokens.device
        for src_rank in range(self.ep_world):
            for local_idx in range(self.num_local_experts):
                count = min(int(state.dispatch_counts[src_rank, local_idx].item()), cap)
                count_tensor = torch.tensor([count], device=device, dtype=torch.int32)
                self._put_or_copy(
                    state.return_counts[self.ep_rank, local_idx : local_idx + 1],
                    count_tensor,
                    remote_ep_rank=src_rank,
                    stream=stream,
                )
                if count == 0:
                    continue

                routed = state.dispatch_tokens[src_rank, local_idx, :count, :]
                expert_out = expert_mlp(
                    routed,
                    up=weights.local_expert_up[local_idx],
                    bias_up=weights.local_bias_up[local_idx],
                    down=weights.local_expert_down[local_idx],
                    bias_down=weights.local_bias_down[local_idx],
                    mlp_type=self.args.mlp_type,
                    act_type=self.args.act_type,
                    up_v=None if weights.local_expert_up_v is None else weights.local_expert_up_v[local_idx],
                    bias_up_v=None if weights.local_bias_up_v is None else weights.local_bias_up_v[local_idx],
                    swish_alpha=swish_alpha,
                    swish_beta=swish_beta,
                ).contiguous()
                self._put_or_copy(
                    state.return_tokens[self.ep_rank, local_idx, :count, :],
                    expert_out,
                    remote_ep_rank=src_rank,
                    stream=stream,
                )
                self._put_or_copy(
                    state.return_ids[self.ep_rank, local_idx, :count],
                    state.dispatch_ids[src_rank, local_idx, :count].contiguous(),
                    remote_ep_rank=src_rank,
                    stream=stream,
                )
                self._put_or_copy(
                    state.return_probs[self.ep_rank, local_idx, :count],
                    state.dispatch_probs[src_rank, local_idx, :count].contiguous(),
                    remote_ep_rank=src_rank,
                    stream=stream,
                )
        backend.sync_all(stream)

    def _compute_and_return_packed(
        self,
        weights: ExpertWeights,
        state: _SymmetricState,
        swish_alpha: float,
        swish_beta: float,
        stream: Any | None,
    ) -> None:
        import torch

        cap = self.args.expert_peer_capacity
        world = self.ep_world
        nlocal = self.num_local_experts
        flat_in = state.dispatch_tokens.reshape(world * nlocal, cap, self.args.token_dim)

        def expand_weight(tensor):
            return tensor.unsqueeze(0).expand(world, *tensor.shape).reshape(world * nlocal, *tensor.shape[1:])

        packed_weights = ExpertWeights(
            local_expert_up=expand_weight(weights.local_expert_up),
            local_bias_up=expand_weight(weights.local_bias_up),
            local_expert_down=expand_weight(weights.local_expert_down),
            local_bias_down=expand_weight(weights.local_bias_down),
            local_expert_up_v=None if weights.local_expert_up_v is None else expand_weight(weights.local_expert_up_v),
            local_bias_up_v=None if weights.local_bias_up_v is None else expand_weight(weights.local_bias_up_v),
        )
        flat_out = expert_mlp_batched(
            flat_in,
            packed_weights,
            mlp_type=self.args.mlp_type,
            act_type=self.args.act_type,
            swish_alpha=swish_alpha,
            swish_beta=swish_beta,
        ).reshape(world, nlocal, cap, self.args.token_dim)

        for src_rank in range(world):
            counts = state.dispatch_counts[src_rank, :].contiguous()
            self._put_or_copy(
                state.return_counts[self.ep_rank, :],
                counts,
                remote_ep_rank=src_rank,
                stream=stream,
            )
            for local_idx in range(nlocal):
                count = min(int(counts[local_idx].item()), cap)
                if count == 0:
                    continue
                self._put_or_copy(
                    state.return_tokens[self.ep_rank, local_idx, :count, :],
                    flat_out[src_rank, local_idx, :count, :].contiguous(),
                    remote_ep_rank=src_rank,
                    stream=stream,
                )
                self._put_or_copy(
                    state.return_ids[self.ep_rank, local_idx, :count],
                    state.dispatch_ids[src_rank, local_idx, :count].contiguous(),
                    remote_ep_rank=src_rank,
                    stream=stream,
                )
                self._put_or_copy(
                    state.return_probs[self.ep_rank, local_idx, :count],
                    state.dispatch_probs[src_rank, local_idx, :count].contiguous(),
                    remote_ep_rank=src_rank,
                    stream=stream,
                )
        backend.sync_all(stream)

    def _combine(self, tokens, state: _SymmetricState, output):
        import torch

        result = output if output is not None else torch.empty_like(tokens)
        result.zero_()
        if self.use_cutedsl_combine and self.args.top_k == 1:
            try:
                from .kernels import combine_top1

                for owner_rank in range(self.ep_world):
                    for local_idx in range(self.num_local_experts):
                        count = min(
                            int(state.return_counts[owner_rank, local_idx].item()),
                            self.args.expert_peer_capacity,
                        )
                        if count:
                            combine_top1(
                                state.return_tokens[owner_rank, local_idx, :count, :],
                                state.return_ids[owner_rank, local_idx, :count],
                                result,
                                stream=self.args.stream,
                            )
                return result
            except Exception:
                if os.environ.get("FLASHMOE_CUTEDSL_STRICT", "0") == "1":
                    raise

        for owner_rank in range(self.ep_world):
            for local_idx in range(self.num_local_experts):
                count = min(int(state.return_counts[owner_rank, local_idx].item()), self.args.expert_peer_capacity)
                if count == 0:
                    continue
                ids = state.return_ids[owner_rank, local_idx, :count].long()
                values = state.return_tokens[owner_rank, local_idx, :count, :]
                if self.args.top_k == 1:
                    result.index_copy_(0, ids, values)
                else:
                    probs = state.return_probs[owner_rank, local_idx, :count].to(values.dtype).unsqueeze(-1)
                    result.index_add_(0, ids, values * probs)
        return result

    def finalize(self) -> None:
        if self._state is not None and self.use_nvshmem:
            import nvshmem.core as nvshmem

            backend.sync_all(self.args.stream)
            for tensor in (
                self._state.dispatch_tokens,
                self._state.dispatch_ids,
                self._state.dispatch_probs,
                self._state.dispatch_counts,
                self._state.return_tokens,
                self._state.return_ids,
                self._state.return_probs,
                self._state.return_counts,
            ):
                nvshmem.free_tensor(tensor)
            self._state = None
            backend.sync_all(self.args.stream)


def initialize(
    args: InitArgs,
    *,
    use_nvshmem: bool = True,
    use_cutedsl_combine: bool | None = None,
    use_packed: bool = True,
    use_cutedsl_gather: bool | None = None,
) -> FlashDMoEPort:
    return FlashDMoEPort(
        args,
        use_nvshmem=use_nvshmem,
        use_cutedsl_combine=use_cutedsl_combine,
        use_packed=use_packed,
        use_cutedsl_gather=use_cutedsl_gather,
    )


def forward(handle: FlashDMoEPort, args: ForwardArgs):
    return handle.forward(args)
