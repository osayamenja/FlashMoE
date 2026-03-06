from . import gate
from . import util

__all__ = ["initialize", "forward", "finalize", "gate", "util"]

class ContextHandle:
    __slots__ = ("_mod", "_ctx")

    def __init__(self, mod, ctx):
        self._mod = mod
        self._ctx = ctx

    @property
    def capsule(self):
        return self._ctx

    @property
    def mod(self):
        return self._mod

class InitializeArgs:
    from typing import Sequence
    tokens_per_rank: int
    token_dim: int
    ffn_size: int
    num_experts: int
    top_k: int
    expert_map: Sequence[int]
    rank_map: Sequence[int]
    gpu_arch: int
    stream_ptr: int
    expert_peer_capacity: int

    def __init__(self, tokens_per_rank: int, token_dim: int, ffn_size: int, num_experts: int,
                 top_k: int, expert_map: Sequence[int], rank_map: Sequence[int], gpu_arch: int,
                 stream_ptr: int, expert_peer_capacity: int = -1) -> None:
        from math import ceil
        assert gpu_arch >= 700
        self.tokens_per_rank = tokens_per_rank
        self.token_dim = token_dim
        self.ffn_size = ffn_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_map = expert_map
        self.rank_map = rank_map
        self.gpu_arch = gpu_arch
        self.stream_ptr = stream_ptr
        if expert_peer_capacity < 0:
            self.expert_peer_capacity = ceil(float(tokens_per_rank) / num_experts) * top_k
        else:
            self.expert_peer_capacity = expert_peer_capacity

def initialize(arg: InitializeArgs) -> None:
    from .jit import _get_compiled
    mod = _get_compiled(arg.tokens_per_rank, arg.token_dim, arg.ffn_size, arg.gpu_arch)
    mod.initialize(arg.expert_peer_capacity, arg.expert_map, arg.rank_map, arg.stream_ptr)
    # return ContextHandle(mod, ctx)

def forward(context: ContextHandle, top_k: int, stream_ptr: int = 0) -> None:
    context.mod.forward(context.capsule, top_k, stream_ptr)

def finalize(context: ContextHandle) -> None:
    pass