from . import gate
from . import util
from .jit import InitArgs
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

def initialize(arg: InitArgs) -> None:
    from .jit import _get_compiled
    import nvshmem.core as nvshmem
    assert nvshmem.init_status() == nvshmem.InitStatus.STATUS_IS_INITIALIZED, "NVSHMEM is not initialized"
    mod = _get_compiled(arg.tokens_per_rank, arg.token_dim, arg.ffn_size, arg.gpu_arch)
    mod.initialize(arg.expert_peer_capacity, arg.expert_map, arg.rank_map, arg.stream_ptr)
    # return ContextHandle(mod, ctx)

def forward(context: ContextHandle, top_k: int, stream_ptr: int = 0) -> None:
    context.mod.forward(context.capsule, top_k, stream_ptr)

def finalize(context: ContextHandle) -> None:
    pass