from . import gate
from . import util
from .jit import InitArgs, ContextHandle, Topology, MLPType, ActivationType, DataType
__all__ = ["initialize", "forward", "finalize", "gate", "util"]

def initialize(arg: InitArgs) -> ContextHandle:
    from .jit import _get_compiled, _module_name
    from .bindings import flashmoe_bindings
    assert arg.ep_rank is None or ((arg.rank_map is not None)
        and (arg.ep_world is not None) and (arg.expert_map is not None)
        and (arg.num_local_experts is not None)
        and (arg.my_pe is not None)), "if rank is set, then so should all dependent metadata"
    if arg.ep_rank is None:
        from . import cb
        cb.initialize()
        arg.ep_rank = cb.get_rank()
        arg.my_pe = cb.get_rank()
        # assume wide-EP
        arg.ep_world = cb.get_world_size()
        assert arg.num_experts % arg.ep_world == 0
        arg.num_local_experts = arg.num_experts // arg.ep_world
        arg.expert_map = []
        # blocked partitioning
        # for 8 experts and 4 ranks
        # ep_rank 0 gets [E0, E1], ep_rank 1 gets [E2, E3] and so on
        for i in range(arg.num_experts):
            arg.expert_map.append(i // arg.num_local_experts)
        arg.rank_map = []
        for i in range(arg.ep_world):
            arg.rank_map.append(i)

    mod_prefix = "flashmoe_moe"
    mod_name = _module_name(arg, mod_prefix)
    src = flashmoe_bindings.substitute(
        arch=arg.gpu_arch,
        s=arg.tokens_per_rank,
        h=arg.token_dim,
        i=arg.ffn_size,
        tk=arg.top_k,
        mod_name=mod_name,
        topo=arg.topo,
        mt=arg.mlp_type,
        act=arg.act_type,
        dt=arg.data_type
    )
    mod = _get_compiled(arg, src, mod_prefix, mod_name)
    ctx = mod.initialize(arg.expert_peer_capacity, arg.expert_map, arg.rank_map, arg.stream_ptr)
    return ContextHandle(mod, ctx)

def forward(handle: ContextHandle, stream_ptr: int) -> None:
    handle.mod.forward(handle.context, stream_ptr)

def finalize(handle: ContextHandle, stream_ptr: int) -> None:
    handle.mod.finalize(handle.context, stream_ptr)