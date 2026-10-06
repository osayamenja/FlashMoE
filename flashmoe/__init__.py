from . import router
from .jit import InitArgs, ContextHandle, Topology, MLPType, ActivationType, DataType, ForwardArgs, QuantType
from . import quant
from .cb import get_local_rank
from . import reference

SHOULD_FINALIZE_NVSHMEM = False
def initialize(arg: InitArgs) -> ContextHandle:
    from .jit import _get_compiled
    from .bindings import flashmoe_bindings
    from . import cb
    import nvshmem.core as nvshmem
    import cuda.core as cuda
    assert arg.ep_rank is None or ((arg.rank_map is not None)
        and (arg.ep_world is not None) and (arg.expert_map is not None)
        and (arg.num_local_experts is not None)
        and (arg.my_pe is not None)), "if rank is set, then so should all dependent metadata"
    if arg.ep_rank is None:
        global SHOULD_FINALIZE_NVSHMEM
        SHOULD_FINALIZE_NVSHMEM = True
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
    nvshmem.sync_all(stream=cuda.Stream.from_handle(arg.stream_ptr)) # <- needed to eagerly initialize state before detecting topology
    def detect_topo():
        assert nvshmem.init_status() == nvshmem.InitStatus.STATUS_IS_INITIALIZED
        if nvshmem.team_n_pes(nvshmem.Teams.TEAM_SHARED) == nvshmem.n_pes():
            return Topology.NVLINK_ONLY
        else:
            return Topology.MIXED
    arg.topo = detect_topo()

    mod_prefix = "flashmoe_moe"
    mod_name = (f"{mod_prefix}_s{arg.tokens_per_rank}_h{arg.token_dim}_i{arg.ffn_size}"
            f"_e{arg.num_experts}_ec{arg.expert_peer_capacity}_k{arg.top_k}"
            f"_topo{arg.topo}_mt{arg.mlp_type}_dt{arg.data_type}_act{arg.act_type}_arch{arg.gpu_arch}"
            f"_q{arg.quant}")

    src = flashmoe_bindings.substitute(
        arch=arg.gpu_arch,
        s=arg.tokens_per_rank,
        h=arg.token_dim,
        i=arg.ffn_size,
        e=arg.num_experts,
        ec=arg.expert_peer_capacity,
        tk=arg.top_k,
        mod_name=mod_name,
        topo=arg.topo,
        mt=arg.mlp_type,
        act=arg.act_type,
        dt=arg.data_type,
        quant=int(arg.quant)
    )
    mod = _get_compiled(arg, src, mod_prefix, mod_name)
    ctx = mod.initialize(
        num_experts=arg.num_experts,
        expert_peer_capacity=arg.expert_peer_capacity,
        ep_world=arg.ep_world,
        my_pe=arg.my_pe,
        ep_rank=arg.ep_rank,
        local_rank=arg.device_id,
        num_local_experts=arg.num_local_experts,
        expert_map=arg.expert_map,
        rank_map=arg.rank_map,
        stream_ptr=arg.stream_ptr
    )
    return ContextHandle(mod, ctx)

def forward(handle: ContextHandle, args: ForwardArgs) -> None:
    handle.mod.forward(handle.context,
                       tokens=args.tokens,
                       expert_counts=args.expert_counts,
                       local_expert_up=args.local_expert_up,
                       local_expert_up_v=args.local_expert_up_v,
                       local_bias_up=args.local_bias_up,
                       local_bias_up_v=args.local_bias_up_v,
                       local_expert_down=args.local_expert_down,
                       local_bias_down=args.local_bias_down,
                       moe_out=args.moe_out,
                       swish_alpha=args.swish_alpha,
                       swish_beta=args.swish_beta,
                       stream_ptr=args.stream_ptr,
                       local_expert_up_sf=args.local_expert_up_sf,
                       local_expert_up_v_sf=args.local_expert_up_v_sf,
                       local_expert_down_sf=args.local_expert_down_sf)

def finalize(handle: ContextHandle, stream_ptr: int) -> None:
    handle.mod.finalize(handle.context, stream_ptr)
    global SHOULD_FINALIZE_NVSHMEM
    if SHOULD_FINALIZE_NVSHMEM:
        SHOULD_FINALIZE_NVSHMEM = False
        import cuda.core as cuda
        import nvshmem.core as nvshmem
        dev = cuda.Device(get_local_rank())
        dev.sync()
        if nvshmem.init_status() == nvshmem.InitStatus.STATUS_IS_INITIALIZED:
            nvshmem.finalize()