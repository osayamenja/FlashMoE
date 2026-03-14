from . import router
from .jit import InitArgs, ContextHandle, Topology, MLPType, ActivationType, DataType
from .cb import get_local_rank, IS_INITIALIZED

def initialize(arg: InitArgs) -> ContextHandle:
    from .jit import _get_compiled
    from .bindings import flashmoe_bindings
    from . import cb
    import nvshmem.core as nvshmem
    import cuda.core.experimental as cuda
    assert arg.ep_rank is None or ((arg.rank_map is not None)
        and (arg.ep_world is not None) and (arg.expert_map is not None)
        and (arg.num_local_experts is not None)
        and (arg.my_pe is not None)), "if rank is set, then so should all dependent metadata"
    if arg.ep_rank is None:
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
            f"_topo{arg.topo}_mt{arg.mlp_type}_dt{arg.data_type}_act{arg.act_type}_arch{arg.gpu_arch}")

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
        dt=arg.data_type
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

class MoEForwardArgs:
    # integers below are for addresses to the underlying contiguous memory of the corresponding tensor
    tokens : int # [S, H]
    expert_counts: int # [E]
    local_expert_up: int # [num_local_experts, H, I]
    local_expert_up_v: int # [num_local_experts, H, I]
    local_bias_up: int # [num_local_experts, I]
    local_bias_up_v: int # [num_local_experts, I]
    local_expert_down: int # [num_local_experts, I, H]
    local_bias_down: int # [num_local_experts, H]
    moe_out: int # [S, H]
    swish_alpha: float = 1.0
    swish_beta: float = 1.0
    stream_ptr: int
    def __init__(self,
                 mt: MLPType,
                 tokens: int,
                 expert_counts: int,
                 local_expert_up: int,
                 local_bias_up: int,
                 local_expert_down: int,
                 local_bias_down: int,
                 moe_out: int,
                 stream_ptr: int,
                 *,
                 swish_alpha: float = 1.0,
                 swish_beta: float = 1.0,
                 local_expert_up_v: int = None,
                 local_bias_up_v: int = None):
        # if GatedMLP, then mlp weights should be specified
        assert not mt == MLPType.GATED or local_expert_up_v is not None
        assert not mt == MLPType.GATED or local_bias_up_v is not None
        self.tokens = tokens
        self.expert_counts = expert_counts
        self.local_expert_up = local_expert_up
        self.local_expert_up_v = local_expert_up_v
        self.local_expert_down = local_expert_down
        self.local_bias_up = local_bias_up
        self.local_bias_up_v = local_bias_up_v
        self.local_bias_down = local_bias_down
        self.moe_out = moe_out
        self.stream_ptr = stream_ptr
        self.swish_alpha = swish_alpha
        self.swish_beta = swish_beta

def forward(handle: ContextHandle, args: MoEForwardArgs) -> None:
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
                       stream_ptr=args.stream_ptr)

def finalize(handle: ContextHandle, stream_ptr: int) -> None:
    handle.mod.finalize(handle.context, stream_ptr)