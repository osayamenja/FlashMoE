from .jit import ContextHandle, InitArgs, _get_compiled

class RouterInitArgs:
    device_id: int
    stream_ptr: int

    def __init__(self, device_id, stream_ptr):
        self.device_id = device_id
        self.stream_ptr = stream_ptr

def initialize(arg: InitArgs, return_logits: bool = False):
    from .bindings import gate_bindings
    mod_prefix = "flashmoe_router"
    mod_name = (f"{mod_prefix}_s{arg.tokens_per_rank}_h{arg.token_dim}"
            f"_e{arg.num_experts}_ec{arg.expert_peer_capacity}_k{arg.top_k}"
            f"_rl{int(return_logits)}_arch{arg.gpu_arch}")
    src = gate_bindings.substitute(
        s=arg.tokens_per_rank,
        h=arg.token_dim,
        e=arg.num_experts,
        top_k=arg.top_k,
        ec=arg.expert_peer_capacity,
        arch=arg.gpu_arch,
        return_logits=int(return_logits),
        dt=arg.data_type,
        mod_name=mod_name
    )
    mod = _get_compiled(arg, src, mod_prefix, mod_name)
    ctx = mod.initialize(device_id=arg.device_id, stream_ptr=arg.stream_ptr)
    return ContextHandle(mod, ctx)

class RouterForwardArgs:
    tokens: int
    weights: int
    routing: int
    expert_counts: int
    token_indices: int
    stream_ptr: int

    def __init__(self, tokens: int, weights: int, expert_counts: int, stream_ptr: int, *, routing: int = 0):
        self.tokens = tokens
        self.weights = weights
        self.routing = routing
        self.expert_counts = expert_counts
        self.stream_ptr = stream_ptr
def forward(router_handle: ContextHandle, flash_handle: ContextHandle, arg: RouterForwardArgs) -> None:
    router_handle.mod.forward(router_handle.context,
                              tokens=arg.tokens,
                              weights=arg.weights,
                              routing=arg.routing,
                              expert_counts=arg.expert_counts,
                              stream_ptr=arg.stream_ptr,
                              token_indices=flash_handle.mod.get_tIdx(flash_handle.context))
def finalize(handle: ContextHandle, stream_ptr: int) -> None:
    handle.mod.finalize(handle.context, stream_ptr)