from math import ceil

from cuda.bindings import runtime as cudart
import flashmoe

if __name__ == "__main__":
    tokens_per_rank = 16
    token_dim = 32
    ffn_size = 32
    num_experts = 16
    k = 2
    ec = ceil(float(tokens_per_rank) / num_experts) * k
    expert_map = []
    rank_map = []
    # initialize NVSHMEM

    world = 1
    my_pe = 0
    ep_rank = 0
    ep_world = world
    device_id = 0
    num_local_experts = num_experts
    err, = cudart.cudaSetDevice(device_id)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(err)

    err, stream = cudart.cudaStreamCreate()
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(err)

    arch = flashmoe.util.get_arch(device_id)
    print("Arch is", arch)

    arg = flashmoe.InitArgs(tokens_per_rank=tokens_per_rank,
                            token_dim=token_dim,
                            ffn_size=ffn_size,
                            num_experts=num_experts,
                            top_k=k,
                            expert_map=expert_map, # which epRank holds which experts
                            rank_map=rank_map, # map from epRank to universal rank
                            gpu_arch=arch,
                            ep_world=world,
                            ep_rank=ep_rank,
                            my_pe=my_pe,
                            num_local_experts=num_local_experts,
                            stream_ptr=int(stream))
    flashmoe.initialize(arg)
    err, = cudart.cudaStreamDestroy(stream)
    assert err == cudart.cudaError_t.cudaSuccess