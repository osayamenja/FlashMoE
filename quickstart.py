import cuda.core as cuda

import torch

import flashmoe
from flashmoe.router import RouterForwardArgs


def run_fused_moe_forward(tokens_per_rank: int, token_dim: int, ffn_size: int,
                          num_experts: int, k: int, device_id: int, use_torch_init: bool=False) -> None:
    if use_torch_init:
        import torch.distributed as dist, os
        world_size = torch.cuda.device_count()
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend="cpu:gloo",
            rank=local_rank,
            world_size=world_size,
            device_id=device
        )
    # setup device ordinals
    dev = cuda.Device(device_id)
    dev.set_current()
    stream = dev.create_stream()
    stream_ptr = int(stream.handle)
    arch = dev.arch * 10

    print("Arch is", arch)

    mlp_type = flashmoe.MLPType.GATED
    data_type = flashmoe.DataType.BF16
    t_dtype = torch.bfloat16 if data_type == flashmoe.DataType.BF16 else torch.float16
    act_type = flashmoe.ActivationType.SILU
    init_args = flashmoe.InitArgs(data_type=data_type,
                                  mlp_type=mlp_type,
                                  act_type=act_type,
                                  tokens_per_rank=tokens_per_rank,
                                  token_dim=token_dim,
                                  ffn_size=ffn_size,
                                  num_experts=num_experts,
                                  top_k=k,
                                  gpu_arch=arch,
                                  stream_ptr=stream_ptr,
                                  device_id=device_id)
    # call initialize
    router_handle = flashmoe.router.initialize(init_args)
    flash_handle = flashmoe.initialize(init_args)
    # construct forward arguments
    tokens = (torch.empty((tokens_per_rank, token_dim), device=device_id, dtype=t_dtype)
              .uniform_(-1.0, 1.0).contiguous())
    expert_counts = torch.zeros(num_experts, device=device_id, dtype=torch.int32).contiguous()
    local_expert_up = torch.empty((init_args.num_local_experts, ffn_size, token_dim),
                                  device=device_id, dtype=t_dtype).uniform_(-1.0, 1.0).contiguous()
    local_expert_up_v = torch.empty((init_args.num_local_experts, ffn_size, token_dim),
                                    device=device_id, dtype=t_dtype).uniform_(-1.0, 1.0).contiguous()
    local_bias_up = torch.empty((init_args.num_local_experts, ffn_size), device=device_id, dtype=t_dtype).contiguous()
    local_bias_up_v = torch.empty((init_args.num_local_experts, ffn_size), device=device_id, dtype=t_dtype).contiguous()
    local_expert_down = torch.empty((init_args.num_local_experts, token_dim, ffn_size),
                                    device=device_id, dtype=t_dtype).uniform_(-1.0, 1.0).contiguous()
    local_bias_down = torch.empty((init_args.num_local_experts, token_dim), device=device_id,
                                  dtype=t_dtype).contiguous()
    moe_out = torch.empty((tokens_per_rank, token_dim), device=device_id, dtype=t_dtype)
    args = flashmoe.MoEForwardArgs(
        mt=flashmoe.MLPType.GATED,
        tokens=tokens.data_ptr(),
        expert_counts=expert_counts.data_ptr(),
        local_expert_up=local_expert_up.data_ptr(),
        local_expert_up_v=local_expert_up_v.data_ptr(),
        local_bias_up=local_bias_up.data_ptr(),
        local_bias_up_v=local_bias_up_v.data_ptr(),
        local_expert_down=local_expert_down.data_ptr(),
        local_bias_down=local_bias_down.data_ptr(),
        moe_out=moe_out.data_ptr(),
        stream_ptr=stream_ptr
    )
    # call router forward
    router_weights = (torch.empty((token_dim, num_experts), device=device_id, dtype=t_dtype)
              .uniform_(-1.0, 1.0).contiguous())
    rfa = RouterForwardArgs(tokens=tokens.data_ptr(), weights=router_weights.data_ptr(),
                            expert_counts=expert_counts.data_ptr(), stream_ptr=stream_ptr)
    #flashmoe.router.forward(router_handle, flash_handle, rfa)
    # call forward of FlashMoE
    # flashmoe.forward(flash_handle, args)
    stream.sync()  # <- ensures the stream is not prematurely garbage collected

    # call finalize
    flashmoe.finalize(flash_handle, stream_ptr)
    flashmoe.router.finalize(router_handle, stream_ptr)
    stream.close()

if __name__ == "__main__":
    # LLama4-Scout-17B-16E shapes
    tokens_per_rank_ = 256
    token_dim_ = 5120
    ffn_size_ = 8192
    num_experts_ = 16
    k_ = 1
    device_id_ = flashmoe.get_local_rank()
    # call kernel
    run_fused_moe_forward(tokens_per_rank_, token_dim_, ffn_size_, num_experts_, k_, device_id_)