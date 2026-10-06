import random

import cuda.core as cuda
import argparse
import torch

import flashmoe

def get_shared_seed(rank_: int, device_id: int, use_torch: bool) -> int:
    torch_device_ = f"cuda:{device_id}"
    shared_seed = 0
    if rank_ == 0:
        shared_seed = random.randint(1, 2**31 - 1)
    if use_torch:
        import torch.distributed as dist
        seed_tensor = torch.tensor([shared_seed], dtype=torch.int64, device=torch_device_)
        dist.broadcast(seed_tensor, src=0)
        return int(seed_tensor.item())
    else:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        shared_seed = comm.bcast(shared_seed, root=0)
        return shared_seed

def run_fused_moe_forward_w_correctness_check(tokens_per_rank: int,
                                              token_dim: int,
                                              ffn_size: int,
                                              num_experts: int,
                                              k: int,
                                              device_id: int,
                                              use_torch_init: bool=False,
                                              fp8: bool=False) -> None:
    if use_torch_init:
        import torch.distributed as dist, os
        world_size = int(os.environ.get("WORLD_SIZE"))
        assert os.environ.get("LOCAL_RANK") is not None, "need to launch with torchrun if set with torch_init=True"
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            rank=int(os.environ['RANK']),
            world_size=world_size,
            device_id=device
        )
    # setup device ordinals
    dev = cuda.Device(device_id)
    dev.set_current()
    stream = dev.create_stream()
    stream_ptr = int(stream.handle)
    arch = int(dev.arch) * 10

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
                                  device_id=device_id,
                                  quant=flashmoe.QuantType.MXFP8 if fp8 else flashmoe.QuantType.NONE)
    # call initialize
    flash_handle = flashmoe.initialize(init_args)
    router_handle = flashmoe.router.initialize(init_args)
    # the reference runs in data_type with the unquantized weights
    ref_handle = flashmoe.reference.initialize(init_args)

    rank = flashmoe.cb.get_rank()
    seed = get_shared_seed(rank, device_id, use_torch_init)
    if rank == 0:
        print("S={},H={},I={},E={},k={},world={}\n"
              "Rank, error(%)".format(tokens_per_rank,token_dim, ffn_size,
                                      num_experts, k, flashmoe.cb.get_world_size()))
    flashmoe.cb.sync_all(stream_ptr)

    torch_device = f"cuda:{device_id}"
    # construct forward arguments for MoE with Gated MLP
    tokens = torch.empty((tokens_per_rank, token_dim), device=torch_device, dtype=t_dtype).uniform_(-1.0, 1.0).contiguous()
    expert_counts = torch.zeros(num_experts, device=torch_device, dtype=torch.int32).contiguous()
    router_weights = torch.empty((token_dim, num_experts), device=torch_device, dtype=t_dtype).uniform_(-1.0,1.0).contiguous()
    torch.manual_seed(seed)
    nlx = init_args.num_experts
    chunk_size = nlx // flashmoe.cb.get_world_size()

    expert_up = torch.empty((nlx, ffn_size, token_dim), device=torch_device, dtype=t_dtype).uniform_(-1.0, 1.0).contiguous()
    expert_up_v = torch.empty((nlx, ffn_size, token_dim), device=torch_device, dtype=t_dtype).uniform_(-1.0, 1.0).contiguous()
    bias_up = torch.empty((nlx, ffn_size), device=torch_device, dtype=t_dtype).uniform_(-1.0, 1.0).contiguous()
    bias_up_v = torch.empty((nlx, ffn_size), device=torch_device, dtype=t_dtype).uniform_(-1.0, 1.0).contiguous()
    expert_down = torch.empty((nlx, token_dim, ffn_size), device=torch_device, dtype=t_dtype).uniform_(-1.0, 1.0).contiguous()
    bias_down = torch.empty((nlx, token_dim), device=torch_device, dtype=t_dtype).uniform_(-1.0, 1.0).contiguous()
    moe_out = torch.empty((tokens_per_rank, token_dim), device=torch_device, dtype=t_dtype).contiguous()

    local_expert_up = expert_up[
        rank * chunk_size: (rank + 1) * chunk_size
    ].contiguous()
    local_expert_up_v = expert_up_v[
        rank * chunk_size: (rank + 1) * chunk_size
    ].contiguous()
    local_bias_up = bias_up[
        rank * chunk_size: (rank + 1) * chunk_size
    ].contiguous()
    local_bias_up_v = bias_up_v[
        rank * chunk_size: (rank + 1) * chunk_size
    ].contiguous()
    local_expert_down = expert_down[
        rank * chunk_size: (rank + 1) * chunk_size
    ].contiguous()
    local_bias_down = bias_down[
        rank * chunk_size: (rank + 1) * chunk_size
    ].contiguous()
    sf = {}
    if fp8:
        # MXFP8: e4m3 expert weights and their block scales
        local_expert_up, sf["local_expert_up_sf"] = flashmoe.quant.quantize_mxfp8(local_expert_up)
        local_expert_up_v, sf["local_expert_up_v_sf"] = flashmoe.quant.quantize_mxfp8(local_expert_up_v)
        local_expert_down, sf["local_expert_down_sf"] = flashmoe.quant.quantize_mxfp8(local_expert_down)
        scales = list(sf.values())
        sf = {name: t.data_ptr() for name, t in sf.items()}
    args = flashmoe.ForwardArgs(
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
        stream_ptr=stream_ptr,
        **sf
    )
    rfa = flashmoe.router.RouterForwardArgs(tokens=tokens.data_ptr(),
                                            weights=router_weights.data_ptr(),
                                            expert_counts=expert_counts.data_ptr(),
                                            stream_ptr=stream_ptr)

    ref_input = torch.empty((tokens_per_rank, token_dim), device=torch_device, dtype=t_dtype).contiguous()
    ref_interim0 = torch.empty((tokens_per_rank, ffn_size), device=torch_device, dtype=t_dtype).contiguous()
    ref_interim1 = torch.empty((tokens_per_rank, token_dim), device=torch_device, dtype=t_dtype).contiguous()
    ref_out = torch.zeros((tokens_per_rank, token_dim), device=torch_device, dtype=t_dtype).contiguous()
    rea = flashmoe.reference.RefForwardArgs(
        expert_up=expert_up.data_ptr(),
        expert_down=expert_down.data_ptr(),
        bias_up=bias_up.data_ptr(),
        bias_down=bias_down.data_ptr(),
        ref_input=ref_input.data_ptr(),
        ref_interim0=ref_interim0.data_ptr(),
        ref_interim1=ref_interim1.data_ptr(),
        ref_out=ref_out.data_ptr(),
        expert_up_v=expert_up_v.data_ptr(),
        bias_up_v=bias_up_v.data_ptr()
    )
    dev.sync() # <- ensures all torch ops are done before we start
    # call forward of fused router
    flashmoe.router.forward(router_handle, flash_handle, rfa)
    # call forward of FlashMoE
    flashmoe.forward(flash_handle, args)
    # call reference
    flashmoe.reference.forward(ref_handle, flash_handle.mod.get_tIdx(flash_handle.context), args, rea)
    stream.sync()
    # compare fused kernel and reference
    match_count = (torch.isclose(moe_out, ref_out, rtol=8e-2, atol=8e-3)).sum().item()
    error_p = 1.0 - (match_count / (tokens_per_rank * token_dim))
    if fp8:
        # against the unquantized reference, the MXFP8 quantization error dominates: report it instead
        rel_frob = (moe_out.float() - ref_out.float()).norm() / ref_out.float().norm()
        print("{}, {:.4f}%, relative Frobenius error vs {}: {:.2f}%".format(rank, error_p, data_type.name,
                                                                         100 * rel_frob.item()))
    else:
        print("{}, {:.4f}%".format(rank, error_p))
    # call finalize
    flashmoe.finalize(flash_handle, stream_ptr)
    flashmoe.router.finalize(router_handle, stream_ptr)
    stream.close()
    if use_torch_init:
        import torch.distributed as dist
        dist.destroy_process_group()

def run_fused_moe_forward(tokens_per_rank: int,
                          token_dim: int,
                          ffn_size: int,
                          num_experts: int,
                          k: int,
                          device_id: int,
                          use_torch_init: bool=False) -> None:
    if use_torch_init:
        import torch.distributed as dist, os
        world_size = int(os.environ.get("WORLD_SIZE"))
        assert os.environ.get("LOCAL_RANK") is not None, "need to launch with torchrun if set with torch_init=True"
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            rank=int(os.environ['RANK']),
            world_size=world_size,
            device_id=device
        )
    # setup device ordinals
    dev = cuda.Device(device_id)
    dev.set_current()
    stream = dev.create_stream()
    stream_ptr = int(stream.handle)
    arch = int(dev.arch) * 10

    torch_device = f"cuda:{device_id}"
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
    flash_handle = flashmoe.initialize(init_args)
    router_handle = flashmoe.router.initialize(init_args)

    rank = flashmoe.cb.get_rank()
    seed = get_shared_seed(rank, device_id, use_torch_init)
    if rank == 0:
        print("S={},H={},I={},E={},k={},world={}\n"
              "Rank, FlashMoE_time(ms)".format(tokens_per_rank, token_dim, ffn_size,
                                      num_experts, k, flashmoe.cb.get_world_size()))
    flashmoe.cb.sync_all(stream_ptr)

    # construct forward arguments for MoE with Gated MLP
    tokens = torch.empty((tokens_per_rank, token_dim), device=torch_device, dtype=t_dtype).uniform_(-1.0, 1.0).contiguous()
    expert_counts = torch.zeros(num_experts, device=torch_device, dtype=torch.int32).contiguous()
    router_weights = torch.empty((token_dim, num_experts), device=torch_device, dtype=t_dtype).uniform_(-1.0,1.0).contiguous()
    torch.manual_seed(seed)
    nlx = init_args.num_local_experts

    local_expert_up = torch.empty((nlx, ffn_size, token_dim), device=torch_device, dtype=t_dtype).uniform_(-1.0,1.0).contiguous()
    local_expert_up_v = torch.empty((nlx, ffn_size, token_dim), device=torch_device, dtype=t_dtype).uniform_(-1.0,1.0).contiguous()
    local_bias_up = torch.empty((nlx, ffn_size), device=torch_device, dtype=t_dtype).uniform_(-1.0, 1.0).contiguous()
    local_bias_up_v = torch.empty((nlx, ffn_size), device=torch_device, dtype=t_dtype).uniform_(-1.0, 1.0).contiguous()
    local_expert_down = torch.empty((nlx, token_dim, ffn_size), device=torch_device, dtype=t_dtype).uniform_(-1.0,1.0).contiguous()
    local_bias_down = torch.empty((nlx, token_dim), device=torch_device, dtype=t_dtype).uniform_(-1.0, 1.0).contiguous()
    moe_out = torch.empty((tokens_per_rank, token_dim), device=torch_device, dtype=t_dtype).contiguous()

    args = flashmoe.ForwardArgs(
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
    rfa = flashmoe.router.RouterForwardArgs(tokens=tokens.data_ptr(),
                                            weights=router_weights.data_ptr(),
                                            expert_counts=expert_counts.data_ptr(),
                                            stream_ptr=stream_ptr)

    dev.sync()  # <- ensures all torch ops are done before we start
    # call forward of fused router
    flashmoe.router.forward(router_handle, flash_handle, rfa)
    # call forward of FlashMoE
    flashmoe.forward(flash_handle, args)

    # benchmark with cuda graph
    capture_stream = torch.cuda.ExternalStream(stream_ptr, device=torch_device)
    g = torch.cuda.CUDAGraph()

    iters = 128
    graph_launches = 4

    with torch.cuda.graph(g, stream=capture_stream):
        for _ in range(iters):
            flashmoe.forward(flash_handle, args)

    # Warmup once
    with torch.cuda.stream(capture_stream):
        g.replay()

    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(capture_stream):
        start.record()
        for _ in range(graph_launches):
            g.replay()
        end.record()
    end.synchronize()
    total_ms = start.elapsed_time(end)

    kernel_time = total_ms / (iters * graph_launches)
    print("{}, {:.5f}".format(rank, kernel_time))
    # call finalize
    flashmoe.finalize(flash_handle, stream_ptr)
    flashmoe.router.finalize(router_handle, stream_ptr)
    stream.close()
    if use_torch_init:
        import torch.distributed as dist
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-init", action="store_true")
    parser.add_argument("--fp8", action="store_true", help="MXFP8 expert weights and activations (sm_120/sm_121)")
    args = parser.parse_args()

    # LLama4-Scout-17B-16E shapes
    tokens_per_rank_ = 1024
    token_dim_ = 5120
    ffn_size_ = 8192
    num_experts_ = 16
    k_ = 1
    device_id_ = flashmoe.get_local_rank()
    # call kernel
    run_fused_moe_forward_w_correctness_check(tokens_per_rank_, token_dim_, ffn_size_, num_experts_, k_, device_id_,
                                              args.torch_init, args.fp8)

if __name__ == "__main__":
    main()