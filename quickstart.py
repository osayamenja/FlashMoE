import cuda.core as cuda
import flashmoe

if __name__ == "__main__":
    tokens_per_rank = 16
    token_dim = 32
    ffn_size = 32
    num_experts = 16
    k = 2

    # setup device ordinals
    device_id = flashmoe.get_local_rank()
    dev = cuda.Device(device_id)
    dev.set_current()
    stream = dev.create_stream()
    stream_ptr = int(stream.handle)
    arch = dev.arch * 10

    print("Arch is", arch)

    # call initialize
    handle = flashmoe.initialize(
        flashmoe.InitArgs(data_type=flashmoe.DataType.BF16,
                            mlp_type=flashmoe.MLPType.GATED,
                            act_type=flashmoe.ActivationType.SILU,
                            tokens_per_rank=tokens_per_rank,
                            token_dim=token_dim,
                            ffn_size=ffn_size,
                            num_experts=num_experts,
                            top_k=k,
                            gpu_arch=arch,
                            stream_ptr=stream_ptr)
    )
    # construct kernel arguments
    args = flashmoe.OpArgs(mt=flashmoe.MLPType.GATED)
    # call forward
    flashmoe.forward(handle, args, stream_ptr)
    # print output tensor or compare with torch
    # call finalize
    flashmoe.finalize(handle, stream_ptr)
    stream.sync() # <- ensures the stream stays alive till work completion.
    stream.close()