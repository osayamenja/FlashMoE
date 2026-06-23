import argparse

import torch

import flashmoe_cutedsl as flashdmoe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--ffn", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no-nvshmem", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    dtype = torch.bfloat16

    init_args = flashdmoe.InitArgs(
        data_type=flashdmoe.DataType.BF16,
        tokens_per_rank=args.tokens,
        token_dim=args.hidden,
        ffn_size=args.ffn,
        num_experts=args.experts,
        top_k=args.top_k,
        mlp_type=flashdmoe.MLPType.GATED,
        act_type=flashdmoe.ActivationType.SILU,
        device_id=args.device,
    )
    moe = flashdmoe.initialize(init_args, use_nvshmem=not args.no_nvshmem)
    n_local = moe.num_local_experts

    tokens = torch.randn(args.tokens, args.hidden, device=device, dtype=dtype)
    router = torch.randn(args.hidden, args.experts, device=device, dtype=dtype)
    weights = flashdmoe.ExpertWeights(
        local_expert_up=torch.randn(n_local, args.ffn, args.hidden, device=device, dtype=dtype),
        local_expert_up_v=torch.randn(n_local, args.ffn, args.hidden, device=device, dtype=dtype),
        local_bias_up=torch.randn(n_local, args.ffn, device=device, dtype=dtype),
        local_bias_up_v=torch.randn(n_local, args.ffn, device=device, dtype=dtype),
        local_expert_down=torch.randn(n_local, args.hidden, args.ffn, device=device, dtype=dtype),
        local_bias_down=torch.randn(n_local, args.hidden, device=device, dtype=dtype),
    )
    out = moe.forward(flashdmoe.ForwardArgs(tokens=tokens, router_weights=router, weights=weights))
    torch.cuda.synchronize()
    print(f"rank={moe.ep_rank} output={tuple(out.shape)} dtype={out.dtype}")
    moe.finalize()


if __name__ == "__main__":
    main()
