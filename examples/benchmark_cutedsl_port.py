import argparse
import math
import statistics

import torch

import flashmoe_cutedsl as flashdmoe


def build_case(args, *, use_packed: bool, use_cutedsl_gather: bool, use_cutedsl_combine: bool = False):
    init_args = flashdmoe.InitArgs(
        data_type=flashdmoe.DataType.BF16 if args.dtype == "bf16" else flashdmoe.DataType.FP16,
        tokens_per_rank=args.tokens,
        token_dim=args.hidden,
        ffn_size=args.ffn,
        num_experts=args.experts,
        top_k=args.top_k,
        mlp_type=flashdmoe.MLPType.GATED if args.gated else flashdmoe.MLPType.VANILLA,
        act_type=flashdmoe.ActivationType.SILU,
        device_id=args.device,
        expert_peer_capacity=args.capacity,
        ep_world=1,
        ep_rank=0,
        num_local_experts=args.experts,
    )
    return flashdmoe.initialize(
        init_args,
        use_nvshmem=False,
        use_packed=use_packed,
        use_cutedsl_gather=use_cutedsl_gather,
        use_cutedsl_combine=use_cutedsl_combine,
    )


def time_case(label, moe, fwd_args, *, warmup: int, iters: int, cuda_graph: bool = False):
    for _ in range(warmup):
        moe.forward(fwd_args)
    torch.cuda.synchronize()

    graph = None
    if cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            moe.forward(fwd_args)
        torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if graph is None:
            moe.forward(fwd_args)
        else:
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    mean = statistics.mean(samples)
    p50 = statistics.median(samples)
    best = min(samples)
    print(f"{label:18s} mean={mean:8.3f} ms  p50={p50:8.3f} ms  min={best:8.3f} ms")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--ffn", type=int, default=8192)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--capacity", type=int, default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--gated", action="store_true", default=True)
    parser.add_argument("--no-gated", dest="gated", action="store_false")
    parser.add_argument("--cutedsl-gather", action="store_true")
    parser.add_argument("--cutedsl-combine", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    capacity = args.capacity or (math.ceil(args.tokens / args.experts) * args.top_k)
    args.capacity = capacity

    torch.manual_seed(123)
    tokens = torch.randn(args.tokens, args.hidden, device=device, dtype=dtype)
    router = torch.randn(args.hidden, args.experts, device=device, dtype=dtype)

    def make_weights():
        up_v = torch.randn(args.experts, args.ffn, args.hidden, device=device, dtype=dtype) if args.gated else None
        bias_up_v = torch.randn(args.experts, args.ffn, device=device, dtype=dtype) if args.gated else None
        return flashdmoe.ExpertWeights(
            local_expert_up=torch.randn(args.experts, args.ffn, args.hidden, device=device, dtype=dtype),
            local_expert_up_v=up_v,
            local_bias_up=torch.randn(args.experts, args.ffn, device=device, dtype=dtype),
            local_bias_up_v=bias_up_v,
            local_expert_down=torch.randn(args.experts, args.hidden, args.ffn, device=device, dtype=dtype),
            local_bias_down=torch.randn(args.experts, args.hidden, device=device, dtype=dtype),
        )

    weights = make_weights()
    packed = build_case(
        args,
        use_packed=True,
        use_cutedsl_gather=args.cutedsl_gather,
        use_cutedsl_combine=args.cutedsl_combine,
    )
    scalar = build_case(args, use_packed=False, use_cutedsl_gather=False)
    routing = packed.route(tokens, router)
    fwd = flashdmoe.ForwardArgs(tokens=tokens, routing=routing, weights=weights)

    print(
        f"S={args.tokens} H={args.hidden} I={args.ffn} E={args.experts} "
        f"k={args.top_k} cap={capacity} dtype={args.dtype} gated={args.gated}"
    )
    if args.cuda_graph:
        print("scalar             skipped under --cuda-graph")
    else:
        time_case("scalar", scalar, fwd, warmup=args.warmup, iters=args.iters)
    packed_graph = args.cuda_graph and args.top_k == 1
    if args.cuda_graph and not packed_graph:
        print("packed graph       unsupported for top-k > 1; using eager")
    time_case("packed", packed, fwd, warmup=args.warmup, iters=args.iters, cuda_graph=packed_graph)


if __name__ == "__main__":
    main()
