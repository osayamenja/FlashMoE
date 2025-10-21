"""
Worker script for distributed FlashMoE execution via nvshmrun
This script is executed by each process in the distributed setup.
"""
import os
import sys
import json
import torch
from pathlib import Path


def main():
    """Main worker function executed by nvshmrun"""
    # Get NVSHMEM process info
    rank = int(os.environ.get("NVSHMEM_RANK", "0"))
    world_size = int(os.environ.get("NVSHMEM_WORLD_SIZE", "1"))
    
    print(f"Process {rank}/{world_size} starting...", flush=True)
    
    # Load config from path passed as command line argument
    if len(sys.argv) < 2:
        print("ERROR: Config path not provided", file=sys.stderr)
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Load config (for tensor dimensions only - kernel uses compile-time config)
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Set device
    torch.cuda.set_device(rank % torch.cuda.device_count())
    
    # Import here to ensure CUDA is initialized first
    from flashmoe import _C as _flashmoe_cuda
    
    # Create tensors matching compile-time config expectations
    batch = config["global_batch"]
    seq = config["sequence_len"]
    H = config["hidden_size"]
    I = config["intermediate_size"]
    E = config["num_experts"]
    
    dtype_map = {0: torch.float32, 1: torch.float16, 2: torch.bfloat16}
    dtype = dtype_map[config["torch_dtype"]]
    
    # Input: [batch, seq, H]
    input_tensor = torch.randn(batch, seq, H, dtype=dtype).cuda()
    
    # Gate weights: [H, E]
    gate_weights = torch.randn(H, E, dtype=dtype).cuda()
    
    # Expert weights: [E, 2, I, H]
    expert_weights = torch.randn(E, 2, I, H, dtype=dtype).cuda()
    
    print(f"Rank {rank}: input={input_tensor.shape}, gate={gate_weights.shape}, "
          f"experts={expert_weights.shape}", flush=True)
    
    # Run MoE forward (no config passed - it's compile-time!)
    output = _flashmoe_cuda.moe_forward(
        input_tensor,
        gate_weights,
        expert_weights
    )
    
    print(f"Process {rank} completed. Output shape: {output.shape}", flush=True)
    
    return output


if __name__ == "__main__":
    main()
