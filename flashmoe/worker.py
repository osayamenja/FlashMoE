"""
Worker script for distributed execution
"""
import os
import sys
import json
import torch

from flashmoe import _C

def main():
    _C.initialize()
    
    # Load config
    if len(sys.argv) < 2:
        print("ERROR: Config path not provided", file=sys.stderr)
        sys.exit(1)
    
    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Get local rank from MPI env vars (for GPU selection only)
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 
                     os.environ.get("PMI_RANK",
                     os.environ.get("SLURM_PROCID", "0"))))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE",
                     os.environ.get("PMI_SIZE", 
                     os.environ.get("SLURM_NTASKS", "1"))))
    
    # Set device
    device_id = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    
    print(f"Process {local_rank}/{world_size} using GPU {device_id}", flush=True)
    
    # Import C++ module
    from flashmoe import _C as _flashmoe_cuda
    
    # Get dimensions
    mini_batch = config["mini_batch"]
    seq_len = config["sequence_len"]
    H = config["hidden_size"]
    I = config["intermediate_size"]
    E = config["num_experts"]
    
    dtype_map = {0: torch.float16, 1: torch.float32}
    dtype = dtype_map[config["torch_dtype"]]
    
    # Get local expert count computed during distributed initialization
    nLx = _flashmoe_cuda.get_num_local_experts()
    
    print(f"Process {local_rank}: Creating {nLx} local experts (total {E})", flush=True)
    
    # Create tensors
    input_tensor = torch.randn(mini_batch, seq_len, H, dtype=dtype, device='cuda')
    gate_weights = torch.randn(H, E, dtype=dtype, device='cuda')
    expert_weights = torch.randn(nLx, 2, I, H, dtype=dtype, device='cuda')
    
    print(f"Process {local_rank}: Calling moe_forward...", flush=True)
    
    # Run MoE
    output = _flashmoe_cuda.moe_forward(
        input_tensor,
        gate_weights,
        expert_weights
    )
    
    print(f"Process {local_rank}: Completed! Output: {output.shape}", flush=True)
    
    return output


if __name__ == "__main__":
    main()
