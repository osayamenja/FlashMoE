"""
Launcher for multi-process FlashMoE using nvshmrun
Mirrors the exact behavior of: nvshmrun -n <n> -ppn <ppn> ./csrc
"""
import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Optional, Dict, Any
import torch


def nvshmrun_launcher(
    config,
    n_processes: int,
    processes_per_node: int,
    hostfile: Optional[str] = None,
    input_tensor: Optional[torch.Tensor] = None,
    expert_weights: Optional[torch.Tensor] = None,
    gate_weights: Optional[torch.Tensor] = None
):
    """
    Launch FlashMoE using nvshmrun (exactly like current C++ workflow)
    
    This creates a temporary Python script that sets up NVSHMEM and runs the kernel,
    then launches it with nvshmrun.
    """
    # Create temporary directory for the run
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save config
        config_path = tmpdir / "kleos_config.json"
        config.to_json(config_path)
        
        # Create the worker script
        worker_script = tmpdir / "flashmoe_worker.py"
        _create_worker_script(
            worker_script,
            config_path,
            input_tensor,
            expert_weights,
            gate_weights
        )
        
        # Run with nvshmrun (exactly like: nvshmrun -n 4 -ppn 4 ./csrc)
        cmd = [
            'nvshmrun',
            '-n', str(n_processes),
            '-ppn', str(processes_per_node),
            sys.executable,
            str(worker_script)
        ]
        
        print(f"Launching FlashMoE with: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
        
        return result


def _create_worker_script(
    script_path: Path,
    config_path: Path,
    input_tensor: Optional[torch.Tensor],
    expert_weights: Optional[torch.Tensor],
    gate_weights: Optional[torch.Tensor]
):
    """Create the Python worker script that will be launched by nvshmrun"""
    
    script_content = f'''
import os
import sys
import torch
from flashmoe import _C as _flashmoe_cuda
import json

def main():
    # Initialize NVSHMEM (happens automatically in CUDA extension)
    rank = int(os.environ.get("NVSHMEM_RANK", "0"))
    world_size = int(os.environ.get("NVSHMEM_WORLD_SIZE", "1"))
    
    print(f"Process {{rank}}/{{world_size}} starting...", flush=True)
    
    # Load config
    with open("{config_path}", "r") as f:
        config = json.load(f)
    
    # Set device
    torch.cuda.set_device(rank % torch.cuda.device_count())
    
    # Create tensors matching C++ expectations
    local_batch = config["global_batch"] // world_size
    S = local_batch * config["sequence_len"]  # Total tokens (flattened)
    H = config["hidden_size"]
    I = config["intermediate_size"]
    E = config["num_experts"]
    
    dtype_map = {{0: torch.float32, 1: torch.float16, 2: torch.bfloat16}}
    dtype = dtype_map[config["torch_dtype"]]
    
    # FIXED: Create flattened input [S, H] instead of [batch, seq, H]
    input_tensor = torch.randn(
        S,  # batch * seq_len (flattened)
        H,
        dtype=dtype
    ).cuda()
    
    # Gate weights [H, E]
    gate_weights = torch.randn(
        H,
        E,
        dtype=dtype
    ).cuda()
    
    # Expert weights [local_experts, 2, I, H]
    experts_per_rank = E // world_size
    expert_weights = torch.randn(
        experts_per_rank,
        2,
        I,
        H,
        dtype=dtype
    ).cuda()
    
    print(f"Rank {{rank}}: input={{input_tensor.shape}}, gate={{gate_weights.shape}}, experts={{expert_weights.shape}}", flush=True)
    
    # Run the MoE forward pass
    output = _flashmoe_cuda.moe_forward(
        input_tensor,
        gate_weights,
        expert_weights,
        config
    )
    
    print(f"Process {{rank}} completed. Output shape: {{output.shape}}", flush=True)
    
    return output

if __name__ == "__main__":
    main()
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)


def create_run_script(
    output_path: str = "run_flashmoe.py",
    config_path: str = "csrc/kleos_config.json"
):
    """
    Create a standalone script for easy distributed execution
    
    This creates a script you can run with:
        nvshmrun -n 4 -ppn 4 python run_flashmoe.py
    
    Args:
        output_path: Where to save the script
        config_path: Path to kleos_config.json
    """
    
    script = f'''#!/usr/bin/env python
"""
FlashMoE Distributed Runner
Run with: nvshmrun -n <n_processes> -ppn <processes_per_node> python {{output_path}}
"""
import os
import sys
import torch
import json
from flashmoe import _C as _flashmoe_cuda

def main():
    rank = int(os.environ.get("NVSHMEM_RANK", "0"))
    world_size = int(os.environ.get("NVSHMEM_WORLD_SIZE", "1"))
    
    print(f"[Rank {{rank}}/{{world_size}}] Starting FlashMoE...", flush=True)
    
    # Load config
    config_path = "{config_path}"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Set device
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    print(f"[Rank {{rank}}] Using GPU {{device}}", flush=True)
    
    # Prepare inputs
    local_batch = config["global_batch"] // world_size
    dtype_map = {{0: torch.float32, 1: torch.float16, 2: torch.bfloat16}}
    dtype = dtype_map[config["torch_dtype"]]
    
    input_tensor = torch.randn(
        local_batch,
        config["sequence_len"],
        config["hidden_size"],
        dtype=dtype,
        device=f"cuda:{{device}}"
    )
    
    gate_weights = torch.randn(
        config["hidden_size"],
        config["num_experts"],
        dtype=dtype,
        device=f"cuda:{{device}}"
    )
    
    experts_per_rank = config["num_experts"] // world_size
    expert_weights = torch.randn(
        experts_per_rank,
        2,
        config["intermediate_size"],
        config["hidden_size"],
        dtype=dtype,
        device=f"cuda:{{device}}"
    )
    
    # Run MoE forward
    print(f"[Rank {{rank}}] Running MoE forward pass...", flush=True)
    output = _flashmoe_cuda.moe_forward(
        input_tensor,
        gate_weights,
        expert_weights,
        config
    )
    
    print(f"[Rank {{rank}}] Completed! Output shape: {{output.shape}}", flush=True)
    
    return output

if __name__ == "__main__":
    main()
'''
    
    with open(output_path, 'w') as f:
        f.write(script)
    
    # Make executable
    os.chmod(output_path, 0o755)
    
    print(f"Created run script: {output_path}")
    print(f"Run with: nvshmrun -n <n> -ppn <ppn> python {output_path}")
    