"""
Launcher for multi-process FlashMoE using nvshmrun
"""
import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Optional
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
    Launch FlashMoE using nvshmrun
    
    Config is used ONLY to create appropriately-sized random tensors
    for testing. The actual kernel uses compile-time config.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save config (for tensor creation only)
        config_path = tmpdir / "kleos_config.json"
        config.to_json(config_path)
        
        # Create worker script
        worker_script = tmpdir / "flashmoe_worker.py"
        _create_worker_script(worker_script, config_path)
        
        # Run with nvshmrun
        cmd = [
            'nvshmrun',
            '-n', str(n_processes),
            '-ppn', str(processes_per_node),
            sys.executable,
            str(worker_script)
        ]
        
        if hostfile:
            cmd.extend(['--hostfile', hostfile])
        
        print(f"Launching FlashMoE with: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
        
        return result


def _create_worker_script(script_path: Path, config_path: Path):
    """Create the Python worker script that will be launched by nvshmrun"""
    
    script_content = f'''
import os
import sys
import torch
from flashmoe import _C as _flashmoe_cuda
import json

def main():
    rank = int(os.environ.get("NVSHMEM_RANK", "0"))
    world_size = int(os.environ.get("NVSHMEM_WORLD_SIZE", "1"))
    
    print(f"Process {{rank}}/{{world_size}} starting...", flush=True)
    
    # Load config (for tensor dimensions only - kernel uses compile-time config)
    with open("{config_path}", "r") as f:
        config = json.load(f)
    
    # Set device
    torch.cuda.set_device(rank % torch.cuda.device_count())
    
    # Create tensors matching compile-time config expectations
    batch = config["global_batch"]
    seq = config["sequence_len"]
    H = config["hidden_size"]
    I = config["intermediate_size"]
    E = config["num_experts"]
    
    dtype_map = {{0: torch.float32, 1: torch.float16, 2: torch.bfloat16}}
    dtype = dtype_map[config["torch_dtype"]]
    
    # Input: [batch, seq, H]
    input_tensor = torch.randn(batch, seq, H, dtype=dtype).cuda()
    
    # Gate weights: [H, E]
    gate_weights = torch.randn(H, E, dtype=dtype).cuda()
    
    # Expert weights: [E, 2, I, H] (all experts, will be distributed internally)
    expert_weights = torch.randn(E, 2, I, H, dtype=dtype).cuda()
    
    print(f"Rank {{rank}}: input={{input_tensor.shape}}, gate={{gate_weights.shape}}, experts={{expert_weights.shape}}", flush=True)
    
    # Run MoE forward (no config passed - it's compile-time!)
    output = _flashmoe_cuda.moe_forward(
        input_tensor,
        gate_weights,
        expert_weights
    )
    
    print(f"Process {{rank}} completed. Output shape: {{output.shape}}", flush=True)
    
    return output

if __name__ == "__main__":
    main()
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)