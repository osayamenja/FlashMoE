"""
Launcher for multi-process FlashMoE using nvshmrun
"""
import os
import sys
import subprocess
import tempfile
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
    # Get path to worker script
    worker_script = Path(__file__).parent / "worker.py"
    
    if not worker_script.exists():
        raise FileNotFoundError(f"Worker script not found: {worker_script}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save config for worker
        config_path = tmpdir / "kleos_config.json"
        config.to_json(config_path)
        
        # Build nvshmrun command
        cmd = [
            'nvshmrun',
            '-n', str(n_processes),
            '-ppn', str(processes_per_node),
        ]
        
        if hostfile:
            cmd.extend(['--hostfile', hostfile])
        
        cmd.extend([
            sys.executable,
            str(worker_script),
            str(config_path)  # Pass config path as argument
        ])
        
        print(f"Launching FlashMoE with: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
        
        return result