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
    config_path: str = "csrc/kleos_config.json",
    n_processes: int = 1,
    processes_per_node: int = 1,
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
    # Get absolute path to config
    config_path = Path(config_path).resolve()
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Get path to worker script
    worker_script = Path(__file__).parent / "worker.py"
    
    if not worker_script.exists():
        raise FileNotFoundError(f"Worker script not found: {worker_script}")
        
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
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print("=" * 60)
        print(f"ERROR: Command failed with exit code {e.returncode}")
        print("=" * 60)
        print("STDOUT:")
        print(e.stdout)
        print("\nSTDERR:")
        print(e.stderr)
        print("=" * 60)
        raise
    
    return result