"""
Core FlashMoE functionality
"""
from pathlib import Path
from typing import Optional
import torch

from .launcher import nvshmrun_launcher

# Import compiled C++ extension
try:
    from flashmoe import _C
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False
    import warnings
    warnings.warn("FlashMoE CUDA extension not found. Install with: pip install -e .")


def run_moe(
    n_processes: int = 1,
    processes_per_node: Optional[int] = None,
    hostfile: Optional[str] = None,
    config_path: str = "csrc/kleos_config.json"
) -> torch.Tensor:
    """
    Run MoE forward pass with random tensors for benchmarking/testing.
    
    Tensors are automatically created based on compiled configuration.
    
    Args:
        n_processes: Number of processes (default=1 for single GPU)
        processes_per_node: Processes per node (default: same as n_processes)
        hostfile: Path to MPI-style hostfile for multi-node execution
        config_path: Path to config file (must match compiled config)
    
    Returns:
        None (prints timing results)
    
    Examples:
        >>> import flashmoe
        >>> 
        >>> # Single GPU
        >>> flashmoe.run_moe()
        >>> 
        >>> # Multi-GPU single node
        >>> flashmoe.run_moe(n_processes=4)
        >>> 
        >>> # Multi-node
        >>> flashmoe.run_moe(n_processes=16, processes_per_node=8, hostfile="hosts.txt")
    
    Note:
        To change dimensions, edit csrc/kleos_config.json and rebuild:
        pip install -e . --no-build-isolation
    """
    if processes_per_node is None:
        processes_per_node = n_processes
    
    if n_processes == 1:
        # Single GPU - run directly
        return _run_single_gpu(config_path)
    else:
        # Multi-GPU - use nvshmrun launcher
        return nvshmrun_launcher(
            config_path=config_path,
            n_processes=n_processes,
            processes_per_node=processes_per_node,
            hostfile=hostfile
        )


def _run_single_gpu(config_path: str = "csrc/kleos_config.json") -> torch.Tensor:
    """Run single GPU with random tensors"""
    import json
    
    # Load config to create tensors
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get dimensions
    mini_batch = config["mini_batch"]
    seq_len = config["sequence_len"]
    H = config["hidden_size"]
    I = config["intermediate_size"]
    E = config["num_experts"]
    
    # Map dtype
    dtype_map = {0: torch.float16, 1: torch.float32}
    dtype = dtype_map[config["torch_dtype"]]
    
    print(f"Creating random tensors: [{mini_batch}, {seq_len}, {H}], dtype={dtype}")
    
    # Create random tensors
    input_tensor = torch.randn(mini_batch, seq_len, H, dtype=dtype, device='cuda')
    gate_weights = torch.randn(H, E, dtype=dtype, device='cuda')
    expert_weights = torch.randn(E, 2, I, H, dtype=dtype, device='cuda')
    
    # Run forward
    output = _C.moe_forward(input_tensor, gate_weights, expert_weights)
    
    return output


def get_compiled_config():
    """
    Get compile-time configuration dimensions.
    
    Returns:
        dict with keys: S, H, E, P, PX, element_size_bytes
    """
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA extension not available. Install with: pip install -e .")
    return _C.get_compiled_config()