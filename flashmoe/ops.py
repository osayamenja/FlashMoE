"""
Core FlashMoE functionality
"""
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
):
    """
    Run MoE forward pass with random tensors for benchmarking/testing.
    
    Tensors are automatically created based on compiled configuration.
    
    Args:
        n_processes: Number of processes (default=1 for single GPU)
        processes_per_node: Processes per node (default: same as n_processes)
        hostfile: Path to MPI-style hostfile for multi-node execution
        config_path: Path to config file (must match compiled config)
    
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
    
    nvshmrun_launcher(
        config_path=config_path,
        n_processes=n_processes,
        processes_per_node=processes_per_node,
        hostfile=hostfile
    )


def get_compiled_config():
    """
    Get compile-time configuration dimensions.
    
    Returns:
        dict with keys: S, H, E, P, PX, element_size_bytes
    """
    if not _CUDA_AVAILABLE:
        raise RuntimeError("CUDA extension not available. Install with: pip install -e .")
    return _C.get_compiled_config()
