"""
Core FlashMoE functionality - mirrors the C++ executable behavior
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Union, Dict, Any, Optional
import torch

# Import compiled C++ extension (standard naming convention)
try:
    from flashmoe import _C  # Compiled extension: flashmoe._C
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False
    import warnings
    warnings.warn("FlashMoE CUDA extension not found. Install with: pip install -e .")


class FlashMoEConfig:
    """Configuration for FlashMoE - matches kleos_config.json structure exactly"""
    
    REQUIRED_KEYS = {
        "capacity_factor", "drop_tokens", "expert_top_k", "global_batch",
        "is_training", "hidden_act", "hidden_size", "intermediate_size",
        "mini_batch", "moe_frequency", "num_experts", "num_layers",
        "sequence_len", "torch_dtype", "vocab_size"
    }
    
    def __init__(self, config: Union[str, Dict[str, Any]]):
        """
        Initialize config from dict or JSON file path
        
        Args:
            config: Either a dict with config params or path to kleos_config.json
        """
        if isinstance(config, str):
            config = Path(config)
            if not config.exists():
                raise FileNotFoundError(f"Config file not found: {config}")
            with open(config, 'r') as f:
                self.config = json.load(f)
        elif isinstance(config, dict):
            self.config = config.copy()
        else:
            raise TypeError("config must be dict or path to JSON file")
        
        # Validate required keys
        missing = self.REQUIRED_KEYS - set(self.config.keys())
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")
    
    def to_json(self, path: str):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __repr__(self):
        return f"FlashMoEConfig({self.config})"


def run_moe(
    config: Union[str, Dict[str, Any]],
    n_processes: int = 1,
    processes_per_node: Optional[int] = None,
    hostfile: Optional[str] = None,
    input_tensor: Optional[torch.Tensor] = None,
    expert_weights: Optional[torch.Tensor] = None,
    gate_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Run MoE forward pass using nvshmrun launcher
    
    Unified interface for all execution modes (always uses nvshmrun):
    - Single GPU (n_processes=1): nvshmrun -n 1 -ppn 1
    - Multi-GPU single node (n_processes>1, no hostfile): nvshmrun -n N -ppn N
    - Multi-node (n_processes>1, with hostfile): nvshmrun -n N -ppn M --hostfile hosts.txt
    
    This mirrors the behavior of:
    - nvshmrun -n 1 -ppn 1 ./csrc                           (single GPU)
    - nvshmrun -n 4 -ppn 4 ./csrc                           (multi-GPU, single node)
    - nvshmrun -n 16 -ppn 8 --hostfile hosts.txt ./csrc    (multi-node)
    
    Args:
        config: Either dict with config params or path to kleos_config.json
        n_processes: Total number of processes across all nodes (default=1 for single GPU)
        processes_per_node: Processes per node (default: same as n_processes for single node)
        hostfile: Path to MPI-style hostfile for multi-node execution.
                 Format: one line per node with "hostname slots=N"
                 Example:
                     node1 slots=4
                     node2 slots=4
                     192.168.1.100 slots=8
        input_tensor: Input tensor [global_batch, sequence_len, hidden_size].
                     If None, uses random data (like C++ version does for benchmarking)
        expert_weights: Expert weights [num_experts, intermediate_size, hidden_size].
                       If None, uses random initialization
        gate_weights: Gate weights [hidden_size, num_experts].
                     If None, uses random initialization
    
    Returns:
        output: MoE layer output [global_batch, sequence_len, hidden_size]
    
    Examples:
        # Single GPU - nvshmrun -n 1 -ppn 1
        >>> config = {
        ...     "capacity_factor": 1,
        ...     "drop_tokens": 1,
        ...     "expert_top_k": 2,
        ...     "global_batch": 256,
        ...     "is_training": 0,
        ...     "hidden_act": 0,
        ...     "hidden_size": 2048,
        ...     "intermediate_size": 2048,
        ...     "mini_batch": 1,
        ...     "moe_frequency": 1,
        ...     "num_experts": 64,
        ...     "num_layers": 1,
        ...     "sequence_len": 8192,
        ...     "torch_dtype": 1,
        ...     "vocab_size": 32000
        ... }
        >>> output = run_moe(config)
        
        # Multi-GPU single node (just like: nvshmrun -n 4 -ppn 4 ./csrc)
        >>> output = run_moe(
        ...     config="csrc/kleos_config.json",
        ...     n_processes=4,
        ...     processes_per_node=4
        ... )
        
        # Multi-node with hostfile (just like: nvshmrun -n 16 -ppn 8 --hostfile hosts.txt ./csrc)
        >>> output = run_moe(
        ...     config="csrc/kleos_config.json",
        ...     n_processes=16,
        ...     processes_per_node=8,
        ...     hostfile="hosts.txt"
        ... )
    """
    
    # Parse config
    cfg = FlashMoEConfig(config) if not isinstance(config, FlashMoEConfig) else config
    
    # Set default processes_per_node
    if processes_per_node is None:
        processes_per_node = n_processes
    
    # Always use nvshmrun launcher (even for single GPU)
    from .launcher import nvshmrun_launcher
    return nvshmrun_launcher(
        config=cfg,
        n_processes=n_processes,
        processes_per_node=processes_per_node,
        hostfile=hostfile,
        input_tensor=input_tensor,
        expert_weights=expert_weights,
        gate_weights=gate_weights
    )


def _get_torch_dtype(dtype_int: int) -> torch.dtype:
    """Convert dtype integer from config to torch dtype"""
    dtype_map = {
        0: torch.float32,
        1: torch.float16,
        2: torch.bfloat16,
    }
    return dtype_map.get(dtype_int, torch.float16)
