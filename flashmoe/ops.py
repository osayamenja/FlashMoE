"""
Core FlashMoE functionality
"""
import json
import os
from pathlib import Path
from typing import Union, Dict, Any, Optional
import torch

# Import compiled C++ extension
try:
    from flashmoe import _C
    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False
    import warnings
    warnings.warn("FlashMoE CUDA extension not found. Install with: pip install -e .")


class FlashMoEConfig:
    """
    Configuration for FlashMoE - matches kleos_config.json structure
    
    NOTE: Config values are compile-time constants. This class is used
    to help create properly-sized tensors for testing. To change config,
    you must edit csrc/kleos_config.json and rebuild.
    """
    
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
    input_tensor: torch.Tensor,
    gate_weights: torch.Tensor,
    expert_weights: torch.Tensor,
    n_processes: int = 1,
    processes_per_node: Optional[int] = None,
    hostfile: Optional[str] = None
) -> torch.Tensor:
    """
    Run MoE forward pass
    
    NOTE: Tensor dimensions must match the compile-time config from
    csrc/kleos_config.json. To use different dimensions, edit the JSON
    file and rebuild with: pip install -e . --no-build-isolation
    
    Args:
        input_tensor: Input activations [batch, seq_len, hidden_size]
        gate_weights: Gate weights [hidden_size, num_experts]
        expert_weights: Expert weights [num_experts, 2, intermediate_size, hidden_size]
        n_processes: Number of processes (default=1 for single GPU)
        processes_per_node: Processes per node (default: same as n_processes)
        hostfile: Path to MPI-style hostfile for multi-node execution
    
    Returns:
        output: MoE layer output [batch, seq_len, hidden_size]
    
    Examples:
        >>> # Tensors must match compiled config dimensions
        >>> input_tensor = torch.randn(256, 8192, 2048, dtype=torch.float16, device='cuda')
        >>> gate_weights = torch.randn(2048, 64, dtype=torch.float16, device='cuda')
        >>> expert_weights = torch.randn(64, 2, 2048, 2048, dtype=torch.float16, device='cuda')
        >>> output = run_moe(input_tensor, gate_weights, expert_weights)
        
        >>> # Multi-GPU
        >>> output = run_moe(input_tensor, gate_weights, expert_weights, n_processes=4)
    """
    if n_processes == 1:
        # Single GPU - call directly
        return _C.moe_forward(input_tensor, gate_weights, expert_weights)
    else:
        # Multi-GPU - use nvshmrun launcher
        raise NotImplementedError("Multi-GPU support via nvshmrun launcher coming soon")


def run_moe_from_config(
    config: Union[str, Dict[str, Any]],
    n_processes: int = 1,
    processes_per_node: Optional[int] = None,
    hostfile: Optional[str] = None
) -> torch.Tensor:
    """
    Run MoE forward pass with random tensors created from config
    
    This is mainly for testing. Config must match compile-time config!
    
    Args:
        config: Config dict or path to kleos_config.json
        n_processes: Number of processes
        processes_per_node: Processes per node
        hostfile: Path to hostfile for multi-node
    
    Returns:
        output: MoE layer output
    """
    cfg = FlashMoEConfig(config) if not isinstance(config, FlashMoEConfig) else config
    
    if processes_per_node is None:
        processes_per_node = n_processes
    
    from .launcher import nvshmrun_launcher
    return nvshmrun_launcher(
        config=cfg,
        n_processes=n_processes,
        processes_per_node=processes_per_node,
        hostfile=hostfile
    )


def get_compiled_config() -> Dict[str, int]:
    """
    Get the compile-time config dimensions from the built extension
    
    Returns:
        dict with keys: S, H, E, P (compile-time constants)
    """
    # This would require adding a function to python_bindings.cu
    # For now, just document that users should check kleos_config.json
    raise NotImplementedError(
        "To check compiled config, see csrc/kleos_config.json. "
        "The extension is compiled with those dimensions."
    )