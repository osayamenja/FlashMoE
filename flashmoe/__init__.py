"""
FlashMoE: Fast Distributed MoE in a Single Kernel

NOTE: Config is compile-time! Tensor shapes must match what was built.
To change dimensions, edit csrc/kleos_config.json and rebuild.

Usage:
    import torch
    import flashmoe
    
    # Check compiled configuration
    config = flashmoe._C.get_compiled_config()
    print(config)  # {'S': 8192, 'H': 2048, 'E': 64, 'P': 2048, 'PX': 64}
    
    # Create tensors matching compiled dimensions
    # For default config: mini_batch=1, seq_len=8192, hidden=2048, experts=64
    input_tensor = torch.randn(1, 8192, 2048, dtype=torch.float32, device='cuda')
    gate_weights = torch.randn(2048, 64, dtype=torch.float32, device='cuda')
    expert_weights = torch.randn(64, 2, 2048, 2048, dtype=torch.float32, device='cuda')
    
    # Single GPU forward pass
    output = flashmoe.run_moe(input_tensor, gate_weights, expert_weights)
    
    # Multi-GPU (via nvshmrun launcher with random tensors for testing)
    # output = flashmoe.run_moe_from_config("csrc/kleos_config.json", n_processes=4)
    
To rebuild with different dimensions:
    1. Edit csrc/kleos_config.json
    2. pip install -e . --no-build-isolation
    3. Create tensors with new dimensions
"""

from .ops import run_moe, run_moe_from_config, FlashMoEConfig

__version__ = '0.1.0'

__all__ = [
    'run_moe',
    'run_moe_from_config',
    'get_compiled_config',
    'FlashMoEConfig',
]
