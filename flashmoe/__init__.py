"""
FlashMoE: Fast Distributed MoE in a Single Kernel

Usage:
    from flashmoe import run_moe
    
    # Single GPU (direct call, no nvshmrun needed)
    config = {
        "capacity_factor": 1,
        "drop_tokens": 1,
        "expert_top_k": 2,
        "global_batch": 256,
        "is_training": 0,
        "hidden_act": 0,
        "hidden_size": 2048,
        "intermediate_size": 2048,
        "mini_batch": 1,
        "moe_frequency": 1,
        "num_experts": 64,
        "num_layers": 1,
        "sequence_len": 8192,
        "torch_dtype": 1,
        "vocab_size": 32000
    }
    
    output = run_moe(config)
    
    # Or from config file (just like: ./csrc with kleos_config.json)
    output = run_moe("csrc/kleos_config.json")
    
    # Multi-GPU (just like: nvshmrun -n 4 -ppn 4 ./csrc)
    output = run_moe(
        config="csrc/kleos_config.json",
        n_processes=4,
        processes_per_node=4
    )
"""

from .ops import run_moe, FlashMoEConfig

__version__ = '0.1.0'

__all__ = ['run_moe', 'FlashMoEConfig']
