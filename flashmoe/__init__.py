"""
FlashMoE: Fast Distributed MoE in a Single Kernel

Simple usage - no manual tensor creation needed:

    import flashmoe
    
    # Single GPU
    flashmoe.run_moe()
    
    # Multi-GPU
    flashmoe.run_moe(n_processes=4)

Configuration is compile-time:
    1. Edit csrc/kleos_config.json
    2. pip install -e . --no-build-isolation
    3. flashmoe.run_moe()
"""

from .ops import run_moe, get_compiled_config

try:
    from . import _C
    _CUDA_AVAILABLE = True
    _C.initialize()
except ImportError:
    _CUDA_AVAILABLE = False
    import warnings
    warnings.warn("FlashMoE CUDA extension not found. Install with: pip install -e .")

__version__ = '0.1.0'

__all__ = ['run_moe', 'get_compiled_config']
