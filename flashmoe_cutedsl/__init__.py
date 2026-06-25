"""Python/NVSHMEM4Py/CuTe DSL port of FlashMoE.

This package is a Python translation of the FlashMoE forward path.  It keeps
the original operator's stages, but expresses orchestration in Python and uses
NVSHMEM4Py symmetric tensors for expert-parallel token exchange.
"""

from .runtime import FlashDMoEPort, forward, initialize
from .packed import local_packed_forward
from .types import (
    ActivationType,
    DataType,
    ExpertWeights,
    ForwardArgs,
    InitArgs,
    MLPType,
    RoutingResult,
    Topology,
)

__all__ = [
    "ActivationType",
    "DataType",
    "ExpertWeights",
    "FlashDMoEPort",
    "ForwardArgs",
    "InitArgs",
    "MLPType",
    "RoutingResult",
    "Topology",
    "forward",
    "initialize",
    "local_packed_forward",
]
