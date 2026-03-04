from typing import Sequence
from .jit import get_compiled
from dataclasses import dataclass

class ContextHandle:
    __slots__ = ("_mod", "_ctx")

    def __init__(self, mod, ctx):
        self._mod = mod
        self._ctx = ctx

    @property
    def capsule(self):
        return self._ctx

    @property
    def mod(self):
        return self._mod

@dataclass(slots=True)
class InitializeArgs:
    s: int
    h: int
    i: int
    ec: int
    expert_map: Sequence[int]
    rank_map: Sequence[int]
    arch: int
    stream_ptr: int

def initialize(arg: InitializeArgs) -> None:
    mod = get_compiled(arg.s, arg.h, arg.i, arg.arch)
    mod.initialize(arg.ec, arg.expert_map, arg.rank_map, arg.stream_ptr)
    # return ContextHandle(mod, ctx)

def forward(context: ContextHandle, top_k: int, stream_ptr: int = 0) -> None:
    context.mod.forward(context.capsule, top_k, stream_ptr)

def finalize(context: ContextHandle) -> None:
    pass