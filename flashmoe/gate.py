from .jit import ContextHandle, InitArgs

def initialize(arg: InitArgs):
    from .bindings import gate_bindings
    pass
def forward(handle: ContextHandle, stream_ptr: int) -> None:
    pass
def finalize(handle: ContextHandle, stream_ptr: int) -> None:
    pass