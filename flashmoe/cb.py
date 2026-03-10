# communication backend
import nvshmem.core as nvshmem
IS_INITIALIZED = False

def _get_local_rank() -> int:
    import os
    import cuda.core as cuda
    if has_package("torch"):
        return int(os.environ['LOCAL_RANK'])
    elif has_package("mpi4py"):
        import mpi4py.MPI as MPI
        return MPI.COMM_WORLD.Get_rank() % cuda.system.get_num_devices()
    else:
        raise RuntimeError("At least one of {torch, mpi4py} must be available")

def has_package(name: str):
    import importlib.util
    return importlib.util.find_spec(name) is not None

def initialize() -> None:
    if nvshmem.init_status() != nvshmem.InitStatus.STATUS_IS_INITIALIZED:
        global IS_INITIALIZED
        IS_INITIALIZED = True
        initialized = False
        if has_package("torch"):
            import torch.distributed as dist
            if dist.is_initialized():
                initialized = True
        if not initialized and has_package("mpi4py"):
            import mpi4py.MPI as MPI
            import cuda.core as cuda
            initialized = True
            dev = cuda.Device(_get_local_rank())
            dev.set_current()
            nvshmem.init(device=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")
        if not initialized:
            raise RuntimeError("At least one of {torch, mpi4py} must be initialized")
        pass

def get_rank() -> int:
    assert IS_INITIALIZED
    return nvshmem.my_pe()

def get_world_size() -> int:
    assert IS_INITIALIZED
    return nvshmem.n_pes()

