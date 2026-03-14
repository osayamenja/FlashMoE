# communication backend
IS_INITIALIZED = False

def get_local_rank() -> int:
    import os
    if has_package("torch") and os.environ.get("LOCAL_RANK") is not None:
        return int(os.environ.get("LOCAL_RANK"))
    elif has_package("mpi4py"):
        import mpi4py.MPI as MPI
        import cuda.core.experimental as cuda
        return MPI.COMM_WORLD.Get_rank() % cuda.system.num_devices
    else:
        raise RuntimeError("At least one of {torch, mpi4py} must be available")

def has_package(name: str):
    import importlib.util
    return importlib.util.find_spec(name) is not None

def initialize() -> None:
    import cuda.core.experimental as cuda
    import nvshmem.core as nvshmem
    if nvshmem.init_status() != nvshmem.InitStatus.STATUS_IS_INITIALIZED:
        global IS_INITIALIZED
        initialized = False
        dev = cuda.Device(get_local_rank())
        dev.set_current()
        if has_package("torch"):
            import torch.distributed as dist
            if dist.is_initialized():
                num_ranks = dist.get_world_size()
                rank_id = dist.get_rank()
                uniqueid = nvshmem.get_unique_id(empty=True)
                src_rank = 0
                if rank_id == src_rank:
                    # Rank 0 gets a real uniqueid
                    uniqueid = nvshmem.get_unique_id()
                    broadcast_objects = [uniqueid]
                else:
                    broadcast_objects = [None]
                dist.broadcast_object_list(broadcast_objects, src=src_rank)
                dist.barrier()
                nvshmem.init(device=dev, uid=broadcast_objects[0], rank=rank_id, nranks=num_ranks,
                                  initializer_method="uid")
                initialized = True
        if not initialized and has_package("mpi4py"):
            import mpi4py.MPI as MPI
            nvshmem.init(device=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")
            initialized = True
        IS_INITIALIZED = initialized
        if not initialized:
            raise RuntimeError("At least one of {torch, mpi4py} must be initialized")
        pass

def get_rank() -> int:
    assert IS_INITIALIZED
    import nvshmem.core as nvshmem
    return nvshmem.my_pe()

def get_world_size() -> int:
    assert IS_INITIALIZED
    import nvshmem.core as nvshmem
    return nvshmem.n_pes()

