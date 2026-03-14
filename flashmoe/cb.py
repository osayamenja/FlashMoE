# communication backend
import os
import cuda.core as cuda
import nvshmem.core as nvshmem

from flashmoe import Topology

IS_INITIALIZED = False

def get_local_rank() -> int:
    if has_package("torch"):
        return int(os.environ['LOCAL_RANK'])
    elif has_package("mpi4py"):
        import mpi4py.MPI as MPI
        return MPI.COMM_WORLD.Get_rank() % cuda.system.get_num_devices()
    else:
        raise RuntimeError("At least one of {torch, mpi4py} must be available")

def detect_topo():
    assert IS_INITIALIZED
    if nvshmem.team_n_pes(nvshmem.Teams.TEAM_SHARED_INDEX) == nvshmem.n_pes():
        return Topology.NVLINK_ONLY
    else:
        return Topology.MIXED

def has_package(name: str):
    import importlib.util
    return importlib.util.find_spec(name) is not None

def initialize() -> None:
    import cuda.core as cuda
    if nvshmem.init_status() != nvshmem.InitStatus.STATUS_IS_INITIALIZED:
        global IS_INITIALIZED
        IS_INITIALIZED = True
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
        if not initialized:
            raise RuntimeError("At least one of {torch, mpi4py} must be initialized")
        pass

def get_rank() -> int:
    assert IS_INITIALIZED
    return nvshmem.my_pe()

def get_world_size() -> int:
    assert IS_INITIALIZED
    return nvshmem.n_pes()

