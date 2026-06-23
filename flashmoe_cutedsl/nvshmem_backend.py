from __future__ import annotations

import importlib.util
import os
from typing import Any


_INITIALIZED_HERE = False


def has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _cuda_device(device_id: int):
    try:
        from cuda.core.experimental import Device

        return Device(device_id)
    except Exception:
        import cuda.core as cuda

        return cuda.Device(device_id)


def _cuda_device_count() -> int:
    try:
        from cuda.core.experimental import system

        return int(system.num_devices)
    except Exception:
        import cuda.core as cuda

        return int(cuda.system.num_devices)


def local_rank(default: int | None = None) -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if default is not None:
        return default
    if has_package("mpi4py"):
        from mpi4py import MPI

        return MPI.COMM_WORLD.Get_rank() % _cuda_device_count()
    return 0


def torch_stream(stream: Any | None = None):
    if stream is not None:
        return stream
    import torch

    return torch.cuda.current_stream()


def initialize(device_id: int | None = None) -> None:
    """Initialize NVSHMEM4Py with torch.distributed or MPI if needed."""
    import nvshmem.core as nvshmem

    if nvshmem.init_status() == nvshmem.InitStatus.STATUS_IS_INITIALIZED:
        return

    dev_id = local_rank(device_id)
    dev = _cuda_device(dev_id)
    dev.set_current()

    initialized = False
    if has_package("torch"):
        import torch
        import torch.distributed as dist

        if torch.cuda.is_available():
            torch.cuda.set_device(dev_id)
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world = dist.get_world_size()
            unique_id = nvshmem.get_unique_id(empty=True)
            payload = [nvshmem.get_unique_id() if rank == 0 else None]
            dist.broadcast_object_list(payload, src=0)
            dist.barrier()
            unique_id = payload[0] if rank != 0 else payload[0]
            nvshmem.init(
                device=dev,
                uid=unique_id,
                rank=rank,
                nranks=world,
                initializer_method="uid",
            )
            initialized = True

    if not initialized and has_package("mpi4py"):
        from mpi4py import MPI

        nvshmem.init(device=dev, mpi_comm=MPI.COMM_WORLD, initializer_method="mpi")
        initialized = True

    if not initialized:
        raise RuntimeError(
            "NVSHMEM4Py needs an initialized torch.distributed process group "
            "or mpi4py/MPI launcher."
        )

    global _INITIALIZED_HERE
    _INITIALIZED_HERE = True


def finalize_if_owned() -> None:
    import nvshmem.core as nvshmem

    global _INITIALIZED_HERE
    if _INITIALIZED_HERE and nvshmem.init_status() == nvshmem.InitStatus.STATUS_IS_INITIALIZED:
        nvshmem.finalize()
    _INITIALIZED_HERE = False


def rank() -> int:
    import nvshmem.core as nvshmem

    return int(nvshmem.my_pe())


def world_size() -> int:
    import nvshmem.core as nvshmem

    return int(nvshmem.n_pes())


def sync_all(stream: Any | None = None) -> None:
    import nvshmem.core as nvshmem

    nvshmem.sync_all(stream=torch_stream(stream))


def symmetric_tensor(shape, dtype, *, release: bool = False):
    import nvshmem.core as nvshmem

    return nvshmem.tensor(tuple(shape), dtype=dtype, release=release)
