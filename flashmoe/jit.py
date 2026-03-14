from __future__ import annotations

from pathlib import Path
from enum import IntEnum

from .cb import get_rank

class ContextHandle:
    __slots__ = ("_mod", "_ctx")

    def __init__(self, mod, ctx):
        self._mod = mod
        self._ctx = ctx

    @property
    def context(self):
        return self._ctx

    @property
    def mod(self):
        return self._mod

class Topology(IntEnum):
    NVLINK_ONLY = 0
    MIXED = 1

class MLPType(IntEnum):
    GATED = 0
    VANILLA = 1

class DataType(IntEnum):
    BF16 = 0
    FP16 = 1
    FP32 = 2
    FP64 = 3

class ActivationType(IntEnum):
    IDENTITY = 0
    SILU = 1
    GELU = 2
    RELU = 3 # as many as CUTLASS supports

class InitArgs:
    from typing import List
    data_type: DataType
    tokens_per_rank: int
    token_dim: int
    ffn_size: int
    num_experts: int
    top_k: int
    expert_map: List[int]
    rank_map: List[int]
    gpu_arch: int
    topo: Topology = Topology.MIXED
    mlp_type: MLPType
    act_type: ActivationType
    device_id: int
    ep_world: int
    ep_rank: int
    my_pe: int
    num_local_experts: int
    stream_ptr: int
    expert_peer_capacity: int

    def __init__(self,
                 data_type: DataType,
                 tokens_per_rank: int,
                 token_dim: int,
                 ffn_size: int,
                 num_experts: int,
                 top_k: int,
                 gpu_arch: int,
                 mlp_type: MLPType,
                 act_type: ActivationType,
                 stream_ptr: int,
                 device_id: int,
                 *,
                 ep_world: int = None,
                 num_local_experts: int = None,
                 ep_rank: int = None,
                 my_pe: int = None,
                 expert_map: List[int] = None,
                 rank_map: List[int] = None,
                 expert_peer_capacity: int = None) -> None:
        from math import ceil
        assert gpu_arch >= 700
        self.data_type = data_type
        self.tokens_per_rank = tokens_per_rank
        self.token_dim = token_dim
        self.ffn_size = ffn_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_map = expert_map
        self.rank_map = rank_map
        self.gpu_arch = gpu_arch
        self.mlp_type = mlp_type
        self.act_type = act_type
        self.device_id=device_id
        self.ep_world = ep_world
        self.ep_rank = ep_rank
        self.my_pe = my_pe
        self.num_local_experts = num_local_experts
        self.stream_ptr = stream_ptr
        if expert_peer_capacity is None:
            self.expert_peer_capacity = ceil(float(tokens_per_rank) / num_experts) * top_k
        else:
            self.expert_peer_capacity = expert_peer_capacity

def _verify_dirs() -> None:
    from pathlib import Path
    root = Path(__file__).resolve().parent

    if not (root / "CMakeLists.txt").exists():
        raise RuntimeError("JIT CMakeLists.txt not found at package root")

def _load_ext(mod_name: str, so_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(mod_name, so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {mod_name} from {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _source_fingerprint() -> str:
    from pathlib import Path
    import hashlib

    root = Path(__file__).resolve().parent.parent
    include_dir = root / "csrc" / "include"

    h = hashlib.sha256()
    h.update(b"flashmoe-jit-v1")

    files = sorted(include_dir.glob("**/*.cuh"))

    for path in files:
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())

    return h.hexdigest()[:16]


def _get_compiled(arg: InitArgs, src: str, mod_prefix: str, mod_name: str):
    import os
    import sys
    import time
    import socket
    import shutil
    import hashlib
    import subprocess
    from pathlib import Path

    _verify_dirs()

    cache = Path(os.environ.get("FLASHMOE_CACHE_DIR", str(Path.home() / ".cache" / "flashmoe_jit")))
    cache.mkdir(parents=True, exist_ok=True)

    key = hashlib.sha256(f"{mod_name}|py{sys.version_info[:2]}|{src}".encode()).hexdigest()[:16]

    build_root = cache / f"{mod_name}_{key}"
    build_root.mkdir(parents=True, exist_ok=True)

    so_path = build_root / f"{mod_name}.so"
    lock_path = build_root / ".build.lock"

    # Fast path
    if so_path.exists():
        return _load_ext(mod_name, so_path)

    # Process-unique tag for temp dirs
    host = socket.gethostname()
    pid = os.getpid()
    rank = get_rank()
    uniq = f"{host}_rank{rank}_pid{pid}"

    gen_dir = build_root / f"gen_{uniq}"
    bdir = build_root / f"build_{uniq}"
    gen_dir.mkdir(exist_ok=True)
    bdir.mkdir(exist_ok=True)

    generated = gen_dir / f"{mod_prefix}_bindings.cu"
    generated.write_text(src)

    root = Path(__file__).resolve().parent.parent
    csrc_dir = root / "csrc"
    cmake_source_dir = root / "flashmoe"  # because root CMakeLists.txt lives here

    def _try_acquire_lock() -> bool:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(f"host={host}\npid={pid}\nrank={rank}\ntime={time.time()}\n")
            return True
        except FileExistsError:
            return False

    def _release_lock() -> None:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    def _wait_for_artifact(timeout_s: float = 1800.0, poll_s: float = 0.1):
        start = time.time()
        while True:
            if so_path.exists():
                return _load_ext(mod_name, so_path)

            if time.time() - start > timeout_s:
                raise TimeoutError(
                    f"Timed out waiting for JIT artifact {so_path} while another process was building it."
                )

            time.sleep(poll_s)

    # Try to become the builder
    have_lock = _try_acquire_lock()

    if not have_lock:
        # Another process is building. Wait for the final .so to appear.
        return _wait_for_artifact()

    try:
        # Double-check after lock acquisition in case another process finished just before us
        if so_path.exists():
            return _load_ext(mod_name, so_path)

        subprocess.run([
            "cmake", "-S", str(cmake_source_dir), "-B", str(bdir), "-G", "Ninja",
            f"-DGENERATED_SRC={generated}",
            f"-DFLASHMOE_KERNELS_SOURCE={csrc_dir}",
            f"-DTARGET_MODULE_NAME={mod_name}",
            f"-DCMAKE_CUDA_ARCHITECTURES={arg.gpu_arch // 10}",
            f"-DCPM_SOURCE_CACHE={Path.home() / '.cache' / 'cpm'}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DARCH={arg.gpu_arch // 10}"
        ], check=True)

        subprocess.run([
            "cmake", "--build", str(bdir), "--parallel"
        ], check=True)

        built = next(bdir.glob(mod_name + "*.so"))

        # Copy into a temp path in build_root, then atomically replace final path
        tmp_so = build_root / f".{mod_name}.{uniq}.tmp.so"
        shutil.copy2(built, tmp_so)
        tmp_so.replace(so_path)

    finally:
        _release_lock()

    return _load_ext(mod_name, so_path)