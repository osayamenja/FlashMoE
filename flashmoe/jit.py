from __future__ import annotations

from pathlib import Path
from enum import IntEnum

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
    topo: Topology
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
                 topo: Topology,
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
        self.topo = topo
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

def _module_name(arg: InitArgs, mod_prefix: str) -> str:
    return (f"{mod_prefix}_s{arg.tokens_per_rank}_h{arg.token_dim}_i{arg.ffn_size}"
            f"_e{arg.num_experts}_ec{arg.expert_peer_capacity}_k{arg.top_k}"
            f"_topo{arg.topo}_mt{arg.mlp_type}_dt{arg.data_type}_act{arg.act_type}_sm{arg.gpu_arch}")

def _load_ext(mod_name: str, so_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(mod_name, so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {mod_name} from {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _source_fingerprint() -> str:
    return "v010"

def _get_compiled(arg: InitArgs, src: str, mod_prefix:str, mod_name: str):
    import os, sys, hashlib, subprocess
    _verify_dirs()
    cache = Path(os.environ.get("FLASHMOE_CACHE_DIR", str(Path.home() / ".cache" / "flashmoe_jit")))
    cache.mkdir(parents=True, exist_ok=True)

    fp = _source_fingerprint()

    key = hashlib.sha256(f"{mod_name}|py{sys.version_info[:2]}|{fp}".encode()).hexdigest()[:16]
    build_root = cache / f"{mod_name}_{key}"
    so_path = build_root / (mod_name + ".so")

    if so_path.exists():
        return _load_ext(mod_name, so_path)

    build_root.mkdir(parents=True, exist_ok=True)
    gen_dir = build_root / "gen"
    gen_dir.mkdir(exist_ok=True)

    generated = gen_dir / "{}_bindings.cu".format(mod_prefix)
    generated.write_text(src)

    bdir = build_root / "build"
    bdir.mkdir(exist_ok=True)
    root = Path(__file__).resolve().parent.parent
    csrc_dir = root / "csrc"
    cmake_source_dir = root / "flashmoe"  # because root CMakeLists.txt lives here

    subprocess.run([
        "cmake", "-S", str(cmake_source_dir), "-B", str(bdir), "-G", "Ninja",
        f"-DGENERATED_SRC={generated}",
        f"-DFLASHMOE_KERNELS_SOURCE={csrc_dir}",
        f"-DTARGET_MODULE_NAME={mod_name}",
        f"-DCMAKE_CUDA_ARCHITECTURES={arg.gpu_arch // 10}",
        f"-DCPM_SOURCE_CACHE={Path.home()/'.cache'/'cpm'}",
        f"-DCMAKE_BUILD_TYPE=Release",
        f"-DARCH={arg.gpu_arch}"
    ], check=True)

    subprocess.run(["cmake", "--build", str(bdir), "--parallel"], check=True)

    built = next(bdir.glob(mod_name + "*.so"))
    built.replace(so_path)

    return _load_ext(mod_name, so_path)