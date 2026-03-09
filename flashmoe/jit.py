from __future__ import annotations
import os, sys, hashlib, subprocess, importlib.util
from pathlib import Path
from string import Template

class InitArgs:
    from typing import Sequence
    tokens_per_rank: int
    token_dim: int
    ffn_size: int
    num_experts: int
    top_k: int
    expert_map: Sequence[int]
    rank_map: Sequence[int]
    gpu_arch: int
    ep_world: int
    ep_rank: int
    my_pe: int
    num_local_experts: int
    stream_ptr: int
    expert_peer_capacity: int

    def __init__(self,
                 tokens_per_rank: int,
                 token_dim: int,
                 ffn_size: int,
                 num_experts: int,
                 top_k: int,
                 expert_map: Sequence[int],
                 rank_map: Sequence[int],
                 gpu_arch: int,
                 ep_world: int,
                 ep_rank: int,
                 my_pe: int,
                 num_local_experts: int,
                 stream_ptr: int,
                 expert_peer_capacity: int = -1) -> None:
        from math import ceil
        assert gpu_arch >= 700
        self.tokens_per_rank = tokens_per_rank
        self.token_dim = token_dim
        self.ffn_size = ffn_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_map = expert_map
        self.rank_map = rank_map
        self.gpu_arch = gpu_arch
        self.ep_world = ep_world
        self.ep_rank = ep_rank
        self.my_pe = my_pe
        self.num_local_experts = num_local_experts
        self.stream_ptr = stream_ptr
        if expert_peer_capacity < 0:
            self.expert_peer_capacity = ceil(float(tokens_per_rank) / num_experts) * top_k
        else:
            self.expert_peer_capacity = expert_peer_capacity

# Base case
ROOT = Path(__file__).resolve().parent.parent
CSRC_INCLUDE = ROOT / "csrc" / "include"
CMAKE_SOURCE_DIR = ROOT / "flashmoe"  # because root CMakeLists.txt lives here

if not (CMAKE_SOURCE_DIR / "CMakeLists.txt").exists():
    raise RuntimeError("JIT CMakeLists.txt not found at package root")
if not CSRC_INCLUDE.exists():
    raise RuntimeError(f"Missing include dir: {CSRC_INCLUDE}")

def _cache_dir() -> Path:
    return Path(os.environ.get("FLASHMOE_CACHE_DIR", str(Path.home() / ".cache" / "flashmoe_jit")))

def _module_name(arg: InitArgs) -> str:
    return f"flashmoe_s{arg.tokens_per_rank}_h{arg.token_dim}_i{arg.ffn_size}_e{arg.num_experts}_k{arg.top_k}_sm{arg.gpu_arch}"

def _load_ext(mod_name: str, so_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {mod_name} from {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _source_fingerprint() -> str:
    return "v010"

def _get_compiled(arg: InitArgs):
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)

    mod_name = _module_name(arg)
    fp = _source_fingerprint()

    key = hashlib.sha256(f"{mod_name}|py{sys.version_info[:2]}|{fp}".encode()).hexdigest()[:16]
    build_root = cache / f"{mod_name}_{key}"
    so_path = build_root / (mod_name + ".so")

    if so_path.exists():
        return _load_ext(mod_name, so_path)

    build_root.mkdir(parents=True, exist_ok=True)
    gen_dir = build_root / "gen"
    gen_dir.mkdir(exist_ok=True)

    generated = gen_dir / "bindings.cu"
    src = Template(r"""
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <nvshmem.h>

// #include "flashmoe/flashmoe.cuh"

namespace py = pybind11;

static void moe_initialize(const size_t&
                                 const std::vector<int>& expertToEpRank,
                                 const std::vector<int>& epRankToGlobalRank,
                                 std::uintptr_t stream_ptr) {
    printf("Hello from inside initialize(sm_%d)\n", $arch);
    /*cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    constexpr int S = $s;
    constexpr int H = $h;
    constexpr int I = $i;
    constexpr int arch = $arch;

    Context ctx = initialize_templated<S, H, I>(
        EC,
        expertToEpRank.data(), (int)expertToEpRank.size(),
        epRankToGlobalRank.data(), (int)epRankToGlobalRank.size(),
        stream
    );

    auto* heap_ctx = new Context(ctx);
    return py::capsule(heap_ctx, "moe.Context",
        [](PyObject* cap) {
            void* p = PyCapsule_GetPointer(cap, "moe.Context");
            auto* c = reinterpret_cast<Context*>(p);
            // TODO: destroy/free GPU/NVSHMEM allocations
            delete c;
        }
    );*/
}

static void gate_initialize() {
    
}

static void py_forward(py::capsule ctx_cap, int top_k, std::uintptr_t stream_ptr) {
    /*cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    auto* ctx = reinterpret_cast<Context*>(ctx_cap.get_pointer("moe.Context"));

    constexpr int S = $s;
    constexpr int H = $h;
    constexpr int I = $i;
    
    // build MoE compile-time config here

    forward_templated<S, H, I>(ctx, top_k, stream);*/
}

PYBIND11_MODULE($mod_name, m) {
    m.def("moe_initialize", &moe_initialize,
          py::arg("EC"),
          py::arg("expertToEpRank"),
          py::arg("epRankToGlobalRank"),
          py::arg("stream_ptr"));
    m.def("forward", &py_forward,
          py::arg("context"),
          py::arg("top_k"),
          py::arg("stream_ptr"));
}
""")

    generated.write_text(src.substitute(
        arch=arg.gpu_arch,
        s=arg.tokens_per_rank,
        h=arg.token_dim,
        i=arg.ffn_size,
        mod_name=mod_name,
    ))

    bdir = build_root / "build"
    bdir.mkdir(exist_ok=True)
    csrc_dir = ROOT / "csrc"

    subprocess.run([
        "cmake", "-S", str(CMAKE_SOURCE_DIR), "-B", str(bdir), "-G", "Ninja",
        f"-DGENERATED_SRC={generated}",
        f"-DFLASHMOE_KERNELS_SOURCE={csrc_dir}",
        f"-DTARGET_MODULE_NAME={mod_name}",
        f"-DCMAKE_CUDA_ARCHITECTURES={arg.gpu_arch}",
        f"-DCPM_SOURCE_CACHE={Path.home()/'.cache'/'cpm'}",
        f"-DCMAKE_BUILD_TYPE=Release",
        f"-DARCH={arg.gpu_arch}"
    ], check=True)

    subprocess.run(["cmake", "--build", str(bdir), "--parallel"], check=True)

    built = next(bdir.glob(mod_name + "*.so"))
    built.replace(so_path)

    return _load_ext(mod_name, so_path)