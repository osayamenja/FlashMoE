# FlashMoE: Fast Distributed MoE in a Single Kernel [NeurIPS'25]
FlashMoE is the first fully fused Distributed MoE system that achieves high tensor core utilization
by eliminating kernel boundaries and enabling fine-grained overlap of communication and computation.
We provide high-performance single- and multi-node EP inference 
and work seamlessly with CUDA graphs. See paper [here](https://arxiv.org/abs/2506.04667).

## Table of Contents
1. [Motivation](#problem-moe-bottlenecks-in-inference)
2. [Our Solution](#our-solution-complete-kernel-fusion)
3. [Installation](#installation)
4. [QuickStart](#-python-quickstart)
5. [Performance Results](#-performance-results)
6. [Running Benchmarks](#run-benchmark-c)

## Problem: MoE Bottlenecks in Inference

<table>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/osayamenja/FlashMoE/main/assets/FlashMoE_motivation.png" alt="Opportunity" width="800"/><br>
      <em>Figure 1: Opportunity. MoE takes 67%-95% of inference runtime.</em>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/osayamenja/FlashMoE/main/assets/FlashMoE_tensor_core_idle_time.png" alt="Tensor core utilization" width="600"/><br>
      <em>Figure 2: Tensor core Utilization. y-axis is percentage of MoE runtime that tensor cores are inactive.</em>
    </td>
  </tr>
</table>

Distributed Mixture-of-Experts (DMoE) is an extremely demanding workload, both compute- and communication-intensive,
accounting for up to **95% of total inference runtime** (Figure 1). 

This makes DMoE the primary bottleneck in distributed inference and a critical target for optimization.

However, existing implementations leave significant performance untapped, achieving only **26% tensor core utilization** (Figure 2).

We identify three key sources of inefficiency:

1. **Exposed communication on the critical path**  
2. **Straggler-induced delays** from load imbalance  
3. **System overheads** from dynamic token routing (e.g., metadata management, inputs preprocessing for compute operators like GroupedGEMM)

As a result, GPUs spend the majority of time stalled, with only **26% of runtime utilizing tensor cores**.

## Our Solution: Complete Kernel Fusion

<div align="center">
  <img src="https://raw.githubusercontent.com/osayamenja/FlashMoE/main/assets/FlashMoE_Arch_title.png" width="700" alt="">
<p><em>Figure 3: FlashMoE Architecture</em></p>
</div>

We address these inefficiencies through **complete kernel fusion**, enabling:

1. **Fine-grained overlap of communication and computation** at tile granularity  
2. **Latency hiding of preprocessing and system overheads** via SM specialization  
3. **Exploitation of task locality at scale**, allowing SMs to execute ready tasks out-of-order, minimizing tensor core idle time and boosting SM utilization.

In contrast, existing implementations rely on tens to hundreds of serialized kernels, enforcing strict execution order
and limiting _task locality_.

This results in unnecessary stalls—for example, during collective synchronization (AllGather, ReduceScatter, AllToAll),
where GPUs idle waiting for stragglers instead of executing independent compute tasks.

## Our Work
We present **FlashMoE** (Figure 3), the first **fully fused Distributed MoE system**.

FlashMoE is a high-throughput, portable system that fuses:
- MoE Dispatch  
- Expert Computation (Gated MLP or standard MLP)  
- MoE Combine  

into a **single tile-pipelined persistent kernel**.

At its core, FlashMoE embeds an **Operating System within the kernel**, enabling concurrent scheduling and execution,
thereby hiding system and communication latency. 

FlashMoE is built from the ground up in **CUDA C++**, with selective inline PTX.
It leverages:

- [cuBLASDx](https://docs.nvidia.com/cuda/cublasdx/) for device-side high-performance compute  
- [NVSHMEM](https://developer.nvidia.com/nvshmem) for asynchronous, device-initiated communication  
- [CCCL](https://github.com/nvidia/cccl) and [CUTLASS](https://github.com/NVIDIA/cutlass) for critical infrastructure

### 🏎️ Portability

We support 
- SM70 and above GPUs, including consumer Blackwell (SM120/SM121: RTX 50 series, RTX PRO 6000 Blackwell, DGX Spark). Boosting compute performance for Hopper and Blackwell is on the roadmap.
- NVLink and multi-node RDMA (EFA, IBGDA, libfabric as NVSHMEM [supports](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html#hardware-requirements)).
- FP16, BF16, FP32 (TF32) and FP64, plus block-scaled [MXFP8, MXFP4 and NVFP4](#quantization-sm120sm121) on SM120/SM121. Other low-precision paths (FP8 and FP4 on Hopper and SM100) are on the roadmap (we welcome contributions!)

## Requirements
- CUDA toolkit
- C++20
- ninja (`sudo apt install ninja-build`)
- CMake (>= 3.28)

### Hardware Requirements
- GPU architecture of at least SM 70. 
- A P2P GPU interconnect (NVLink, some PCIe and GPUDirect RDMA). NVSHMEM will fail if this criterion is not met.

## Installation
### cuBLASDx
- Download from [here](https://developer.nvidia.com/cublasdx-downloads) and save in `<your_directory>`, e.g `~/.local`.

### NVSHMEM
- Install as directed [here](https://developer.nvidia.com/nvshmem-downloads).

### Env Variables
```shell
export NVSHMEM_LIB_HOME=/usr/lib/x86_64-linux-gnu/nvshmem/<12 or 13>. #Do confirm this directory exists!
export MATHDX_ROOT=<your_directory>/nvidia-<...>/mathdx/yy.mm/
export CMAKE_PREFIX_PATH=$NVSHMEM_LIB_HOME:$MATHDX_ROOT:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=$NVSHMEM_LIB_HOME:$LD_LIBRARY_PATH
```

> 👉 Tip: add the above exports to your `.bashrc`

## 🚀 Python QuickStart
```bash
pip install flashmoe-py[cu12] # or cu13
```
## Python API Showcase
See `quickstart.py` for a complete example, the below is just a showcase.
```python
import flashmoe

if __name__ == "__main__":
    # Llama4-Scout-17B-16E shapes
    # model description which flashmoe.initialize uses to JIT compile the kernel
    tokens_per_rank = 1024
    token_dim = 5120
    ffn_size = 8192
    num_experts = 16
    k = 1
    mlp_type = flashmoe.MLPType.GATED # Gated MLP
    data_type = flashmoe.DataType.BF16
    act_type = flashmoe.ActivationType.SILU
    
    init_args = flashmoe.InitArgs(...)
    
    flash_handle = flashmoe.initialize(init_args)
    router_handle = flashmoe.router.initialize(init_args)

    router_forward_args = ...
    # single kernel for GEMM + Softmax + topk selection
    flashmoe.router.forward(router_handle, flash_handle, router_forward_args)
    
    flashmoe_forward_args = ...
    # single kernel for Dispatch + Experts + Combine
    flashmoe.forward(flash_handle, flashmoe_forward_args)
    
    # call finalize
    flashmoe.finalize(flash_handle)
    flashmoe.router.finalize(router_handle)
```
## Running a Python Program
We suggest running these to verify that you meet all installation requirements.

## Single-Node
### Torchrun

```shell
torchrun --nproc_per_node=<number of GPUs> quickstart.py --torch-init
```

### MPI
```shell
pip install mpi4py
mpirun -n <number of GPUs> python3 quickstart.py
```

### Multi-node
Getting this to work would be dependent on the launcher in your cluster. 
Below, we suggest some launch recipes. Use what works for you.
```shell
# SLURM with libfabric (tested)
export NVSHMEM_REMOTE_TRANSPORT=libfabric
export NVSHMEM_LIBFABRIC_PROVIDER=... # efa,cxi,or verbs
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_BOOTSTRAP=MPI
srun -N <number of nodes> -n <total number of gpus> \
    --ntasks-per-node=<gpus per node> --gpus-per-task=1 --gpu-bind=closest python3 quickstart.py
```
```shell 
# torchrun with Connect-x NICs (not tested)
export NVSHMEM_IB_ENABLE_IBGDA=true
torchrun \
    --nproc_per_node=<number of GPUs> \
    --nnodes<...> \
    --rdzv_endpoint=<master address, like hostname of rank 0> \
    --rdzv_backend=c10d \
    --rdzv-id=<some id, like 123456789> \
    --node_rank=<...> python3 quickstart.py
```

## Use C++ API (header-only)
Add the following to your `CMakeLists.txt`
```CMake
set(CPM_SOURCE_CACHE
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cache"
        CACHE PATH "Shared CPM source cache"
)
set(CMAKE_CUDA_ARCHITECTURES "native") # or your own architecture

#...
CPMAddPackage(
  NAME flashmoe
  GITHUB_REPOSITORY osayamenja/flashmoe
  GIT_TAG v0.1.2
)

target_link_libraries(app PRIVATE flashmoe::flashmoe)

FlashMoESetRDC(app)
FlashMoEAddOptions(app)
```
and include the header file like below. See `csrc/tests/flashmoe.cu` for more usage details.
```cpp
#include <flashmoe/flashmoe.cuh>
```

## Quantization (SM120/SM121)
On RTX 50 series, RTX PRO 6000 Blackwell and DGX Spark GPUs, the expert GEMMs can run block-scaled, in
[MXFP8](#mxfp8), [MXFP4 or NVFP4](#fp4-sm120sm121): quantized weights and activations with one scale per 16 or 32
elements along K, FP32 accumulation.
The GEMMs use the block-scaled tensor core instructions of these GPUs (`mma.sync ... .block_scale`),
which, unlike BF16 with FP32 accumulation (about 230 TFLOPS on an RTX 5090), run at full rate:
the nominal dense peak is about 1014 TFLOPS for MXFP8 and about 2029 TFLOPS for FP4.
All formats share the data path:
- Tokens are quantized as they are dispatched, so a dispatched token takes fewer bytes. Scale factors included, a token of
  H elements takes 2H bytes in BF16, about H + H/32 in MXFP8, H/2 + H/32 in MXFP4 and H/2 + H/16 in NVFP4,
  i.e., 1.9x, 3.8x and 3.6x fewer bytes (arithmetic on the sizes, not a measured speedup).
- The GEMM0 epilogue quantizes the activations for GEMM1.
- GEMM1 writes BF16, so combine and the output are unchanged.
- Tokens, biases and the output stay BF16. H and I must be multiples of 128.

### MXFP8
E4M3 weights and activations with one UE8M0 scale per 32 elements along K
(`kind::mxf8f6f4.block_scale`).
Quantize the expert weights once with `flashmoe.quant.quantize_mxfp8`.
Block FP8 checkpoints (128x128 scales, e.g. DeepSeek-V3) can be converted with `flashmoe.quant.blockfp8_to_mxfp8`.
```python
init_args = flashmoe.InitArgs(..., data_type=flashmoe.DataType.BF16, quant=flashmoe.QuantType.MXFP8)
up_q, up_sf = flashmoe.quant.quantize_mxfp8(local_expert_up)          # [num_local_experts, I, H]
up_v_q, up_v_sf = flashmoe.quant.quantize_mxfp8(local_expert_up_v)    # [num_local_experts, I, H]
down_q, down_sf = flashmoe.quant.quantize_mxfp8(local_expert_down)    # [num_local_experts, H, I]
args = flashmoe.ForwardArgs(...,
                            local_expert_up=up_q.data_ptr(), local_expert_up_sf=up_sf.data_ptr(),
                            local_expert_up_v=up_v_q.data_ptr(), local_expert_up_v_sf=up_v_sf.data_ptr(),
                            local_expert_down=down_q.data_ptr(), local_expert_down_sf=down_sf.data_ptr())
```
`quickstart.py --quant mxfp8` (or `--fp8`) is a complete example. In C++, set `QuantRecipe::mxfp8` in `moe::MoEConfig`, and see
`csrc/tests/flashmoe.cu` (target `testFlashMoEFP8`) and `csrc/tests/gemmMXFP8.cu`.

Single RTX 5090, gated SiLU MLP, latency of one MoE layer in ms (`testFlashMoE` vs. `testFlashMoEFP8`):

| Model (H, I, E, k) | Tokens | BF16 | MXFP8 | Speedup |
|---|---|---|---|---|
| Llama4-Scout (5120, 8192, 16, 1) | 8192 | 12.50 | 4.38 | 2.86x |
| Mixtral-8x7B (4096, 14336, 8, 2) | 8192 | 34.43 | 12.20 | 2.82x |
| DeepSeek-V2-Lite (2048, 1408, 64, 6) | 4096 | 3.28 | 1.49 | 2.20x |
| Qwen3-30B-A3B (2048, 768, 128, 8) | 4096 | 3.00 | 1.59 | 1.89x |

On random data, the MXFP8 output deviates from the BF16 output by about 6% (relative Frobenius norm).
The kernel itself matches a reference that applies the same quantization with FP32 GEMMs.

### FP4 (SM120/SM121)
The FP4 tensor core instruction (`mma.sync ... kind::mxf4nvf4.block_scale`, m16n8k64) has twice the nominal peak rate of
the MXFP8 one, and FP4 weights and tokens take about half the bytes of MXFP8 (scale factors included).
Two formats share one tile GEMM, one data path and one set of tests.
Both store E2M1 elements, two per byte along K (the even element in the low nibble), rounded to nearest even:

| | MXFP4 | NVFP4 |
|---|---|---|
| Scale per block | UE8M0 (a power of two) per 32 elements along K | UE4M3 per 16 elements along K |
| Block scale | `2^ceil(log2(amax / 6))`, blocks never saturate | `e4m3(amax_block / (6 * gs))` |
| Tensor scale | none | one FP32 `gs = amax / (6 * 448)` per expert weight matrix |
| Activation scales | dynamic, per block | dynamic per block, plus two static per-layer scales |
| Python | `QuantType.MXFP4`, `quantize_mxfp4` | `QuantType.NVFP4`, `quantize_nvfp4` |
| C++ | `QuantRecipe::mxfp4` | `QuantRecipe::nvfp4` |

A dequantized NVFP4 element is `e2m1 * float(sf) * gs`.
NVFP4 has smaller blocks and non-power-of-two scales, MXFP4 needs no calibration.
Quantizing a random matrix gives a relative Frobenius error of 9.5% (NVFP4) and 11.5% (MXFP4) for Gaussian data,
and 10.2% and 9.9% for data uniform in [-1, 1].

Quantize the expert weights once with `flashmoe.quant.quantize_mxfp4` or `flashmoe.quant.quantize_nvfp4`.
`quantize_mxfp4` returns `(q, sf)` like `quantize_mxfp8` (`q` is `uint8 [num_local_experts, N, K / 2]`),
`quantize_nvfp4` returns `(q, sf, gs)` with `gs` a float32 tensor `[num_local_experts]`, the tensor scale of each expert matrix.
```python
init_args = flashmoe.InitArgs(..., data_type=flashmoe.DataType.BF16, quant=flashmoe.QuantType.NVFP4)  # or MXFP4
up_q, up_sf, up_gs = flashmoe.quant.quantize_nvfp4(local_expert_up)          # [num_local_experts, I, H]
up_v_q, up_v_sf, up_v_gs = flashmoe.quant.quantize_nvfp4(local_expert_up_v)  # [num_local_experts, I, H]
down_q, down_sf, down_gs = flashmoe.quant.quantize_nvfp4(local_expert_down)  # [num_local_experts, H, I]
args = flashmoe.ForwardArgs(...,
    local_expert_up=up_q.data_ptr(), local_expert_up_sf=up_sf.data_ptr(), local_expert_up_gs=up_gs.data_ptr(),
    local_expert_up_v=up_v_q.data_ptr(), local_expert_up_v_sf=up_v_sf.data_ptr(), local_expert_up_v_gs=up_v_gs.data_ptr(),
    local_expert_down=down_q.data_ptr(), local_expert_down_sf=down_sf.data_ptr(), local_expert_down_gs=down_gs.data_ptr(),
    input_scale=input_scale, interim_scale=interim_scale)
```
For MXFP4, pass only the `*_sf` pointers: it has no `*_gs`, `input_scale` and `interim_scale`.

**NVFP4 activation scales.**
The fused kernel cannot reduce a tensor-wide amax at run time, so NVFP4 activations use two static per-layer scales
that you calibrate once on representative data (NVFP4 checkpoints do the same, e.g., ModelOpt stores a static
`input_scale` per layer):
`input_scale` for the dispatched tokens and `interim_scale` for the output of GEMM0 (the input of GEMM1),
both `flashmoe.quant.nvfp4_scale(amax)` (= `amax / (6 * 448)`).
```python
# x: calibration tokens [S, H], expert_up, expert_up_v: unquantized weights [E, I, H], biases [E, I]
input_scale = flashmoe.quant.nvfp4_scale(x.abs().max()).item()
y_amax = 0.0
for e in range(num_experts):  # y: output of GEMM0 for the tokens x, ideally only those routed to expert e
    y = swish_alpha * torch.nn.functional.silu(swish_beta * (x @ expert_up[e].T + bias_up[e])) \
        * (x @ expert_up_v[e].T + bias_up_v[e])
    y_amax = max(y_amax, y.abs().max().item())
interim_scale = flashmoe.quant.nvfp4_scale(y_amax).item()
```
A scale that is too small clips (block scales saturate at 448), a scale that is too large only costs some resolution,
so take the maximum over the calibration set.
Data with a very wide dynamic range is better served by MXFP4: with one tensor scale, NVFP4 blocks far below the
tensor's amax flush to zero.
With expert parallelism, use the same scales on every rank: the dispatch quantizes with the sender's `input_scale`.

`quickstart.py --quant mxfp4` and `--quant nvfp4` are complete examples (`nvfp4` calibrates both scales on its own data,
as above, using all tokens against every expert as an upper bound). They print the relative Frobenius error of the
output against the unquantized BF16 reference.
In C++, set `QuantRecipe::mxfp4` or `QuantRecipe::nvfp4` in `moe::MoEConfig`;
for NVFP4, `moe::KernelArgs` also takes the tensor scales `expert_up_gs`, `expert_up_v_gs`, `expert_down_gs`
and the static scales `input_scale`, `interim_scale`.
See `csrc/tests/flashmoe.cu` (targets `testFlashMoEMXFP4` and `testFlashMoENVFP4`, which take the same arguments as `testFlashMoE`)
and `csrc/tests/gemmFP4.cu` (target `gemmFP4`: `--check` tests the quantizer, the scale factor layout against CUTLASS and the tile GEMM
against FP32 references, with cuBLASLt NVFP4 on the same data as a cross-check; `--bench` compares the tile GEMM with the
MXFP8 tile and with cuBLASLt).
The tile GEMM is `FP4Mainloop` in `csrc/include/flashmoe/tile_sm120_fp4.cuh`, the quantizer is `csrc/include/flashmoe/infra/fp4.cuh`.

Single RTX 5090, gated SiLU MLP, latency of one MoE layer in ms
(`testFlashMoE`, `testFlashMoEFP8`, `testFlashMoEMXFP4`, `testFlashMoENVFP4`):

| Model (H, I, E, k) | Tokens | BF16 | MXFP8 | MXFP4 | NVFP4 | MXFP4 / NVFP4 vs. BF16 | MXFP4 / NVFP4 vs. MXFP8 |
|---|---|---|---|---|---|---|---|
| Llama4-Scout (5120, 8192, 16, 1) | 8192 | 12.65 | 4.54 | 2.33 | 2.47 | 5.43x / 5.13x | 1.95x / 1.84x |
| Mixtral-8x7B (4096, 14336, 8, 2) | 8192 | 35.52 | 13.14 | 6.36 | 6.87 | 5.59x / 5.17x | 2.07x / 1.91x |
| DeepSeek-V2-Lite (2048, 1408, 64, 6) | 4096 | 3.33 | 1.50 | 1.11 | 1.22 | 3.01x / 2.74x | 1.35x / 1.23x |
| Qwen3-30B-A3B (2048, 768, 128, 8) | 4096 | 3.01 | 1.59 | 1.38 | 1.50 | 2.18x / 2.01x | 1.15x / 1.07x |

The two large models gain about 1.8-2.1x over MXFP8. The smaller models gain less (1.07-1.35x), and the GEMMs are not the
only cost there.
BF16 and MXFP8 were measured again together with the FP4 columns, so they differ from the MXFP8 table above by up to 8%.

The kernels match a reference that applies the same quantization and runs FP32 GEMMs: at most 0.17% of the output
elements are outside `isclose` (rtol 8e-2, atol 8e-3), the `error(%)` column of the tests.
Against BF16 the picture is different. On uniform random data, the output of the whole layer deviates from the BF16
output by about 26% (MXFP4) and 25% (NVFP4), against about 6% for MXFP8
(relative Frobenius norm, as reported by the tests for the quantized reference and by `quickstart.py` for the kernel
output; it depends on the random data of a run).
The tokens, both weight matrices of the MLP and the output of GEMM0 are all quantized to 4 bits, and a single quantized
matrix already deviates by about 10% (see above), so with four quantization steps in a row on random data, FP4 is far
less accurate than MXFP8.
The accuracy on real models is not evaluated here, check it on yours; NVFP4 activations additionally depend on the
calibrated static scales.

The tile GEMM (`gemmFP4 --bench`, TFLOPS on an RTX 5090). Our columns include the epilogue; the gated shape is the fused GEMM0,
counted as 4MNK flops and compared with two cuBLASLt GEMMs; cuBLASLt is the GEMM alone and has no MXFP4 algorithm on SM120:

| M x N x K | MXFP8 | MXFP4 | NVFP4 | cuBLASLt NVFP4 | cuBLASLt MXFP8 |
|---|---|---|---|---|---|
| 8192 x 8192 x 8192 | 524-528 | 1120 | 1050 | 1282 | 653 (603 in the NVFP4 run) |
| 8192 x 8192 x 5120, gated | 541-543 | 1191 | 1170 | 1147 | 580 (578 in the NVFP4 run) |
| 4096 x 2048 x 768 | 288-315 (noisy) | 484 | 395 | 735 | 420-458 |

The FP4 tile is about 2x the MXFP8 tile at large sizes, and on the gated shape NVFP4 matches cuBLASLt;
at 8192^3 it reaches 0.82x of cuBLASLt NVFP4, and the small shape, which is only 6 k-tiles deep and whose timing varies
by up to about 20% between runs, is further behind (0.54x).

**Limits.**
- SM120/SM121 only, with BF16 tokens, biases and output. H and I must be multiples of 128, and bM = 128.
- NVFP4 activations use the static scales above, which must be calibrated.
- Tested on a single GPU. The RDMA path, which `nvshmem_putmem`s the scale factors before the signaled data put, is compile-checked only.

---

### ✅ Roadmap
- [ ] Improve MMA for Hopper (WGMMA) and Blackwell (UTCMMA).
- [x] FP8: MXFP8 on SM120/SM121 (Hopper and SM100 to follow)
- [x] FP4: MXFP4 and NVFP4 on SM120/SM121 (SM100 to follow)
- [ ] Shared experts
- [ ] AMD support
- [ ] Backward pass

---

## 📊 Performance Results
- We measure with the EP+DP parallelism scheme.
- We compare against: 
  - [COMET](https://github.com/bytedance/flux) (MLSys '25) at commit: 19831ca2d820e3e782ed1d15d8b52d0898b78b26
  - [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) at v0.15.3
  - [Triton-Distributed](https://github.com/ByteDance-Seed/Triton-distributed) at commit: 3644f0586d14591c8a43a10aa6b47fe98a95aea7 
- We measure a single layer's execution only. 
- For every model we evaluated, 
we use model shapes and data types as defined in its corresponding `config.json` on HuggingFace. 
- We **do not** execute any shared experts.
> 👉 On frontier MoE models, FlashMoE gives up to 5x speedup and 69% increase in tensor core utilization compared to SOTA baselines.

## Gated MLP

<div align="center">
  <img src="https://raw.githubusercontent.com/osayamenja/FlashMoE/main/assets/FlashMoE_A100_single_node-2.png" width="4101" alt="">
<p><em>Figure 4: Up to 5.1x faster MoE layer runtime on Qwen-30B with single-node EP</em></p>
</div>

---

## Conventional MLP
<div align="center">
  <img src="https://raw.githubusercontent.com/osayamenja/FlashMoE/main/assets/FlashMoE_A100_vs_COMET.png" width="2946" alt="">
<p><em>Figure 5: Up to 2.6x faster runtime DeepSeek-V2-Lite</em></p>
</div>

---

## Multi-node (libfabric on Slingshot 11)
<div align="center">
  <img src="https://raw.githubusercontent.com/osayamenja/FlashMoE/main/assets/FlashMoE_A100_multi_node.png" width="5592" alt="">
<p><em>Figure 6: Up to 3x speedup on Llama4-Scout for multi-node EP!</em></p>
</div>

--- 

## H100s
<div align="center">
  <img src="https://raw.githubusercontent.com/osayamenja/FlashMoE/main/assets/FlashMoE_H100_single_node.png" width="2940" alt="">
<p><em>Figure 7: Up to 2.5x speedup on H100s.</em></p>
</div>

---

## Run Benchmark (C++)
```shell
cd csrc
mkdir cmake-build-release && cd cmake-build-release
cmake -DCMAKE_BUILD_TYPE=Release -Wno-dev -G Ninja -S.. -B.
cmake --build . --target testFlashMoE --parallel
export NVSHMEM_BOOTSTRAP=MPI
mpirun -n <world> ./testFlashMoE <num tokens per rank> <token dim> <ffn dim> <num experts total> <top k>
```
`testFlashMoEFP8`, `testFlashMoEMXFP4` and `testFlashMoENVFP4` take the same arguments and run the MXFP8, MXFP4 and NVFP4 paths
(SM120/SM121; so far tested on a single GPU).


## IDEs
The codebase integrates well with CLion: open the project at `csrc`.

## Contributions
We welcome them! Submit a PR!

## Acknowledgements
Super grateful to the amazing folks behind
- cuBLASDx 
- CUTLASS
- NVSHMEM
- CCCL

This work would not have been possible without the critical building blocks they provide.

# 📖 Citation
If you can, please cite as below:
```bibtex
@inproceedings{NEURIPS2025_918d938b,
 author = {Aimuyo, Osayamen and Oh, Byungsoo and Singh, Rachee},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {D. Belgrave and C. Zhang and H. Lin and R. Pascanu and P. Koniusz and M. Ghassemi and N. Chen},
 pages = {100676--100699},
 publisher = {Curran Associates, Inc.},
 title = {FlashMoE: Fast Distributed MoE in a Single Kernel},
 url = {https://proceedings.neurips.cc/paper_files/paper/2025/file/918d938bd209e5b56072777366f8a211-Paper-Conference.pdf},
 volume = {38},
 year = {2025}
}
```
