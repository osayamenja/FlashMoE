# FlashMoE Python/CuTe DSL Port

This directory adds a Python port under `flashmoe_cutedsl`.

The upstream CUDA implementation fuses routing, dispatch, expert compute, and
combine into one persistent kernel with an in-kernel scheduler.  The Python port
keeps the same dataflow but moves scheduling to Python:

1. `route_topk` computes softmax + top-k and emits expert-major token tables.
2. NVSHMEM4Py symmetric tensors hold dispatch, metadata, return, and signal-like
   count buffers.
3. Host NVSHMEM `put` operations move token payloads to owner PEs.
4. Each PE runs its local experts using either the torch packed baseline or the
   Python/CuTe DSL expert MLP kernels.
5. Owner PEs put results back to origin PEs, which combine by `index_copy_` or
   `index_add_`.

CuTe DSL hooks live in `flashmoe_cutedsl/kernels.py`.  The implemented kernels
are the expert-major token gather, fused gated up/value projection, down
projection, and top-1 combine scatter.  Enable them with
`FLASHMOE_CUTEDSL_GATHER=1`, `FLASHMOE_CUTEDSL_MLP=1`,
`FLASHMOE_CUTEDSL_COMBINE=1`, or the matching `initialize(...)` keyword
arguments.  The first expert MLP kernel is intentionally scalar-per-output for
validation; the next optimization step is replacing those loops with CuTe
MMA/tensor-core tiling.

The default fast path packs routed tokens into `[experts, capacity, hidden]` and
uses batched tensor-core GEMMs for expert compute.  Disable it with
`initialize(..., use_packed=False)` to compare against the scalar reference.

Example:

```bash
torchrun --nproc_per_node=8 examples/cutedsl_nvshmem_port_quickstart.py
```

For a single-process correctness smoke run:

```bash
python examples/cutedsl_nvshmem_port_quickstart.py --no-nvshmem
```

Single-rank benchmark:

```bash
python examples/benchmark_cutedsl_port.py --tokens 1024 --hidden 5120 --ffn 8192 --experts 16 --top-k 1
```

Top-1 can also be measured with CuTe gather/combine and CUDA Graph replay:

```bash
python examples/benchmark_cutedsl_port.py --tokens 1024 --hidden 5120 --ffn 8192 --experts 16 --top-k 1 --cutedsl-gather --cutedsl-combine --cuda-graph
```

To force the expert compute itself through Python/CuTe DSL kernels:

```bash
FLASHMOE_CUTEDSL_STRICT=1 python examples/benchmark_cutedsl_port.py --tokens 256 --hidden 1024 --ffn 2048 --experts 8 --top-k 1 --cutedsl-gather --cutedsl-mlp --cutedsl-combine
```
