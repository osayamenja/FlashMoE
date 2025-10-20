/**
 * Python bindings for FlashMoE CUDA kernels
 * This wraps your existing CUDA code to be callable from Python
 */
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>

// Include your existing headers
// #include "correctness.cuh"
#include "include/kleos/bootstrap.cuh"
#include "include/kleos/moe/moe.cuh"

namespace py = pybind11;

// Forward declare the Element type from your config
// This will be set based on your kleos_config.json at compile time
using Element = kleos::ACC::Element;


/**
 * Modified version of runOS() that accepts tensors from Python
 * instead of generating random data
 */
torch::Tensor moe_forward(
    torch::Tensor input,              // [local_batch, seq_len, hidden_size] - Activations
    torch::Tensor gate_weights,        // [hidden_size, num_experts] - Gate weights  
    torch::Tensor expert_weights,      // [local_experts, 2, intermediate_size, hidden_size] - Expert weights
    py::dict config                    // Configuration dict (from kleos_config.json)
) {
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(gate_weights.is_cuda(), "Gate weights must be CUDA tensor");
    TORCH_CHECK(expert_weights.is_cuda(), "Expert weights must be CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(gate_weights.is_contiguous(), "Gate weights must be contiguous");
    TORCH_CHECK(expert_weights.is_contiguous(), "Expert weights must be contiguous");
    
    // Extract config values
    int capacity_factor = config["capacity_factor"].cast<int>();
    int drop_tokens = config["drop_tokens"].cast<int>();
    int expert_top_k = config["expert_top_k"].cast<int>();
    int global_batch = config["global_batch"].cast<int>();
    int is_training = config["is_training"].cast<int>();
    int hidden_act = config["hidden_act"].cast<int>();
    int hidden_size = config["hidden_size"].cast<int>();
    int intermediate_size = config["intermediate_size"].cast<int>();
    int mini_batch = config["mini_batch"].cast<int>();
    int moe_frequency = config["moe_frequency"].cast<int>();
    int num_experts = config["num_experts"].cast<int>();
    int num_layers = config["num_layers"].cast<int>();
    int sequence_len = config["sequence_len"].cast<int>();
    int torch_dtype = config["torch_dtype"].cast<int>();
    int vocab_size = config["vocab_size"].cast<int>();
    
    // Initialize Kleos (NVSHMEM, etc.)
    kleos::initialize();
    const auto rank = kleos::getRank();
    
    // Get dimensions from your config (matching your constants)
    constexpr auto S = kleos::ACC::S::value;
    constexpr auto H = kleos::ACC::H::value;
    constexpr auto E = kleos::ACC::E::value;
    constexpr auto P = kleos::ACC::P::value;
    constexpr auto PX = kleos::ACC::PX::value;
    const auto nLx = kleos::hostBookkeeping.nLx;
    
    // Validate input dimensions match config
    TORCH_CHECK(input.size(0) * input.size(1) == S, 
                "Input batch*seq must equal S");
    TORCH_CHECK(input.size(2) == H, "Input hidden_size must equal H");
    TORCH_CHECK(gate_weights.size(0) == H && gate_weights.size(1) == E,
                "Gate weights shape mismatch");
    
    // Calculate memory layout (matching your runOS() layout)
    constexpr unsigned long aZ = S * H;              // Activations
    constexpr auto gwZ = aZ + PX * H;                // + Gate weights
    const auto bZ = gwZ + nLx * P * H;               // + Expert up weights
    const auto b2Z = bZ + nLx * P * H;               // + Expert down weights
    const auto dZ = b2Z + nLx * (P + H);             // + Bias
    const auto gZ = dZ + S * PX;                     // + Gate output
    const auto cZ = gZ + S * H;                      // + MoE output
    
    using Element = kleos::ACC::Element;
    
    // Allocate device memory for all data (matching your layout)
    cuda::std::byte* p;
    KLEOS_CHECK_CUDA(cudaMallocAsync(&p, cZ * sizeof(Element), kleos::kleosStream));
    KLEOS_CHECK_CUDA(cudaMemsetAsync(p, 0, cZ * sizeof(Element), kleos::kleosStream));
    
    // Copy input tensors to the allocated memory in the correct layout
    // Layout: [activations | gate_weights | expert_up | expert_down | bias | gate_output | moe_output]
    
    auto* __restrict__ dP = reinterpret_cast<Element*>(p);
    
    // Copy activations
    KLEOS_CHECK_CUDA(cudaMemcpyAsync(
        dP,                           // dest: start of buffer
        input.data_ptr(),             // src: input tensor
        aZ * sizeof(Element),
        cudaMemcpyDeviceToDevice,
        kleos::kleosStream
    ));
    
    // Copy gate weights
    KLEOS_CHECK_CUDA(cudaMemcpyAsync(
        dP + aZ,                      // dest: after activations
        gate_weights.data_ptr(),      // src: gate weights tensor
        (gwZ - aZ) * sizeof(Element), // size: E * H
        cudaMemcpyDeviceToDevice,
        kleos::kleosStream
    ));
    
    // Copy expert weights (assuming expert_weights is [nLx, 2, P, H])
    // where dimension 1 has [up_proj, down_proj]
    TORCH_CHECK(expert_weights.size(0) == nLx, "Expert count mismatch");
    TORCH_CHECK(expert_weights.size(1) == 2, "Expert weights must have up and down projections");
    
    for (uint i = 0; i < nLx; ++i) {
        // Copy up projection for expert i
        KLEOS_CHECK_CUDA(cudaMemcpyAsync(
            dP + gwZ + i * (P * H),
            expert_weights[i][0].data_ptr(),
            P * H * sizeof(Element),
            cudaMemcpyDeviceToDevice,
            kleos::kleosStream
        ));
        
        // Copy down projection for expert i  
        KLEOS_CHECK_CUDA(cudaMemcpyAsync(
            dP + bZ + i * (P * H),
            expert_weights[i][1].data_ptr(),
            P * H * sizeof(Element),
            cudaMemcpyDeviceToDevice,
            kleos::kleosStream
        ));
    }
    
    // Bias is already zeroed by cudaMemsetAsync above
    
    // Call your kernel (matching your runOS() call)
    float timed = 0;
    kleos::moe::forwardHostBench<32, 32>(p, p + dZ * sizeof(Element), timed);
    
    if (rank == 0) {
        printf("FlashMoE forward pass took %.2fms\n", timed);
    }
    
    KLEOS_CHECK_CUDA(cudaPeekAtLastError());
    
    // Extract output (MoE output is at offset gZ)
    auto output = torch::empty({input.size(0), input.size(1), H}, 
                               torch::TensorOptions()
                                   .dtype(input.dtype())
                                   .device(input.device()));
    
    KLEOS_CHECK_CUDA(cudaMemcpyAsync(
        output.data_ptr(),
        dP + gZ,                      // MoE output location
        S * H * sizeof(Element),
        cudaMemcpyDeviceToDevice,
        kleos::kleosStream
    ));
    
    // Synchronize and check for errors
    KLEOS_CHECK_CUDA(cudaStreamSynchronize(kleos::kleosStream));
    
    // Free allocated memory
    KLEOS_CHECK_CUDA(cudaFreeAsync(p, kleos::kleosStream));
    
    // Note: We don't call kleos::finalize() here because we may want to
    // call moe_forward multiple times. Finalize should be called when done.
    
    return output;
}


/**
 * Initialize NVSHMEM/Kleos
 * Call this once before first forward pass if needed
 */
void init_nvshmem() {
    kleos::initialize();
}


/**
 * Finalize NVSHMEM/Kleos  
 * Call this when completely done with all forward passes
 */
void finalize_nvshmem() {
    kleos::finalize();
}


/**
 * Benchmark function
 */
py::dict benchmark_moe(
    torch::Tensor input,
    torch::Tensor gate_weights,
    torch::Tensor expert_weights,
    py::dict config,
    int num_warmup = 10,
    int num_iterations = 100
) {
    // Warmup
    for (int i = 0; i < num_warmup; i++) {
        auto output = moe_forward(input, gate_weights, expert_weights, config);
    }
    
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        auto output = moe_forward(input, gate_weights, expert_weights, config);
    }
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    py::dict results;
    results["total_time_ms"] = milliseconds;
    results["avg_time_ms"] = milliseconds / num_iterations;
    results["num_iterations"] = num_iterations;
    
    return results;
}


/**
 * PyBind11 module definition
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMoE: Fast Distributed MoE in a Single Kernel";
    
    m.def("moe_forward", &moe_forward, 
          "MoE forward pass (inference only)",
          py::arg("input"),
          py::arg("gate_weights"),
          py::arg("expert_weights"),
          py::arg("config"));
    
    m.def("benchmark", &benchmark_moe,
          "Benchmark MoE forward pass",
          py::arg("input"),
          py::arg("gate_weights"),
          py::arg("expert_weights"),
          py::arg("config"),
          py::arg("num_warmup") = 10,
          py::arg("num_iterations") = 100);
    
    m.def("init_nvshmem", &init_nvshmem,
          "Initialize NVSHMEM/Kleos");
    
    m.def("finalize_nvshmem", &finalize_nvshmem,
          "Finalize NVSHMEM/Kleos");
}
