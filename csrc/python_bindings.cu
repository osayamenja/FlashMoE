/**
 * Python bindings for FlashMoE CUDA kernels
 * This wraps your existing CUDA code to be callable from Python
 */
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>

// Include your existing headers
#include "include/kleos/bootstrap.cuh"
#include "include/kleos/moe/moe.cuh"

namespace py = pybind11;

// Forward declare the Element type from your config
using Element = kleos::ACC::Element;


/**
 * Modified version of runOS() that accepts tensors from Python
 * Config is compile-time only - tensor shapes must match compiled config
 */
torch::Tensor moe_forward(
    torch::Tensor input,              // [batch, seq_len, hidden_size] - Activations
    torch::Tensor gate_weights,        // [hidden_size, num_experts] - Gate weights  
    torch::Tensor expert_weights       // [local_experts, 2, intermediate_size, hidden_size] - Expert weights
) {
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(gate_weights.is_cuda(), "Gate weights must be CUDA tensor");
    TORCH_CHECK(expert_weights.is_cuda(), "Expert weights must be CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(gate_weights.is_contiguous(), "Gate weights must be contiguous");
    TORCH_CHECK(expert_weights.is_contiguous(), "Expert weights must be contiguous");
    
    // Initialize Kleos (NVSHMEM, etc.)
    kleos::initialize();
    const auto rank = kleos::getRank();
    
    // Get dimensions from compile-time config
    constexpr auto S = kleos::ACC::S::value;
    constexpr auto H = kleos::ACC::H::value;
    constexpr auto E = kleos::ACC::E::value;
    constexpr auto P = kleos::ACC::P::value;
    constexpr auto PX = kleos::ACC::PX::value;
    const auto nLx = kleos::hostBookkeeping.nLx;
    
    // Validate input dimensions match compile-time config
    TORCH_CHECK(input.dim() == 3, "Input must be 3D [batch, seq, H]");
    TORCH_CHECK(input.size(0) * input.size(1) == S, 
                "Input batch*seq must equal compiled S=" + std::to_string(S) + 
                ". Got batch=" + std::to_string(input.size(0)) + 
                ", seq=" + std::to_string(input.size(1)) + 
                " (product=" + std::to_string(input.size(0) * input.size(1)) + ")");
    TORCH_CHECK(input.size(2) == H, 
                "Input hidden_size must equal compiled H=" + std::to_string(H) + 
                ". Got " + std::to_string(input.size(2)));
    TORCH_CHECK(gate_weights.size(0) == H && gate_weights.size(1) == E,
                "Gate weights must be [H=" + std::to_string(H) + 
                ", E=" + std::to_string(E) + "]. Got [" + 
                std::to_string(gate_weights.size(0)) + ", " + 
                std::to_string(gate_weights.size(1)) + "]");
    // TORCH_CHECK(expert_weights.size(0) == nLx, 
    //             "Expert count mismatch. Expected " + std::to_string(nLx) + 
    //             " local experts, got " + std::to_string(expert_weights.size(0)));
    TORCH_CHECK(expert_weights.size(1) == 2, 
                "Expert weights must have up and down projections [nLx, 2, P, H]");
    TORCH_CHECK(expert_weights.size(2) == P && expert_weights.size(3) == H,
                "Expert weights must be [*, 2, P=" + std::to_string(P) + 
                ", H=" + std::to_string(H) + "]. Got [*, 2, " + 
                std::to_string(expert_weights.size(2)) + ", " + 
                std::to_string(expert_weights.size(3)) + "]");
    
    // Calculate memory layout (matching your runOS() layout)
    constexpr unsigned long aZ = S * H;              // Activations
    constexpr auto gwZ = aZ + PX * H;                // + Gate weights
    const auto bZ = gwZ + nLx * P * H;               // + Expert up weights
    const auto b2Z = bZ + nLx * P * H;               // + Expert down weights
    const auto dZ = b2Z + nLx * (P + H);             // + Bias
    const auto gZ = dZ + S * PX;                     // + Gate output
    const auto cZ = gZ + S * H;                      // + MoE output
    
    // Allocate device memory for all data
    cuda::std::byte* p;
    KLEOS_CHECK_CUDA(cudaMallocAsync(&p, cZ * sizeof(Element), kleos::kleosStream));
    KLEOS_CHECK_CUDA(cudaMemsetAsync(p, 0, cZ * sizeof(Element), kleos::kleosStream));
    
    auto* __restrict__ dP = reinterpret_cast<Element*>(p);
    
    // Copy activations (flatten if 3D)
    // auto input_flat = input.reshape({S, H});
    KLEOS_CHECK_CUDA(cudaMemcpyAsync(
        dP,
        // input_flat.data_ptr(),
        input.data_ptr(),
        aZ * sizeof(Element),
        cudaMemcpyDeviceToDevice,
        kleos::kleosStream
    ));
    
    // Copy gate weights
    KLEOS_CHECK_CUDA(cudaMemcpyAsync(
        dP + aZ,
        gate_weights.data_ptr(),
        // (gwZ - aZ) * sizeof(Element),
        E * H * sizeof(Element),
        cudaMemcpyDeviceToDevice,
        kleos::kleosStream
    ));
    
    // Copy expert weights
    for (uint i = 0; i < nLx; ++i) {
        // Copy up projection
        KLEOS_CHECK_CUDA(cudaMemcpyAsync(
            dP + gwZ + i * (P * H),
            expert_weights[i][0].data_ptr(),
            P * H * sizeof(Element),
            cudaMemcpyDeviceToDevice,
            kleos::kleosStream
        ));
        
        // Copy down projection
        KLEOS_CHECK_CUDA(cudaMemcpyAsync(
            dP + bZ + i * (P * H),
            expert_weights[i][1].data_ptr(),
            P * H * sizeof(Element),
            cudaMemcpyDeviceToDevice,
            kleos::kleosStream
        ));
    }
    
    // Call kernel
    float timed = 0;
    kleos::moe::forwardHostBench<32, 32>(p, p + dZ * sizeof(Element), timed);
    
    if (rank == 0) {
        printf("FlashMoE forward pass took %.2fms\n", timed);
    }
    
    KLEOS_CHECK_CUDA(cudaPeekAtLastError());
    
    // Extract output
    auto output = torch::empty({input.size(0), input.size(1), H}, 
                               torch::TensorOptions()
                                   .dtype(input.dtype())
                                   .device(input.device()));
    
    KLEOS_CHECK_CUDA(cudaMemcpyAsync(
        output.data_ptr(),
        dP + gZ,
        S * H * sizeof(Element),
        cudaMemcpyDeviceToDevice,
        kleos::kleosStream
    ));
    
    // Synchronize
    KLEOS_CHECK_CUDA(cudaStreamSynchronize(kleos::kleosStream));
    
    // Free memory
    KLEOS_CHECK_CUDA(cudaFreeAsync(p, kleos::kleosStream));
    
    return output;
}


/**
 * Initialize NVSHMEM/Kleos
 */
void init_nvshmem() {
    kleos::initialize();
}


/**
 * Finalize NVSHMEM/Kleos
 */
void finalize_nvshmem() {
    kleos::finalize();
}

// Helper function
py::dict get_compiled_config() {
    py::dict result;
    result["S"] = kleos::ACC::S::value;
    result["H"] = kleos::ACC::H::value;
    result["E"] = kleos::ACC::E::value;
    result["P"] = kleos::ACC::P::value;
    result["PX"] = kleos::ACC::PX::value;
    result["Element_size"] = sizeof(Element);
    return result;
}


/**
 * PyBind11 module definition
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMoE: Fast Distributed MoE in a Single Kernel";
    
    m.def("moe_forward", &moe_forward, 
          "MoE forward pass. Tensors must match compiled config dimensions.",
          py::arg("input"),
          py::arg("gate_weights"),
          py::arg("expert_weights"));
    
    m.def("init_nvshmem", &init_nvshmem,
          "Initialize NVSHMEM/Kleos");
    
    m.def("finalize_nvshmem", &finalize_nvshmem,
          "Finalize NVSHMEM/Kleos");

    m.def("get_compiled_config", &get_compiled_config,
      "Get compile-time configuration values");
}