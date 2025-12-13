/**
 * Python bindings for FlashMoE CUDA kernels
 * This wraps FlashMoE CUDA code to be callable from Python
 */
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>

#include "include/flashmoe/bootstrap.cuh"
#include "include/flashmoe/moe/moe.cuh"

namespace py = pybind11;

using Element = flashmoe::ACC::Element;

torch::Tensor moe_forward(
    torch::Tensor input,               // [batch, seq_len, hidden_size] - Activations
    torch::Tensor gate_weights,        // [hidden_size, num_experts] - Gate weights
    torch::Tensor expert_weights       // [local_experts, 2, intermediate_size, hidden_size] - Expert weights
) {
    TORCH_CHECK(flashmoe::isInitialized, "Must call initialize() before moe_forward");
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(gate_weights.is_cuda(), "Gate weights must be CUDA tensor");
    TORCH_CHECK(expert_weights.is_cuda(), "Expert weights must be CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(gate_weights.is_contiguous(), "Gate weights must be contiguous");
    TORCH_CHECK(expert_weights.is_contiguous(), "Expert weights must be contiguous");
    
    const auto rank = flashmoe::getRank();
    
    // Get dimensions from compile-time config
    constexpr auto S = flashmoe::ACC::S::value;
    constexpr auto H = flashmoe::ACC::H::value;
    constexpr auto E = flashmoe::ACC::E::value;
    constexpr auto P = flashmoe::ACC::P::value;
    constexpr auto PX = flashmoe::ACC::PX::value;
    const auto nLx = flashmoe::hostBookkeeping.nLx;
    
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
    TORCH_CHECK(expert_weights.size(0) == nLx, 
                "Expert count mismatch. Expected " + std::to_string(nLx) + 
                " local experts, got " + std::to_string(expert_weights.size(0)));
    TORCH_CHECK(expert_weights.size(1) == 2, 
                "Expert weights must have up and down projections [nLx, 2, P, H]");
    TORCH_CHECK(expert_weights.size(2) == P && expert_weights.size(3) == H,
                "Expert weights must be [*, 2, P=" + std::to_string(P) + 
                ", H=" + std::to_string(H) + "]. Got [*, 2, " + 
                std::to_string(expert_weights.size(2)) + ", " + 
                std::to_string(expert_weights.size(3)) + "]");
    
    // Calculate memory layout
    constexpr unsigned long aZ = S * H;              // Activations
    constexpr auto gwZ = aZ + PX * H;                // + Gate weights
    const auto bZ = gwZ + nLx * P * H;               // + Expert up weights
    const auto b2Z = bZ + nLx * P * H;               // + Expert down weights
    const auto dZ = b2Z + nLx * (P + H);             // + Bias
    const auto gZ = dZ + S * PX;                     // + Gate output
    const auto cZ = gZ + S * H;                      // + MoE output
    
    // Allocate device memory for all data
    cuda::std::byte* p;
    FLASHMOE_CHECK_CUDA(cudaMallocAsync(&p, cZ * sizeof(Element), flashmoe::flashmoeStream));
    FLASHMOE_CHECK_CUDA(cudaMemsetAsync(p, 0, cZ * sizeof(Element), flashmoe::flashmoeStream));
    
    auto* __restrict__ dP = reinterpret_cast<Element*>(p);
    
    // Copy activations
    FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(
        dP,
        input.data_ptr(),
        aZ * sizeof(Element),
        cudaMemcpyDeviceToDevice,
        flashmoe::flashmoeStream
    ));
    
    // Copy gate weights
    FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(
        dP + aZ,
        gate_weights.data_ptr(),
        E * H * sizeof(Element),
        cudaMemcpyDeviceToDevice,
        flashmoe::flashmoeStream
    ));
    
    // Copy expert weights
    for (uint i = 0; i < nLx; ++i) {
        // Copy up projection
        FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(
            dP + gwZ + i * (P * H),
            expert_weights[i][0].data_ptr(),
            P * H * sizeof(Element),
            cudaMemcpyDeviceToDevice,
            flashmoe::flashmoeStream
        ));
        
        // Copy down projection
        FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(
            dP + bZ + i * (P * H),
            expert_weights[i][1].data_ptr(),
            P * H * sizeof(Element),
            cudaMemcpyDeviceToDevice,
            flashmoe::flashmoeStream
        ));
    }
    
    // Call kernel
    float timed = 0;
    flashmoe::moe::forwardHostBench<32, 32>(p, p + dZ * sizeof(Element), timed);
    
    printf("Process %d: FlashMoE forward pass took %.2f ms\n", rank, timed);
    
    FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
    
    // Extract output
    auto output = torch::empty({input.size(0), input.size(1), H}, 
                               torch::TensorOptions()
                                   .dtype(input.dtype())
                                   .device(input.device()));
    
    FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(
        output.data_ptr(),
        dP + gZ,
        S * H * sizeof(Element),
        cudaMemcpyDeviceToDevice,
        flashmoe::flashmoeStream
    ));
    
    // Synchronize
    FLASHMOE_CHECK_CUDA(cudaStreamSynchronize(flashmoe::flashmoeStream));
    
    // Free memory
    FLASHMOE_CHECK_CUDA(cudaFreeAsync(p, flashmoe::flashmoeStream));
    
    return output;
}


/**
 * Initialize NVSHMEM/Flashmoe
 */
void initialize() {
    flashmoe::initialize();
}


/**
 * Finalize NVSHMEM/Flashmoe
 */
void finalize() {
    flashmoe::finalize();
}

// Helper function
py::dict get_compiled_config() {
    py::dict result;
    result["S"] = flashmoe::ACC::S::value;
    result["H"] = flashmoe::ACC::H::value;
    result["E"] = flashmoe::ACC::E::value;
    result["P"] = flashmoe::ACC::P::value;
    result["PX"] = flashmoe::ACC::PX::value;
    result["Element_size"] = sizeof(Element);
    return result;
}

py::dict get_bookkeeping() {
    py::dict result;
    result["nLx"] = flashmoe::hostBookkeeping.nLx;
    return result;
}

uint16_t get_num_local_experts() {
    return flashmoe::hostBookkeeping.nLx;
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
    
    m.def("initialize", &initialize,
          "Initialize NVSHMEM/Flashmoe");
    
    m.def("finalize", &finalize,
          "Finalize NVSHMEM/Flashmoe");

    m.def("get_compiled_config", &get_compiled_config,
      "Get compile-time configuration values");

    m.def("get_bookkeeping", &get_bookkeeping,
      "Get internal bookkeeping values");

    m.def("get_num_local_experts", &get_num_local_experts,
      "Get the number of local experts");
}
