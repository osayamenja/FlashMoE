//
// Created by oja7 on 2/15/25.
//

#ifndef THROUGHPUT_CUH
#define THROUGHPUT_CUH

#include <torch/torch.h>

#include "debug.cuh"
#include "moe/expert.cuh"
#include "types.cuh"

namespace aristos {
    template<
        typename GPUType,
        unsigned int skip = 128U,
        CombineMode c,
        typename Activation,
        UseBarrier u = UseBarrier::no,
        typename Element,
        typename ElementAccum = float,
        unsigned int blocks = GPUType::OS::processorBlocks::value,
        unsigned int threads = GPUType::OS::threads::value
    >
    requires(cuda::std::is_invocable_r_v<ElementAccum, Activation, ElementAccum>)
    __host__ __forceinline__
    void mFT(WorkerAttribute* __restrict__ const& dWa,
        const unsigned int& M, const unsigned int& N, const unsigned int& K,
        Element* __restrict__ const& iP, Element* __restrict__ oP) {
        cuda::std::byte* p;
        const auto tSz = sizeof(uint) * (M / BLOCK_M) * umin(K / BLOCK_N, blocks);
        const auto stateSize = sizeof(cuda::barrier<cuda::thread_scope_device>) + sizeof(float) + tSz;
        CHECK_ERROR_EXIT(cudaMallocAsync(&p, stateSize, aristosStream));
        CHECK_ERROR_EXIT(cudaMemsetAsync(p, 0, stateSize, aristosStream));
        const auto hB = new cuda::barrier<cuda::thread_scope_device>{blocks};
        CHECK_ERROR_EXIT(cudaMemcpyAsync(p, hB,
            sizeof(cuda::barrier<cuda::thread_scope_device>),
            cudaMemcpyHostToDevice, aristosStream));

        auto pS = cute::make_tuple(M, N, K);
        auto* dT = CAST_TO(float, p + sizeof(cuda::barrier<cuda::thread_scope_device>));
        auto* tileSync = CAST_TO(uint, dT + 1);
        #pragma unroll
        for (uint i = 0; i < skip; ++i) {
            expert<GPUType, Activation, c, ElementAccum, u><<<blocks, threads, 0, aristosStream>>>(pS,
                CAST_TO(cuda::barrier<cuda::thread_scope_device>, p), dT, tileSync, iP, oP);
            if constexpr (u == UseBarrier::no) {
                CHECK_ERROR_EXIT(cudaMemsetAsync(tileSync, 0, tSz, aristosStream));
            }
            // Needed to clear accumulator buffer
            if constexpr (c == CombineMode::multithreaded) {
                CHECK_ERROR_EXIT(cudaMemsetAsync(oP + M * N, 0, sizeof(Element) * (M * K), aristosStream));
            }
        }
        expert<GPUType, Activation, c, ElementAccum, u><<<blocks, threads, 0, aristosStream>>>(pS,
        CAST_TO(cuda::barrier<cuda::thread_scope_device>, p), dT, tileSync, iP, oP, false);
        CHECK_ERROR_EXIT(cudaPeekAtLastError());
        float latency = 0;
#if TIME_EXPERT
        CHECK_ERROR_EXIT(cudaMemcpyAsync(&latency, dT, sizeof(float), cudaMemcpyDeviceToHost, aristosStream));
        CHECK_ERROR_EXIT(cudaStreamSynchronize(aristosStream));
        printf("Kernel took %fms\n", latency);
        // quantize to half-precision, this should be safe as the value is very small: in the hundreds
        // we use float for compatibility with device atomics
#endif
        dWa->throughput = cute::half_t(1 / latency); // latency should be > 0
        CHECK_ERROR_EXIT(cudaFreeAsync(p, aristosStream));
        delete hB;
    }

    template<
        typename GPUType,
        typename Element,
        CombineMode c,
        typename Activation,
        unsigned int trials = 128U
    >
    requires (cuda::std::is_invocable_r_v<GEA, Activation, GEA>)
    __host__ __forceinline__
    void mT(WorkerAttribute* __restrict__ const& dWa,
        const unsigned int& M, const unsigned int& N, const unsigned int& K, uint const& devId) {
        // create torch tensors
        const auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided)
            .device(torch::kCUDA, devId);
        at::globalContext().setAllowTF32CuBLAS(true);
        at::globalContext().setAllowTF32CuDNN(true);
        const torch::Device device(torch::kCUDA, devId);

        // Clean way to initialize the memory needed
        torch::nn::Sequential expert(
            torch::nn::Linear(torch::nn::LinearOptions(K, N).bias(true)),
            torch::nn::Linear(torch::nn::LinearOptions(N, K).bias(true))
            );
        expert->to(device);

        const auto aZ =  M * K;
        const auto bZ =  aZ + N * K;
        const auto b2Z =  bZ + N * K;
        const auto dZ =  b2Z + N;
        const auto d2Z =  dZ + K;
        const auto sZ =  d2Z + M;
        const auto cWz =  sZ + M;
        const auto cZ =  cWz + M * N;
        const auto hZ =  cZ + M * K;

        // Pack A, B, D, S into a single, linear tensor
        const auto hT = torch::ones({1, hZ}, options).contiguous();
        const auto activations = torch::rand({M, K}, options);
        const auto scaleWeights = torch::ones({M, 1}, options);
        // Pack A
        hT.index({0, torch::indexing::Slice(torch::indexing::None, aZ)}) =
            activations.view({aZ}).contiguous();
        // Pack expert weights
        hT.index({0, torch::indexing::Slice(aZ, bZ)}) =
            expert->named_parameters()[0].value().view({N * K}).contiguous();
        hT.index({0, torch::indexing::Slice(bZ, b2Z)}) =
            expert->named_parameters()[2].value().view({N * K}).contiguous();
        // Pack expert bias
        hT.index({0, torch::indexing::Slice(b2Z, dZ)}) =
            expert->named_parameters()[1].value().view({N}).contiguous();
        hT.index({0, torch::indexing::Slice(dZ, d2Z)}) =
            expert->named_parameters()[3].value().view({K}).contiguous();
        // Pack Scale
        hT.index({0, torch::indexing::Slice(d2Z, sZ)}) =
            scaleWeights.view({M}).contiguous();
        CHECK_ERROR_EXIT(cudaDeviceSynchronize());
        using VT = typename VCT<c, Element>::Element;
        mFT<GPUType, trials, c, Activation>(dWa, M, N, K,
            CAST_TO(VT, hT.mutable_data_ptr()), CAST_TO(VT, hT.mutable_data_ptr()) + cWz);
    }
}
#endif //THROUGHPUT_CUH
