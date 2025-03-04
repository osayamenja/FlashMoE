//
// Created by oja7 on 2/15/25.
//

#ifndef THROUGHPUT_CUH
#define THROUGHPUT_CUH

#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/random.h>

#include "debug.cuh"
#include "moe/expert.cuh"
#include "types.cuh"

namespace aristos {
    template<
        UseBarrier u = UseBarrier::no,
        unsigned int trials = 128U,
        unsigned int N = ACC::P::value,
        unsigned int K = ACC::H::value,
        unsigned int blocks = ACC::PeakHardware::OS::processorBlocks::value,
        unsigned int threads = ACC::PeakHardware::OS::threads::value,
        CombineMode c = ACC::CM::value,
        typename Element
    >
    __host__ __forceinline__
    void mFT(unsigned int M,
        cuda::barrier<cuda::thread_scope_device>* dB,
        float* deviceThroughput, uint* tileSync,
        const Element* __restrict__ iP /* A, B, D, S, W*/, Element* __restrict__ oP /*C*/) {
        const auto tSz = sizeof(uint) * (M / BLOCK_M) * cute::min(K / BLOCK_N, blocks);
        #pragma unroll
        for (uint i = 0; i < trials; ++i) {
            expert<u, N, K><<<blocks, threads, 0, aristosStream>>>(M, dB, deviceThroughput, tileSync, iP, oP);
            if constexpr (u == UseBarrier::no) {
                CHECK_ERROR_EXIT(cudaMemsetAsync(tileSync, 0, tSz, aristosStream));
            }
            // Needed to clear accumulator buffer
            if constexpr (c == CombineMode::multithreaded) {
                CHECK_ERROR_EXIT(cudaMemsetAsync(oP + M * N, 0, sizeof(Element) * (M * K),
                    aristosStream));
            }
        }
        expert<u, N, K><<<blocks, threads, 0, aristosStream>>>(M, dB, deviceThroughput, tileSync, iP, oP, false);
        CHECK_ERROR_EXIT(cudaPeekAtLastError());
    }
    template<
        UseBarrier u = UseBarrier::no,
        unsigned int trials = 128U
    >
    __host__ __forceinline__
    void mT(WorkerAttribute* __restrict__ const& dWa) {
        constexpr unsigned int M = ACC::S::value;
        constexpr unsigned int N = ACC::P::value;
        constexpr unsigned int K = ACC::H::value;
        using Element = ACC::Element;

        constexpr auto aZ =  M * K;
        constexpr auto bZ =  aZ + N * K;
        constexpr auto b2Z =  bZ + N * K;
        constexpr auto dZ =  b2Z + N;
        constexpr auto d2Z =  dZ + K;
        constexpr auto sZ =  d2Z + M;
        constexpr auto cWz =  sZ + M;

        constexpr auto cZ =  cWz + M * N;
        constexpr auto hZ =  cZ + M * K;

        cuda::std::byte* p;
        constexpr unsigned int blocks = ACC::PeakHardware::OS::processorBlocks::value;
        // Scheduling state
        constexpr auto tSz = sizeof(uint) * ACC::TM::value * cute::min(ACC::TNx::value, blocks);
        constexpr auto stateSize = sizeof(cuda::barrier<cuda::thread_scope_device>) + sizeof(float) + tSz;
        CHECK_ERROR_EXIT(cudaMallocAsync(&p, hZ + stateSize, aristosStream));
        CHECK_ERROR_EXIT(cudaMemsetAsync(p, 0, stateSize, aristosStream));
        const auto hB = new cuda::barrier<cuda::thread_scope_device>{blocks};
        CHECK_ERROR_EXIT(cudaMemcpyAsync(p, hB,
            sizeof(cuda::barrier<cuda::thread_scope_device>),
            cudaMemcpyHostToDevice, aristosStream));

        thrust::default_random_engine rng(42);
        thrust::normal_distribution<float> dist(2,3);
        thrust::host_vector<float> hV(cWz);
        thrust::generate(hV.begin(), hV.end(), [&] { return dist(rng); });
        constexpr cutlass::NumericConverter<Element, float> conv{};
        auto* hVd = CAST_TO(Element, hV.data());
        #pragma unroll 16
        for (uint i = 0; i < cWz; ++i) {
            hVd[i] = conv(hV[i]);
        }
        CHECK_ERROR_EXIT(cudaMemcpyAsync(p + stateSize, hVd, cWz, cudaMemcpyHostToDevice, aristosStream));

        auto* dB = CAST_TO(cuda::barrier<cuda::thread_scope_device>, p);
        static_assert(alignof(cuda::barrier<cuda::thread_scope_device>) % alignof(float) == 0);
        auto* deviceThroughput = CAST_TO(float, dB + 1);
        static_assert(alignof(float) % alignof(uint) == 0);
        auto* tileSync = CAST_TO(uint, deviceThroughput + 1);
        static_assert(alignof(uint) % alignof(Element) == 0);
        auto* iP = CAST_TO(Element, p + stateSize);
        auto* oP = iP + cWz;
        mFT<u, trials>(M, dB, deviceThroughput, tileSync, iP, oP);
        CHECK_ERROR_EXIT(cudaPeekAtLastError());
        float latency = 0;
        CHECK_ERROR_EXIT(cudaMemcpyAsync(&latency, deviceThroughput, sizeof(float),
            cudaMemcpyDeviceToHost, aristosStream));
        CHECK_ERROR_EXIT(cudaStreamSynchronize(aristosStream));
#if TIME_EXPERT
        printf("Kernel took %fms\n", latency);
#endif
        dWa->throughput = cute::half_t(1 / latency); // latency should be > 0
        CHECK_ERROR_EXIT(cudaFreeAsync(p, aristosStream));
        delete hB;
    }
}
#endif //THROUGHPUT_CUH
