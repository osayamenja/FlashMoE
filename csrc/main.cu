/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
#include <fmt/ranges.h>
#include <thrust/generate.h>
#include <thrust/random.h>

#include "correctness.cuh"
#include "include/bootstrap.cuh"
#include "include/moe/moe.cuh"

__host__ __forceinline__
void runOS() {
    aristos::initialize();
    const auto rank = aristos::getRank();
    // generate random input tile and eye weights
    constexpr auto S = aristos::ACC::S::value;
    constexpr auto H = aristos::ACC::H::value;
    constexpr auto E = aristos::ACC::E::value;
    constexpr auto P = aristos::ACC::P::value;
    constexpr auto PX = aristos::ACC::PX::value;
    const auto nLx = aristos::hostBookkeeping.nLx;
    constexpr unsigned long aZ =  S * H;
    constexpr auto gwZ = aZ + PX * H;
    // scale this to number of experts
    const auto bZ =  gwZ + nLx * P * H;
    const auto b2Z =  bZ + nLx * P * H;
    const auto dZ =  b2Z + nLx * (P + H);
    const auto gZ = dZ + S * PX;
    const auto cZ = gZ + S * H;
    cuda::std::byte* p;
    ARISTOS_CHECK_CUDA(cudaMallocAsync(&p, cZ * sizeof(float), aristos::aristosStream));
    ARISTOS_CHECK_CUDA(cudaMemsetAsync(p, 0, cZ * sizeof(float), aristos::aristosStream));
    auto* hP = std::calloc(cZ, sizeof(float));
    auto* fHp = static_cast<float*>(hP);
    using Element = aristos::ACC::Element;
    auto* __restrict__ eHp = static_cast<Element*>(hP);
    {
        #if ARISTOS_NVTX
        aristos::aristosRange forwardRange{"Host Data Prep"};
        #endif
        thrust::default_random_engine rng(47 * (rank + 42));
        thrust::normal_distribution<float> dist(0, 5);
        // Activations
        thrust::generate(fHp, fHp + aZ, [&] { return dist(rng); });
        // gate weights
        thrust::generate(fHp + aZ, fHp + aZ + E * H, [&] { return dist(rng); });
        // Expert weights
        // loop for number of experts
        for (uint i = 0; i < nLx; ++i) {
            // expert up
            thrust::generate(fHp + gwZ + i * (P * H), fHp + gwZ + (i + 1) * (P * H),
                [&] { return dist(rng); });
            thrust::generate(fHp + bZ + i * (P * H), fHp + bZ + (i + 1) * (P * H),
                [&] { return dist(rng); });
        }
        // bias
        std::ranges::fill(fHp + b2Z, fHp + dZ, 0.0f);
        constexpr cutlass::NumericConverter<Element, float> conv{};
        for (uint i = 0; i < dZ; ++i) {
            eHp[i] = conv(fHp[i]);
        }
    }
    ARISTOS_CHECK_CUDA(cudaMemcpyAsync(p, eHp, sizeof(Element) * dZ,
        cudaMemcpyHostToDevice,
        aristos::aristosStream));
    float timed = 0;
    aristos::moe::forwardHostBench<32, 32>(p, p + dZ * sizeof(Element), timed);
    printf("epRank: %u took %.2fms\n", aristos::hostBookkeeping.rank, timed);
    ARISTOS_CHECK_CUDA(cudaPeekAtLastError());
    aristos::finalize();
    std::free(hP);
}

__host__ __forceinline__
void runReference() {
    constexpr auto S = 32;
    constexpr auto H = 32;
    constexpr auto E = 16;
    constexpr auto P = 32;
    constexpr auto PX = E;
    constexpr unsigned long aZ =  S * H;
    constexpr auto gwZ = aZ + PX * H;
    // scale this to number of experts
    constexpr auto nLx = E;
    constexpr auto bZ =  gwZ + nLx * P * H;
    constexpr auto b2Z =  bZ + nLx * P * H;
    constexpr auto dZ =  b2Z + nLx * (P + H);
    constexpr auto gZ = dZ + S * PX;
    constexpr auto cZ = gZ + S * H;
    void* p;
    ARISTOS_CHECK_CUDA(cudaMallocAsync(&p, cZ * sizeof(float), aristos::aristosStream));
    ARISTOS_CHECK_CUDA(cudaMemsetAsync(p, 0, cZ * sizeof(float), aristos::aristosStream));
    auto* hP = std::calloc(cZ, sizeof(float));
    auto* fHp = static_cast<float*>(hP);
    using ET = float;
    auto* __restrict__ eHp = static_cast<ET*>(hP);
    auto* __restrict__ dP = static_cast<ET*>(p);
    {
        #if ARISTOS_NVTX
        aristos::aristosRange forwardRange{"Host Data Prep"};
        #endif
        thrust::default_random_engine rng(47 * 42);
        thrust::normal_distribution<float> dist(0, 5);
        // Activations, Gate weights, expert weights
        thrust::generate(fHp, fHp + b2Z, [&] { return dist(rng); });
        if constexpr (!cuda::std::is_same_v<ET, float>) {
            constexpr cutlass::NumericConverter<ET, float> conv{};
            for (uint i = 0; i < dZ; ++i) {
                eHp[i] = conv(fHp[i]);
            }
        }
    }
    ARISTOS_CHECK_CUDA(cudaMemcpyAsync(p, eHp, sizeof(ET) * dZ,
        cudaMemcpyHostToDevice,
        aristos::aristosStream));
    auto* __restrict__ act = dP;
    auto* __restrict__ gateWeights = dP + aZ;
    auto* __restrict__ expertWeights = dP + gwZ;
    auto* __restrict__ bias = dP + b2Z;
    auto* __restrict__ gateOutput = dP + dZ;
    auto* __restrict__ moeOutput = dP + gZ;
    aristos::rExpert<S, H, P, E>(act,
        gateWeights, expertWeights, bias, gateOutput, moeOutput, nLx);
    ARISTOS_CHECK_CUDA(cudaMemcpyAsync(eHp, gateOutput, sizeof(ET) * S * PX, cudaMemcpyDeviceToHost,
        aristos::aristosStream));
    ARISTOS_CHECK_CUDA(cudaStreamSynchronize(aristos::aristosStream));
    const auto cGo = make_tensor(eHp,
            cute::Layout<cute::Shape<cute::Int<S>, cute::Int<E>>,
            cute::Stride<cute::Int<E>, cute::_1>>{});
    print_tensor(cGo);
}
int main() {
    runOS();
}
