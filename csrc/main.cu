/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
#include <fmt/ranges.h>
#include <thrust/generate.h>
#include <thrust/random.h>

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
    ARISTOS_CHECK_CUDA(cudaMemcpyAsync(p, eHp, sizeof(Element) * dZ,
        cudaMemcpyHostToDevice,
        aristos::aristosStream));
    float timed = 0;
    aristos::moe::forwardHostBench<50, 100>(p, p + dZ * sizeof(Element), timed);
    printf("epRank: %u took %.2fms\n", aristos::hostBookkeeping.rank, timed);
    ARISTOS_CHECK_CUDA(cudaPeekAtLastError());
    aristos::finalize();
    std::free(hP);
}

int main() {
    runOS();
}
