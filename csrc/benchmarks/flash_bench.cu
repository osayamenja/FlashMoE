/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
#include <fmt/ranges.h>

#include "../include/flashmoe/flashmoe.cuh"

__host__ __forceinline__
void runOS() {
    flashmoe::initialize();
    const auto rank = flashmoe::getRank();
    constexpr auto S = flashmoe::ACC::S::value;
    constexpr auto H = flashmoe::ACC::H::value;
    constexpr auto E = flashmoe::ACC::E::value;
    constexpr auto P = flashmoe::ACC::P::value;
    constexpr auto PX = flashmoe::ACC::PX::value;
    const auto nLx = flashmoe::hostBookkeeping.nLx;

    using Element = flashmoe::ACC::Element;
    auto stream = flashmoe::flashmoeStream;
    Element* activations = nullptr;
    Element* gateWeights = nullptr;
    Element* expertUpWeights = nullptr;
    Element* biasUp = nullptr;
    Element* expertDownWeights = nullptr;
    Element* biasDown = nullptr;
    Element* gateOut = nullptr;
    Element* moeOut = nullptr;

    const int localExperts = flashmoe::hostBookkeeping.nLx;
    cudaMallocAsync(&activations, sizeof(Element) * S * H, stream);
    cudaMallocAsync(&gateWeights, sizeof(Element) * H * E, stream);
    cudaMallocAsync(&expertUpWeights, sizeof(Element) * localExperts * H * P, stream);
    cudaMallocAsync(&biasUp, sizeof(Element) * localExperts * P, stream);
    cudaMallocAsync(&expertDownWeights, sizeof(Element) * localExperts * P * H, stream);
    cudaMallocAsync(&biasDown, sizeof(Element) * localExperts * H, stream);
    cudaMallocAsync(&gateOut, sizeof(Element) * S * E, stream);
    cudaMallocAsync(&moeOut, sizeof(Element) * S * H, stream);
    FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
    {
        #if FLASHMOE_NVTX
        flashmoe::flashmoeRange forwardRange{"Host Data Prep"};
        #endif
        // fill input matrices
    }
    float timed = 0;
    flashmoe::moe::forwardHostBench<32, 32>(p, p + dZ * sizeof(Element), timed);
    printf("epRank: %u took %.2fms\n", flashmoe::hostBookkeeping.rank, timed);
    FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
    flashmoe::finalize();
}

int main() {
    runOS();
}
