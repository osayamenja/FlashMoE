//
// Created by osayamen on 12/22/25.
//

// place to experime
#include <random>
#include "common.cuh"
#include "debug.cuh"

int main() {
    CHECK_CUDA(cudaSetDevice(0));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    __half* p = nullptr;
    constexpr auto E = 128;
    constexpr auto H = 8192;
    constexpr auto I = 14336;
    constexpr auto bytes = sizeof(__half) * E * H * I;
    CHECK_CUDA(cudaMallocAsync(&p, bytes, stream));
    std::random_device rd;
    randUniform<FLASHMOE_ARCH>(p, static_cast<size_t>(E) * H * I, rd(), -1.0f, 1.0f, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    printf("Success!\n");
    CHECK_CUDA(cudaFreeAsync(p, stream));
    CHECK_CUDA(cudaStreamDestroy(stream));
}