//
// Created by osayamen on 12/22/25.
//

#include <random>

#include <curanddx.hpp>
#include <cublasdx.hpp>

#include "common.cuh"

__host__ __forceinline__
void driver(const int& M, const int& N) {
    const long int n = M * N;
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    using Element = __half;
    using CuteElement = cute::half_t;
    Element* p = nullptr;
    const auto nBytes = n * sizeof(Element);
    cudaMallocAsync(&p, nBytes, stream);
    auto* q = static_cast<Element*>(std::malloc(nBytes));
    std::random_device rd;
    constexpr int TPB = 256;
    const int blocks = (n + 4ull * TPB - 1) / (4ull * TPB);
    generateRandUniform<FLASHMOE_ARCH><<<blocks, TPB>>>(p, n, /*seed=*/rd(),
        -1.f, 1.f);
    cudaMemcpyAsync(q, p, nBytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    CHECK_CUDA(cudaPeekAtLastError());
    // print tensor
    const auto t = cute::make_tensor(reinterpret_cast<CuteElement*>(q),
        cute::make_layout(cute::make_shape(M, N), cute::LayoutRight{}));
    print_tensor(t);
    cudaFreeAsync(p, stream);
    std::free(q);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}
int main() {
    driver(8, 8);
}