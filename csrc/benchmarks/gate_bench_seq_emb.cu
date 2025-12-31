//
// Created by osayamen on 12/30/25.
//
#include "common.cuh"

__host__ __forceinline__
void kickStart(const int argc, char** argv) {
    // fixed to minimize template instantiations
    constexpr int E = 8;
    int MK = 8;
    int MK_max = 8192;
    float rtol = 2e-2f;
    float atol = 2e-3f;
    using Element = __half;
    using ElementC = Element;
    printf("S, E, H, bM, bN, bK, pipeStages, threads, blocks/SM, SMs, blocks, rtol, atol, error(%%), Kernel_Time(ms), "
           "Matx_Time(ms)\n");
    if (argc > 2) {
        MK = std::stoi(argv[2]);
    }
    if (argc > 3) {
        MK_max = std::stoi(argv[3]);
    }
    if (argc > 4) {
        rtol = std::stof(argv[4]);
    }
    if (argc > 5) {
        atol = std::stof(argv[5]);
    }
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    matx::cudaExecutor exec{stream, true};
    // tiling heuristics
    constexpr int bN = cute::min(E, 64);
    cudaStreamDestroy(stream);
}
int main(const int argc, char** argv) {

}