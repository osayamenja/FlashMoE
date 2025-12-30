//
// Created by osayamen on 12/22/25.
//

#include <random>
#include <curanddx.hpp>
#include <cublasdx.hpp>
#include "common.cuh"

int main() {
    cudaSetDevice(0);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    matx::cudaExecutor exec{stream};
    auto tA = matx::make_tensor<int>({2,2});
    constexpr auto cutoff = 1;
    tA(0,0) = 2; tA(0, 1) = 1; tA(1, 0) = 4; tA(1, 1) = 3;
    auto tAx = tA.Slice<2>({0, 0}, {matx::matxEnd, cutoff});
    print(tA);
    auto stAx = matx::make_tensor<int>(tAx.Shape());
    (stAx = matx::sort(tAx, matx::SORT_DIR_ASC)).run(exec);
    auto tB = matx::make_tensor<matx::index_t>({2,2});
    tB(0,0) = 2; tB(0, 1) = 1; tB(1, 0) = 4; tB(1, 1) = 3;
    auto tBx = tB.Slice<2>({0, 0}, {matx::matxEnd, cutoff});
    print(tB);
    auto stBx = matx::make_tensor<matx::index_t>(tBx.Shape());
    (stBx = matx::sort(tBx, matx::SORT_DIR_ASC)).run(exec);
    exec.sync();
    print(stAx);print(stBx);
    auto isEqual = matx::make_tensor<matx::index_t>(stAx.Shape());
    (isEqual = stAx == stBx).run(exec);
    auto tC = matx::make_tensor<float>({2,2});
    auto topK_idx = matx::make_tensor<matx::index_t>(tC.Shape());
    (tC = matx::random<float>(tC.Shape(), matx::UNIFORM)).run(exec);
    (topK_idx = matx::argsort(tC, matx::SORT_DIR_DESC)).run(exec);
    exec.sync();
    print(isEqual);
    print(tC);print(topK_idx);
    cudaStreamDestroy(stream);
}