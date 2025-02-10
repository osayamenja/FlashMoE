/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/

#include "include/bootstrap.cuh"
#include "include/types.cuh"

template<typename Element, typename ProblemShape>
__host__ __forceinline__
void mT(aristos::WorkerAttribute* __restrict__ const& dWa,
    ProblemShape const& pShape,
    cuda::std::byte const& hP, cuda::std::byte const& hC) {
    const auto [M, N, K] = pShape;
    // malloc memory for all matrices
    constexpr auto batch = 2U;
    cuda::std::byte* abc;
    const auto stateSize = sizeof(uint) * (1 + M / BLOCK_M); // tileSync + dT
    const auto aSize = stateSize + sizeof(Element) * M * K;
    const auto abSize = aSize + 2 * sizeof(Element) * K * N;
    const auto abBSize = abSize + sizeof(Element) * (N + K); // bias
    const auto abBSSize = abBSize + sizeof(Element) * M; // scale/combine weights
    const auto abBScSize = abBSize + sizeof(Element) * (M * N); // output

    // malloc
    CHECK_ERROR_EXIT(cudaMallocAsync(&abc, abBSSize, aristos::aristosStream));

    auto* __restrict__ p = CAST_TO(uint, abc);
    const auto pS = cute::make_tuple(M, N, K);
    using ElementAccum = float;
    using Activation = cutlass::epilogue::thread::ReLU<ElementAccum>;
    constexpr auto blocks = aristos::Hardware<>::blocks::value - 1U;

    auto* __restrict__ pBp = CAST_TO(Element, abc + aSize);
    auto* __restrict__ pDp = CAST_TO(Element, abc + abSize);
    const auto pB = cuda::std::array<Element*, batch>{pBp, pBp + K * N};
    const auto pD = cuda::std::array<Element*, batch>{pDp, pDp + N};
    auto* __restrict__ pSC = CAST_TO(Element, abc + abBSize);
    auto* __restrict__ pC = CAST_TO(Element, abc + abBSSize);

    // copy A, B1, B2, D1, D2, M to device
    CHECK_ERROR_EXIT(cudaMemcpyAsync(abc + stateSize, hP, abBSSize, cudaMemcpyHostToDevice,
        aristos::aristosStream));

    expert<800, Activation, batch, ElementAccum><<<blocks, ARISTOS_BLOCK_SIZE, 0, aristos::aristosStream>>>(pS, p,
        p + 1, CAST_TO(Element, abc + stateSize), pB, pC, pD, pSC, pSC);
    CHECK_ERROR_EXIT(cudaMemcpyAsync(hC, pC, sizeof(Element) * (M * N),
        cudaMemcpyDeviceToHost, aristos::aristosStream));
    uint stage = 0;
    CHECK_ERROR_EXIT(cudaMemcpyAsync(&stage, p, sizeof(uint), cudaMemcpyDeviceToHost, aristos::aristosStream));
    CHECK_ERROR_EXIT(cudaFreeAsync(abc, aristos::aristosStream));
    CHECK_ERROR_EXIT(cudaPeekAtLastError());
    CHECK_ERROR_EXIT(cudaStreamSynchronize(aristos::aristosStream));
    dWa->throughput = static_cast<uint16_t>(stage);
}
__host__ __forceinline__
void evalExpert() {
    // create torch tensors
    // gemm 1 -> ReLu -> gemm 2 -> scale
    // compute fused expert
    // verify correctness
    // compare times
    aristos::WorkerAttribute wA{};
    aristos::measureThroughput<800, cute::tfloat32_t>(&wA, 128, 64, 8);
}
/*__host__ __forceinline__
void startAristos() {
    using Element = cute::tfloat32_t;
    // GPT3-MoE 1.3B
    const auto iC = aristos::InitialConfig{
        50 * 1024,
        24,
        512,
        4,
        2,
        2048,
        2048,
        2048 * 4,
        1,
        1,
        16,
        true,
        true,
    };
    aristos::initialize<Element>(iC);
    //aristos::moe::forwardHost();
    aristos::finalize();
}*/
int main() {
    return 0;
}
