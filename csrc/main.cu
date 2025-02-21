/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
#include <torch/torch.h>

#include "include/moe/moe.cuh"
#include "include/throughput.cuh"
#include "include/types.cuh"

__host__ __forceinline__
void evalExpert() {
    using GPUType = aristos::Hardware<ARISTOS_ARCH, 255>;
    constexpr auto blocks = GPUType::OS::processorBlocks::value;
    constexpr auto M = 8192UL;
    constexpr auto N = 4096UL;
    constexpr auto K = 1024UL;
    static_assert(M % BLOCK_M == 0 && M < BLOCK_M * blocks * 128 &&
        N % BLOCK_N == 0 && K % BLOCK_N == 0);
    using clk = std::chrono::high_resolution_clock;
    std::chrono::duration<float> end {};
    // create torch tensors
    CHECK_ERROR_EXIT(cudaSetDevice(0));
    constexpr auto sT = torch::kFloat16;
    at::globalContext().setAllowTF32CuBLAS(true);
    at::globalContext().setAllowTF32CuDNN(true);
    const auto options = torch::TensorOptions().dtype(sT).layout(torch::kStrided).device(torch::kCUDA, 0);
    const torch::Device device(torch::kCUDA, 0);
    torch::nn::Sequential expert(
        torch::nn::Linear(torch::nn::LinearOptions(K, N).bias(true)),
        torch::nn::ReLU(),
        torch::nn::Linear(torch::nn::LinearOptions(N, K).bias(true))
        );
    // set expert weights to identity and bias to zero to avoid accumulation errors for this test
    expert->to(device, sT);

    using ElementAccum = float;
    using Activation = cutlass::epilogue::thread::ReLU<ElementAccum>;
    using Element = cute::half_t;
    constexpr auto aZ =  M * K;
    constexpr auto bZ =  aZ + N * K;
    constexpr auto b2Z =  bZ + N * K;
    constexpr auto dZ =  b2Z + N;
    constexpr auto d2Z =  dZ + K;
    constexpr auto sZ =  d2Z + M;
    constexpr auto cWz =  sZ + M;
    constexpr auto cZ =  cWz + M * N;
    constexpr auto hZ =  cZ + M * K;

    // Pack A, B, D, S into a single, linear tensor
    const auto hT = torch::ones({1, hZ}, options).contiguous();
    const auto activations = torch::ones({M, K}, options);
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
    // set output to zero
    hT.index({0, torch::indexing::Slice(cZ, hZ)}) = torch::zeros({M * K}, options).contiguous();
    const auto combineWeights = hT.index({0, torch::indexing::Slice(sZ, cWz)}).view({M, 1});
    // gemm 1 -> ReLU -> gemm 2 -> scale
    constexpr auto trials = 128U;
    const auto start = clk::now();
    const auto result = mul(expert->forward(activations), scaleWeights);
    CHECK_ERROR_EXIT(cudaDeviceSynchronize());
    end = clk::now() - start;
    printf("Torch takes %fms\n", end.count() * 1000);

    // Get a copy of the reference result
    aristos::WorkerAttribute wA{};
    // compute & measure fused expert
    aristos::mFT<GPUType, trials, aristos::CombineMode::single, Activation, aristos::UseBarrier::no>(&wA, M, N, K,
        CAST_TO(Element, hT.mutable_data_ptr()),
        CAST_TO(Element, hT.mutable_data_ptr()) + cWz);
    // verify and compare
    std::cout << "Passed? " << (result.view({M * K})
        .allclose(hT.index({0, torch::indexing::Slice(cZ, hZ)}),
            1e-03, 1e-05, true) ? "Yes!" : "No")
    << std::endl;
    std::cout << result.index({1024,
        torch::indexing::Slice(256, 266)}).view({1, 10})
    << std::endl;
    std::cout << hT.index({0, torch::indexing::Slice(cZ, hZ)}).view({M, K}).
        index({1024, torch::indexing::Slice(256, 266)}).view({1, 10})
    << std::endl;
    CHECK_LAST();
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
        aristos::ActivationFunction::ReLu,
        true,
    };
    aristos::initialize(iC, torch::kFloat32);
    //aristos::moe::forwardHost();
    aristos::finalize();
}*/
int main() {
    cutlass::AlignedArray<float, 2> f{};
    auto d = static_cast<uint>(f[0]);
    evalExpert();
    return 0;
}
