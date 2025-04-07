/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
#include <fmt/ranges.h>
#include <thrust/generate.h>
#include <thrust/random.h>

#include "include/bootstrap.cuh"
#include "include/moe/moe.cuh"
template<unsigned int M, unsigned int N = M>
requires(M > 0 && cutlass::is_pow2<M>::value && M == N)
__host__ __forceinline__
void makeIdentity(float* const& __restrict__ p, const bool print = false) {
    // we assume row major
    const auto t = cute::make_tensor(p,
        cute::Layout<cute::Shape<cute::Int<M>, cute::Int<N>>,
            cute::Stride<cute::Int<N>, cute::_1>>{});
    for (uint i = 0; i < M; ++i) {
        for (uint j = 0; j < N; ++j) {
            if (i == j) {
                t(i, j) = 1.0f;
            }
            else {
                t(i, j) = 0.0f;
            }
        }
    }

    if (print) {
        print_tensor(t);
    }
}

__host__ __forceinline__
void runOS() {
    aristos::initialize();
    const auto rank = aristos::getRank();
    // device should already be set from above
    // generate random input tile and eye weights
    constexpr auto S = aristos::ACC::S::value;
    constexpr auto H = aristos::ACC::H::value;
    constexpr auto E = aristos::ACC::E::value;
    constexpr auto P = aristos::ACC::P::value;
    constexpr auto PX = aristos::ACC::PX::value;
    constexpr auto pEC = aristos::ACC::pEC::value;
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
    CHECK_ERROR_EXIT(cudaMallocAsync(&p, cZ * sizeof(float), aristos::aristosStream));
    CHECK_ERROR_EXIT(cudaMemsetAsync(p, 0, cZ * sizeof(float), aristos::aristosStream));
    auto* hP = std::calloc(cZ, sizeof(float));
    auto* fHp = static_cast<float*>(hP);
    using Element = aristos::ACC::Element;
    auto* __restrict__ eHp = static_cast<Element*>(hP);
    thrust::default_random_engine rng(47 * (rank + 42));
    thrust::normal_distribution<float> dist(0, 5);
    // Activations
    //std::ranges::fill(fHp, fHp + aZ, 1.0f);
    thrust::generate(fHp, fHp + aZ, [&] { return dist(rng); });
    // gate weights
    //std::ranges::fill(fHp + aZ, fHp + aZ + E * H, 1.0f);
    thrust::generate(fHp + aZ, fHp + aZ + E * H, [&] { return dist(rng); });
    // Expert weights
    // loop for number of experts
    for (uint i = 0; i < nLx; ++i) {
        // expert up
        thrust::generate(fHp + gwZ + i * (P * H), fHp + gwZ + (i + 1) * (P * H),
            [&] { return dist(rng); });
        //makeIdentity<P, H>(fHp + gwZ + i * (P * H));
        // expert down
        //makeIdentity<P, H>(fHp + bZ +  i * (P * H));
        thrust::generate(fHp + bZ + i * (P * H), fHp + bZ + (i + 1) * (P * H),
            [&] { return dist(rng); });
    }
    // bias
    std::ranges::fill(fHp + b2Z, fHp + dZ, 0.0f);

    constexpr cutlass::NumericConverter<Element, float> conv{};
    for (uint i = 0; i < dZ; ++i) {
        eHp[i] = conv(fHp[i]);
    }
    CHECK_ERROR_EXIT(cudaMemcpyAsync(p, eHp, sizeof(Element) * dZ,
        cudaMemcpyHostToDevice,
        aristos::aristosStream));
    for (uint i = 0; i < 1024; ++i) {
        aristos::moe::forwardHost(p, p + dZ * sizeof(Element));
    }
    //aristos::moe::forwardHost<false>(p, p + dZ * sizeof(Element));
    /*using clk = std::chrono::high_resolution_clock;
    std::chrono::duration<float> end {};
    const auto start = clk::now();*/
    //aristos::moe::forwardHost(p, p + dZ * sizeof(Element));
    /*end = clk::now() - start;
    printf("Initialize takes %fms\n", end.count() * 1000);*/

    CHECK_ERROR_EXIT(cudaPeekAtLastError());
    auto* __restrict__ oH = eHp + dZ;
    CHECK_ERROR_EXIT(cudaMemcpyAsync(oH, p + dZ * sizeof(Element),
        sizeof(Element) * S * (PX + H),
        cudaMemcpyDeviceToHost, aristos::aristosStream));
    cuda::std::array<aristos::TPS, E * pEC> tokenOrdering{};
    cuda::std::array<uint, E * pEC> tokenOrderingIds{};
    cuda::std::array<uint, E> expertCounts{};
    CHECK_ERROR_EXIT(cudaMemcpyAsync(tokenOrdering.data(), aristos::hostBookkeeping.tP(),
        sizeof(decltype(tokenOrdering)::value_type) * tokenOrdering.size(),
        cudaMemcpyDeviceToHost, aristos::aristosStream));
    CHECK_ERROR_EXIT(cudaMemcpyAsync(expertCounts.data(),
        aristos::hostBookkeeping.eC(),
        sizeof(decltype(expertCounts)::value_type) * expertCounts.size(),
        cudaMemcpyDeviceToHost, aristos::aristosStream));
    CHECK_ERROR_EXIT(cudaStreamSynchronize(aristos::aristosStream));

    auto* file = std::fopen(std::string("eC_")
            .append(std::to_string(rank)).append(".txt").c_str(), "w");
    for (uint i = 0; i < E; ++i) {
        for (uint j = 0; j < pEC; ++j) {
            tokenOrderingIds[j] = tokenOrdering[j + i * pEC].tokenIdx;
        }
        fmt::println(file, "Rank {}->Expert {}->{} tokens: {}", rank, i, expertCounts[i], tokenOrderingIds);
    }
    std::fclose(file);
    /*if (rank == 1) {
        const auto o = make_tensor(oH + S * PX,
            make_layout(cute::make_shape(S, H), cute::LayoutRight{}));
        using Tiler = cute::Shape<cute::Int<BLOCK_M>, cute::_8>;
        print_tensor(local_tile(o, Tiler{}, 0));
        //print_tensor(o);
    }*/
    CHECK_ERROR_EXIT(cudaFreeAsync(p, aristos::aristosStream));
    aristos::finalize();
    /*const auto og = make_tensor(oH,
        make_layout(cute::make_shape(S, PX), cute::LayoutRight{}));
    const auto o = make_tensor(oH + S * PX,
        make_layout(cute::make_shape(S, H), cute::LayoutRight{}));
    print_tensor(og);
    print_tensor(o);*/
    /*const auto o = make_tensor(eHp,
        cute::Layout<cute::Shape<cute::Int<S>, cute::Int<PX>>, cute::Stride<cute::Int<PX>, cute::_1>>{});
    print_tensor(o);
    for (const auto &[idx, p] : b) {
        printf("{idx: %u, p: %f}\n", idx, p);
    }*/
    /*using clk = std::chrono::high_resolution_clock;
    std::chrono::duration<float> end {};
    const auto start = clk::now();
    aristos::initialize();
    printf("Number of local experts is %u\n", aristos::hostBookkeeping.nLx);
    end = clk::now() - start;
    printf("Initialize takes %fms\n", end.count() * 1000);
    aristos::finalize();*/
    std::free(hP);
}

int main() {
    runOS();
}
