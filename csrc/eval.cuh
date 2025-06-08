/*
 * Copyright (c) 2025, Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by oja7 on 12/20/24.
//

#ifndef EVAL_CUH
#define EVAL_CUH

#include <vector>
#include <cuda/std/cmath>
#include <cuda/std/cstddef>
#include <cuda/std/memory>
#include <cute/tensor.hpp>
#include <fmt/ranges.h>
#include <fmt/core.h>
#include <nvshmem.h>

#include "include/kleos/debug.cuh"
#include "include/os/decider/decider.cuh"
#include "include/kleos/types.cuh"
#include "include/kleos/topo.cuh"

namespace kleos {
    __forceinline__ __host__
    void testTopologyDiscovery() {
        char hostname[HOST_NAME_MAX];
        gethostname(hostname, HOST_NAME_MAX);
         /// Initialization
        nvshmem_init();
        const int n = nvshmem_n_pes();
        const unsigned int ax = n*n;
        const int localRank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
        const int globalRank = nvshmem_my_pe();

        /// Logging
        cudaDeviceProp prop{};
        int dev_count = 0;
        KLEOS_CHECK_CUDA(cudaGetDeviceCount(&dev_count));
        KLEOS_CHECK_CUDA(cudaGetDeviceProperties(&prop, localRank));
        if (globalRank == 0) {
            fmt::println("Starting Topology Discovery...");
        }
        fmt::println("GlobalRank: {}, LocalRank: {}, Device: {}, Bus ID: {}, Devices: {}",
            globalRank, localRank, prop.name, prop.pciBusID, dev_count);

        KLEOS_CHECK_CUDA(cudaSetDevice(localRank));
        const size_t heapBytes = n * sizeof(flagsType) + ax * sizeof(floatPair) + sizeof(WorkerAttribute) * n +
            sizeof(uint) * 2 + n * BETA_BUFFER;
        auto* symHeap = nvshmem_calloc(heapBytes, sizeof(cuda::std::byte));
        // Pointer orchestration
        // Starting index of flags array
        auto* flags = CAST_TO(flagsType, symHeap);
        auto* adj = CAST_TO(floatPair, flags + n);
        // Navigate to our slice of the adjacency matrix
        auto* results = adj + globalRank * n;
        // Repurpose a subset of the symmetric heap for local storage.
        auto* workerAttributes = CAST_TO(WorkerAttribute, adj + ax);
        auto* syncArray = CAST_TO(uint, workerAttributes + n);
        // Starting index of heap
        auto* sHeap = CAST_TO(cuda::std::byte, syncArray + 2);
        const uint16_t pr = globalRank + 1;
        const auto self = WorkerAttribute{cute::half_t(pr), pr};
        const auto remotePresent = [&n, &results] {
            for (int i = 0; i< n; ++i) {
                if (nvshmem_ptr(results, i) == nullptr) return true;
            }
            return false;
        };
        const auto isRemotePresent = remotePresent();
        const auto sharedSize = n * (sizeof(floatPair) + sizeof(unsigned int));
        constexpr auto skip = 256U;
        constexpr auto blocks = 32U;
        constexpr auto threads = KLEOS_BLOCK_SIZE;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float duration;

        auto seqNo = 1U;
        #pragma unroll
        for (uint i = 0; i < skip; ++i) {
            topology::discover<<<blocks, threads, sharedSize, kleosStream>>>(n, globalRank, isRemotePresent,
            self, sHeap, flags, results, syncArray, workerAttributes, seqNo);
            seqNo++;
        }
        KLEOS_CHECK_CUDA(cudaMemsetAsync(results, 0, sizeof(floatPair) * n, kleosStream));
        KLEOS_CHECK_CUDA(cudaEventRecord(start, kleos::kleosStream));
        topology::discover<<<blocks, threads, sharedSize, kleosStream>>>(n, globalRank, isRemotePresent,
            self, sHeap, flags, results, syncArray, workerAttributes, seqNo);
        KLEOS_CHECK_CUDA(cudaEventRecord(stop, kleos::kleosStream));
        KLEOS_CHECK_CUDA(cudaPeekAtLastError());
        KLEOS_CHECK_CUDA(cudaStreamSynchronize(kleosStream));
        cudaEventElapsedTime(&duration, start, stop);
        auto* aP = std::calloc(ax * sizeof(floatPair) + n * sizeof(WorkerAttribute),
            sizeof(cuda::std::byte));
        auto* adjMatrix = static_cast<floatPair*>(aP);
        const auto aMt = make_tensor(adjMatrix, make_layout(cute::make_shape(n, n), cute::LayoutRight{}));
        auto* attributesPtr = CAST_TO(WorkerAttribute, adjMatrix + ax);

        /// Epilogue
        KLEOS_CHECK_CUDA(cudaMemcpyAsync(aP, adj, ax * sizeof(floatPair) + n * sizeof(WorkerAttribute),
            cudaMemcpyDeviceToHost, kleosStream));
        KLEOS_CHECK_CUDA(cudaPeekAtLastError());
        KLEOS_CHECK_CUDA(cudaStreamSynchronize(kleosStream));

        std::vector<std::array<float, 2>> temp(n);
        std::vector<float> aT(n);
        std::vector<uint16_t> aM(n);
        auto* file = std::fopen(std::string("adjMatrix_Rank")
            .append(std::to_string(globalRank)).append(".txt").c_str(), "w");
        fmt::print(file, "----> {} processes pair-wise (𝛼 ms, 𝛽 ms/MB) costs <------\n", n);
        for (uint i = 0; i < n; ++i){
            for (uint j = 0; j < n; ++j){
                const auto [alpha, beta] = aMt(i, j);
                temp[j] = {alpha, beta};
            }
            fmt::print(file, "Rank {}: {:::.2e}\n", i, temp);
        }

        for (uint i = 0; i < n; ++i){
            const auto [t, m] = attributesPtr[i];
            aT[i] = static_cast<float>(t);
            aM[i] = m;
        }
        //fmt::print(file, "Rank {}: \n\t Throughput: {}\n\t MemoryCapacity: {}\n", globalRank, aT, aM);
        fmt::println(file, "Duration is {}ms", duration);
        std::fclose(file);
        nvshmem_free(symHeap);
        std::free(aP);
        nvshmem_finalize();
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    __host__ __forceinline__
    void testDecider() {
        using clk = std::chrono::high_resolution_clock;
        std::chrono::duration<float> end {};
        auto constexpr dim = 8U;
        auto constexpr nExp = 16U;
        auto constexpr intraWidth = 4.0;

        //AdjMatrix A(dim);
        /*std::pair constexpr intraBW = {4.35e-04, 1.29e-02}; // (ms, ms/MB)
        std::pair constexpr interBW = {1.12e-02, 5e-02}; // (ms, ms/MB)*/
        auto constexpr intraBW = floatPair{4.35e-04f, 1.29e-02f}; // (ms, ms/MB)
        auto constexpr interBW = floatPair{1.12e-02f, 5e-02f}; // (ms, ms/MB)

        constexpr auto hPz = 2 * sizeof(Worker) * dim +
            sizeof (Expert) * nExp + sizeof(floatPair) * dim * dim +
                sizeof(uint) * (2 * dim + nExp);
        auto* hP = std::calloc(hPz, sizeof(cuda::std::byte));
        // Pointer salami slicing
        auto* __restrict__ ePwG = CAST_TO(Worker, hP);
        auto* __restrict__ wG = ePwG + dim;
        static_assert(alignof(Worker) % alignof(Expert) == 0);
        auto* __restrict experts = CAST_TO(Expert, wG + dim);
        static_assert(alignof(Expert) % alignof(floatPair) == 0);
        auto* __restrict__ aP = CAST_TO(floatPair, experts + nExp);
        static_assert(alignof(floatPair) % alignof(uint) == 0);
        auto* __restrict__ pT = CAST_TO(uint, aP + dim * dim);
        auto* __restrict__ dTg = pT + dim;
        auto* __restrict__ ePs = dTg + dim;
        const auto adj = make_tensor(aP,
            cute::Layout<cute::Shape<cute::Int<dim>, cute::Int<dim>>,
                        cute::Stride<cute::Int<dim>, cute::_1>>{});

        for(int i = 0; i < dim; ++i){
            for(int j = 0; j < dim; ++j){
                if(i == j)[[unlikely]]
                    continue;
                if(static_cast<int>(std::floor(j / static_cast<float>(intraWidth))) == static_cast<int>(std::floor(i / static_cast<float>(intraWidth)))){
                    // intra node partitions
                    adj(i, j) = intraBW;
                }
                else{
                    adj(i, j) = interBW;
                }
            }
        }

        auto constexpr expertGigaFlops = 128U; // Giga == 2^30
        auto constexpr deviceGigaFlopsPerMs = static_cast<float>(0.43*312U * (1e9 / (1024*1024*1024)));
        for(uint16_t i = 0; i < dim; ++i){
            auto constexpr deviceMem = 8U;
            wG[i] = Worker{i, deviceGigaFlopsPerMs, deviceMem};
        }

        for(uint i = 0; i < nExp; ++i){
            experts[i] = Expert{i, expertGigaFlops};
        }

        auto start = clk::now();
        constexpr Decider<JobType::training> judge{};
        judge(adj, wG, nExp*expertGigaFlops, nExp, dTg);
        end = clk::now() - start;
        std::array<uint, dim> spec{};
        std::memcpy(spec.data(), dTg, sizeof(uint) * dim);
        //judge(adj, wG, nExp*expertGigaFlops, nExp, dTg);
        fmt::println("Measured time for the Decider is {}s", end.count());
        fmt::println("Device to Groups: {}", spec);

        for (uint i = 0 ; i < spec.size(); ++i) {
            dTg[i] = spec[i];
        }
        const auto epWorld = subsets(dTg, pT, dim, 0);
        std::vector<uint> pS (epWorld);
        for (uint i = 0; i < epWorld; ++i) {
            pS[i] = pT[i];
        }
        fmt::println("Rank {} Group {}", 0, pS); // peer translation
        for(uint16_t i = 0; i < epWorld; ++i){
            const auto wId = pT[i];
            ePwG[i] = Worker{i, wG[wId].processingRate, wG[wId].memoryCapacity};
        }
        start = clk::now();
        assign(ePwG, epWorld, experts, nExp, ePs);
        end += clk::now() - start;
        fmt::println("Total time is {}s", end.count());
        std::vector<uint> assignment (nExp);
        for (uint i = 0; i < nExp; ++i) {
            assignment[i] = ePs[i];
        }
        fmt::println("Experts to Devices {}", assignment); // sharding spec
        free(hP);
    }
}
#endif //EVAL_CUH
