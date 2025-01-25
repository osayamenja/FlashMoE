//
// Created by oja7 on 12/20/24.
//

#ifndef EVAL_CUH
#define EVAL_CUH

#include <vector>

#include <cuda/std/memory>
#include <cuda/std/cstddef>
#include <fmt/ranges.h>
#include <fmt/core.h>
#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h>

#include "include/debug.cuh"
#include "include/os/decider/decider.cuh"
#include "include/types.cuh"
#include "include/topo.cuh"

namespace aristos {
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
        CHECK_ERROR_EXIT(cudaGetDeviceCount(&dev_count));
        CHECK_ERROR_EXIT(cudaGetDeviceProperties(&prop, localRank));
        if (globalRank == 0) {
            fmt::println("Starting Topology Discovery...");
        }
        fmt::println("GlobalRank: {}, LocalRank: {}, Device: {}, Bus ID: {}, Devices: {}",
            globalRank, localRank, prop.name, prop.pciBusID, dev_count);

        CHECK_ERROR_EXIT(cudaSetDevice(localRank));
        const size_t heapBytes = ax * sizeof(floatPair) + sizeof(uint) * (n + 2) + n * sizeof(flagsType) + BETA_BUFFER;
        auto* symHeap = static_cast<cuda::std::byte*>(nvshmem_calloc(heapBytes, sizeof(cuda::std::byte)));
        // Pointer orchestration
        // Starting index of flags array
        auto* flags = CAST_TO(flagsType, symHeap);
        auto* adj = CAST_TO(floatPair, flags + n);
        // Navigate to our slice of the adjacency matrix
        auto* results = adj + globalRank * n;
        // Repurpose a subset of the symmetric heap for local storage.
        auto* rates = CAST_TO(uint, adj + ax * sizeof(floatPair));
        auto* syncArray = rates + n;
        // Starting index of heap
        auto* sHeap = CAST_TO(cuda::std::byte, syncArray + 2);
        const auto pr = globalRank + 1;
        const auto remotePresent = [&n, &results] {
            for (int i = 0; i< n; ++i) {
                if (nvshmem_ptr(results, i) == nullptr) return true;
            }
            return false;
        };
        const auto isRemotePresent = remotePresent();
        const auto sharedSize = n * (sizeof(floatPair) + sizeof(unsigned int));
        constexpr auto skip = 256U;
        constexpr auto blocks = 1U;
        constexpr auto threads = ARISTOS_BLOCK_SIZE;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float duration;
        #pragma unroll
        for (uint i = 0; i < skip; ++i) {
            topology::discover<<<blocks, threads, sharedSize, aristosStream>>>(n, globalRank, isRemotePresent,
            pr, sHeap, flags, results, syncArray, rates);
        }
        CHECK_ERROR_EXIT(cudaMemsetAsync(results, 0, sizeof(floatPair) * n, aristos::aristosStream));
        CHECK_ERROR_EXIT(cudaEventRecord(start, aristos::aristosStream));
        topology::discover<<<blocks, threads, sharedSize, aristosStream>>>(n, globalRank, isRemotePresent,
            pr, sHeap, flags, results, syncArray, rates);
        CHECK_ERROR_EXIT(cudaEventRecord(stop, aristos::aristosStream));
        CHECK_ERROR_EXIT(cudaPeekAtLastError());
        CHECK_ERROR_EXIT(cudaStreamSynchronize(aristosStream));
        cudaEventElapsedTime(&duration, start, stop);
        auto* __restrict__ adjMatrix = static_cast<floatPair*>(calloc(ax * sizeof(floatPair) + n * sizeof(uint),
            sizeof(cuda::std::byte)));

        /// Epilogue
        CHECK_ERROR_EXIT(cudaMemcpyAsync(adjMatrix, adj, ax * sizeof(floatPair) + n * sizeof(uint),
            cudaMemcpyDeviceToHost, aristosStream));
        CHECK_ERROR_EXIT(cudaPeekAtLastError());
        CHECK_ERROR_EXIT(cudaStreamSynchronize(aristosStream));

        std::vector<std::array<float, 2>> temp(n);
        std::vector<uint> temp_f(n);
        auto* file = std::fopen(std::string("adjMatrix_Rank")
            .append(std::to_string(globalRank)).append(".txt").c_str(), "w");
        fmt::print(file, "----> {} processes pair-wise (𝛼 ms, 𝛽 ms/MB) costs <------\n", n);
        for (uint i = 0; i < n; ++i){
            for (uint j = 0; j < n; ++j){
                temp[j] = {adjMatrix[i*n + j].alpha, adjMatrix[i*n + j].beta};
            }
            fmt::print(file, "Rank {}: {:::.2e}\n", i, temp);
        }

        auto* flagsPtr = CAST_TO(uint, adjMatrix + ax);
        for (uint i = 0; i < n; ++i){
            temp_f[i] = flagsPtr[i];
        }
        temp_f[globalRank] = pr;
        fmt::print(file, "Rank {} Flags: {}\n", globalRank, temp_f);
        fmt::println(file, "Duration is {}ms", duration);
        std::fclose(file);
        nvshmem_free(symHeap);
        free(adjMatrix);
        nvshmem_finalize();
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    __host__ __forceinline__
    void testDecider() {
        using clk = std::chrono::high_resolution_clock;
        auto end = std::chrono::duration<double>::zero();
        auto constexpr dim = 4096U;
        auto constexpr intraWidth = 4.0;

        AdjMatrix A(dim);
        std::pair constexpr intraBW = {4.35e-04, 1.29e-02}; // (ms, ms/MB)
        std::pair constexpr interBW = {1.12e-02, 5e-02}; // (ms, ms/MB)

        for(int i = 0; i < dim; ++i){
            A[i] = std::vector<std::pair<double, double>>(dim);
            for(int j = 0; j < dim; ++j){
                if(i == j )[[unlikely]]
                    continue;
                if(static_cast<int>(std::floor(j / static_cast<double>(intraWidth))) == static_cast<int>(std::floor(i / static_cast<double>(intraWidth)))){
                    // intra node partitions
                    A[i][j] = intraBW;
                }
                else{
                    A[i][j] = interBW;
                }
            }
        }

        auto constexpr deviceMem = 16U;
        auto constexpr deviceGigaFlopsPerMs = static_cast<unsigned int>(0.43*312U * (1e9 / (1024*1024*1024)));
        std::vector<aristos::Worker> w;
        for(int i = 0; i < dim; ++i){
            w.emplace_back(i, deviceGigaFlopsPerMs, deviceMem);
        }

        auto constexpr nExp = 4096U;
        auto constexpr expertGigaFlops = 128U; // Giga == 2^30
        std::vector<aristos::Expert> e;
        for(int i = 0; i < nExp; ++i){
            e.emplace_back(i, expertGigaFlops);
        }
        // GPT-3 350M MoE
        const auto m = aristos::ModelConfig(24, 1, 256, 4, 24, 16, 512);

        auto start = clk::now();
        aristos::decider::decide(A, w, e.size()*expertGigaFlops, e.size(), m);
        end = clk::now() - start;
        const auto spec = aristos::decider::decide(A, w, e.size()*expertGigaFlops, e.size(), m);
        fmt::println("Measured time for the Decider is {}s", end.count());
        //fmt::println("Device to Groups: {}", spec);

        const auto g = aristos::subsets(spec, 0);

        //fmt::println("Rank {} Group {}", 0, g);
        std::vector<aristos::Worker> wG;
        for(int i = 0; i < g.size(); ++i){
            wG.emplace_back(g[i], w[i].processingRate, w[i].memoryCapacity);
        }
        start = clk::now();
        const auto assignment = aristos::decider::assign(e, wG);
        end += clk::now() - start;
        fmt::println("Total time is {}s", end.count());
        //fmt::println("Experts to Devices {}", assignment);
    }
}
#endif //EVAL_CUH
