/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
#include <climits>
#include <cstdio>
#include <iostream>
#include <array>

#include <cuda/std/chrono>
#include <cuda/atomic>
#include <cuda/barrier>
#include <cuda_runtime.h>

#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h>

#include "include/aristos.cuh"

#include <cooperative_groups/memcpy_async.h>
#include <cub/cub.cuh>
#include <cute/tensor.hpp>
#include <cute/config.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>

#include <fmt/ranges.h>
#include <fmt/core.h>
#include <spdlog/spdlog.h>

#define THREADS_PER_WARP 32
#define THREADS_PER_BLOCK 256
#define NANO_TO_MICRO (cuda::std::nano::den / cuda::std::micro::den)

__device__ __constant__ cuda::atomic<unsigned int, cuda::thread_scope_device> last{1};

#define SEQ 23U
__global__ void benchAtomics(CUTE_GRID_CONSTANT const int iter, unsigned int* flag, bool skip = false, bool shouldPersist = false){
    // initialization
    using Nano = cuda::std::chrono::duration<double, cuda::std::nano>;
    unsigned int* pUnderTest = flag;
    __shared__ unsigned int sharedFlag;
    if(shouldPersist){
        cuda::associate_access_property(flag, cuda::access_property::persisting{});
        //pUnderTest = &sharedFlag;
    }
    __syncthreads();
    atomicExch(&sharedFlag, 0);
    Nano a_flag = Nano::zero(), a_cas = Nano::zero(), a_or = Nano::zero(), a_and = Nano::zero();
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> aFlag (*pUnderTest);
    CUTE_UNROLL
    for(int i = 0; i < iter; ++i){
        uint64_t start, end;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        aFlag.load();
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        a_flag += static_cast<cuda::std::chrono::duration<double, cuda::std::nano>>(end - start);

        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        atomicCAS(pUnderTest, 0, 0);
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        a_cas += static_cast<cuda::std::chrono::duration<double, cuda::std::nano>>(end - start);

        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        atomicOr(pUnderTest, 0U);
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        a_or += static_cast<cuda::std::chrono::duration<double, cuda::std::nano>>(end - start);

        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        atomicAnd(pUnderTest, 1U);
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        a_and += static_cast<cuda::std::chrono::duration<double, cuda::std::nano>>(end - start);

    }
    using BlockReduce = cub::BlockReduce<Nano, THREADS_PER_BLOCK>;
    __shared__ BlockReduce::TempStorage temp_storage;

    a_flag = BlockReduce(temp_storage).Reduce(a_flag, cub::Max());
    a_cas = BlockReduce(temp_storage).Reduce(a_cas, cub::Max());
    a_or = BlockReduce(temp_storage).Reduce(a_or, cub::Max());
    a_and = BlockReduce(temp_storage).Reduce(a_and, cub::Max());
    cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds, double>(a_flag);

    if(aristos::block::threadID() == 0 && !skip){
        printf("Block Id is %u, a_flag: {T: %f, V: %d}, a_cas: {T: %f, V:%u}, a_or: {T: %f, V:%u}, a_and: {T: %f, V: %u},"
               "isShared: %s\n",
               aristos::grid::blockID(),
               cuda::std::chrono::duration_cast<cuda::std::chrono::duration<double, cuda::std::micro>>(a_flag / (iter*1.0)).count(),
               aFlag.load(),
               cuda::std::chrono::duration_cast<cuda::std::chrono::duration<double, cuda::std::micro>>(a_cas/(iter*1.0)).count(),
               atomicCAS(pUnderTest, 0, 0),
               cuda::std::chrono::duration_cast<cuda::std::chrono::duration<double, cuda::std::micro>>(a_or/(iter*1.0)).count(),
               atomicOr(pUnderTest, 0U),
               cuda::std::chrono::duration_cast<cuda::std::chrono::duration<double, cuda::std::micro>>(a_and/(iter*1.0)).count(),
               atomicAnd(pUnderTest, 1U),
               (shouldPersist)? "Yes" : "No");
    }
}

template<unsigned int bM=128, unsigned int bN=128, unsigned int bK=8, unsigned int bP=3>
__global__ void occupancyTestKernel(){
    __shared__ float sharedA[cute::cosize_v<decltype(cute::make_layout(cute::make_shape(cute::Int<bM>{}, cute::Int<bK>{}, cute::Int<bP>{})))>];
    __shared__ float sharedB[cute::cosize_v<decltype(cute::make_layout(cute::make_shape(cute::Int<bN>{}, cute::Int<bK>{}, cute::Int<bP>{})))>];
}

#define SUPPORTED = 1;
namespace aristos{
    bool isInitialized = false;
    int blocksPerSM = 0;
    auto aristosStream = cudaStreamPerThread;
    void aristosInit(const unsigned int& seqLen, const unsigned int& embedDim, const unsigned int& hiddenProjDim,
                     const unsigned int& k, const unsigned int& capacityFactor, const unsigned int& globalWorld,
                     const unsigned int& numExperts) {
        assert(!isInitialized);
        isInitialized = true;
        int cudaDevAttribute = 0;
        int dev = 0;
        CUTE_CHECK_ERROR(cudaGetDevice(&dev));
        CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&cudaDevAttribute, cudaDevAttrMultiProcessorCount, dev));
        //TODO make below flexible to tiling
        const auto GEMMBlocks = cute::ceil_div(seqLen, ARISTOS_M_BATCH) * cute::ceil_div(hiddenProjDim, ARISTOS_N_BATCH);
        CUTE_CHECK_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocksPerSM,
                occupancyTestKernel<ARISTOS_M_BATCH, ARISTOS_N_BATCH, ARISTOS_K_BATCH, ARISTOS_PIPELINE_STAGES>,
                ARISTOS_BLOCK_SIZE,
                0));
        const int maxActiveBlocks = blocksPerSM * cudaDevAttribute;
        assert(GEMMBlocks + 2 <= maxActiveBlocks);
        CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&cudaDevAttribute,
                                                cudaDevAttrMemoryPoolsSupported, dev));
        assert(cudaDevAttribute);
        CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&cudaDevAttribute, cudaDevAttrComputeCapabilityMajor, dev));
        /// Due to NVSHMEM: https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html#hardware-requirements
        assert(cudaDevAttribute >= 7);

        // Good to go! Let's do some initialization
        // Run decider
        // ...
        // generates the below
        const unsigned int numLocalExperts = 0;
        std::vector<specType> parallelSpec{};
        std::vector<specType> translation{};
        const unsigned int numNeighbors = translation.size();

        // initialize NVSHMEM
        nvshmem_init();
        CUTE_CHECK_ERROR(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
        // Allocate Symmetric Heap + Flags
        size_t heapBytes = ((4 * seqLen * (embedDim + k + 2))*sizeof(maxPrecision)) + (STAGES * numNeighbors * sizeof(flagsType));
        heapBytes = std::max(heapBytes, globalWorld * (BETA_BUFFER + ((2*globalWorld + 1)*sizeof(double))));
        /// Allocates aligned memory
        const auto sHeap = nvshmem_align(HEAP_ALIGNMENT, heapBytes);
        CUTE_CHECK_ERROR(cudaMemsetAsync(sHeap, 0, heapBytes, aristosStream));

        //Final Initialization
        hostMoEConfig = Config();
        specType* bookKeeping;
        /// pubQueueLen -> Multiplied by 2 to simulate pair: {index, numTokens}
        /// + translationLen + shardSpecLen +
        /// syncVectorLen -> {syncGrid, checkpoints}
        hostMoEConfig.bookKeepingLen = (numLocalExperts * numNeighbors * 2) + numNeighbors + numExperts + (numNeighbors * 2);
        CUTE_CHECK_ERROR(cudaMallocAsync(&bookKeeping,
                                    sizeof(specType)*hostMoEConfig.bookKeepingLen,
                                    aristosStream));
        CUTE_CHECK_ERROR(cudaMemsetAsync(bookKeeping, 0, hostMoEConfig.bookKeepingLen, aristosStream));

        //TODO init with host memcpy?
        hostMoEConfig.numP2PPublisherBlocks = maxActiveBlocks - GEMMBlocks - 1;
        hostMoEConfig.worldSize = numNeighbors;
        hostMoEConfig.bookKeeping = bookKeeping;
        hostMoEConfig.sHeap = static_cast<cuda::std::byte*>(sHeap);
        CUTE_CHECK_ERROR(cudaMemcpyToSymbolAsync(moeConfig,
                                            &hostMoEConfig, sizeof(Config), 0, cudaMemcpyHostToDevice, aristosStream));
        CUTE_CHECK_ERROR(cudaPeekAtLastError());
        CUTE_CHECK_ERROR(cudaStreamSynchronize(aristosStream));
    }

    void forwardHost(){

    }

    void backwardHost(){
    }

    void aristosFinalize(){
        assert(isInitialized);
        isInitialized = false;
        CUTE_CHECK_ERROR(cudaFreeAsync(hostMoEConfig.bookKeeping, aristosStream));
        nvshmem_free(hostMoEConfig.sHeap);
        nvshmem_finalize();
        CUTE_CHECK_ERROR(cudaPeekAtLastError());
        CUTE_CHECK_ERROR(cudaStreamSynchronize(aristosStream));
    }
}

extern constexpr int peers = 1;
extern constexpr int stages = 2;
extern constexpr int cells = 2;

extern constexpr int capacity = 1;
extern constexpr int k = 0;
extern constexpr int embedDim = 0;
extern constexpr int tokens = capacity * (embedDim + k + 2);

extern constexpr int peerStride = stages * cells * tokens;
extern constexpr int stageStride = cells * tokens;
extern constexpr int cellStride = tokens;
extern constexpr int tokenStride = (embedDim + k + 2);

template<typename T>
CUTE_DEVICE
T* getTokenPointer(T* const& __restrict__ addr, unsigned int const& peer, unsigned int const& stage, unsigned int const& cell, unsigned int const& token){
    return addr + ((peer * peerStride) + (stage * stageStride) + (cell * cellStride) + (token * tokenStride));
}

__global__ void benchTen(unsigned int* foo, bool skip = false){
    auto t = make_tensor(cute::make_gmem_ptr(foo), make_shape(peers, make_shape(cute::make_shape(stages, cells), tokens)), cute::LayoutRight{});
    double cuteDuration = 0.0, rawDuration = 0.0;
    CUTE_UNROLL
    for(unsigned int i = 0; i < 1024; ++i){
        uint64_t start = 0, end = 0;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        &t(0, make_coord(cute::make_coord(0,1),0));
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        cuteDuration = static_cast<double>(end - start);

        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        getTokenPointer(foo, 0, 0, 1, 0);
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        rawDuration = static_cast<double>(end - start);
    }
    if(aristos::block::threadID() == 0 && !skip){
        printf("Block Id is %u, cute_t: {T: %f, V: %d}, raw_t: {T: %f, V:%d}\n",
               aristos::block::threadID(),
               cuteDuration,
               t(0, make_coord(cute::make_coord(0,1),0)),
               rawDuration,
               *getTokenPointer(foo, 0, 0, 1, 0));
    }
}

void testBenchTen() {
    unsigned int* p;
    constexpr std::array<unsigned int, 4> arr {{1234, 56, 78, 0}};
    CUTE_CHECK_ERROR(cudaMalloc(&p, sizeof(unsigned int) * arr.size()));
    CUTE_CHECK_ERROR(cudaMemcpy(p, arr.data(), sizeof(unsigned int) * arr.size(), cudaMemcpyHostToDevice));
    benchTen<<<1,1>>>(p);
    CUTE_CHECK_LAST();
}

__device__ unsigned int phases = 0U;
__device__ unsigned int makeshiftBarrier = 0U;
__global__ void benchBarrier(cuda::barrier<cuda::thread_scope_device>* __restrict__ bar,
    CUTE_GRID_CONSTANT const unsigned int n, CUTE_GRID_CONSTANT const bool skip = false){
    double makeshiftDuration = 0.0, quoDuration = 0.0;
    constexpr auto iter = 1024;
    CUTE_UNROLL
    for(unsigned int i = 0; i < iter; ++i){
        uint64_t start = 0, end = 0;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        if(!aristos::block::threadID()){
            /// Arrive
            if(auto nextPhase = aristos::atomicLoad(&phases) + 1;
                (atomicAdd(&makeshiftBarrier, 1U) + 1) == n * nextPhase){
                atomicAdd(&phases, 1U);
                /// Could execute completion function here
            }
            else{
                /// You could do some other task prior to waiting
                /// Wait
                while(aristos::atomicLoad(&phases) != nextPhase){
                }
                /// Could execute completion function here
            }
        }
        __syncthreads();
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        makeshiftDuration += static_cast<double>(end - start) / static_cast<double>(iter*NANO_TO_MICRO);
    }

    CUTE_UNROLL
    for(unsigned int i = 0; i < iter; ++i){
        uint64_t start = 0, end = 0;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        if(!aristos::block::threadID()){
            bar->arrive_and_wait();
        }
        __syncthreads();
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        quoDuration += static_cast<double>(end - start) / static_cast<double>(iter*NANO_TO_MICRO);
    }

    using BlockReduce = cub::BlockReduce<double, ARISTOS_BLOCK_SIZE>;
    __shared__ BlockReduce::TempStorage temp_storage;

    makeshiftDuration = BlockReduce(temp_storage).Reduce(makeshiftDuration, cub::Max());
    quoDuration = BlockReduce(temp_storage).Reduce(quoDuration, cub::Max());
    if(aristos::block::threadID() == 0 && !skip){
        printf("Block Id is %u, makeshiftTime: {T: %f}, quoTime: {T: %f}\n", aristos::grid::blockID(),
            makeshiftDuration, quoDuration);
    }
}

void testBenchBarrier() {
    constexpr unsigned int n = 32;
    cuda::barrier<cuda::thread_scope_device>* deviceBarrier;
    const auto hostBarrier = new cuda::barrier<cuda::thread_scope_device>{n};
    CUTE_CHECK_ERROR(cudaMalloc(&deviceBarrier, sizeof(cuda::barrier<cuda::thread_scope_device>)));
    CUTE_CHECK_ERROR(cudaMemcpy(deviceBarrier, hostBarrier, sizeof(cuda::barrier<cuda::thread_scope_device>), cudaMemcpyHostToDevice));
    benchBarrier<<<n, ARISTOS_BLOCK_SIZE>>>(deviceBarrier, n);
    CUTE_CHECK_LAST();
    delete hostBarrier;
    cudaFree(deviceBarrier);
}

__global__ void testAlign(float* result) {
    extern __shared__ float remoteDurations[];
    extern __shared__ int x[];
    unsigned long int start, end;
    remoteDurations[2*aristos::block::threadID()] = static_cast<float>(aristos::block::threadID()*(aristos::grid::blockID() + 1));
    remoteDurations[2*aristos::block::threadID() + 1] = static_cast<float>(aristos::block::threadID()*(aristos::grid::blockID() + 1)) * 2.0f;
    asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
    memcpy_async(cooperative_groups::this_thread_block(),
                 result + aristos::grid::blockID()*2*ARISTOS_BLOCK_SIZE,
                 remoteDurations,
                 2*sizeof(float)*ARISTOS_BLOCK_SIZE);
    cooperative_groups::wait_prior<1>(cooperative_groups::this_thread_block());
    asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
    if (aristos::block::threadID() == 0) {
        printf("Addr is %p, Latency is %fms, W is %f, raw is %fus\n",
            remoteDurations,
            cuda::std::chrono::duration_cast<Milli>(static_cast<Nano>(end - start)).count(),
            cuda::std::chrono::duration<double>(end - start).count(),
            static_cast<double>(end - start) / NANO_TO_MICRO);
    }
}

__forceinline__
void testTopologyDiscovery() {
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
     /// Initialization
    nvshmem_init();
    const int n = nvshmem_n_pes();
    const unsigned int ax = 2*n*n;
    const int localRank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    const int globalRank = nvshmem_my_pe();

    /// Logging
    cudaDeviceProp prop{};
    int dev_count = 0;
    CUTE_CHECK_ERROR(cudaGetDeviceCount(&dev_count));
    CUTE_CHECK_ERROR(cudaGetDeviceProperties(&prop, localRank));
    spdlog::set_pattern(std::string("[").append(std::string(hostname))
    .append(":Rank ").append(std::to_string(localRank)).append("] [%c] [%^%l%$] %v"));
    spdlog::info("Starting Topology Discovery...");
    spdlog::info("GlobalRank: {}, LocalRank: {}, Device: {}, Bus ID: {}, Devices: {}",
        globalRank, localRank, prop.name, prop.pciBusID, dev_count);

    CUTE_CHECK_ERROR(cudaSetDevice(localRank));
    const size_t heapBytes = n * (BETA_BUFFER + ((2*n + 1)*sizeof(double)));
    void* symHeap = nvshmem_calloc(heapBytes, sizeof(cuda::std::byte));
    /// Pointer orchestration
    static_assert(sizeof(aristos::flagsType) == sizeof(uint64_t));
    // Navigate to our slice of the adjacency matrix
    auto* results = static_cast<double*>(symHeap) + 2*globalRank*n;
    // Starting index of flags array
    auto* flags = static_cast<uint64_t*>(symHeap) + ax;
    // Starting index of heap
    auto* sHeap = static_cast<double*>(symHeap) + ax + n;
    const auto pr = globalRank + 1;
    const auto remotePresent = [&n, &results] {
        for (int i = 0; i< n; ++i) {
            if (nvshmem_ptr(results, i) == nullptr) return true;
        }
        return false;
    };
    const auto isRemotePresent = remotePresent();
    constexpr auto skip = 32U;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    constexpr unsigned int zeroVal = GRAND_MASTER;
    float duration;
    uint64_t hostSeqNo = 0U;
    static_assert(sizeof(decltype(hostSeqNo)) == sizeof(decltype(aristos::seqNo)));
    CUTE_CHECK_ERROR(cudaMemcpyToSymbol(aristos::seqNo, &hostSeqNo, sizeof(decltype(aristos::seqNo))));
    #pragma unroll
    for (uint i = 0; i < skip; ++i) {
        aristos::topology::discover<<<ARISTOS_SUPER_BLOCK_SIZE, ARISTOS_BLOCK_SIZE>>>(n, globalRank, isRemotePresent,
        pr, sHeap, flags, results);
        CUTE_CHECK_ERROR(cudaMemcpyToSymbol(aristos::publisher::blockade, &zeroVal, sizeof(decltype(aristos::publisher::blockade))));
        CUTE_CHECK_ERROR(cudaMemcpyToSymbol(aristos::publisher::baton, &zeroVal, sizeof(decltype(aristos::publisher::baton))));
        hostSeqNo = hostSeqNo + 1;
        CUTE_CHECK_ERROR(cudaMemcpyToSymbol(aristos::seqNo, &hostSeqNo, sizeof(decltype(aristos::seqNo))));
    }
    CUTE_CHECK_ERROR(cudaMemset(results, 0, sizeof(double)*2*n));
    CUTE_CHECK_ERROR(cudaEventRecord(start));
    aristos::topology::discover<<<ARISTOS_SUPER_BLOCK_SIZE, ARISTOS_BLOCK_SIZE>>>(n, globalRank, isRemotePresent,
        pr, sHeap, flags, results);
    CUTE_CHECK_ERROR(cudaEventRecord(stop));
    CUTE_CHECK_LAST();
    cudaEventElapsedTime(&duration, start, stop);
    void* adjMatrix = malloc((ax + n) * sizeof(double));

    /// Epilogue
    CUTE_CHECK_ERROR(cudaMemcpy(adjMatrix, symHeap, (ax + n)*sizeof(double), cudaMemcpyDeviceToHost));
    CUTE_CHECK_LAST();
    auto* adjPtr = static_cast<double*>(adjMatrix);
    std::vector<std::array<double, 2>> temp(n);
    std::vector<aristos::flagsType> temp_f(n);
    auto* file = std::fopen(std::string("adjMatrix_Rank")
        .append(std::to_string(globalRank)).append(".txt").c_str(), "w");
    fmt::print(file, "----> {} processes pair-wise (𝛼 ms, 𝛽 ms/MB) costs <------\n", n);
    for (uint i = 0; i < n; ++i){
        for (uint j = 0; j < 2*n; j+=2){
            temp[j/2] = {adjPtr[(i*2*n) + j], adjPtr[(i*2*n) + j + 1]};
        }
        fmt::print(file, "Rank {}: {:::.2e}\n", i, temp);
    }
    static_assert(sizeof(aristos::flagsType) == sizeof(double));
    auto* flagsPtr = static_cast<aristos::flagsType*>(adjMatrix) + ax;
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
}

__host__
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

int main() {
    testTopologyDiscovery();
    return 0;
}

