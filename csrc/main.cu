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
T* getTokenPointer(T* const& addr, unsigned int const& peer, unsigned int const& stage, unsigned int const& cell, unsigned int const& token){
    return addr + ((peer * peerStride) + (stage * stageStride) + (cell * cellStride) + (token * tokenStride));
}

__global__ void benchTen(unsigned int* foo, bool skip = false, bool shouldPersist = false){
    using Nano = cuda::std::chrono::duration<double, cuda::std::nano>;
    Nano cute_t = Nano::zero();
    Nano raw_t = Nano::zero();
    auto t = cute::make_tensor(cute::make_gmem_ptr(foo), cute::make_shape(peers, cute::make_shape(cute::make_shape(stages, cells), tokens)), cute::LayoutRight{});
    if(shouldPersist){
        cuda::associate_access_property(foo, cuda::access_property::persisting{});
    }

    CUTE_UNROLL
    for(unsigned int i = 0; i < 1000; ++i){
        uint64_t start = 0, end = 0;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        &t(0, cute::make_coord(cute::make_coord(0,1),0));
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        cute_t += static_cast<cuda::std::chrono::duration<double, cuda::std::nano>>(end - start);

        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        getTokenPointer(foo, 0, 0, 1, 0);
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        raw_t += static_cast<cuda::std::chrono::duration<double, cuda::std::nano>>(end - start);
    }

    using BlockReduce = cub::BlockReduce<Nano, THREADS_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    cute_t = BlockReduce(temp_storage).Reduce(cute_t, cub::Max());
    raw_t = BlockReduce(temp_storage).Reduce(raw_t, cub::Max());
    if(aristos::block::threadID() == 0 && !skip){
        printf("Block Id is %u, cute_t: {T: %f, V: %d, Micro: %f}, raw_t: {T: %f, V:%d, Micro: %f} "
               "persist: %s\n",
               aristos::block::threadID(),
               (cute_t / (1000*1.0)).count(),
               t(0, cute::make_coord(cute::make_coord(0,1),0)),
               cuda::std::chrono::duration_cast<cuda::std::chrono::duration<double, cuda::std::micro>>(cute_t/(1000*1.0)).count(),
               (raw_t / (1000*1.0)).count(),
               *getTokenPointer(foo, 0, 0, 1, 0),
               cuda::std::chrono::duration_cast<cuda::std::chrono::duration<double, cuda::std::micro>>(raw_t/(1000*1.0)).count(),
               (shouldPersist)? "Yes" : "No");
    }
}

__device__ unsigned int testStages = 0;
__global__ void benchBarrier(unsigned int* b, cuda::barrier<cuda::thread_scope_device>* bar, unsigned int n, bool skip = false, bool persist = false){
    using Nano = cuda::std::chrono::duration<double, cuda::std::nano>;
    Nano bar_ptr = Nano::zero();
    Nano bar_obj = Nano::zero();
    if(persist){
        associate_access_property(b, cuda::access_property::persisting{});
        associate_access_property(&testStages, cuda::access_property::persisting{});
        associate_access_property(bar, cuda::access_property::persisting{});
    }
    /*aristos::barrier::init(n, persist);*/
    constexpr auto iter = 1024;
    CUTE_UNROLL
    for(unsigned int i = 0; i < iter; ++i){
        uint64_t start = 0, end = 0;
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(start)::);
        if(!aristos::block::threadID()){
            auto nextStage = aristos::atomicLoad(&testStages) + 1;
            /// Arrive
            if((atomicAdd(b, 1U) + 1) == n * nextStage){
                atomicAdd(&testStages, 1U);
                /// Could execute completion function here
            }
            else{
                /// You could do some other task prior to waiting
                /// Wait
                while(aristos::atomicLoad(&testStages) != nextStage){
                    __nanosleep(2);
                }
                /// Could execute completion function here
            }
            /*aristos::barrier::wait(aristos::barrier::arrive());*/
        }
        __syncthreads();
        asm volatile("mov.u64 %0, %%globaltimer;": "=l"(end)::);
        bar_ptr += static_cast<cuda::std::chrono::duration<double, cuda::std::nano>>(end - start);
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
        bar_obj += static_cast<cuda::std::chrono::duration<double, cuda::std::nano>>(end - start);
    }

    if(!aristos::block::threadID() && !skip){
        printf("Block Id is %u, bar_ptr: {T: %f}, bar_obj: {T: %f}, persist: %s\n",
               aristos::grid::blockID(),
               cuda::std::chrono::duration_cast<cuda::std::chrono::duration<double, cuda::std::micro>>(bar_ptr / (iter*1.0)).count(),
               cuda::std::chrono::duration_cast<cuda::std::chrono::duration<double, cuda::std::micro>>(bar_obj / (iter*1.0)).count(),
               persist? "Yes" : "No");
    }
}

template <class DispatchPolicy,
        class TMMA, class TiledCopyA, class TiledCopyB,
                class ElementA = cute::half_t, class ElementB = cute::half_t,
        class ElementC = cute::half_t,unsigned int stages = 1>
__global__ void processorSpec(DispatchPolicy dispatchPolicy, ElementA* A,
                              ElementA* B, ElementC* C,
                              TMMA tmma, TiledCopyA tca, TiledCopyB tcb,
                              const int& M, const int& N, const int& K){
    using namespace cute;
    using ProblemShape = decltype(make_shape(M, N, K));
    using StrideA = Underscore;
    using StrideB = Underscore;
    using StrideC = Underscore;
    using tiledMma = TMMA;
    using GmemCopyA = TiledCopyA;
    using SmemLayoutAtomA = decltype(
    composition(Swizzle<1,2,3>{},
                Layout<Shape<Int<ARISTOS_M_BATCH>, Int<ARISTOS_K_BATCH>>>{}));
    using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;
    using TransformA = identity; // upcast from fp8 to fp16
    using GmemCopyB = TiledCopyB;
    using SmemLayoutAtomB = decltype(
    composition(Swizzle<1,2,3>{},
                Layout<Shape<Int<ARISTOS_N_BATCH>, Int<ARISTOS_K_BATCH>>>{}));
    using SmemCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, ElementB>;
    using TransformB = identity; // upcast from fp8 to fp16
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveMma<
            DispatchPolicy,
            ProblemShape,
            ElementA,
            StrideA,
            ElementB,
            StrideB,
            tiledMma,
            GmemCopyA,
            SmemLayoutAtomA,
            SmemCopyAtomA,
            TransformA,
            GmemCopyB,
            SmemLayoutAtomB,
            SmemCopyAtomB,
            TransformB>;
    auto problemShape = ProblemShape{};
    auto ctaTiler = make_shape(ARISTOS_M_BATCH, ARISTOS_N_BATCH, ARISTOS_K_BATCH);
    auto ma = make_tensor(make_gmem_ptr(A), select<0,2>(problemShape), StrideA{});
    auto mb = make_tensor(make_gmem_ptr(B), select<1,2>(problemShape), StrideB{});
    auto mc = make_tensor(make_gmem_ptr(C), select<0,1>(problemShape), StrideC{});
    auto altGridDimX = cute::ceil_div(get<0>(problemShape), cute::get<0>(ctaTiler));
    auto cta_coord = make_coord((blockIdx.x % altGridDimX), (blockIdx.x / altGridDimX), _);
    auto gA = local_tile(ma, ctaTiler, cta_coord, Step<_1, X,_1>{});
    auto gB = local_tile(mb, ctaTiler, cta_coord, Step< X,_1,_1>{});
    auto gC = local_tile(mc, ctaTiler, cta_coord, Step<_1,_1, X>{});
    auto k_tile_iter = cute::make_coord_iterator(size<2>(gA));
    int k_tile_count = size<2>(gA);
    auto accum = partition_fragment_C(tmma, Shape<Int<ARISTOS_M_BATCH>, Int<ARISTOS_N_BATCH>>{});
    clear(accum);
    extern __shared__ cuda::std::byte sharedBuf[];
    
    CollectiveMainloop  expert;
    expert(accum, gA, gB, accum, k_tile_iter, k_tile_count, Underscore{}, threadIdx.x, sharedBuf);
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

    /// Logging
    cudaDeviceProp prop{};
    int dev_count = 0;
    CUTE_CHECK_ERROR(cudaGetDeviceCount(&dev_count));
    CUTE_CHECK_ERROR(cudaGetDeviceProperties(&prop, localRank));
    spdlog::set_pattern(std::string("[").append(std::string(hostname))
    .append(":Rank ").append(std::to_string(localRank)).append("] [%c] [%^%l%$] %v"));
    spdlog::info("Starting Topology Discovery...");
    spdlog::info("GlobalRank: {}, LocalRank: {}, Device: {}, Bus ID: {}, Devices: {}",
        nvshmem_my_pe(), localRank, prop.name, prop.pciBusID, dev_count);

    CUTE_CHECK_ERROR(cudaSetDevice(localRank));
    const size_t heapBytes = n * (BETA_BUFFER + ((2*n + 1)*sizeof(double)));
    void* symHeap = nvshmem_align(16, heapBytes);
    CUTE_CHECK_ERROR(cudaMemset(symHeap, 0, heapBytes));
    CUTE_CHECK_ERROR(cudaPeekAtLastError());
    /// Pointer orchestration
    static_assert(sizeof(aristos::flagsType) == sizeof(uint64_t));
    auto* results = static_cast<double*>(symHeap);
    auto* flags = static_cast<uint64_t*>(symHeap) + ax;
    auto* sHeap = static_cast<double*>(symHeap) + ax + n;
    const auto pr = nvshmem_my_pe() + 1;
    const auto remotePresent = [&n, &symHeap] {
        for (int i = 0; i< n; ++i) {
            if (nvshmem_ptr(symHeap, i) == nullptr) return true;
        }
        return false;
    };

    aristos::topology::discover<<<2, ARISTOS_BLOCK_SIZE>>>(n, nvshmem_my_pe(), remotePresent(),
        pr, sHeap, flags, results + 2*nvshmem_my_pe()*n);
    CUTE_CHECK_ERROR(cudaPeekAtLastError());
    void* adjMatrix = malloc((ax + n) * sizeof(double));

    /// Epilogue
    CUTE_CHECK_ERROR(cudaMemcpy(adjMatrix, symHeap, (ax + n)*sizeof(double), cudaMemcpyDeviceToHost));
    CUTE_CHECK_LAST();
    auto* adjPtr = static_cast<double*>(adjMatrix);
    std::vector<std::array<double, 2>> temp(n);
    std::vector<aristos::flagsType> temp_f(n);
    auto* file = std::fopen(std::string("adjMatrix_Rank")
        .append(std::to_string(nvshmem_my_pe())).append(".txt").c_str(), "w");
    fmt::print(file, "----> {} processes pair-wise (𝛼 ms, 𝛽 ms/MB) costs <------\n", n);
    for (uint i = 0; i < n; ++i){
        for (uint j = 0; j < 2*n; j+=2){
            temp[j/2] = {adjPtr[(i*2*n) + j], adjPtr[(i*2*n) + j + 1]};
        }
        fmt::print(file, "Rank {}: {:::.2e}\n", i, temp);
    }
    static_assert(sizeof(aristos::flagsType) == sizeof(double));
    auto* flagsPtr = static_cast<aristos::flagsType*>(adjMatrix) + ax;
    for (uint i = 0; i < n; ++i)
    {
        temp_f[i] = flagsPtr[i];
    }
    fmt::print(file, "Rank {} Flags: {}\n", nvshmem_my_pe(), temp_f);
    std::fclose(file);
    nvshmem_free(symHeap);
    free(adjMatrix);
    nvshmem_finalize();
}

int main() {
    /*auto size = 64;
    unsigned int* p;
    cuda::barrier<cuda::thread_scope_device>* b;
    auto host_b = new cuda::barrier<cuda::thread_scope_device>{size};
    CUTE_CHECK_ERROR(cudaMallocAsync(&b, sizeof(cuda::barrier<cuda::thread_scope_device>), cudaStreamPerThread));
    CUTE_CHECK_ERROR(cudaMallocAsync(&p, sizeof(unsigned int), cudaStreamPerThread));
    CUTE_CHECK_ERROR(cudaMemcpyAsync(b, host_b, sizeof(cuda::barrier<cuda::thread_scope_device>), cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUTE_CHECK_ERROR(cudaMemsetAsync(p, 0, sizeof(unsigned int), cudaStreamPerThread));
    CUTE_CHECK_LAST();

    CUTE_UNROLL
    for(int i = 0; i < 16; ++i){
        benchBarrier<<<size, THREADS_PER_BLOCK>>>(p, b, size, true);
    }
    benchBarrier<<<size, THREADS_PER_BLOCK>>>(p, b, size, false);
    benchBarrier<<<size, THREADS_PER_BLOCK>>>(p, b, size, false, true);
    CUTE_CHECK_LAST();
    CUTE_CHECK_ERROR(cudaFreeAsync(p, cudaStreamPerThread));
    CUTE_CHECK_ERROR(cudaFreeAsync(b, cudaStreamPerThread));
    CUTE_CHECK_ERROR(cudaStreamSynchronize(cudaStreamPerThread));
    free(host_b);*/
    /*using djs = boost::disjoint_sets_with_storage<boost::identity_property_map,
    boost::identity_property_map, boost::find_with_path_halving>;
    auto constexpr n = 5;
    djs groups(n);
    for(int i = 0; i < n; ++i){
        groups.make_set(i);
    }
    auto p = groups.parents();
    std::cout << "Before Merges: ";
    aristos::printContainer(p);
    std::cout << std::endl;
    groups.link(0, 1);
    std::cout << "Merged 0 and 1: ";
    p = groups.parents();
    aristos::printContainer(p);
    std::cout << std::endl;
    groups.link(2, 3);
    std::cout << "Merged 2 and 3: ";
    p = groups.parents();
    aristos::printContainer(p);
    std::cout << std::endl;

    std::unordered_map<decltype(p)::value_type, std::vector<int>> sets{};
    for(int i = 0; i < n; ++i){
        sets[p[i]].push_back(i);
    }

    aristos::printMapCV(sets);
    std::cout << ']' << std::endl;
    std::cout << aristos::Streamable<decltype(sets)::key_type> << std::endl;

    std::array<aristos::Expert, 4> exps {{aristos::Expert(0, 3),
                                          aristos::Expert(1, 12),
                                          aristos::Expert(2, 6),
                                          aristos::Expert(3, 3)}};
    std::array<aristos::Worker, 4> workers {{aristos::Worker(0, 4, 4),
                                             aristos::Worker(1, 8, 6),
                                             aristos::Worker(2, 2, 6),
                                             aristos::Worker(3, 2, 6)}};
    std::array<std::string, 4> sv;
    std::ranges::transform(workers.begin(), workers.end(), sv.begin(), [](const aristos::Worker& e){
       return e.toString();
    });
    aristos::printContainer<sv.size()>(sv);
    std::cout << std::endl << "Sorted 👇" << std::endl;
    std::ranges::sort(workers, std::greater<>());
    std::ranges::transform(workers.begin(), workers.end(), sv.begin(), [](const aristos::Worker& e){
        return e.toString();
    });
    aristos::printContainer<sv.size()>(sv);*/
    /*printf("Number of blocks per SM %u\n", aristos::blocksPerSM);
    auto constexpr dim = 4U;
    auto constexpr intraWidth = 4.0;

    AdjMatrix A(dim);
    std::pair constexpr intraBW = {0.0, 1}; // (ms, ms/MB)
    std::pair constexpr interBW = {0.03, 0.054};

    CUTE_UNROLL
    for(int i = 0; i < dim; ++i){
        A[i] = std::vector<std::pair<double, double>>(dim);
        CUTE_UNROLL
        for(int j = 0; j < dim; ++j){
            if(i == j )[[unlikely]]
                continue;
            if(static_cast<int>(std::floor(j / static_cast<double>(intraWidth)) == static_cast<int>(std::floor(i / static_cast<double>(intraWidth))))){
                // intra node partitions
                A[i][j] = intraBW;
            }
            else{
                A[i][j] = interBW;
            }
        }
    }

    auto const A100Rate = static_cast<unsigned long>(std::ceil(19UL * 1E12));
    auto constexpr deviceMem = 16U;
    std::vector<aristos::Worker> w;
    CUTE_UNROLL
    for(int i = 0; i < dim; ++i){
        w.emplace_back(i, 1, deviceMem);
    }

    auto constexpr nExp = 16U;
    std::vector<aristos::Expert> e;
    //TODO KiloFLOPs
    unsigned long constexpr expC = 16L * 4L * 2048L * (1024L * 1024L);
    CUTE_UNROLL
    for(int i = 0; i < nExp; ++i){
        e.emplace_back(i, 1);
    }

    aristos::hostMoEConfig = aristos::Config();
    aristos::hostMoEConfig.redAmount = 1;
    aristos::hostMoEConfig.globalBatch = 256;
    aristos::hostMoEConfig.miniBatch = 64;
    aristos::hostMoEConfig.numLayers = 24;
    aristos::hostMoEConfig.p2pBuffer = 1; // MB
    aristos::hostMoEConfig.gradBuffer = 1;
    aristos::hostMoEConfig.moeFreq = 24; // every other layer

    const auto spec = aristos::decider::decide(A, w, e.size(), e.size());
    const auto g = aristos::subsets(spec, 0);
    aristos::printContainer(g);
    std::vector<aristos::Worker> wG;
    for(int i = 0; i < g.size(); ++i){
        wG.emplace_back(g[i], w[i].processingRate, w[i].memoryCapacity);
    }
    const auto assignment = aristos::decider::assign(e, wG);
    aristos::printContainer(assignment);

    /// TODO tomorrow, Invoke CUTLASS test kernel instead of executing subprocess
    /// TODO remove ibgda and compile with libfabric and compare performance
    int fd[2];
    assert(pipe(fd) != -1);
    const pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        close(fd[1]);
        int msg; //nvtx3::message
        read(fd[0], &msg, sizeof(msg));
        std::cout << "Child received " << msg << std::endl;
        close(fd[0]);
        _Exit(0);
    }
    close(fd[0]);
    auto buf = 34;
    write(fd[1], &buf, sizeof(int));
    close(fd[1]);
    wait(&buf);
    std::cout << "Child status is: " << buf << std::endl;*/
    testTopologyDiscovery();
    return 0;
}

