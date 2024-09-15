/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/
#include <iostream>
#include <array>
#include <atomic>

#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/chrono>
#include <cuda/atomic>
#include <cuda/barrier>

#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h>

#include "include/aristos.cuh"
#include <functional>
#include <queue>

#include <cub/cub.cuh>
#include <cute/tensor.hpp>
#include <cute/config.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>
#include <boost/pending/disjoint_sets.hpp>

#define THREADS_PER_WARP 32
#define THREADS_PER_BLOCK 256

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
    __shared__ typename BlockReduce::TempStorage temp_storage;

    a_flag = BlockReduce(temp_storage).Reduce(a_flag, cub::Max());
    a_cas = BlockReduce(temp_storage).Reduce(a_cas, cub::Max());
    a_or = BlockReduce(temp_storage).Reduce(a_or, cub::Max());
    a_and = BlockReduce(temp_storage).Reduce(a_and, cub::Max());

    if(aristos::block::threadID() == 0 && !skip){
        printf("Block Id is %u, a_flag: {T: %f, V: %d}, a_cas: {T: %f, V:%u}, a_or: {T: %f, V:%u}, a_and: {T: %f, V: %u},"
               "isShared: %s\n",
               aristos::grid::blockID(),
               static_cast<cuda::std::chrono::duration<double, cuda::std::micro>>(a_flag / (iter*1.0)).count(),
               aFlag.load(),
               static_cast<cuda::std::chrono::duration<double, cuda::std::micro>>(a_cas/(iter*1.0)).count(),
               atomicCAS(pUnderTest, 0, 0),
               static_cast<cuda::std::chrono::duration<double, cuda::std::micro>>(a_or / (iter * 1.0)).count(),
               atomicOr(pUnderTest, 0U),
               static_cast<cuda::std::chrono::duration<double, cuda::std::micro>>(a_and/(iter*1.0)).count(),
               atomicAnd(pUnderTest, 1U),
               (shouldPersist)? "Yes" : "No");
    }
}

template<unsigned int bM=128, unsigned int bN=128, unsigned int bK=8, unsigned int bP=3>
__global__ void occupancyTestKernel(){
    __shared__ float sharedA[cute::cosize_v<decltype(cute::make_layout(cute::make_shape(cute::Int<bM>{}, cute::Int<bK>{}, cute::Int<bP>{})))>];
    __shared__ float sharedB[cute::cosize_v<decltype(cute::make_layout(cute::make_shape(cute::Int<bN>{}, cute::Int<bK>{}, cute::Int<bP>{})))>];
}

namespace aristos{
    bool isInitialized = false;
    cudaStream_t aristosStream = cudaStreamPerThread;
    void aristosInit(const unsigned int& seqLen, const unsigned int& embedDim, const unsigned int& hiddenProjDim,
                     const unsigned int& k, const unsigned int& capacityFactor,
                     const unsigned int& numExperts) {
        assert(!isInitialized);
        isInitialized = true;
        int numSMs = 0;
        int numBlocksPerSM = 0;
        constexpr int minCommunicatorBlocks = 2;
        int localRank = 0;
        int computeCapability = 0;
        CUTE_CHECK_ERROR(cudaGetDevice(&localRank));
        CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, localRank));
        const auto GEMMBlocks = cute::ceil_div(seqLen, bM) * cute::ceil_div(hiddenProjDim, bN);
        const auto minBlocks = GEMMBlocks + minCommunicatorBlocks;
        CUTE_CHECK_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &numBlocksPerSM,
                occupancyTestKernel<bM, bN, bK, bP>,
                blockSize,
                sizeof(aristos::maxPrecision) * ((bK * bP) * (bM + bN))));
        const int maxActiveBlocks = numBlocksPerSM * numSMs;
        assert(minBlocks <= maxActiveBlocks);
        int deviceSupportsMemoryPools = 0;
        CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&deviceSupportsMemoryPools,
                                                cudaDevAttrMemoryPoolsSupported, localRank));
        assert(deviceSupportsMemoryPools);
        CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&computeCapability, cudaDevAttrComputeCapabilityMajor, localRank));
        /// Due to NVSHMEM: https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html#hardware-requirements
        assert(computeCapability >= 7);

        // Good to go! Let's do some initialization
        // initialize NVSHMEM
        nvshmem_init();
        CUTE_CHECK_ERROR(cudaSetDevice(localRank));
        // Run Lysi
        // ...
        // generates the below
        unsigned int numNeighbors = 0;
        unsigned int numLocalExperts = 0;
        std::vector<specType> parallelSpec{};
        std::vector<specType> translation{};

        // Allocate Symmetric Heap + Flags
        auto trailer = k + 2U;
        size_t payload = Config::getCapacity(seqLen, numExperts, capacityFactor, k)
                         * (embedDim + trailer);
        size_t heapBytes = numNeighbors * stages * numCells * payload;
        heapBytes += numNeighbors * (sizeof(flagsType) / sizeof(maxPrecision));
        auto sHeap = nvshmem_calloc(heapBytes, sizeof(maxPrecision));

        // Final Initialization
        hostMoEConfig = Config();
        unsigned int* bookKeeping;
        /// pubQueueLen -> Multiplied by 2 to simulate pair: {index, numTokens}
        /// + translationLen + shardSpecLen +
        /// syncVectorLen -> {syncGrid, checkpoints}
        hostMoEConfig.bookKeepingLen = (numLocalExperts * numNeighbors * 2) + numNeighbors + numExperts + (numNeighbors * 2);
        CUTE_CHECK_ERROR(cudaMallocAsync(&bookKeeping,
                                    sizeof(specType)*hostMoEConfig.bookKeepingLen,
                                    aristosStream));
        //TODO init with host memcpy?
        hostMoEConfig.numPublisherBlocks = maxActiveBlocks - GEMMBlocks;
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
extern constexpr int tokens = capacity * (embedDim + k + 1);

extern constexpr int peerStride = stages * cells * tokens;
extern constexpr int stageStride = cells * tokens;
extern constexpr int cellStride = tokens;
extern constexpr int tokenStride = (embedDim + k + 1);
extern constexpr int finalTokenStride = (embedDim + 1);

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
               static_cast<cuda::std::chrono::duration<double, cuda::std::micro>>(cute_t/(1000*1.0)).count(),
               (raw_t / (1000*1.0)).count(),
               *getTokenPointer(foo, 0, 0, 1, 0),
               static_cast<cuda::std::chrono::duration<double, cuda::std::micro>>(raw_t/(1000*1.0)).count(),
               (shouldPersist)? "Yes" : "No");
    }
}

__device__ unsigned int testStages = 0;
__global__ void benchBarrier(unsigned int* b, cuda::barrier<cuda::thread_scope_device>* bar, unsigned int n, bool skip = false, bool persist = false){
    using Nano = cuda::std::chrono::duration<double, cuda::std::nano>;
    Nano bar_ptr = Nano::zero();
    Nano bar_obj = Nano::zero();
    if(persist){
        cuda::associate_access_property(b, cuda::access_property::persisting{});
        cuda::associate_access_property(&testStages, cuda::access_property::persisting{});
        cuda::associate_access_property(bar, cuda::access_property::persisting{});
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
               static_cast<cuda::std::chrono::duration<double, cuda::std::micro>>(bar_ptr/(iter*1.0)).count(),
               static_cast<cuda::std::chrono::duration<double, cuda::std::micro>>(bar_obj/(iter*1.0)).count(),
               (persist)? "Yes" : "No");
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
                Layout<Shape<Int<aristos::bM>, Int<aristos::bK>>>{}));
    using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;
    using TransformA = identity; // upcast from fp8 to fp16
    using GmemCopyB = TiledCopyB;
    using SmemLayoutAtomB = decltype(
    composition(Swizzle<1,2,3>{},
                Layout<Shape<Int<aristos::bN>, Int<aristos::bK>>>{}));
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
    auto ctaTiler = make_shape(aristos::bM, aristos::bN, aristos::bK);
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
    auto accum = partition_fragment_C(tmma, Shape<Int<aristos::bM>, Int<aristos::bN>>{});
    clear(accum);
    extern __shared__ cuda::std::byte sharedBuf[];
    
    CollectiveMainloop  expert;
    expert(accum, gA, gB, accum, k_tile_iter, k_tile_count, Underscore{}, threadIdx.x, sharedBuf);
}

template<typename T>
void pop_println(std::string_view rem, T& pq)
{
    std::cout << rem << ": ";
    for (; !pq.empty(); pq.pop())
        std::cout << pq.top().toString() << ' ';
    std::cout << '\n';
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
    free(host_b);*/

    /// Necessary to use path halving to ensure amortized "practical constant" time
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
    std::cout << aristos::Streamable<decltype(sets)::key_type> << std::endl;*/

    const std::vector data = {{aristos::Edge(0,1,0.3), aristos::Edge(0,2,0.2)}};
    std::priority_queue q(data.begin(), data.end(), std::greater<>());
    pop_println("Min Priority Queue", q);
    auto b = data[1] > data[0];
    std::cout << b << std::endl;
}