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
#include <cooperative_groups.h>
#include <functional>

#include <cub/cub.cuh>
#include <cute/tensor.hpp>
#include <cute/config.hpp>

#define THREADS_PER_WARP 32
#define THREADS_PER_BLOCK 256

__global__ void aristos_sandbox(int *destination, const int my_pe, const int n_pes, const bool skip = true) {
    cuda::std::array<int, 4>activations{{my_pe + 1, 2,my_pe + 3, 4}};
    int n_elems = 2;
    int k = 0;
    const auto start{cuda::std::chrono::high_resolution_clock::now()};
    for(int i = 0; i < n_pes; ++i){
        for(uint j = 0; j < n_elems; ++j){
            destination[j] = activations[k++];
        }
        nvshmemx_int_put_block((destination + ((my_pe+1)*n_elems)),
                        destination,
                        n_elems,
                        (i % n_pes));
    }
    const auto end{cuda::std::chrono::high_resolution_clock::now()};
    const cuda::std::chrono::duration<double> elapsed_seconds{end - start};

    if (!skip && cute::thread(0)){
        printf("PE %d.t0 Elapsed time is %lld\n", my_pe,
               cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(elapsed_seconds).count());
    }
    __syncthreads();
}
void nvshmem_test(){
    std::cout << "Hello, World!" << std::endl;
    int my_pe_node;
    std::array<int, 6> msg{};
    cudaStream_t stream;

    nvshmem_init();
    my_pe_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    int my_pe = nvshmem_my_pe();
    int n_pe = nvshmem_n_pes();
    int dev_count;
    CUTE_CHECK_ERROR(cudaGetDeviceCount(&dev_count));
    cudaDeviceProp prop{};
    CUTE_CHECK_ERROR(cudaGetDeviceProperties(&prop, my_pe_node % dev_count));
    int clockrate;
    CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&clockrate, cudaDevAttrClockRate, my_pe_node));

    fprintf(stderr, "mype: %d mype_node: %d device name: %s bus id: %d n_pes: %d\n", my_pe, my_pe_node,
            prop.name, prop.pciBusID, n_pe);
    cudaSetDevice(my_pe_node);
    cudaStreamCreate(&stream);

    auto destination_void = nvshmem_calloc(msg.size(), sizeof(int));
    auto destination = static_cast<int*>(destination_void);

    for (int i = 0; i < 5; ++i) {
        aristos_sandbox<<<1, 1>>>(destination, my_pe, n_pe);
        CUTE_CHECK_ERROR(cudaPeekAtLastError());
    }
    aristos_sandbox<<<1, 1>>>(destination, my_pe, n_pe, false);
    CUTE_CHECK_ERROR(cudaPeekAtLastError());
    CUTE_CHECK_ERROR(cudaMemcpyAsync(msg.data(), destination, msg.size() * sizeof(int), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();
    std::stringstream result_stream;
    result_stream << "PE " << nvshmem_my_pe() << " buffer: { ";
    for(auto i: msg){
        result_stream << i << ", ";
    }
    result_stream << "}\n";
    std::cout << std::string(result_stream.str());
    nvshmem_free(destination);
    nvshmem_finalize();
}

__global__ void memory_heterogeneity(void* symmetric,  uint64_t* flags, int my_pe){
    // Arrange
    const int peer = !my_pe;
    constexpr int k = 2;
    constexpr int n_tok = 2;
    constexpr int total_mem = n_tok + k;
    auto* scratchpad = static_cast<cute::half_t*>(symmetric);
    auto* scratchpad_begin = scratchpad + (my_pe*total_mem);
    scratchpad_begin[0] = cute::half_t(0.67);
    scratchpad_begin[1] = cute::half_t(0.02);
    auto* trailer = static_cast<uint_fast16_t*>(static_cast<void*>((scratchpad_begin + n_tok)));
    trailer[0] = 2;
    trailer[1] = 5;
    constexpr uint64_t set_flag = 1;

    // Send
    nvshmemx_putmem_signal_nbi_block(static_cast<void*>(scratchpad_begin),
                              static_cast<void*>(scratchpad_begin),
                              (total_mem * 2),
                              (flags + my_pe),
                              set_flag,
                              NVSHMEM_SIGNAL_SET,
                              peer);
    nvshmem_signal_wait_until((flags + my_pe), NVSHMEM_CMP_EQ, set_flag);

    // Reconstitute
    auto* floats = scratchpad + (peer*total_mem);
    cute::print("Received floats: %f, %f", floats[0].operator float(), floats[1].operator float());
    auto* r_trailers = static_cast<uint_fast16_t*>(static_cast<void*>(floats + n_tok));
    cute::print("Received trailers: %d, %d", r_trailers[0], r_trailers[1]);
}

__device__ __constant__ cuda::atomic<unsigned int, cuda::thread_scope_device> last{1};

#define SEQ 23U
__global__ void benchLoad(CUTE_GRID_CONSTANT const int iter, unsigned int* sync_p, bool shouldPersist = false){
    if(aristos::grid::threadID() == 0){
        printf("Initial Value is %u\n", sync_p[0]);
    }
    // initialization
    if(shouldPersist){
        cuda::associate_access_property(&last, cuda::access_property::persisting{});
        cuda::associate_access_property(&sync_p, cuda::access_property::persisting{});
    }
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> loader(*sync_p);
    if(shouldPersist){
        cuda::associate_access_property(&loader, cuda::access_property::persisting{});
    }

    auto start = cuda::std::chrono::high_resolution_clock::now();
    auto end = cuda::std::chrono::high_resolution_clock::now();
    auto elapsed_seconds{end - start};
    size_t half_cas = 0, a_add = 0, a_cas = 0, freq_at = 0, a_or = 0;
    /*auto t = aristos::grid_tid();
    auto tt = t;*/
    CUTE_UNROLL
    for(int i = 0; i < iter; ++i){
        start = cuda::std::chrono::high_resolution_clock::now();
        atomicAdd(sync_p, 0U); // equivalent to a load
        elapsed_seconds = cuda::std::chrono::high_resolution_clock::now() - start;
        a_add += cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(elapsed_seconds).count();

        start = cuda::std::chrono::high_resolution_clock::now();
        atomicCAS(sync_p, 0U, 0U); // equivalent to a load
        elapsed_seconds = cuda::std::chrono::high_resolution_clock::now() - start;
        a_cas += cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(elapsed_seconds).count();

        start = cuda::std::chrono::high_resolution_clock::now();
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device>(*sync_p).load();
        elapsed_seconds = cuda::std::chrono::high_resolution_clock::now() - start;
        freq_at += cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(elapsed_seconds).count();

        start = cuda::std::chrono::high_resolution_clock::now();
        atomicOr(sync_p, 0U);
        elapsed_seconds = cuda::std::chrono::high_resolution_clock::now() - start;
        a_or += cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(elapsed_seconds).count();
    }
    // Specialize BlockReduce for a 1D block of 128 threads of type int
    using BlockReduce = cub::BlockReduce<size_t, THREADS_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    half_cas = BlockReduce(temp_storage).Reduce(half_cas, cub::Max());
    a_add = BlockReduce(temp_storage).Reduce(a_add, cub::Max());
    a_or = BlockReduce(temp_storage).Reduce(a_or, cub::Max());
    a_cas = BlockReduce(temp_storage).Reduce(a_cas, cub::Max());
    freq_at = BlockReduce(temp_storage).Reduce(freq_at, cub::Max());

    if(aristos::block::threadID() == 0){
        printf("Block Id is %u, a_add: {T: %f, V:%u}, "
               "a_cas: {T: %f, V: %u}, a_or: {T: %f, V: %u}, freq_at: {T: %f, V: %u},"
               "persist: %s\n",
               aristos::grid::blockID(),
               a_add/(iter*1.0),
               atomicAdd(sync_p, 0U),
               a_cas/(iter*1.0),
               atomicCAS(sync_p, 0U, 0U),
               a_or/(iter*1.0),
               atomicOr(sync_p, 0U),
               freq_at/(iter*1.0),
               cuda::atomic_ref<unsigned int, cuda::thread_scope_device>(*sync_p).load(),
               (shouldPersist)? "Yes" : "No");
    }
}

__global__ void benchAtomics(CUTE_GRID_CONSTANT const int iter, unsigned int* flag, bool shouldPersist = false){
    // initialization
    cuda::std::atomic_flag stopFlag(false);
    if(shouldPersist){
        cuda::associate_access_property(flag, cuda::access_property::persisting{});
        cuda::associate_access_property(&stopFlag, cuda::access_property::persisting{});
    }
    auto start = cuda::std::chrono::high_resolution_clock::now();
    auto end = cuda::std::chrono::high_resolution_clock::now();
    auto elapsed_seconds{end - start};
    size_t a_flag = 0, a_cas = 0, a_or = 0, a_and = 0;
    CUTE_UNROLL
    for(int i = 0; i < iter; ++i){
        start = cuda::std::chrono::high_resolution_clock::now();
        stopFlag.test();
        elapsed_seconds = cuda::std::chrono::high_resolution_clock::now() - start;
        a_flag += cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(elapsed_seconds).count();

        start = cuda::std::chrono::high_resolution_clock::now();
        atomicCAS(flag, 0, 0);
        elapsed_seconds = cuda::std::chrono::high_resolution_clock::now() - start;
        a_cas += cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(elapsed_seconds).count();

        start = cuda::std::chrono::high_resolution_clock::now();
        atomicOr(flag, 0U);
        elapsed_seconds = cuda::std::chrono::high_resolution_clock::now() - start;
        a_or += cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(elapsed_seconds).count();

        start = cuda::std::chrono::high_resolution_clock::now();
        atomicAnd(flag, 1U);
        elapsed_seconds = cuda::std::chrono::high_resolution_clock::now() - start;
        a_and += cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(elapsed_seconds).count();

    }
    using BlockReduce = cub::BlockReduce<size_t, THREADS_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    a_flag = BlockReduce(temp_storage).Reduce(a_flag, cub::Max());
    a_cas = BlockReduce(temp_storage).Reduce(a_cas, cub::Max());
    a_or = BlockReduce(temp_storage).Reduce(a_or, cub::Max());
    a_and = BlockReduce(temp_storage).Reduce(a_and, cub::Max());

    if(aristos::block::threadID() == 0){
        printf("Block Id is %u, a_flag: {T: %f, V: %d}, a_cas: {T: %f, V:%u}, a_or: {T: %f, V:%u}, a_and: {T: %f, V: %u},"
               "persist: %s\n",
               aristos::grid::blockID(),
               a_flag / (iter*1.0),
               stopFlag.test(),
               a_cas/(iter*1.0),
               atomicCAS(flag, 0, 0),
               a_or / (iter * 1.0),
               atomicOr(flag, 0U),
               a_and/(iter*1.0),
               atomicAnd(flag, 1U),
               (shouldPersist)? "Yes" : "No");
    }
}

template<unsigned int bM=128, unsigned int bN=128, unsigned int bK=8, unsigned int bP=3>
__global__ void occupancyTestKernel(){
    __shared__ float sharedA[cute::cosize_v<decltype(cute::make_layout(cute::make_shape(cute::Int<bM>{}, cute::Int<bK>{}, cute::Int<bP>{})))>];
    __shared__ float sharedB[cute::cosize_v<decltype(cute::make_layout(cute::make_shape(cute::Int<bN>{}, cute::Int<bK>{}, cute::Int<bP>{})))>];
}

namespace  aristos{
    bool isInitialized = false;
    Config hostMoEConfig;
    void aristosInit(unsigned int seqLen, unsigned int embedDim, unsigned int hiddenProjDim,
                     unsigned int k, int deviceID, unsigned int capacityFactor,
                     unsigned int numExperts) {
        assert(!isInitialized);
        isInitialized = true;
        int numSMs = 0;
        CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceID));

        int numBlocksPerSM = 0;
        int minCommunicatorBlocks = 2;
        auto GEMMBlocks = cute::ceil_div(seqLen, bM) * cute::ceil_div(hiddenProjDim, bN);
        auto minBlocks = GEMMBlocks + minCommunicatorBlocks;
        CUTE_CHECK_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &numBlocksPerSM,
                occupancyTestKernel<bM, bN, bK, bP>,
                blockSize,
                sizeof(aristos::maxPrecision) * ((bK * bP) * (bM + bN))));
        int maxActiveBlocks = numBlocksPerSM * numSMs;
        assert(minBlocks <= maxActiveBlocks);

        // Good to go! Let's do some initialization
        // initialize NVSHMEM
        nvshmem_init();
        auto localRank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
        auto globalRank = nvshmem_my_pe();
        unsigned int numNodes = nvshmem_n_pes();
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
        size_t payload = aristos::Config::getCapacity(seqLen, numExperts, capacityFactor, k)
                         * (embedDim + trailer);
        size_t heapBytes = numNeighbors * aristos::stages * aristos::numCells * payload;
        heapBytes += numNeighbors * (sizeof(aristos::flagsType) / sizeof(aristos::maxPrecision));
        auto sHeap = nvshmem_calloc(heapBytes, sizeof(aristos::maxPrecision));

        // Final Initialization
        hostMoEConfig = Config();
        unsigned int* bookKeeping;
        /// pubQueueLen -> Multiplied by 2 to simulate pair: {index, numTokens}
        /// + translationLen + shardSpecLen +
        /// syncVectorLen -> {syncGrid, checkpoints}
        hostMoEConfig.bookKeepingLen = (numLocalExperts * numNeighbors * 2) + numNeighbors + numExperts + (numNeighbors * 2);
        CUTE_CHECK_ERROR(cudaMallocAsync(&bookKeeping,
                                    sizeof(specType)*hostMoEConfig.bookKeepingLen,
                                    cudaStreamDefault));
        //TODO init with host memcpy?
        hostMoEConfig.numPublisherBlocks = maxActiveBlocks - GEMMBlocks;
        hostMoEConfig.worldSize = numNeighbors;
        hostMoEConfig.bookKeeping = bookKeeping;
        hostMoEConfig.sHeap = static_cast<cuda::std::byte*>(sHeap);
        CUTE_CHECK_ERROR(cudaMemcpyToSymbol(moeConfig,
                                            &hostMoEConfig, sizeof(Config)));
        CUTE_CHECK_ERROR(cudaStreamSynchronize(cudaStreamDefault));
        CUTE_CHECK_LAST();
    }

    void forwardHost(){

    }

    void backwardHost(){
    }

    void aristosFinalize(){
        assert(isInitialized);
        isInitialized = false;
        CUTE_CHECK_ERROR(cudaFree(hostMoEConfig.bookKeeping));
        nvshmem_free(hostMoEConfig.sHeap);
        nvshmem_finalize();
        CUTE_CHECK_LAST();
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

__global__ void benchTen(int* foo, bool shouldPersist = false){
    auto start = cuda::std::chrono::high_resolution_clock::now();
    auto elapsed_seconds{cuda::std::chrono::high_resolution_clock::now() - start};
    size_t cute_t = 0, raw_t = 0;
    auto t = cute::make_tensor(cute::make_gmem_ptr(foo), cute::make_shape(peers, cute::make_shape(cute::make_shape(stages, cells), tokens)), cute::LayoutRight{});
    if(shouldPersist){
        cuda::associate_access_property(foo, cuda::access_property::persisting{});
    }

    CUTE_UNROLL
    for(unsigned int i = 0; i < 1000; ++i){
        start = cuda::std::chrono::high_resolution_clock::now();
        &t(0, cute::make_coord(cute::make_coord(0,1),0));
        elapsed_seconds = cuda::std::chrono::high_resolution_clock::now() - start;
        cute_t += cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(elapsed_seconds).count();

        start = cuda::std::chrono::high_resolution_clock::now();
        getTokenPointer(foo, 0, 0, 1, 0);
        elapsed_seconds = cuda::std::chrono::high_resolution_clock::now() - start;
        raw_t += cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(elapsed_seconds).count();
    }

    using BlockReduce = cub::BlockReduce<size_t, THREADS_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    cute_t = BlockReduce(temp_storage).Reduce(cute_t, cub::Max());
    raw_t = BlockReduce(temp_storage).Reduce(raw_t, cub::Max());
    if(aristos::block::threadID() == 0){
        printf("Block Id is %u, cute_t: {T: %f, V: %d}, raw_t: {T: %f, V:%d} "
               "persist: %s\n",
               aristos::block::threadID(),
               cute_t / (1000*1.0),
               t(0, cute::make_coord(cute::make_coord(0,1),0)),
               raw_t / (1000*1.0),
               *getTokenPointer(foo, 0, 0, 1, 0),
               (shouldPersist)? "Yes" : "No");
    }
}

#include <condition_variable>
int main() {
    std::array<int, 4> a = {{0,1,2,3}};
    void* pp = static_cast<void*>(a.begin());
    int* p;
    CUTE_CHECK_ERROR(cudaMallocAsync(&p, sizeof(int)*a.size(), cudaStreamDefault));
    CUTE_CHECK_ERROR(cudaMemcpyAsync(p, a.cbegin(), sizeof(int)*a.size(), cudaMemcpyHostToDevice, cudaStreamDefault));
    CUTE_CHECK_ERROR(cudaStreamSynchronize(cudaStreamDefault));
    benchTen<<<1, THREADS_PER_BLOCK>>>(p);
    benchTen<<<1, THREADS_PER_BLOCK>>>(p, true);
    CUTE_CHECK_LAST();
}