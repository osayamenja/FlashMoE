//
// Created by oja7 on 11/12/24.
//

#ifndef PREP_CUH
#define PREP_CUH

#include <cute/layout.hpp>
#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h>
#include "types.cuh"

#define SUPPORTED = 1;
namespace aristos{
    __inline__ bool isInitialized = false;
    __inline__ auto aristosStream = cudaStreamPerThread;
    template<typename Element>
    requires(aristos::TensorValueType<Element>)
    __host__ __forceinline__
    void aristosInit(const unsigned int& seqLen, const unsigned int& embedDim, const unsigned int& hiddenProjDim,
                     const unsigned int& k, const unsigned int& capacityFactor, const unsigned int& numExperts,
                     const bool& shouldDrop) {
        // TODO assert inputs are correct and check for cudaDevP2PAttrNativeAtomicSupported, cudaDevP2PAttrAccessSupported
        assert(embedDim % BLOCK_N == 0 && hiddenProjDim % BLOCK_N == 0 && "Must be multiple of BLOCK_N");
        assert(seqLen % BLOCK_M == 0 && "Must be a multiple of BLOCK_M");
        int l2CacheSize = 0;
        assert(!isInitialized);
        isInitialized = true;
        int cudaDevAttribute = 0;
        int dev = 0;
        int blocks = 0;
        CUTE_CHECK_ERROR(cudaGetDevice(&dev));
        CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&cudaDevAttribute,
                                                cudaDevAttrMemoryPoolsSupported, dev));
        assert(cudaDevAttribute);
        CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&cudaDevAttribute, cudaDevAttrComputeCapabilityMajor, dev));
        assert(cudaDevAttribute >= 7);
        CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, dev));

        // Good to go! Let's do some initialization
        // Run decider
        // ...
        // generates the below
        const unsigned int numLocalExperts = 0;
        std::vector<specType> parallelSpec{};
        std::vector<specType> translation{};
        const unsigned int numNeighbors = translation.size();

        // initialize communication backend
        nvshmem_init();
        CUTE_CHECK_ERROR(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
        const auto globalWorld = nvshmem_n_pes();
        // TODO: dual allocation to precisely get memory cost, namely max {|W| * |X_l|} across all EP groups
        // Allocate Symmetric Heap + Flags
        auto memoryBytes = STAGES * CELLS * seqLen * (embedDim + k + 2) * sizeof(Element) +
            STAGES * numNeighbors * sizeof(flagsType);
        memoryBytes = cute::max(memoryBytes, globalWorld * (BETA_BUFFER + (globalWorld + 1) * sizeof(floatPair)));
        /// Allocates symmetric heap
        /// assert alignment 16 bytes
        auto sHeap = static_cast<cuda::std::byte*>(nvshmem_calloc(memoryBytes, sizeof(cuda::std::byte)));

        //Final Initialization
        void* bookKeeping;
        const auto taskBound = cute::ceil_div(seqLen, BLOCK_M) *
            (cute::ceil_div(embedDim, BLOCK_N) + cute::ceil_div(hiddenProjDim, BLOCK_N) + 1);
        const auto tilesN = embedDim / BLOCK_N;
        const auto tilesM = seqLen / BLOCK_M;
        const auto paddedNumExperts = Config::pad<BLOCK_N>(numExperts);
        const auto flagCount = numNeighbors * numLocalExperts + tilesM * tilesN;

        const auto brsData = (numExperts > BLOCK_N) *
            (sizeof(unsigned int) * (seqLen * (paddedNumExperts / BLOCK_N)) + // sync flags for gate
            2 * sizeof(mp_t) * seqLen + // m and d for softmax
            sizeof(cuda::std::pair<mp_t, unsigned int>) * k * seqLen);  // binary min heap

        // Allocate all memory needed once
        memoryBytes = sizeof(mp_t) * seqLen * hiddenProjDim + // intermediary results of expert GEMM
            brsData +
            // flags for ring aggregation of token indices
            sizeof(unsigned int) * (tilesM + tilesN) +
            sizeof(sizeof(TokenIdxTuple)) * seqLen + // token ids and probabilities
            sizeof(mp_t) * seqLen * paddedNumExperts + // gate routing
            sizeof(mp_t) * (2 * paddedNumExperts + 1) + // gate loss vectors, loss value
            sizeof(unsigned int) * paddedNumExperts + // expert counts,
            sizeof(unsigned int) * (numExperts + numNeighbors + 1) + // GPU -> expert lookup table and sentinel for prefix
            sizeof(unsigned int) * numExperts + // sync array for packet Construction
            sizeof(unsigned int) * numNeighbors + // EP rank -> global rank
            sizeof(unsigned int) * numExperts * 2  + // Expert parallelism specification and EP -> heap
            sizeof(unsigned int) * blocks + // readyQ
            sizeof(unsigned int) * blocks + // statusQ
            sizeof(unsigned int) * numNeighbors * numLocalExperts * cute::ceil_div(seqLen, numExperts * BLOCK_M) + // taskSync
            sizeof(unsigned int) * flagCount +  // flag checkpoints
            sizeof(Task) * (taskBound + blocks * tilesN) + // taskQ
            sizeof(unsigned int) * (blocks + THREADS - 2); // tQHeads
        // Initialize hostConfig
        CUTE_CHECK_ERROR(cudaMallocAsync(&bookKeeping, memoryBytes, aristosStream));
        CUTE_CHECK_ERROR(cudaMemsetAsync(bookKeeping, 0, memoryBytes, aristosStream));
        CUTE_CHECK_ERROR(cudaMemcpyToSymbolAsync(moeConfig, &hostMoEConfig, sizeof(Config), 0,
                                            cudaMemcpyHostToDevice, aristosStream));
        CUTE_CHECK_ERROR(cudaPeekAtLastError());
        CUTE_CHECK_ERROR(cudaStreamSynchronize(aristosStream));

#if (__CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800)
        cudaStreamAttrValue stream_attribute;
        stream_attribute.accessPolicyWindow.base_ptr  = bookKeeping;
        stream_attribute.accessPolicyWindow.num_bytes = cuda::std::min(static_cast<size_t>(0.25 * l2CacheSize),
            sizeof(unsigned int) * numNeighbors + sizeof(unsigned int) * numExperts * 2);
        stream_attribute.accessPolicyWindow.hitRatio  = 1.0;
        stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
        stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

        //Set the attributes to a CUDA stream of type cudaStream_t
        cudaStreamSetAttribute(aristosStream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
#endif

    }

    __forceinline__
    void forwardHost(){
        assert(isInitialized);
    }

    __forceinline__
    void backwardHost(){
        assert(isInitialized);
    }

    __forceinline__
    void aristosFinalize(){
        assert(isInitialized);
        isInitialized = false;
        CUTE_CHECK_ERROR(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
#if (__CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800)
        cudaStreamAttrValue stream_attribute;
        stream_attribute.accessPolicyWindow.num_bytes = 0; // Setting the window size to 0 disable it
        // Overwrite the access policy attribute to a CUDA Stream
        cudaStreamSetAttribute(aristosStream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
        cudaCtxResetPersistingL2Cache();
#endif

        CUTE_CHECK_ERROR(cudaFreeAsync(hostMoEConfig.bookKeeping, aristosStream));
        nvshmem_free(hostMoEConfig.sHeap);
        nvshmem_finalize();
        CUTE_CHECK_ERROR(cudaPeekAtLastError());
        CUTE_CHECK_ERROR(cudaStreamSynchronize(aristosStream));
    }
}
#endif //PREP_CUH
