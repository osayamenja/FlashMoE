//
// Created by oja7 on 11/12/24.
//

#ifndef PREP_CUH
#define PREP_CUH

#include <cute/layout.hpp>
#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h>

#include "debug.cuh"
#include "types.cuh"

#define SUPPORTED = 1;
namespace aristos{
    template<typename Element>
    requires(aristos::TensorValueType<Element>)
    __host__ __forceinline__
    void aristosInit(const unsigned int& seqLen, const unsigned int& embedDim, const unsigned int& hiddenProjDim,
                     const unsigned int& k, const unsigned int& capacityFactor, const unsigned int& numExperts,
                     const bool& shouldDrop) {
        reportError(!isInitialized, "Already Initialized");
        isInitialized = true;
        reportError(embedDim % BLOCK_N == 0 && hiddenProjDim % BLOCK_N == 0,
            "Must be multiple of BLOCK_N");
        reportError(seqLen % BLOCK_M == 0, "Must be a multiple of BLOCK_M");
        int l2CacheSize = 0;
        int cudaDevAttribute = 0;
        int dev = 0;
        int blocks = 0;
        int arch = 0;
        CUTE_CHECK_ERROR(cudaGetDevice(&dev));
        CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&cudaDevAttribute, cudaDevAttrMemoryPoolsSupported, dev));
        reportError(cudaDevAttribute, "Memory Pools support required");
        CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&arch, cudaDevAttrComputeCapabilityMajor, dev));
        reportError(arch, "At least Volta required");
        CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, dev));
        CUTE_CHECK_ERROR(cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, dev));

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

    template<typename Element>
    requires(aristos::TensorValueType<Element>)
    __host__ __forceinline__
    void forwardHost(){
        assert(isInitialized);
    }

    template<typename Element>
    requires(aristos::TensorValueType<Element>)
    __host__ __forceinline__
    void backwardHost(){
        assert(isInitialized);
    }

    __host__ __forceinline__
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
