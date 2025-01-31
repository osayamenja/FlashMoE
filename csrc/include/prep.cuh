//
// Created by oja7 on 11/12/24.
//

#ifndef PREP_CUH
#define PREP_CUH

#include <cutlass/epilogue/thread/activation.h>
#include <cute/layout.hpp>
#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h>

#include "debug.cuh"
#include "types.cuh"
#include "moe/expert.cuh"

#define SUPPORTED = 1;
namespace aristos{
    template<
        unsigned int Arch,
        typename Element
    >
    __host__ __forceinline__
    void measureThroughput(uint* __restrict__ hDt,
        const unsigned int& M, const unsigned int& N, const unsigned int& K) {
        // malloc memory for all matrices
        constexpr auto batch = 2U;
        cuda::std::byte* abc;
        constexpr auto eS = sizeof(Element);
        const auto stateSize = sizeof(uint) * (1 + M / BLOCK_M); // tileSync + dT
        const auto aSize = stateSize + eS * M * K;
        const auto abSize = aSize + eS * K * N;
        const auto abcSize = abSize + eS * M * N;
        const auto abcBSize = abcSize + eS * cute::max(N, K); // bias
        const auto abcBSSize = abcBSize + eS * M; // scale/combine weights
        CHECK_ERROR_EXIT(cudaMallocAsync(&abc, abcBSSize, aristosStream));
        CHECK_ERROR_EXIT(cudaMemsetAsync(abc, 0, abcBSSize, aristosStream));

        // memcpy to device memory

        auto* __restrict__ p = CAST_TO(uint, abc);
        const auto pS = cute::make_tuple(M, N, K);
        using Activation = cutlass::epilogue::thread::ReLU<Element>;
        constexpr auto blocks = Hardware<Arch>::blocks::value - 1U;
        auto* __restrict__ pBp = CAST_TO(Element, abc + aSize);
        auto* __restrict__ pDp = CAST_TO(Element, abc + abcSize);
        const auto pB = cuda::std::array<Element*, batch>{pBp, pBp};
        const auto pD = cuda::std::array<Element*, batch>{pDp, pDp};
        auto* __restrict__ pC = CAST_TO(Element, abc + abSize);
        auto* __restrict__ pSC = CAST_TO(Element, abc + abcBSize);
        expert<Arch, Activation, batch><<<blocks, ARISTOS_BLOCK_SIZE, 0, aristosStream>>>(pS, p,
            p + 1, CAST_TO(Element, abc + stateSize), pB, pC, pD, pSC, pSC);
        CHECK_ERROR_EXIT(cudaMemcpyAsync(hDt, p, sizeof(uint), cudaMemcpyDeviceToHost, aristosStream));
        CHECK_ERROR_EXIT(cudaFreeAsync(abc, aristosStream));
        CHECK_ERROR_EXIT(cudaPeekAtLastError());
        CHECK_ERROR_EXIT(cudaStreamSynchronize(aristosStream));
    }
    __host__ __forceinline__
    void discoverTopology(void* hAp, const uint& n, const uint& globalRank,
        const uint& eT) {
        const auto aD = n * n;
        const size_t heapBytes = n * sizeof(flagsType) + aD * sizeof(floatPair)
        + sizeof(uint) * (n + 2) + n * BETA_BUFFER;
        auto* symHeap = nvshmem_calloc(heapBytes, sizeof(cuda::std::byte));
        // Pointer orchestration
        // Starting index of flags array
        auto* flags = CAST_TO(flagsType, symHeap);
        auto* adj = CAST_TO(floatPair, flags + n);
        // Navigate to our slice of the adjacency matrix
        auto* results = adj + globalRank * n;
        // Repurpose a subset of the symmetric heap for local storage.
        auto* rates = CAST_TO(uint, adj + aD);
        auto* syncArray = rates + n;
        // Starting index of heap
        auto* sHeap = CAST_TO(cuda::std::byte, syncArray + 2);
        const auto remotePresent = [&n, &results] {
            for (int i = 0; i< n; ++i) {
                if (nvshmem_ptr(results, i) == nullptr) return true;
            }
            return false;
        };
        const auto isRemotePresent = remotePresent();
        const auto sharedSize = n * (sizeof(floatPair) + sizeof(unsigned int));
        topology::discover<<<ARISTOS_SUPER_BLOCK_SIZE, ARISTOS_BLOCK_SIZE, sharedSize, aristosStream>>>(n, globalRank,
            isRemotePresent, eT, sHeap, flags, results, syncArray, rates);
        CHECK_ERROR_EXIT(cudaMemcpyAsync(hAp, adj, aD * sizeof(floatPair) + n * sizeof(uint),
            cudaMemcpyDeviceToHost, aristosStream));
        CHECK_ERROR_EXIT(cudaPeekAtLastError());
        CHECK_ERROR_EXIT(cudaStreamSynchronize(aristosStream));
        nvshmem_free(symHeap);
    }

    __host__ __forceinline__
    std::vector<size_t> runDecider(const InitialConfig& iC, const floatPair* __restrict__ const& aP,
        const uint* __restrict__ const& rates, const uint& world) {
        const auto mC = ModelConfig{
            iC.numLayers,
            iC.redAmount,
            iC.globalBatch,
            iC.miniBatch,
            iC.moeFrequency,
            cute::ceil_div(iC.p2pBuffer, 1024U * 1024U), // in MB
            cute::ceil_div(iC.p2pBuffer, 1024U * 1024U) // in MB
        };
        std::vector<Worker> workers(world);

        // TODO estimate available device memory
        return {};
    }
    template<typename Element>
    requires(aristos::TensorValueType<Element>)
    __host__ __forceinline__
    void aristosInit(const InitialConfig& iC) {
        reportError(!isInitialized, "Already Initialized");
        isInitialized = true;
        reportError(iC.embedDim % BLOCK_N == 0 && iC.hiddenProjDim % BLOCK_N == 0,
            "Must be multiple of BLOCK_N");
        reportError(iC.seqLen % BLOCK_M == 0, "Must be a multiple of BLOCK_M");
        int l2CacheSize = 0;
        int cudaDevAttribute = 0;
        int dev = 0;
        int blocks = 0;
        int arch = 0;
        CHECK_ERROR_EXIT(cudaGetDevice(&dev));
        CHECK_ERROR_EXIT(cudaDeviceGetAttribute(&cudaDevAttribute, cudaDevAttrMemoryPoolsSupported, dev));
        reportError(cudaDevAttribute, "Memory Pools support required");
        CHECK_ERROR_EXIT(cudaDeviceGetAttribute(&arch, cudaDevAttrComputeCapabilityMajor, dev));
        reportError(arch, ">= Volta is required");
        CHECK_ERROR_EXIT(cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, dev));
        CHECK_ERROR_EXIT(cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, dev));

        // initialize communication backend
        nvshmem_init();
        CUTE_CHECK_ERROR(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
        const auto globalWorld = nvshmem_n_pes();
        const auto rank = nvshmem_my_pe();

        // Pointer to adjacency matrix and throughput of all devices
        const auto aD = globalWorld * globalWorld;
        auto* aP = std::calloc(globalWorld * globalWorld * sizeof(floatPair) +
            globalWorld * sizeof(uint), sizeof(cuda::std::byte));
        // measure device throughput
        auto* hA = static_cast<floatPair*>(aP);
        auto* eTp = CAST_TO(uint, hA + aD);
        switch (arch) {
            case 7:
                measureThroughput<700, Element>(eTp + rank, iC.seqLen, iC.hiddenProjDim, iC.embedDim);
            break;
            case 8:
                measureThroughput<800, Element>(eTp + rank, iC.seqLen, iC.hiddenProjDim, iC.embedDim);
            break;
            default:
                measureThroughput<900, Element>(eTp + rank, iC.seqLen, iC.hiddenProjDim, iC.embedDim);
            break;
        }
        const auto dT = eTp[rank];
        discoverTopology(aP, globalWorld, globalWorld, dT);

        // The topology adjacency matrix is ready, let's map devices to optimal cooperative process groups
        const auto dTg = runDecider(iC, hA, eTp, globalWorld);

        // Good to go! Let's do some initialization
        // Run decider
        // ...
        // generates the below
        const unsigned int numLocalExperts = 0;
        std::vector<specType> translation{};
        const unsigned int numNeighbors = translation.size();


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
        std::free(aP);
    }

    template<typename Element>
    requires(aristos::TensorValueType<Element>)
    __host__ __forceinline__
    void forwardHost(){
        // This function would likely be called frequently--per layer per iteration--thus, we elide the error check.
        // Of course, calling this function without initializing the aristos runtime yields undefined behavior.
        // reportError(isInitialized, "Not initialized")
    }

    template<typename Element>
    requires(aristos::TensorValueType<Element>)
    __host__ __forceinline__
    void backwardHost(){
        // reportError(isInitialized, "Not initialized")
    }

    __host__ __forceinline__
    void aristosFinalize(){
        reportError(isInitialized, "Not initialized!");
        isInitialized = false;
        CHECK_ERROR_EXIT(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
#if (__CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800)
        cudaStreamAttrValue stream_attribute;
        stream_attribute.accessPolicyWindow.num_bytes = 0; // Setting the window size to 0 disable it
        // Overwrite the access policy attribute to a CUDA Stream
        cudaStreamSetAttribute(aristosStream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
        cudaCtxResetPersistingL2Cache();
#endif

        CHECK_ERROR_EXIT(cudaFreeAsync(hostBookkeeping.book, aristosStream));
        nvshmem_free(hostMoEConfig.sHeap);
        nvshmem_finalize();
        CHECK_ERROR_EXIT(cudaPeekAtLastError());
        CHECK_ERROR_EXIT(cudaStreamSynchronize(aristosStream));
    }
}
#endif //PREP_CUH
