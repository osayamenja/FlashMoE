//
// Created by oja7 on 11/12/24.
//

#ifndef BOOTSRAP_CUH
#define BOOTSTRAP_CUH

#include <vector>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h>

#include "topo.cuh"
#include "debug.cuh"
#include "types.cuh"
#include "moe/moe.cuh"
#include "moe/expert.cuh"
#include "os/decider/decider.cuh"
#include "os/decider/comps/expert.cuh"
#include "os/decider/comps/niche.cuh"
#include "os/decider/comps/worker.cuh"

#define SUPPORTED = 1;
namespace aristos{
    // Expert Parallel Group details
    struct __align__(8) EPG {
        uint16_t epRank;
        uint16_t expertSlots;
        uint16_t nLx;
        uint16_t epWorld;

        EPG(const uint16_t& _epR,
            const uint16_t& _eS,
            const uint16_t& _nLx,
            const uint16_t& _epW):
        epRank(_epR), expertSlots(_eS), nLx(_nLx), epWorld(_epW) {}
    };

    template<typename Element>
    __host__ __forceinline__
    void estimateMemory(const InitialConfig& iC, WorkerAttribute* __restrict__ const& dWa) {
        // estimate available device memory
        size_t free = 0, total = 0;
        CHECK_ERROR_EXIT(cudaMemGetInfo(&free, &total));
        // Deduct cost for the dense case, assuming at least one expert per device
        const auto bP = iC.bytesPerParameter<Element>();
        free -= bP * (iC.numParameters + iC.p2pBuffer);
        const size_t mX = cute::ceil_div(iC.numLayers, iC.moeFrequency) * bP * 2UL * (iC.hiddenProjDim * iC.embedDim);
        dWa->memoryCapacity = free / mX;
    }

    template<
        unsigned int Arch,
        typename Element
    >
    __host__ __forceinline__
    void measureThroughput(WorkerAttribute* __restrict__ const& dWa,
        const unsigned int& M, const unsigned int& N, const unsigned int& K) {
        // malloc memory for all matrices
        constexpr auto batch = 2U;
        cuda::std::byte* abc;
        const auto stateSize = sizeof(uint) * (1 + M / BLOCK_M); // tileSync + dT
        const auto aSize = stateSize + sizeof(Element) * M * K;
        const auto abSize = aSize + sizeof(Element) * K * N;
        const auto abcSize = abSize + sizeof(Element) * M * N;
        const auto abcBSize = abcSize + sizeof(Element) * cute::max(N, K); // bias
        const auto abcBSSize = abcBSize + sizeof(Element) * M; // scale/combine weights
        CHECK_ERROR_EXIT(cudaMallocAsync(&abc, abcBSSize, aristosStream));
        CHECK_ERROR_EXIT(cudaMemsetAsync(abc, 0, abcBSSize, aristosStream));
        auto* __restrict__ p = CAST_TO(uint, abc);
        const auto pS = cute::make_tuple(M, N, K);
        using ElementAccum = float;
        using Activation = cutlass::epilogue::thread::ReLU<ElementAccum>;
        constexpr auto blocks = Hardware<Arch>::blocks::value - 1U;
        auto* __restrict__ pBp = CAST_TO(Element, abc + aSize);
        auto* __restrict__ pDp = CAST_TO(Element, abc + abcSize);
        const auto pB = cuda::std::array<Element*, batch>{pBp, pBp};
        const auto pD = cuda::std::array<Element*, batch>{pDp, pDp};
        auto* __restrict__ pC = CAST_TO(Element, abc + abSize);
        auto* __restrict__ pSC = CAST_TO(Element, abc + abcBSize);
        expert<Arch, Activation, batch, ElementAccum><<<blocks, ARISTOS_BLOCK_SIZE, 0, aristosStream>>>(pS, p,
            p + 1, CAST_TO(Element, abc + stateSize), pB, pC, pD, pSC, pSC);
        uint stage = 0;
        CHECK_ERROR_EXIT(cudaMemcpyAsync(&stage, p, sizeof(uint), cudaMemcpyDeviceToHost, aristosStream));
        CHECK_ERROR_EXIT(cudaFreeAsync(abc, aristosStream));
        CHECK_ERROR_EXIT(cudaPeekAtLastError());
        CHECK_ERROR_EXIT(cudaStreamSynchronize(aristosStream));
        // quantize to uint16_t, this should be safe as the value is very small: in the hundreds
        // we use uint due to the API requirement of CUDA atomicMin
        dWa->throughput = static_cast<uint16_t>(stage);
    }

    __host__ __forceinline__
    void discoverTopology(void* hAp, const uint& n, const uint& globalRank, const WorkerAttribute& lWa) {
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
        auto* attributes = CAST_TO(WorkerAttribute, adj + aD);
        static_assert(sizeof(WorkerAttribute) >= sizeof(uint));
        auto* syncArray = CAST_TO(uint, attributes + n);
        // Starting index of heap
        auto* sHeap = CAST_TO(cuda::std::byte, syncArray + 2);
        const auto remotePresent = [&n, &results] {
            for (int i = 0; i < n; ++i) {
                if (nvshmem_ptr(results, i) == nullptr) return true;
            }
            return false;
        };
        const auto isRemotePresent = remotePresent();
        const auto sharedSize = n * (sizeof(floatPair) + sizeof(unsigned int));
        topology::discover<<<ARISTOS_SUPER_BLOCK_SIZE, ARISTOS_BLOCK_SIZE, sharedSize, aristosStream>>>(n, globalRank,
            isRemotePresent, lWa, sHeap, flags, results, syncArray, attributes);
        CHECK_ERROR_EXIT(cudaMemcpyAsync(hAp, adj, aD * sizeof(floatPair) + n * sizeof(uint),
            cudaMemcpyDeviceToHost, aristosStream));
        CHECK_ERROR_EXIT(cudaPeekAtLastError());
        CHECK_ERROR_EXIT(cudaStreamSynchronize(aristosStream));
        nvshmem_free(symHeap);
    }

    __host__ __forceinline__
    void runDecider(const InitialConfig& iC,
        EPG* __restrict__ const& ePg,
        Expert* __restrict__ const& experts,
        Worker* __restrict__ const& wG,
        Worker* __restrict__ const& ePwG,
        uint* __restrict__ const& pT,
        uint* __restrict__ const& ePs,
        uint* __restrict__ const& ePsX,
        uint16_t* __restrict__ const& scratch, // assume zeroed out
        const floatPair* __restrict__ const& aP,
        const WorkerAttribute* __restrict__ const& attributes,
        const uint& rank, const uint& world) {
        const auto mC = ModelConfig{
            iC.numLayers,
            iC.redAmount,
            iC.globalBatch,
            iC.miniBatch,
            iC.moeFrequency,
            iC.p2pBuffer,
            iC.gradBuffer,
        };

        for (uint i = 0; i < world; ++i) {
            const auto [t, m] = attributes[i];
            wG[i] = Worker{i, t, m};
        }
        const auto adj = make_tensor(aP, make_layout(cute::make_shape(world, world), cute::LayoutRight{}));
        const auto dTg = decider::decide(adj, wG, iC.numExperts,
            iC.numExperts, mC);
        const auto epWorld = subsets(dTg, pT, rank);
        for (uint i = 0; i < iC.numExperts; ++i) {
            // assuming homogenous experts, where each has normalized compute cost of 1
            experts[i] = Expert{i, 1};
        }
        // repurpose memory as the expert parallel group
        uint16_t epRank = 0U;
        for (uint i = 0; i < epWorld; ++i) {
            const auto wRank = pT[i];
            if (wRank == rank) {
                epRank = i;
            }
            ePwG[i] = Worker{i, wG[wRank].processingRate, wG[wRank].memoryCapacity};
        }
        decider::assign(ePwG, epWorld, experts, iC.numExperts, ePs);
        uint16_t expertSlots = 0U;
        // compute expert slots for our group
        for (uint16_t i = 0; i < iC.numExperts; ++i) {
            const auto wIdx = ePs[i];
            const uint16_t tally = scratch[wIdx] + 1U;
            scratch[wIdx] = tally;
            expertSlots = cuda::std::max(tally, expertSlots);
        }
        const auto numLocalExperts = scratch[epRank];
        *ePg = EPG{
            epRank,
            expertSlots,
            numLocalExperts,
            epWorld
        };
        if (epWorld == world) {
            // everyone is in one group; thus, we end early
            // peer translation, sharding spec, expert slots
            //return cuda::std::tuple{peerTranslation, assignment, epRank, expertSlots, numLocalExperts};
            return;
        }
        // Get other group ids
        std::unordered_set<uint> groups{};
        for (const auto& i: dTg) {
            groups.emplace(i);
        }
        const auto myGroup = dTg[rank];

        auto* __restrict__ pTs = pT + epWorld;

        for (const auto& group : groups) {
            if (group != myGroup) {
                std::ranges::fill(scratch, scratch + world, 0U);
                const auto ePw = subsets(dTg, pTs, group);
                for (uint i = 0; i < ePw; ++i) {
                    const auto wRank = pTs[i];
                    ePwG[i] = Worker{
                        wRank,
                        wG[wRank].processingRate,
                        wG[wRank].memoryCapacity
                    };
                }
                decider::assign(ePwG, epWorld, experts, iC.numExperts, ePsX);
                for (uint16_t i = 0; i < iC.numExperts; ++i) {
                    const auto wIdx = ePsX[i];
                    const uint16_t tally = scratch[wIdx] + 1U;
                    scratch[wIdx] = tally;
                    expertSlots = cuda::std::max(tally, expertSlots);
                }
            }
        }
    }

    template<unsigned int Arch, typename Element>
    requires(aristos::TensorValueType<Element> && SupportedArch<Arch>)
    __host__ __forceinline__
    void archSpecificInit(const InitialConfig& iC) {
        // initialize communication backend
        nvshmem_init();
        CUTE_CHECK_ERROR(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
        const auto globalWorld = nvshmem_n_pes();
        const auto rank = nvshmem_my_pe();

        // Pointer to adjacency matrix and throughput of all devices
        const auto aD = globalWorld * globalWorld;
        const auto dZ = sizeof(EPG) + 2 * sizeof(Worker) * globalWorld + sizeof(Expert) * iC.numExperts;
        const auto sZ = sizeof(EDT) * iC.numExperts + sizeof(uint) * (globalWorld + 2 * iC.numExperts + 1) +
            sizeof(uint16_t) * globalWorld + sizeof(bool) * iC.numExperts;
        const auto cZ = dZ + aD * sizeof(floatPair) + globalWorld * sizeof(WorkerAttribute) + sZ;

        // allocate all memory in one go
        auto* mP = static_cast<cuda::std::byte*>(std::calloc(cZ, sizeof(cuda::std::byte)));

        // Pointer salami slicing
        auto* ePg = CAST_TO(EPG, mP);
        static_assert(alignof(EPG) % alignof(Expert) == 0);
        auto* workers = CAST_TO(Worker, mP + sizeof(EPG));
        auto* ePWorkers = workers + globalWorld;
        static_assert(alignof(Worker) % alignof(Expert) == 0);
        auto* experts = CAST_TO(Expert, mP + sizeof(Worker) + globalWorld);
        static_assert(alignof(Expert) % alignof(floatPair) == 0);
        auto* aP = CAST_TO(floatPair, mP + dZ);
        static_assert(alignof(floatPair) % alignof(WorkerAttribute) == 0);
        auto* wAp = CAST_TO(WorkerAttribute, aP + aD);
        static_assert(alignof(WorkerAttribute) % alignof(EDT) == 0);
        auto* __restrict__ eDr = CAST_TO(EDT, wAp + globalWorld);
        static_assert(alignof(EDT) == 0 % alignof(BookType) == 0);
        auto* pT = CAST_TO(uint, eDr + iC.numExperts);
        auto* ePs = pT + globalWorld;
        auto* ePsX = ePs + iC.numExperts; // scratch
        auto* nRXp = ePsX + iC.numExperts;
        auto* scratch = CAST_TO(uint16_t, nRXp + 1);
        auto* __restrict__ eRs = CAST_TO(bool, scratch + globalWorld);

        estimateMemory<Element>(iC, &wAp[rank]);
        measureThroughput<Arch, Element>(wAp + rank, iC.seqLen, iC.hiddenProjDim, iC.embedDim);
        discoverTopology(aP, globalWorld, globalWorld, wAp[rank]);

        // The topology adjacency matrix is ready, let's map devices to optimal cooperative process groups
        runDecider(iC, ePg, experts, workers, ePWorkers, pT, ePs, ePsX,
            scratch, aP, wAp, rank, globalWorld);
        const auto ePgD = *ePg;
        // Now allocate memory
        /// Symmetric memory
        const auto eCap = iC.shouldDrop? iC.expertCapacity(ePgD.epWorld) : iC.seqLen;
        const auto heapBytes = STAGES * CELLS * ePgD.epWorld * ePgD.expertSlots * eCap * iC.embedDim * sizeof(Element);
        const auto tMc = Bookkeeping::tiles<BLOCK_M>(eCap);
        const auto tN = Bookkeeping::tiles<BLOCK_N>(iC.embedDim);
        const auto flagBytes = (ePgD.epWorld * ePgD.nLx + iC.numExperts * tMc * tN) * sizeof(flagsType);
        auto* sHeap = nvshmem_calloc(flagBytes + heapBytes, sizeof(cuda::std::byte));
        
        auto* sHb = static_cast<cuda::std::byte*>(sHeap);

        // local bookkeeping memory
        constexpr auto blocks = Hardware<Arch>::blocks::value - 1U;
        const auto bookSize = Bookkeeping::bookLength(iC.seqLen,
            iC.numExperts, ePgD.nLx, iC.hiddenProjDim,
            iC.embedDim, eCap, blocks, ePgD.epWorld, sizeof(Element));

        cuda::std::byte* book;
        CHECK_ERROR_EXIT(cudaMallocAsync(&book, bookSize, aristosStream));
        CHECK_ERROR_EXIT(cudaMemsetAsync(&book, 0, bookSize, aristosStream));
        const uint16_t functionId = 8 * ((Arch - MIN_ARCH) / 100) + 4 * (iC.numExperts > BLOCK_N) +
            2 * iC.shouldDrop + (iC.k > 1);
        // Initialize bookkeeping
        hostBookkeeping = Bookkeeping{
            sHb + flagBytes,
            static_cast<flagsType *>(sHeap),
            book,
            iC.seqLen * iC.miniBatch,
            iC.numExperts,
            ePgD.nLx,
            ePgD.expertSlots,
            ePgD.epRank,
            iC.hiddenProjDim,
            iC.embedDim,
            eCap,
            blocks,
            ePgD.epWorld,
            functionId,
            sizeof(Element)
        };
        // copy peer translation and parallelism spec to device memory
        CHECK_ERROR_EXIT(cudaMemcpyAsync(hostBookkeeping.pT(), pT,
            sizeof(BookType) * ePgD.epWorld, cudaMemcpyHostToDevice, aristosStream));
        CHECK_ERROR_EXIT(cudaMemcpyAsync(hostBookkeeping.ePs(), ePs,
            sizeof(BookType) * iC.numExperts, cudaMemcpyHostToDevice, aristosStream));
        // copy device-wide barrier
        const auto hB = new cuda::barrier<cuda::thread_scope_device>{blocks};
        CHECK_ERROR_EXIT(cudaMemcpyAsync(hostBookkeeping.dB(), hB, sizeof(cuda::barrier<cuda::thread_scope_device>),
            cudaMemcpyHostToDevice, aristosStream));
        CHECK_ERROR_EXIT(cudaMemcpyToSymbolAsync(bookkeeping, &hostBookkeeping, sizeof(Bookkeeping), 0,
            cudaMemcpyHostToDevice, aristosStream));
        // Construct eD and eLs and compute nRx
        // eLs, reusing ePsX
        auto* __restrict__ eLs = ePsX;
        auto nRx = 0U;
        std::ranges::fill(scratch, scratch + ePgD.epWorld, 0U);
        for (uint i = 0; i < iC.numExperts; ++i) {
            const auto ePrank = ePs[i];
            const auto gRank = pT[ePrank];
            eLs[i] = scratch[ePrank]++;
            if (nvshmem_ptr(sHeap, gRank) == nullptr) {
                ++nRx;
                // This expert is hosted on a remote device
                eRs[i] = true;
            }
        }
        *nRXp = nRx;
        // Copy eLs and nRx
        CHECK_ERROR_EXIT(cudaMemcpyAsync(hostBookkeeping.eLs(), eLs,
            sizeof(BookType) * (iC.numExperts + 1), cudaMemcpyHostToDevice, aristosStream));

        // eD
        auto* __restrict__ eDp = eDr + nRx;
        auto rI = 0U;
        auto pI = 0U;
        for (uint i = 0; i < iC.numExperts; ++i) {
            if (eRs[i]) {
                // remote expert
                eDr[rI++] = {i, eLs[i], ePs[i]};
            }
            else {
                // p2p expert
                eDp[pI++] = {i, eLs[i], ePs[i]};
            }
        }
        CHECK_ERROR_EXIT(cudaMemcpyAsync(hostBookkeeping.eD(), eDr,
            sizeof(EDT) * iC.numExperts, cudaMemcpyHostToDevice, aristosStream));
        CHECK_ERROR_EXIT(cudaPeekAtLastError());
        CHECK_ERROR_EXIT(cudaStreamSynchronize(aristosStream));
        delete hB;
        std::free(aP);
    }

    // Should be called before loading the model
    template<typename Element>
    requires(aristos::TensorValueType<Element>)
    __host__ __forceinline__
    void initialize(const InitialConfig& iC) {
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
        switch (arch) {
            case 7:
                archSpecificInit<700, Element>(iC);
            break;
            case 8:
                archSpecificInit<800, Element>(iC);
            break;
            default:
                // TODO hopper
                break;
        }
#if (__CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800)
        cudaStreamAttrValue stream_attribute;
        stream_attribute.accessPolicyWindow.base_ptr  = hostBookkeeping.pDs();
        stream_attribute.accessPolicyWindow.num_bytes = cuda::std::min(static_cast<size_t>(0.25 * l2CacheSize),
            sizeof(BookType) * (3 * hostBookkeeping.nx + hostBookkeeping.world + 1));
        stream_attribute.accessPolicyWindow.hitRatio  = 1.0;
        stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
        stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

        //Set the attributes to a CUDA stream of type cudaStream_t
        cudaStreamSetAttribute(aristosStream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
#endif
    }

    __host__ __forceinline__
    void finalize(){
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
        nvshmem_free(hostBookkeeping.sHeap);
        nvshmem_finalize();
        CHECK_ERROR_EXIT(cudaPeekAtLastError());
        CHECK_ERROR_EXIT(cudaStreamSynchronize(aristosStream));
    }
}
#endif //BOOTSTRAP_CUH
