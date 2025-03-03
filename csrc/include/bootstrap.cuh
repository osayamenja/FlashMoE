//
// Created by oja7 on 11/12/24.
//

#ifndef BOOTSRAP_CUH
#define BOOTSTRAP_CUH

#include <vector>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/epilogue/thread/activation.h>

#include "throughput.cuh"
#include "topo.cuh"
#include "debug.cuh"
#include "types.cuh"
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

    __host__ __forceinline__
    void estimateMemory(WorkerAttribute* __restrict__ const& dWa) {
        // estimate available device memory
        size_t free = 0, total = 0;
        CHECK_ERROR_EXIT(cudaMemGetInfo(&free, &total));
        // Deduct cost for the dense case, assuming at least one expert per device
        free -= ACC::BPP::value * (ACC::PC::value + ACC::P2PB::value);
        constexpr size_t mX = cute::ceil_div(ACC::L::value, ACC::F::value) * ACC::BPP::value * 2UL *
            (ACC::P::value * ACC::H::value);
        dWa->memoryCapacity = free / mX;
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

        for (uint16_t i = 0; i < world; ++i) {
            const auto [t, m] = attributes[i];
            wG[i] = Worker{i, t, m};
        }
        const auto adj = make_tensor(aP, make_layout(cute::make_shape(world, world), cute::LayoutRight{}));
        const auto dTg = decide(adj, wG, iC.numExperts,
            iC.numExperts, mC);
        auto epWorld = subsets(dTg, pT, rank);
        for (uint i = 0; i < iC.numExperts; ++i) {
            // assuming homogenous experts, where each has normalized compute cost of 1
            experts[i] = Expert{i, 1};
        }
        // repurpose memory as the expert parallel group
        uint16_t epRank = 0U;
        for (uint16_t i = 0; i < epWorld; ++i) {
            const auto wRank = pT[i];
            if (wRank == rank) {
                epRank = i;
            }
            ePwG[i] = Worker{i, wG[wRank].processingRate, wG[wRank].memoryCapacity};
        }
        assign(ePwG, epWorld, experts, iC.numExperts, ePs);
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
                epWorld = cute::max(epWorld, ePw);
                for (uint i = 0; i < ePw; ++i) {
                    const auto wRank = pTs[i];
                    ePwG[i] = Worker{
                        static_cast<uint16_t>(wRank),
                        wG[wRank].processingRate,
                        wG[wRank].memoryCapacity
                    };
                }
                assign(ePwG, epWorld, experts, iC.numExperts, ePsX);
                for (uint16_t i = 0; i < iC.numExperts; ++i) {
                    const auto wIdx = ePsX[i];
                    const uint16_t tally = scratch[wIdx] + 1U;
                    scratch[wIdx] = tally;
                    expertSlots = cuda::std::max(tally, expertSlots);
                }
            }
        }
        // The global maximum of the below values is necessary for allocating a uniformly sized symmetric heap
        // across all PGAS PEs.
        ePg->expertSlots = expertSlots;
        ePg->epWorld = epWorld;
    }

    __host__ __forceinline__
    void internalInitialize() {
        using GPUType = Hardware<ARISTOS_ARCH, 255>;
        using Element = ACC::Element;
        using Activation = ACC::ActivationOp;
        // initialize communication backend
        nvshmem_init();
        const uint devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
        CHECK_ERROR_EXIT(cudaSetDevice(devId));
        const auto globalWorld = nvshmem_n_pes();
        const auto rank = nvshmem_my_pe();

        constexpr uint E = ACC::E::value;
        constexpr uint S = ACC::S::value;
        constexpr uint P = ACC::P::value;
        constexpr uint H = ACC::H::value;
        constexpr uint EC = ACC::EC::value;

        // Pointer to adjacency matrix and throughput of all devices
        const auto aD = globalWorld * globalWorld;
        const auto dZ = sizeof(EPG) + 2 * sizeof(Worker) * globalWorld + sizeof(Expert) * E;
        const auto sZ = sizeof(EDT) * E + sizeof(uint) * (2 * globalWorld +
            2 * E + 1) + sizeof(uint16_t) * globalWorld + sizeof(bool) * E;
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
        auto* pT = CAST_TO(uint, eDr + E);
        auto* ePs = pT + globalWorld;
        auto* ePsX = ePs + E; // scratch
        auto* nRXp = ePsX + E;
        auto* scratch = CAST_TO(uint16_t, nRXp + 1);
        auto* __restrict__ eRs = CAST_TO(bool, scratch + globalWorld);

        estimateMemory(&wAp[rank]);
        mT(wAp + rank);

        discoverTopology(aP, globalWorld, globalWorld, wAp[rank]);

        // The topology adjacency matrix is ready, let's map devices to optimal cooperative process groups
        runDecider(iC, ePg, experts, workers, ePWorkers, pT, ePs, ePsX,
            scratch, aP, wAp, rank, globalWorld);
        const auto ePgD = *ePg;
        // Now allocate memory
        /// Symmetric memory
        const auto eCap = iC.shouldDrop ? iC.expertCapacity(ePgD.epWorld) : iC.seqLen;
        const auto heapBytes = STAGES * CELLS * ePgD.epWorld * ePgD.expertSlots * eCap *
            iC.embedDim * sizeof(Element);
        const auto tMc = Bookkeeping::tiles<BLOCK_M>(eCap);
        const auto tN = Bookkeeping::tiles<BLOCK_N>(iC.embedDim);
        const auto flagBytes = (ePgD.epWorld * ePgD.expertSlots + iC.numExperts * tMc * tN) *
            sizeof(flagsType);
        // Note this allocation's size has to be identical across all PEs
        auto* sHeap = nvshmem_calloc(flagBytes + heapBytes, sizeof(cuda::std::byte));
        
        auto* sHb = static_cast<cuda::std::byte*>(sHeap);

        // local bookkeeping memory
        constexpr auto blocks = GPUType::blocks::value - 1U;
        const auto bookSize = Bookkeeping::bookLength(iC.seqLen,
            iC.numExperts, ePgD.nLx, iC.hiddenProjDim,
            iC.embedDim, eCap, blocks, ePgD.epWorld, sizeof(Element));

        cuda::std::byte* book;
        CHECK_ERROR_EXIT(cudaMallocAsync(&book, bookSize, aristosStream));
        CHECK_ERROR_EXIT(cudaMemsetAsync(&book, 0, bookSize, aristosStream));
        // Initialize bookkeeping
        hostBookkeeping = Bookkeeping{
            sHb + flagBytes,
            static_cast<flagsType *>(sHeap),
            book,
            iC.seqLen * iC.miniBatch,
            iC.numExperts,
            iC.k,
            ePgD.nLx,
            ePgD.expertSlots,
            ePgD.epRank,
            iC.hiddenProjDim,
            iC.embedDim,
            eCap,
            blocks,
            ePgD.epWorld,
            sizeof(Element)
        };
        // copy peer translation and parallelism spec to device memory
        CHECK_ERROR_EXIT(cudaMemcpyAsync(hostBookkeeping.pT(), pT,
            sizeof(BookType) * ePgD.epWorld, cudaMemcpyHostToDevice, aristosStream));
        CHECK_ERROR_EXIT(cudaMemcpyAsync(hostBookkeeping.ePs(), ePs,
            sizeof(BookType) * iC.numExperts, cudaMemcpyHostToDevice, aristosStream));
        // copy device-wide barrier
        const auto hB = new cuda::barrier<cuda::thread_scope_device>{blocks};
        CHECK_ERROR_EXIT(cudaMemcpyAsync(hostBookkeeping.dB(), hB,
            sizeof(cuda::barrier<cuda::thread_scope_device>),
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
        std::free(mP);
    }

    // Should be called before loading the model
    __host__ __forceinline__
    void initialize() {
        reportError(!isInitialized, "Already Initialized");
        using GPUType = aristos::Hardware<ARISTOS_ARCH, 255>;
        constexpr auto blocks = GPUType::OS::processorBlocks::value;
        static_assert(ARISTOS_ARCH >= 700, "Volta and above is required!");
        isInitialized = true;
        static_assert(SEQ_LEN % BLOCK_M == 0 && SEQ_LEN < BLOCK_M * blocks * 128 &&
        I_SIZE % BLOCK_N == 0 && HIDDEN_SIZE % BLOCK_N == 0);
        static_assert(NUM_EXPERTS <= cuda::std::numeric_limits<uint16_t>::max(),
            "For performance, we assume number of experts <= UINT16_MAX");
        int cudaDevAttribute = 0;
        int dev = 0;
        CHECK_ERROR_EXIT(cudaGetDevice(&dev));
        CHECK_ERROR_EXIT(cudaDeviceGetAttribute(&cudaDevAttribute, cudaDevAttrMemoryPoolsSupported, dev));
        reportError(cudaDevAttribute, "Memory Pools support required");
        internalInitialize();
    }

    __host__ __forceinline__
    void finalize(){
        reportError(isInitialized, "Not initialized!");
        isInitialized = false;
        CHECK_ERROR_EXIT(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
        CHECK_ERROR_EXIT(cudaFreeAsync(hostBookkeeping.book, aristosStream));
        nvshmem_free(hostBookkeeping.sHeap);
        nvshmem_finalize();
        CHECK_ERROR_EXIT(cudaPeekAtLastError());
        CHECK_ERROR_EXIT(cudaStreamSynchronize(aristosStream));
    }
}
#endif //BOOTSTRAP_CUH
