//
// Created by oja7 on 11/12/24.
//

#ifndef BOOTSRAP_CUH
#define BOOTSTRAP_CUH

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

#include "throughput.cuh"
#include "topo.cuh"
#include "debug.cuh"
#include "telemetry.cuh"
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
        #if ARISTOS_TRACE
        aristosInitRange estRange{__PRETTY_FUNCTION__};
        #endif
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
        #if ARISTOS_TRACE
        aristosInitRange discRange{__func__};
        #endif
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
        constexpr auto seqNo = 1U;
        topology::discover<<<ARISTOS_SUPER_BLOCK_SIZE, ARISTOS_BLOCK_SIZE, sharedSize, aristosStream>>>(n, globalRank,
            isRemotePresent, lWa, sHeap, flags, results, syncArray, attributes, seqNo);
        CHECK_ERROR_EXIT(cudaMemcpyAsync(hAp, adj, aD * sizeof(floatPair) + n * sizeof(uint),
            cudaMemcpyDeviceToHost, aristosStream));
        CHECK_ERROR_EXIT(cudaPeekAtLastError());
        CHECK_ERROR_EXIT(cudaStreamSynchronize(aristosStream));
        nvshmem_free(symHeap);
    }

    __host__ __forceinline__
    void runDecider(EPG* __restrict__ const& ePg,
        Expert* __restrict__ const& experts,
        Worker* __restrict__ const& wG,
        Worker* __restrict__ const& ePwG,
        uint* __restrict__ const& dTg,
        uint* __restrict__ const& pT,
        uint* __restrict__ const& ePs,
        uint* __restrict__ const& ePsX,
        uint16_t* __restrict__ const& scratch, // assume zeroed out
        const floatPair* __restrict__ const& aP,
        const WorkerAttribute* __restrict__ const& attributes,
        const uint& rank, const uint& world) {
        #if ARISTOS_TRACE
        aristosInitRange decRange{__func__};
        #endif
        constexpr auto E = ACC::E::value;

        for (uint16_t i = 0; i < world; ++i) {
            const auto [t, m] = attributes[i];
            wG[i] = Worker{i, __half2float(t.to_half()), m};
        }
        const auto adj = make_tensor(aP, make_layout(cute::make_shape(world, world), cute::LayoutRight{}));
        constexpr Decider<ACC::JT::value> decider{};
        decider(adj, wG, E, E, dTg);
        auto epWorld = subsets(dTg, pT, world, rank);
        for (uint i = 0; i < E; ++i) {
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
        assign(ePwG, epWorld, experts, E, ePs);
        uint16_t expertSlots = 0U;
        // compute expert slots for our group
        for (uint16_t i = 0; i < E; ++i) {
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
        for (uint i = 0; i < world; ++i) {
            groups.emplace(dTg[i]);
        }
        const auto myGroup = dTg[rank];

        auto* __restrict__ pTs = pT + epWorld;

        for (const auto& group : groups) {
            if (group != myGroup) {
                std::ranges::fill(scratch, scratch + world, 0U);
                const auto ePw = subsets(dTg, pTs, world, group);
                epWorld = cute::max(epWorld, ePw);
                for (uint i = 0; i < ePw; ++i) {
                    const auto wRank = pTs[i];
                    ePwG[i] = Worker{
                        static_cast<uint16_t>(wRank),
                        wG[wRank].processingRate,
                        wG[wRank].memoryCapacity
                    };
                }
                assign(ePwG, epWorld, experts, E, ePsX);
                for (uint16_t i = 0; i < E; ++i) {
                    const auto wIdx = ePsX[i];
                    const uint16_t tally = scratch[wIdx] + 1U;
                    scratch[wIdx] = tally;
                    expertSlots = cuda::std::max(tally, expertSlots);
                }
            }
        }
        // The global maximum of the below values is necessary for allocating a uniformly sized symmetric heap
        // across all PEs.
        ePg->expertSlots = expertSlots;
        ePg->epWorld = epWorld;
    }

    template<EP e = EP::yes>
    __host__ __forceinline__
    void distributedInit() {
        #if ARISTOS_TRACE
        aristosInitRange distRange{__PRETTY_FUNCTION__};
        #endif
        static_assert(e == EP::yes);
        constexpr auto blocks = ACC::PeakHardware::blocks::value;
        using Element = ACC::Element;
        constexpr uint E = ACC::E::value;
        // initialize communication backend
        nvshmem_init();
        const uint devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
        CHECK_ERROR_EXIT(cudaSetDevice(devId));
        const auto globalWorld = nvshmem_n_pes();
        const auto rank = nvshmem_my_pe();

        // Pointer to adjacency matrix and throughput of all devices
        const auto aD = globalWorld * globalWorld;
        const auto dZ = sizeof(EPG) +
                2 * sizeof(Worker) * globalWorld +
                sizeof(Expert) * E +
                aD * sizeof(floatPair) +
                globalWorld * sizeof(WorkerAttribute) +
                sizeof(uint) * globalWorld;
        const auto aXz = (sizeof(ELI) + sizeof(PEL)) * E + sizeof(PLI) * globalWorld;
        const auto pZ = umax(dZ, aXz);
        const auto sZ = sizeof(uint) * (globalWorld + 2 * E) + sizeof(uint16_t) * globalWorld;

        // allocate all memory in one go
        auto* mP = static_cast<cuda::std::byte*>(std::calloc(pZ + sZ, sizeof(cuda::std::byte)));

        // Pointer salami slicing
        auto* ePg = CAST_TO(EPG, mP);
        static_assert(alignof(EPG) % alignof(Expert) == 0);
        auto* workers = CAST_TO(Worker, mP + sizeof(EPG));
        auto* ePWorkers = workers + globalWorld;
        static_assert(alignof(Worker) % alignof(Expert) == 0);
        auto* experts = CAST_TO(Expert, ePWorkers + globalWorld);
        static_assert(alignof(Expert) % alignof(floatPair) == 0);
        auto* aP = CAST_TO(floatPair, experts + E);
        static_assert(alignof(floatPair) % alignof(WorkerAttribute) == 0);
        auto* wAp = CAST_TO(WorkerAttribute, aP + aD);
        static_assert(alignof(WorkerAttribute) % alignof(uint) == 0);
        auto* dTg = CAST_TO(uint, wAp + globalWorld);

        // Result buffers
        auto* pT = CAST_TO(uint, mP + pZ);
        auto* ePs = pT + globalWorld;
        auto* ePsX = ePs + E; // scratch
        auto* scratch = CAST_TO(uint16_t, ePsX + 1);

        estimateMemory(&wAp[rank]);
        mT(wAp + rank);

        discoverTopology(aP, globalWorld, rank, wAp[rank]);

        // The topology adjacency matrix is ready, let's map devices to optimal cooperative process groups
        runDecider(ePg, experts, workers, ePWorkers, dTg, pT, ePs, ePsX,
            scratch, aP, wAp, rank, globalWorld);
        const auto ePgD = *ePg;
        // Now allocate memory
        /// Symmetric memory
        const auto heapBytes = STAGES * CELLS * ePgD.epWorld * ePgD.expertSlots * ACC::EC::value *
            ACC::H::value * sizeof(Element);
        const auto syncArrayBytes = sizeof(flagsType) * (ePgD.epWorld + 1);
        const auto flagBytes = (ePgD.epWorld * ePgD.expertSlots + E * ACC::TCM::value * ACC::TNx::value) *
            sizeof(flagsType);
        // Note this allocation's size has to be identical across all PEs
        auto* sHeap = nvshmem_calloc(syncArrayBytes + flagBytes + heapBytes, sizeof(cuda::std::byte));

        auto* sHb = static_cast<cuda::std::byte*>(sHeap);

        // local bookkeeping memory
        const auto bookSize = Bookkeeping::bookLength(ePgD.nLx, ePgD.epWorld);
        cuda::std::byte* book;
        CHECK_ERROR_EXIT(cudaMallocAsync(&book, bookSize, aristosStream));
        CHECK_ERROR_EXIT(cudaMemsetAsync(book, 0, bookSize, aristosStream));
        // Initialize bookkeeping
        auto* sA = static_cast<flagsType*>(sHeap);
        auto* flags = CAST_TO(flagsType, sHb + syncArrayBytes);
        auto* wSHeap = sHb + flagBytes + syncArrayBytes;
        hostBookkeeping = Bookkeeping{
            sHb,
            sA,
            flags,
            wSHeap,
            book,
            ePgD.nLx,
            ePgD.epRank,
            ePgD.epWorld,
            ePgD.expertSlots,
        };
        // copy device-wide barrier
        const auto hB = new cuda::barrier<cuda::thread_scope_device>{blocks};
        CHECK_ERROR_EXIT(cudaMemcpyAsync(hostBookkeeping.dB(), hB,
            sizeof(cuda::barrier<cuda::thread_scope_device>),
            cudaMemcpyHostToDevice, aristosStream));
        CHECK_ERROR_EXIT(cudaMemcpyToSymbolAsync(bookkeeping, &hostBookkeeping, sizeof(Bookkeeping), 0,
            cudaMemcpyHostToDevice, aristosStream));

        // reuse pre-allocated memory for device data structures
        auto* __restrict__ pEL = CAST_TO(PEL, mP);
        static_assert(alignof(PEL) % alignof(ELI) == 0);
        auto* __restrict__ eLI = CAST_TO(ELI, pEL + E);
        static_assert(alignof(ELI) % alignof(PLI) == 0);
        auto* __restrict__ pLI = CAST_TO(PLI, eLI + E);

        auto nRx = 0U;
        auto pel = PEL{};
        auto eli = ELI{};
        auto pli = PLI{};
        std::ranges::fill(scratch, scratch + ePgD.epWorld, 0U);
        for (uint i = 0; i < E; ++i) {
            const auto ePrank = ePs[i];
            const auto gRank = pT[ePrank];
            auto* rSHeap = CAST_TO(cuda::std::byte, nvshmem_ptr(wSHeap, gRank));
            auto* rFlags = CAST_TO(cuda::std::byte, nvshmem_ptr(flags, gRank));
            const auto xLi = scratch[ePrank]++;
            const auto isRemote = rSHeap == nullptr;
            if (isRemote) {
                ++nRx;
            }
            // PEL
            pel.isRemote = isRemote;
            pel.expertLocalIdx = xLi;
            pel.pe = gRank;
            pel.remoteSFlags = rFlags;
            pel.remoteSHeap = rSHeap;
            pel.peer = ePrank;

            // ELI
            eli.epRank = ePrank;
            eli.isRemote = isRemote;
            eli.localExpertIndex = xLi;

            // PLI
            pli.isRemote = isRemote;
            pli.pe = gRank;

            pEL[i] = pel;
            eLI[i] = eli;
            pLI[ePrank] = pli;
        }

        CHECK_ERROR_EXIT(cudaMemcpyAsync(hostBookkeeping.pEL(), pEL, sizeof(PEL) * E,
            cudaMemcpyHostToDevice, aristosStream));
        // Copy eLI and pLI in one fell sweep
        CHECK_ERROR_EXIT(cudaMemcpyAsync(hostBookkeeping.eL(), eLI,
            sizeof(ELI) * E + sizeof(PLI) * ePgD.epWorld,
            cudaMemcpyHostToDevice, aristosStream));
        CHECK_ERROR_EXIT(cudaMemcpyAsync(hostBookkeeping.nRx(), &nRx,
            sizeof(BookType), cudaMemcpyHostToDevice, aristosStream));
        CHECK_ERROR_EXIT(cudaPeekAtLastError());
        CHECK_ERROR_EXIT(cudaStreamSynchronize(aristosStream));
        delete hB;
        std::free(mP);
    }

    template<>
    __host__ __forceinline__
    void distributedInit<EP::no>() {
        constexpr auto bookSize = Bookkeeping::bookLength();
        cuda::std::byte* book;
        CHECK_ERROR_EXIT(cudaMallocAsync(&book, bookSize, aristosStream));
        CHECK_ERROR_EXIT(cudaMemsetAsync(&book, 0, bookSize, aristosStream));
        hostBookkeeping = Bookkeeping{};
        CHECK_ERROR_EXIT(cudaMemcpyToSymbolAsync(bookkeeping, &hostBookkeeping,
            sizeof(Bookkeeping), 0,
            cudaMemcpyHostToDevice, aristosStream));
        CHECK_ERROR_EXIT(cudaPeekAtLastError());
        CHECK_ERROR_EXIT(cudaStreamSynchronize(aristosStream));
    }

    // Should be called before loading the model
    __host__ __forceinline__
    void initialize() {
        #if ARISTOS_TRACE
        aristosInitRange initRange{__PRETTY_FUNCTION__};
        #endif
        reportError(!isInitialized, "Already Initialized");
        using GPUType = aristos::Hardware<ARISTOS_ARCH, 255>;
        constexpr auto blocks = GPUType::OS::processorBlocks::value;
        static_assert(ARISTOS_ARCH >= 700, "Volta and above is required!");
        isInitialized = true;
        static_assert(ACC::S::value % BLOCK_M == 0 && ACC::S::value < BLOCK_M * blocks * ACC::TMU::value &&
        ACC::P::value % BLOCK_N == 0 && ACC::H::value % BLOCK_N == 0);
        static_assert(NUM_EXPERTS <= cuda::std::numeric_limits<uint16_t>::max(),
            "For performance, we assume number of experts <= UINT16_MAX");
        int cudaDevAttribute = 0;
        int dev = 0;
        CHECK_ERROR_EXIT(cudaGetDevice(&dev));
        CHECK_ERROR_EXIT(cudaDeviceGetAttribute(&cudaDevAttribute, cudaDevAttrMemoryPoolsSupported, dev));
        reportError(cudaDevAttribute, "Memory Pools support required");
        distributedInit<(ACC::E::value > 1) ? EP::yes : EP::no>();
    }

    __host__ __forceinline__
    void finalize(){
        #if ARISTOS_TRACE
        aristosInitRange finalRange{__PRETTY_FUNCTION__};
        #endif
        reportError(isInitialized, "Not initialized!");
        isInitialized = false;
        CHECK_ERROR_EXIT(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
        CHECK_ERROR_EXIT(cudaFreeAsync(hostBookkeeping.book, aristosStream));
        nvshmem_free(hostBookkeeping.symHeap);
        nvshmem_finalize();
        CHECK_ERROR_EXIT(cudaPeekAtLastError());
        CHECK_ERROR_EXIT(cudaStreamSynchronize(aristosStream));
    }
}
#endif //BOOTSTRAP_CUH
