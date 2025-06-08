/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by oja7 on 11/12/24.
//

#ifndef BOOTSRAP_CUH
#define BOOTSTRAP_CUH

#include <cstdlib>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <fmt/ranges.h>

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
namespace kleos{
    __host__ __forceinline__
    void imposeStrategy(EPG* __restrict__ const& ePg,
        uint* __restrict__ const& pT, uint* __restrict__ const& ePs, const uint& rank, const uint& globalWorld) {
        constexpr auto E = ACC::E::value;
        *ePg = EPG{
            static_cast<uint16_t>(rank),
            static_cast<uint16_t>(E / globalWorld),
            static_cast<uint16_t>(E / globalWorld),
            static_cast<uint16_t>(globalWorld)
        };
        for (uint i = 0; i < globalWorld; ++i) {
            pT[i] = i;
        }
        const auto split = E / globalWorld;
        for (uint i = 0; i < E; ++i) {
            ePs[i] = i / split;
        }
    }

    __host__ __forceinline__
    auto gEI(const char* const& eV, const int& eVd) {
        if (std::getenv(eV) == nullptr) {
            return eVd;
        }
        return std::stoi(std::getenv(eV));
    }

    __host__ __forceinline__
    void uEI(const char* const& eV, const int& v) {
        if (setenv(eV, std::to_string(v).c_str(), 1)) {
            perror(std::string("failed to set environment variable: " + std::string(eV)).c_str());
        }
    }

    __host__ __forceinline__
    void exportTopo(const floatPair* __restrict__ const& aP,
        const WorkerAttribute* __restrict__ const& attributes,
        const uint& world, const uint& rank) {
        const auto aM = make_tensor(aP,
            make_layout(cute::make_shape(world, world), cute::LayoutRight{}));
        std::vector<std::array<float, 2>> tAM(world);
        std::vector<float> tWT(world);
        std::vector<uint16_t> tWM(world);

        auto* file = std::fopen(std::string("adjMatrix_Rank")
            .append(std::to_string(rank)).append(".txt").c_str(), "w");
        fmt::print(file, "----> {} processes pair-wise (𝛼 ms, 𝛽 ms/MB) costs <------\n", world);
        for (uint i = 0; i < world; ++i){
            for (uint j = 0; j < world; ++j){
                const auto [alpha, beta] = aM(i, j);
                tAM[j] = {alpha, beta};
            }
            fmt::print(file, "Rank {}: {:::.2e}\n", i, tAM);
        }
        for (uint i = 0; i < world; ++i) {
            const auto [t, m] = attributes[i];
            tWT[i] = static_cast<float>(t);
            tWM[i] = m;
        }
        fmt::print(file, "Rank {}: \n\t Throughput: {}\n\t MemoryCapacity: {}\n", rank, tWT, tWM);
        std::fclose(file);
    }

    __host__ __forceinline__
    void estimateMemory(WorkerAttribute* __restrict__ const& dWa) {
        #if KLEOS_NVTX
        kleosRange estRange{__PRETTY_FUNCTION__};
        #endif
        // estimate available device memory
        size_t free = 0, total = 0;
        KLEOS_CHECK_CUDA(cudaMemGetInfo(&free, &total));
        // Deduct cost for the dense case, assuming at least one expert per device
        free -= ACC::BPP::value * (ACC::PC::value + ACC::S::value * ACC::H::value);
        constexpr size_t mX = cute::ceil_div(ACC::L::value, ACC::F::value) * ACC::BPP::value * 2UL *
            (ACC::P::value * ACC::H::value);
        dWa->memoryCapacity = free / mX;
    }

    __host__ __forceinline__
    void discoverTopology(void* const& hAp, const uint& n, const uint& globalRank,
        const WorkerAttribute& lWa, WorkerAttribute* __restrict__ const& wAp) {
        #if KLEOS_NVTX
        kleosRange discRange{__func__};
        #endif
        const auto aD = n * n;
        const auto heapBytes = n * BETA_BUFFER;
        const auto sBkz =  n + aD;
        WorkerAttribute* attributes;
        cuda::barrier<cuda::thread_scope_device>* dvB;
        KLEOS_CHECK_CUDA(cudaMallocAsync(&attributes, sizeof(WorkerAttribute) * n,
            kleosStream));
        KLEOS_CHECK_CUDA(cudaMallocAsync(&dvB,
            sizeof(cuda::barrier<cuda::thread_scope_device>), kleosStream));
        const auto hB = new cuda::barrier<cuda::thread_scope_device>{KLEOS_STATIC_SBZ};
        KLEOS_CHECK_CUDA(cudaMemcpyAsync(dvB, hB,
            sizeof(cuda::barrier<cuda::thread_scope_device>),
            cudaMemcpyHostToDevice, kleosStream));
        static_assert(sizeof(floatPair) == sizeof(flagsType) &&
            alignof(floatPair) == sizeof(flagsType));
        auto* symBook = nvshmem_calloc(sBkz, sizeof(flagsType));
        auto* symHeap = nvshmem_align(16, heapBytes);
        KLEOS_ASSERT(symBook != nullptr, "nvshmem_calloc failed");
        KLEOS_ASSERT(symHeap != nullptr, "nvshmem_align failed");
        // Pointer orchestration
        // Starting index of flags array
        auto* flags = static_cast<flagsType*>(symBook);
        auto* adj = CAST_TO(floatPair, flags + n);
        // Navigate to our slice of the adjacency matrix
        auto* results = adj + globalRank * n;
        const auto remotePresent = [&n, &results] {
            for (int i = 0; i < n; ++i) {
                if (nvshmem_ptr(results, i) == nullptr) return true;
            }
            return false;
        };
        const auto isRemotePresent = remotePresent();
        const auto sharedSize = n * (sizeof(floatPair) + sizeof(unsigned int));
        constexpr auto seqNo = 1U;
        topology::discover<<<KLEOS_STATIC_SBZ, KLEOS_BLOCK_SIZE, sharedSize, kleosStream>>>(n, globalRank,
            isRemotePresent, lWa,
            static_cast<cuda::std::byte*>(symHeap),
            flags, results, dvB, attributes, seqNo);
        KLEOS_CHECK_CUDA(cudaMemcpyAsync(hAp, adj, aD * sizeof(floatPair),
            cudaMemcpyDeviceToHost, kleosStream));
        KLEOS_CHECK_CUDA(cudaMemcpyAsync(wAp, attributes, n * sizeof(WorkerAttribute),
            cudaMemcpyDeviceToHost, kleosStream));
        KLEOS_CHECK_CUDA(cudaFreeAsync(attributes, kleosStream));
        KLEOS_CHECK_CUDA(cudaFreeAsync(dvB, kleosStream));
        KLEOS_CHECK_CUDA(cudaPeekAtLastError());
        KLEOS_CHECK_CUDA(cudaStreamSynchronize(kleosStream));
        delete hB;
        nvshmem_free(symHeap);
        nvshmem_free(symBook);
    }

    __host__ __forceinline__
    bool runDecider(EPG* __restrict__ const& ePg,
        Expert* __restrict__ const& experts,
        Worker* __restrict__ const& wG,
        Worker* __restrict__ const& ePwG,
        uint* __restrict__ const& dTg,
        uint* __restrict__ const& pT,
        uint* __restrict__ const& pTs,
        uint* __restrict__ const& ePs,
        uint* __restrict__ const& ePsX,
        uint16_t* __restrict__ const& scratch,
        const floatPair* __restrict__ const& aP,
        const WorkerAttribute* __restrict__ const& attributes,
        const uint& rank, const uint& world) {
        #if KLEOS_NVTX
        kleosRange decRange{__func__};
        #endif
        constexpr auto E = ACC::E::value;
        constexpr cutlass::NumericConverter<float, cute::half_t> h2f{};
        for (uint16_t i = 0; i < world; ++i) {
            const auto [t, m] = attributes[i];
            wG[i] = Worker{i, h2f(t), m};
        }
        const auto adj = make_tensor(aP, make_layout(cute::make_shape(world, world), cute::LayoutRight{}));
        constexpr Decider<ACC::JT::value> decider{};
        if (!decider(adj, wG, E, E, dTg)) {
            // this means there isn't enough device memory for the input model configuration.
            // we propagate this information above
            return false;
        }
        const auto epWorld = subsets(dTg, pT, world, dTg[rank]);
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
        std::ranges::fill(scratch, scratch + world, 0U);
        // compute expert slots for our group
        for (uint16_t i = 0; i < E; ++i) {
            const auto wIdx = ePs[i];
            const uint16_t tally = scratch[wIdx] + 1U;
            scratch[wIdx] = tally;
            expertSlots = cuda::std::max(tally, expertSlots);
        }
        const auto numLocalExperts = scratch[epRank];
        auto epg = EPG{
            epRank,
            expertSlots,
            numLocalExperts,
            epWorld
        };
        if (epWorld < world) {
            // Get other group ids
            // The global maximum of epW and expertSlots is necessary for allocating a uniformly sized symmetric heap
            // across all PEs.
            std::unordered_set<uint> groups{};
            for (uint i = 0; i < world; ++i) {
                groups.emplace(dTg[i]);
            }
            const auto myGroup = dTg[rank];
            for (const auto& group : groups) {
                if (group != myGroup) {
                    std::ranges::fill(scratch, scratch + world, 0U);
                    const auto ePw = subsets(dTg, pTs, world, group);
                    epg.epWorldM = cute::max(epg.epWorldM, ePw);
                    for (uint i = 0; i < ePw; ++i) {
                        const auto wRank = pTs[i];
                        ePwG[i] = Worker{
                            static_cast<uint16_t>(wRank),
                            wG[wRank].processingRate,
                            wG[wRank].memoryCapacity
                        };
                    }
                    assign(ePwG, ePw, experts, E, ePsX);
                    for (uint16_t i = 0; i < E; ++i) {
                        const auto wIdx = ePsX[i];
                        const uint16_t tally = scratch[wIdx] + 1U;
                        scratch[wIdx] = tally;
                        epg.expertSlots = cuda::std::max(tally, epg.expertSlots);
                    }
                }
            }
        }
        *ePg = epg;
        return true;
    }

    __host__ __forceinline__
    void cleanup() {
        #if KLEOS_NVTX
        kleosRange finalRange{__PRETTY_FUNCTION__};
        #endif
        KLEOS_ASSERT(isInitialized, "Not initialized!");
        isInitialized = false;
        KLEOS_CHECK_CUDA(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
        nvshmem_finalize();
    }

    template<EP e = EP::yes>
    __host__ __forceinline__
    void distributedInit() {
        #if KLEOS_NVTX
        kleosRange distRange{__PRETTY_FUNCTION__};
        #endif
        static_assert(e == EP::yes);
        constexpr auto blocks = ACC::PeakHardware::blocks::value;
        using Element = ACC::Element;
        constexpr uint E = ACC::E::value;
        // Below forces NVSHMEM to statically allocate memory, which is desirable
        uEI("NVSHMEM_DISABLE_CUDA_VMM", 1);
        // initialize communication backend
        {
            #if KLEOS_NVTX
            kleosRange cR{"distributedInit::nvshmem_init()"};
            #endif
            nvshmem_init();
        }
        const uint devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
        KLEOS_CHECK_CUDA(cudaSetDevice(devId));
        const auto globalWorld = nvshmem_n_pes();
        const auto rank = nvshmem_my_pe();

        // Pointer to adjacency matrix and throughput of all devices
        const auto aD = globalWorld * globalWorld;
        const auto dZ = 2 * sizeof(Worker) * globalWorld +
                sizeof(Expert) * E +
                aD * sizeof(floatPair) +
                globalWorld * sizeof(WorkerAttribute) +
                2 * sizeof(uint) * globalWorld;
        const auto aXz = (sizeof(ELI) + sizeof(PEL)) * E +
            (sizeof(PLI) * globalWorld) + sizeof(LXI) * E +
                (sizeof(uint) * (E * ACC::TCM::value * ACC::TNx::value));
        const auto pZ = cuda::std::max(dZ, aXz);
        const auto sZ = sizeof(uint) * (globalWorld + 2 * E) + sizeof(uint16_t) * globalWorld;
        auto* mP = static_cast<cuda::std::byte*>(std::calloc(pZ + sZ, sizeof(cuda::std::byte)));;

        // Pointer salami slicing
        auto* workers = CAST_TO(Worker, mP);
        auto* ePWorkers = workers + globalWorld;
        static_assert(alignof(Worker) % alignof(Expert) == 0);
        auto* experts = CAST_TO(Expert, ePWorkers + globalWorld);
        static_assert(alignof(Expert) % alignof(floatPair) == 0);
        auto* aP = CAST_TO(floatPair, experts + E);
        static_assert(alignof(floatPair) % alignof(WorkerAttribute) == 0);
        auto* wAp = CAST_TO(WorkerAttribute, aP + aD);
        static_assert(alignof(WorkerAttribute) % alignof(uint) == 0);
        auto* dTg = CAST_TO(uint, wAp + globalWorld);
        auto* pTs = CAST_TO(uint, dTg + globalWorld);

        // Result buffers
        auto* pT = CAST_TO(uint, mP + pZ);
        auto* ePs = pT + globalWorld;
        auto* ePsX = ePs + E; // scratch
        auto* scratch = CAST_TO(uint16_t, ePsX + E);

        estimateMemory(&wAp[rank]);
        mT(wAp + rank);

        discoverTopology(aP, globalWorld, rank, wAp[rank], wAp);
        auto ePgD = EPG{};
        // The topology adjacency matrix is ready, let's map devices to optimal cooperative process groups
        const auto isFeasible = runDecider(&ePgD, experts, workers, ePWorkers, dTg, pT, pTs, ePs, ePsX,
            scratch, aP, wAp, rank, globalWorld);
        if (!isFeasible) {
            cleanup();
            KLEOS_ASSERT(isFeasible, "Insufficient Memory for Experts");
        }
        // Now allocate memory
        const auto heapElems = STAGES * CELLS * ePgD.epWorldM * ePgD.expertSlots * ACC::pEC::value *
            ACC::H::value;
        const auto flagElems = (ePgD.epWorldM * ePgD.expertSlots + E * ACC::TCM::value * ACC::TNx::value);
        auto tHB = flagElems * sizeof(flagsType) + heapElems * sizeof(Element);
        // Required for large allocations
        const auto nss = gEI("NVSHMEM_SYMMETRIC_SIZE", ACC::SZD::value); // default is 1GB
        if (tHB >= nss) {
            const auto nGB = cute::ceil_div(tHB, nss) * nss;
            uEI("NVSHMEM_SYMMETRIC_SIZE", nGB);
        }
        // Note every symmetric memory allocation's size has to be identical across all PEs
        auto* flags = static_cast<flagsType*>(nvshmem_calloc(flagElems, sizeof(flagsType)));
        auto* sHeap = static_cast<cuda::std::byte*>(nvshmem_align(16, heapElems * sizeof(Element)));
        KLEOS_ASSERT(flags != nullptr, "nvshmem_calloc failed");
        KLEOS_ASSERT(sHeap != nullptr, "nvshmem_align failed");

        // local bookkeeping memory
        Task* bookTask = nullptr;
        PEL* bookPEL = nullptr;
        PLI* bookPLI = nullptr;
        TPS* bookTPS = nullptr;
        cuda::barrier<cuda::thread_scope_device>* bookDB = nullptr;
        TQSignal* bookTQS = nullptr;
        RingSoftmaxPayload* bookRSP = nullptr;
        RingTopKPayload* bookRTP = nullptr;
        ELI* bookELI = nullptr;
        BookType* book = nullptr;
        cuda::std::byte* bookElement = nullptr;

        KLEOS_CHECK_CUDA(cudaMallocAsync(&bookTask, sizeof(Task) * Bookkeeping::tQlt(ePgD.nLx, ePgD.epWorld), kleosStream));
        KLEOS_CHECK_CUDA(cudaMallocAsync(&bookPEL, sizeof(PEL) * ACC::E::value, kleosStream));
        KLEOS_CHECK_CUDA(cudaMallocAsync(&bookPLI, sizeof(PLI) * ePgD.epWorld, kleosStream));
        KLEOS_CHECK_CUDA(cudaMallocAsync(&bookTPS, sizeof(TPS) * Bookkeeping::tPlt(), kleosStream));
        KLEOS_CHECK_CUDA(cudaMallocAsync(&bookDB, sizeof(cuda::barrier<cuda::thread_scope_device>) *
            Bookkeeping::dBlt(), kleosStream));
        KLEOS_CHECK_CUDA(cudaMallocAsync(&bookTQS, sizeof(TQSignal) * Bookkeeping::pDBlt(),
            kleosStream));
        KLEOS_CHECK_CUDA(cudaMallocAsync(&bookRSP, sizeof(RingSoftmaxPayload) * Bookkeeping::rSlt(),
            kleosStream));
        KLEOS_CHECK_CUDA(cudaMallocAsync(&bookRTP, sizeof(RingTopKPayload) * Bookkeeping::rTlt(),
            kleosStream));
        KLEOS_CHECK_CUDA(cudaMallocAsync(&bookELI, sizeof(ELI) * Bookkeeping::eLlt(), kleosStream));
        KLEOS_CHECK_CUDA(cudaMallocAsync(&book, sizeof(BookType) * Bookkeeping::b4lt(ePgD.nLx, ePgD.epWorld), kleosStream));
        KLEOS_CHECK_CUDA(cudaMallocAsync(&bookElement, sizeof(ACC::Element) * Bookkeeping::xMlt(ePgD.nLx, ePgD.epWorld), kleosStream));
        // Initialize bookkeeping
        KLEOS_CHECK_CUDA(cudaMemsetAsync(book, 0, sizeof(BookType) * Bookkeeping::b4lt(ePgD.nLx, ePgD.epWorld),
            kleosStream));
        KLEOS_CHECK_CUDA(cudaMemsetAsync(bookTQS, 0, sizeof(TQSignal) * Bookkeeping::pDBlt(), kleosStream));
        KLEOS_CHECK_CUDA(cudaMemsetAsync(bookRSP, 0, sizeof(RingSoftmaxPayload) * Bookkeeping::rSlt(), kleosStream));
        KLEOS_CHECK_CUDA(cudaMemsetAsync(bookRTP, 0, sizeof(RingTopKPayload) * Bookkeeping::rTlt(), kleosStream));
        hostBookkeeping = Bookkeeping{
            flags,
            sHeap,
            bookTask,
            bookPEL,
            bookPLI,
            bookTPS,
            bookDB,
            bookTQS,
            bookRSP,
            bookRTP,
            bookELI,
            book,
            bookElement,
            ePgD
        };
        // copy device-wide barrier
        const auto hB = new cuda::barrier<cuda::thread_scope_device>{blocks};
        KLEOS_CHECK_CUDA(cudaMemcpyAsync(hostBookkeeping.dB(), hB,
            sizeof(cuda::barrier<cuda::thread_scope_device>),
            cudaMemcpyHostToDevice, kleosStream));
        KLEOS_CHECK_CUDA(cudaMemcpyToSymbolAsync(bookkeeping, &hostBookkeeping, sizeof(Bookkeeping), 0,
            cudaMemcpyHostToDevice, kleosStream));

        // reuse pre-allocated memory for device data structures
        auto* __restrict__ pEL = CAST_TO(PEL, mP);
        static_assert(alignof(PEL) % alignof(PLI) == 0);
        auto* __restrict__ pLI = CAST_TO(PLI, pEL + E);
        static_assert(alignof(PLI) % alignof(ELI) == 0);
        auto* __restrict__ eLI = CAST_TO(ELI, pLI + ePgD.epWorld);
        static_assert(alignof(ELI) % alignof(LXI) == 0);
        auto* __restrict__ lxI = CAST_TO(LXI, eLI + E);
        static_assert(alignof(LXI) % alignof(uint) == 0);
        auto* __restrict__ tileIndices = CAST_TO(uint, lxI + ePgD.nLx);

        auto pel = PEL{};
        auto eli = ELI{};
        auto pli = PLI{};
        auto tileIndex = 0U;
        auto current = 0U;
        std::ranges::fill(scratch, scratch + ePgD.epWorld, 0U);
        for (uint i = 0; i < E; ++i) {
            const auto ePrank = ePs[i];
            const auto gRank = pT[ePrank];
            auto* rSHeap = CAST_TO(cuda::std::byte, nvshmem_ptr(sHeap, gRank));
            auto* rFlags = CAST_TO(flagsType, nvshmem_ptr(flags, gRank));
            rFlags = rFlags == nullptr ? flags : rFlags;
            const auto xLi = scratch[ePrank]++;
            const auto isRemote = rSHeap == nullptr;
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
            pli.remoteSFlags = rFlags;
            pli.remoteSHeap = rSHeap;

            // LXI
            if (gRank == rank) {
                (lxI + xLi)->expertIndex = i;
            }

            pEL[i] = pel;
            eLI[i] = eli;
            pLI[ePrank] = pli;
            if (isRemote) {
                for (uint j = 0; j < ACC::TCM::value; ++j) {
                    tileIndices[current++] = tileIndex;
                    tileIndex += ACC::TNx::value;
                }
            }
            else {
                for (uint j = 0; j < ACC::TCM::value * ACC::TNx::value; ++j) {
                    tileIndices[current++] = tileIndex++;
                }
            }
        }

        for (uint i = 0; i < E; ++i) {
            pel = pEL[i];
            pel.nLocalExperts = scratch[pel.peer];
            pEL[i] = pel;
        }

        KLEOS_CHECK_CUDA(cudaMemcpyAsync(hostBookkeeping.pEL(), pEL,
            sizeof(PEL) * E, cudaMemcpyHostToDevice, kleosStream));
        KLEOS_CHECK_CUDA(cudaMemcpyAsync(hostBookkeeping.pL(), pLI,
            sizeof(PLI) * ePgD.epWorld,
            cudaMemcpyHostToDevice, kleosStream));
        KLEOS_CHECK_CUDA(cudaMemcpyAsync(hostBookkeeping.eL(), eLI,
            sizeof(ELI) * E,
            cudaMemcpyHostToDevice, kleosStream));
        KLEOS_CHECK_CUDA(cudaMemcpyAsync(hostBookkeeping.lX(), lxI,
            sizeof(LXI) * ePgD.nLx,
            cudaMemcpyHostToDevice, kleosStream));
        KLEOS_CHECK_CUDA(cudaMemcpyAsync(hostBookkeeping.ssFc(), &current,
            sizeof(BookType), cudaMemcpyHostToDevice, kleosStream));
        KLEOS_CHECK_CUDA(cudaMemcpyAsync(hostBookkeeping.tIx(), tileIndices,
            sizeof(uint) * current, cudaMemcpyHostToDevice, kleosStream));
        KLEOS_CHECK_CUDA(cudaPeekAtLastError());
        KLEOS_CHECK_CUDA(cudaStreamSynchronize(kleosStream));
        delete hB;
        std::free(mP);
    }

    template<>
    __host__ __forceinline__
    void distributedInit<EP::no>() {
        BookType* book = nullptr;
        cuda::std::byte* bookElement = nullptr;
        KLEOS_CHECK_CUDA(cudaMallocAsync(&book,
            sizeof(BookType) * Bookkeeping::b4lt(), kleosStream));
        KLEOS_CHECK_CUDA(cudaMallocAsync(&bookElement,
            sizeof(ACC::Element) * Bookkeeping::xMlt(), kleosStream));
        hostBookkeeping = Bookkeeping{book, bookElement};
        KLEOS_CHECK_CUDA(cudaMemcpyToSymbolAsync(bookkeeping, &hostBookkeeping,
            sizeof(Bookkeeping), 0,
            cudaMemcpyHostToDevice, kleosStream));
        KLEOS_CHECK_CUDA(cudaPeekAtLastError());
        KLEOS_CHECK_CUDA(cudaStreamSynchronize(kleosStream));
    }

    // Should be called before loading the model
    __host__ __forceinline__
    void initialize() {
        #if KLEOS_NVTX
        kleosRange initRange{__PRETTY_FUNCTION__};
        #endif
        KLEOS_ASSERT(!isInitialized, "Already Initialized");
        using GPUType = kleos::Hardware<KLEOS_ARCH, 255>;
        constexpr auto blocks = GPUType::OS::processorBlocks::value;
        static_assert(KLEOS_ARCH >= 700, "Volta and above is required!");
        isInitialized = true;
        static_assert(ACC::S::value % BLOCK_M == 0 && ACC::S::value < BLOCK_M * blocks * ACC::TMU::value &&
        ACC::P::value % BLOCK_N == 0 && ACC::H::value % BLOCK_N == 0);
        static_assert(NUM_EXPERTS <= cuda::std::numeric_limits<uint16_t>::max(),
            "For performance, we assume number of experts <= UINT16_MAX");
        distributedInit<(ACC::E::value > 1) ? EP::yes : EP::no>();
    }

    __host__ __forceinline__
    void setDevice() {
        KLEOS_ASSERT(isInitialized, "Not initialized!");
        KLEOS_CHECK_CUDA(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
    }

    __host__ __forceinline__
    auto getRank() {
        KLEOS_ASSERT(isInitialized, "Not initialized!");
        return nvshmem_my_pe();
    }

    __host__ __forceinline__
    void finalize(){
        #if KLEOS_NVTX
        kleosRange finalRange{__PRETTY_FUNCTION__};
        #endif
        KLEOS_ASSERT(isInitialized, "Not initialized!");
        isInitialized = false;
        KLEOS_CHECK_CUDA(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
        KLEOS_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.bookTask, kleosStream));
        KLEOS_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.bookPEL, kleosStream));
        KLEOS_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.bookPLI, kleosStream));
        KLEOS_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.bookTPS, kleosStream));
        KLEOS_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.bookDB, kleosStream));
        KLEOS_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.bookTQS, kleosStream));
        KLEOS_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.bookRSP, kleosStream));
        KLEOS_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.bookRTP, kleosStream));
        KLEOS_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.bookELI, kleosStream));
        KLEOS_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.book, kleosStream));
        KLEOS_CHECK_CUDA(cudaPeekAtLastError());
        KLEOS_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.bookElement, kleosStream));
        // Below ensures all work is done before deallocating via the external API
        KLEOS_CHECK_CUDA(cudaStreamSynchronize(kleosStream));
        nvshmem_free(hostBookkeeping.flags);
        nvshmem_free(hostBookkeeping.sHeap);
        nvshmem_finalize();
        KLEOS_CHECK_CUDA(cudaPeekAtLastError());
        KLEOS_CHECK_CUDA(cudaStreamSynchronize(kleosStream));
    }
}
#endif //BOOTSTRAP_CUH
