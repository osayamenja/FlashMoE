/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
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
#include "experimental/topo.cuh"
#include "../../tests/debug.cuh"
#include "telemetry.cuh"
#include "types.cuh"
#include "moe/expert.cuh"
#include "experimental/decider/decider.cuh"
#include "experimental/decider/comps/expert.cuh"
#include "experimental/decider/comps/niche.cuh"
#include "experimental/decider/comps/worker.cuh"

#define SUPPORTED = 1;
namespace flashmoe{
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
    void cleanup() {
        #if FLASHMOE_NVTX
        flashmoeRange finalRange{__PRETTY_FUNCTION__};
        #endif
        FLASHMOE_ASSERT(isInitialized, "Not initialized!");
        isInitialized = false;
        FLASHMOE_CHECK_CUDA(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
        nvshmem_finalize();
    }

    struct MoEConfig {
        int S;
        int E;
        int H;
        int typeBytes;
    };
    __host__ __forceinline__
    void distributedInit() {
        #if FLASHMOE_NVTX
        flashmoeRange distRange{__PRETTY_FUNCTION__};
        #endif
        const auto nss = gEI("NVSHMEM_SYMMETRIC_SIZE", ACC::SZD::value); // default is 1GB
        if (tHB >= nss) {
            const auto nGB = cute::ceil_div(tHB, nss) * nss;
            uEI("NVSHMEM_SYMMETRIC_SIZE", nGB);
        }
        // initialize communication backend
        {
            #if FLASHMOE_NVTX
            flashmoeRange cR{"distributedInit::nvshmem_init()"};
            #endif
            nvshmem_init();
        }
        const uint devId = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
        FLASHMOE_CHECK_CUDA(cudaSetDevice(devId));
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

        auto ePgD = EPG{};
        // Now allocate memory
        const auto heapElems = STAGES * CELLS * ePgD.epWorldM * ePgD.expertSlots * ACC::pEC::value *
            ACC::H::value;
        const auto flagElems = (ePgD.epWorldM * ePgD.expertSlots + E * ACC::TCM::value * ACC::TNx::value);
        auto tHB = flagElems * sizeof(flagsType) + heapElems * sizeof(Element);
        // Required for large allocations
        // Note every symmetric memory allocation's size has to be identical across all PEs
        auto* flags = static_cast<flagsType*>(nvshmem_calloc(flagElems, sizeof(flagsType)));
        auto* sHeap = static_cast<cuda::std::byte*>(nvshmem_align(16, heapElems * sizeof(Element)));
        FLASHMOE_ASSERT(flags != nullptr, "nvshmem_calloc failed");
        FLASHMOE_ASSERT(sHeap != nullptr, "nvshmem_align failed");

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

        FLASHMOE_CHECK_CUDA(cudaMallocAsync(&bookTask, sizeof(Task) * Bookkeeping::tQlt(ePgD.nLx, ePgD.epWorld), flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaMallocAsync(&bookPEL, sizeof(PEL) * ACC::E::value, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaMallocAsync(&bookPLI, sizeof(PLI) * ePgD.epWorld, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaMallocAsync(&bookTPS, sizeof(TPS) * Bookkeeping::tPlt(), flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaMallocAsync(&bookDB, sizeof(cuda::barrier<cuda::thread_scope_device>) *
            Bookkeeping::dBlt(), flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaMallocAsync(&bookTQS, sizeof(TQSignal) * Bookkeeping::pDBlt(),
            flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaMallocAsync(&bookRSP, sizeof(RingSoftmaxPayload) * Bookkeeping::rSlt(),
            flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaMallocAsync(&bookRTP, sizeof(RingTopKPayload) * Bookkeeping::rTlt(),
            flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaMallocAsync(&bookELI, sizeof(ELI) * Bookkeeping::eLlt(), flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaMallocAsync(&book, sizeof(BookType) * Bookkeeping::b4lt(ePgD.nLx, ePgD.epWorld), flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaMallocAsync(&bookElement, sizeof(ACC::Element) * Bookkeeping::xMlt(ePgD.nLx, ePgD.epWorld), flashmoeStream));
        // Initialize bookkeeping
        FLASHMOE_CHECK_CUDA(cudaMemsetAsync(book, 0, sizeof(BookType) * Bookkeeping::b4lt(ePgD.nLx, ePgD.epWorld),
            flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaMemsetAsync(bookTQS, 0, sizeof(TQSignal) * Bookkeeping::pDBlt(), flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaMemsetAsync(bookRSP, 0, sizeof(RingSoftmaxPayload) * Bookkeeping::rSlt(), flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaMemsetAsync(bookRTP, 0, sizeof(RingTopKPayload) * Bookkeeping::rTlt(), flashmoeStream));
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
        FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(hostBookkeeping.dB(), hB,
            sizeof(cuda::barrier<cuda::thread_scope_device>),
            cudaMemcpyHostToDevice, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaMemcpyToSymbolAsync(bookkeeping, &hostBookkeeping, sizeof(Bookkeeping), 0,
            cudaMemcpyHostToDevice, flashmoeStream));

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

        FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(hostBookkeeping.pEL(), pEL,
            sizeof(PEL) * E, cudaMemcpyHostToDevice, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(hostBookkeeping.pL(), pLI,
            sizeof(PLI) * ePgD.epWorld,
            cudaMemcpyHostToDevice, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(hostBookkeeping.eL(), eLI,
            sizeof(ELI) * E,
            cudaMemcpyHostToDevice, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(hostBookkeeping.lX(), lxI,
            sizeof(LXI) * ePgD.nLx,
            cudaMemcpyHostToDevice, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(hostBookkeeping.ssFc(), &current,
            sizeof(BookType), cudaMemcpyHostToDevice, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaMemcpyAsync(hostBookkeeping.tIx(), tileIndices,
            sizeof(uint) * current, cudaMemcpyHostToDevice, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
        FLASHMOE_CHECK_CUDA(cudaStreamSynchronize(flashmoeStream));
        delete hB;
        std::free(mP);
    }

    // Should be called before loading the model
    __host__ __forceinline__
    void initialize() {
        #if FLASHMOE_NVTX
        flashmoeRange initRange{__PRETTY_FUNCTION__};
        #endif
        FLASHMOE_ASSERT(!isInitialized, "Already Initialized");
        FLASHMOE_CHECK_CUDA(cudaStreamCreate(&flashmoeStream));
        isInitialized = true;
        distributedInit();
    }

    __host__ __forceinline__
    void setDevice() {
        FLASHMOE_ASSERT(isInitialized, "Not initialized!");
        FLASHMOE_CHECK_CUDA(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
    }

    __host__ __forceinline__
    auto getRank() {
        FLASHMOE_ASSERT(isInitialized, "Not initialized!");
        return nvshmem_my_pe();
    }

    __host__ __forceinline__
    void finalize(){
        #if FLASHMOE_NVTX
        flashmoeRange finalRange{__PRETTY_FUNCTION__};
        #endif
        FLASHMOE_ASSERT(isInitialized, "Not initialized!");
        isInitialized = false;
        FLASHMOE_CHECK_CUDA(cudaSetDevice(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
        FLASHMOE_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.bookTask, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.bookPEL, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.bookPLI, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.bookTPS, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.bookDB, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.bookTQS, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.bookRSP, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.bookRTP, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.bookELI, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.book, flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
        FLASHMOE_CHECK_CUDA(cudaFreeAsync(hostBookkeeping.bookElement, flashmoeStream));
        // Below ensures all work is done before deallocating via the external API
        FLASHMOE_CHECK_CUDA(cudaStreamSynchronize(flashmoeStream));
        nvshmem_free(hostBookkeeping.flags);
        nvshmem_free(hostBookkeeping.sHeap);
        nvshmem_finalize();
        FLASHMOE_CHECK_CUDA(cudaPeekAtLastError());
        FLASHMOE_CHECK_CUDA(cudaStreamSynchronize(flashmoeStream));
        FLASHMOE_CHECK_CUDA(cudaStreamDestroy(flashmoeStream));
    }
}
#endif //BOOTSTRAP_CUH
