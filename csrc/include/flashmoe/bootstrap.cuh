//
// Created by osayamen on 1/17/26.
//

#ifndef FLASHMOE_BOOTSTRAP_CUH
#define FLASHMOE_BOOTSTRAP_CUH
#include <algorithm>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <cstdio>
#include <nvshmem.h>

#include <cuda/cmath>

#include "infra/constants.cuh"
#include "infra/telemetry.cuh"
#include "infra/bitset.cuh"
#include "context.cuh"
#include "infra/atomics.cuh"
#include "infra/signal.cuh"
#include "infra/heap.cuh"
#if !defined(CHECK_CUDA)
#  define CHECK_CUDA(e)                                      \
do {                                                         \
    cudaError_t code = (e);                                  \
    if (code != cudaSuccess) {                               \
        fprintf(stderr, "<%s:%d> %s:\n    %s: %s\n",         \
            __FILE__, __LINE__, #e,                          \
            cudaGetErrorName(code),                          \
            cudaGetErrorString(code));                       \
        fflush(stderr);                                      \
        exit(1);                                             \
    }                                                        \
} while (0)
#endif
namespace flashmoe
{
    struct MoEArgs {
        const size_t elementBytes;
        const uint sequenceLength;
        const uint EC;
        const uint tokenDim;
        const uint ffnIntermediateSize;
        const uint bM;
        const uint bN0;
        const uint bN1;
        const uint bK0;
        const uint bK1;
        const uint threads;
        const uint blocks; //CTAs
        const uint16_t epRank;
        const uint16_t epWorld;
        const uint16_t myPE; // NVSHMEM PE
        const uint16_t numExperts;
        const uint16_t numLocalExperts;
        const Topology topo;

        MoEArgs(const size_t& eb, const uint& S, const uint& H, const uint& I, const uint& _EC,
            const uint& bm, const uint& bn0, const uint& bn1, const uint& bk0, const uint bk1,
            const uint& _threads, const uint& ctas, const uint16_t& ep_rank, const uint16_t& ep_world,
            const uint16_t& mype, const uint16_t& experts,
            const uint16_t& nlx, const Topology& topo_):
        elementBytes(eb), sequenceLength(S),
        EC(_EC),
        tokenDim(H), ffnIntermediateSize(I), bM(bm), bN0(bn0), bN1(bn1), bK0(bk0), bK1(bk1),
        threads(_threads), blocks(ctas), epRank(ep_rank), epWorld(ep_world), myPE(mype), numExperts(experts),
        numLocalExperts(nlx), topo(topo_) {}
    };

    __global__ void bI(cuda::barrier<cuda::thread_scope_device>* db, const uint blocks) {
        init(db, blocks);
    }

    __host__ __forceinline__
    void expertParallelBookkeeping(const uint* __restrict__ const& expertToEpRank,
        const int* __restrict__ const& epRankToGlobalRank, const uint& epWorld,
        const int& myPE, const uint& E, const uint& nLx,
        cuda::std::byte* const& sHeap, uint64_t* const& signals,
        PEL* const& pel, PLI* const& pli, ELI* const& eli, LXI* const& lxi,
        cudaStream_t stream) {
#if defined(FLASHMOE_NVTX) && FLASHMOE_NVTX
        const flashmoeRange range{"FlashMoE::expertParallelBookkeeping"};
#endif
        if (nvshmemx_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED) {
            throw std::runtime_error("nvshmem is not initialized");
        }
        std::vector<uint> lxIndices(epWorld);
        std::vector<PEL> pelHost(E);
        std::vector<PLI> pliHost(epWorld);
        std::vector<ELI> eliHost(E);
        std::vector<LXI> lxiHost(nLx);

        std::ranges::fill(lxIndices.begin(), lxIndices.end(), 0);
        for (uint i = 0; i < E; ++i) {
            const auto epRank = expertToEpRank[i];
            const auto pe = epRankToGlobalRank[epRank];
            auto* rSheap = static_cast<cuda::std::byte*>(nvshmem_ptr(sHeap, pe));
            auto* rFlags = static_cast<uint64_t*>(nvshmem_ptr(signals, pe));
            const uint lxIdx = lxIndices[epRank]++;
            const auto isRemote = rSheap == nullptr;

            // PEL
            pelHost[i].isRemote = isRemote;
            pelHost[i].expertLocalIdx = lxIdx;
            pelHost[i].pe = pe;
            pelHost[i].remoteSFlags = rFlags;
            pelHost[i].remoteSHeap = rSheap;
            pelHost[i].peer = epRank;

            // ELI
            eliHost[i].epRank = epRank;
            eliHost[i].isRemote = isRemote;
            eliHost[i].localExpertIndex = lxIdx;

            // PLI
            pliHost[epRank].isRemote = isRemote;
            pliHost[epRank].pe = pe;
            pliHost[epRank].remoteSFlags = rFlags;
            pliHost[epRank].remoteSHeap = rSheap;

            //LXI
            if (pe == myPE) {
                lxiHost[lxIdx].expertIndex = i;
            }
        }

        const auto nlxUniform = lxIndices[0];
        for (uint i = 0; i < E; ++i) {
            auto pt = pelHost[i];
            pt.nLocalExperts = lxIndices[pt.peer];
            if (pt.nLocalExperts != nlxUniform) {
                // may relax this later
                throw std::runtime_error("Number of local experts should be equal across the ep group");
            }
            pelHost[i] = pt;
        }

        cudaMemcpyAsync(pel, pelHost.data(), sizeof(PEL) * pelHost.size(),cudaMemcpyHostToDevice,stream);
        cudaMemcpyAsync(pli, pliHost.data(), sizeof(PLI) * pliHost.size(),cudaMemcpyHostToDevice,stream);
        cudaMemcpyAsync(eli, eliHost.data(), sizeof(ELI) * eliHost.size(), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(lxi, lxiHost.data(), sizeof(LXI) * lxiHost.size(), cudaMemcpyHostToDevice, stream);
    }

    __host__ __forceinline__
    Topology detectTopo() {
        if (nvshmemx_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED) {
            throw std::runtime_error("nvshmem is not initialized");
        }
        return nvshmem_team_n_pes(NVSHMEM_TEAM_SHARED_INDEX) == nvshmem_n_pes() ? Topology::NVLINK_ONLY : Topology::MIXED;
    }
    __host__ __forceinline__
    Context initialize(const MoEArgs& args, const int& arch, const uint* __restrict__ const& expertToEpRank,
        const int* __restrict__ const& epRankToGlobalRank, cudaStream_t stream){
#if defined(FLASHMOE_NVTX) && FLASHMOE_NVTX
        const flashmoeRange range{"FlashMoE::initialize"};
#endif
        // fused gate + moe layer
        if (args.tokenDim % args.bK0 != 0 || args.tokenDim % args.bN1 != 0) {
            throw std::runtime_error("token dimension should be multiples of tile dimensions");
        }
        if (args.ffnIntermediateSize % args.bN0 != 0 || args.ffnIntermediateSize % args.bK1 != 0) {
            throw std::runtime_error("Intermediate size should be multiples of tile dimensions");
        }
        if (args.blocks < 2) {
            throw std::runtime_error("blocks must be at least 2");
        }
        const auto processors = args.blocks - 1;
        if (processors > scheduler::MAX_PROCESSORS) {
            const auto errmsg = std::string("processor count: ").append(std::to_string(processors))
            .append(" is too high");
            throw std::runtime_error(errmsg);
        }
        // maximum tiles that a peer will send to another peer in aggregate.
        const auto maxPeerTaskTiles = cute::ceil_div(args.EC, args.bM) * args.numExperts;
        if (maxPeerTaskTiles > cuda::std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("Max peer task tiles exceeds supported limit. Inform the maintainer.");
        }
        const auto roundEC = cute::ceil_div(args.EC, args.bM) * args.bM;
        const auto ecTilesM = cute::ceil_div(roundEC, args.bM);
        const auto tilesN0 = cute::ceil_div(args.ffnIntermediateSize, args.bN0);
        const auto tilesN1 = cute::ceil_div(args.tokenDim, args.bN1);

        if (nvshmemx_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED) {
            throw std::runtime_error("nvshmem is not initialized");
        }
        const bool elementBytesConditions = cutlass::ispow2(args.elementBytes) &&
            (args.elementBytes == 2 || args.elementBytes == 4 || args.elementBytes == 8);
        if (!elementBytesConditions) {
            throw std::runtime_error("elementBytes not supported");
        }
        // signals ~= tiles(S) * tiles(H)
        // below ensures the following calloc initializes our signals to the expected value 0
        static_assert(SignalConstants::ground == 0);
        const size_t signalLength = (args.epWorld * args.numLocalExperts) + (args.numExperts * ecTilesM * tilesN1);
        auto* signals = static_cast<uint64_t*>(nvshmem_calloc(signalLength, sizeof(uint64_t)));
        if (signals == nullptr) {
            throw std::runtime_error("failed to allocate signals via NVSHMEM");
        }
        // symmetric heap ~= 4*S*H
        const auto heapLength = args.elementBytes * HEAP_STAGES * HEAP_CELLS * args.epWorld * args.numLocalExperts * roundEC * args.tokenDim;
        auto* sHeap = static_cast<cuda::std::byte*>(nvshmem_malloc(heapLength));
        if (sHeap == nullptr) {
            throw std::runtime_error("failed to allocate heap via NVSHMEM");
        }
        const auto supports32 = arch >= 1000;
        checkAlignment(sHeap, supports32);

        Task* tQ = nullptr;
        const bool threadConditions = args.threads >= WARP_SIZE * 2 && args.threads % WARP_SIZE == 0;
        if (!threadConditions) {
            throw std::runtime_error("threads not supported");
        }
        const auto subscriberCount = args.threads - WARP_SIZE;
        const size_t tQLength = subscriberTQLength<WARP_SIZE>(args.epWorld, args.numLocalExperts, ecTilesM, args.numExperts, tilesN0,
            tilesN1, args.threads - WARP_SIZE);
        const size_t secondaryTQL = secondaryTQLength(args.epWorld, args.numLocalExperts, ecTilesM, tilesN1);
        CHECK_CUDA(cudaMallocAsync(&tQ, sizeof(Task) * (tQLength + secondaryTQL), stream));
        Task* pTq = tQ + tQLength;
        if (tQLength + secondaryTQL > cuda::std::numeric_limits<uint>::max()) {
            throw std::runtime_error("Task Queue length > UINT32_MAX. Not an error: need to migrate to uint64");
        }
        const size_t gRQIdxMax = (args.numLocalExperts * args.epWorld * ecTilesM * (tilesN0 + tilesN1)) +
            (cute::ceil_div(roundEC, args.bM) * tilesN1) + (cute::ceil_div(processors, scheduler::SCHEDULER_COUNT));
        if (gRQIdxMax >= cuda::std::numeric_limits<uint>::max()) {
            // catches overflow in scheduler. See circularIdx function
            throw std::runtime_error("gRQIdxMax >= UINT32_MAX. Not an error: need to migrate to uint64");
        }
        checkAlignment(tQ);
        checkAlignment(pTq);

        cuda::std::byte* GEMM0Staging = nullptr;
        const size_t stagingLength = static_cast<size_t>(args.epWorld * args.numLocalExperts * roundEC) * args.ffnIntermediateSize;
        CHECK_CUDA(cudaMallocAsync(&GEMM0Staging, stagingLength * args.elementBytes, stream));

        BitSet* consumerBitMap = nullptr;
        const auto cbmLength = nSI(args.numExperts * ecTilesM * tilesN1, subscriberCount);
        CHECK_CUDA(cudaMallocAsync(&consumerBitMap, sizeof(BitSet) * cbmLength, stream));
        CHECK_CUDA(cudaMemsetAsync(consumerBitMap, 0, sizeof(BitSet) * cbmLength, stream));

        uint8_t* producerBitMap = nullptr;
        const auto pbmLength = args.epWorld * args.numLocalExperts * ecTilesM * tilesN1;
        CHECK_CUDA(cudaMallocAsync(&producerBitMap, sizeof(uint8_t) * pbmLength, stream));
        CHECK_CUDA(cudaMemsetAsync(producerBitMap, 0, sizeof(uint8_t) * pbmLength, stream));

        PEL* pel = nullptr;
        CHECK_CUDA(cudaMallocAsync(&pel, sizeof(PEL) * args.numExperts, stream));

        PLI* pli = nullptr;
        CHECK_CUDA(cudaMallocAsync(&pli, sizeof(PLI) * args.epWorld, stream));

        ELI* eli = nullptr;
        CHECK_CUDA(cudaMallocAsync(&eli, sizeof(ELI) * args.numExperts, stream));

        LXI* lxi = nullptr;
        CHECK_CUDA(cudaMallocAsync(&lxi, sizeof(LXI) * args.numLocalExperts, stream));

        TPS* tps = nullptr;
        CHECK_CUDA(cudaMallocAsync(&tps, sizeof(TPS) * args.numExperts * roundEC, stream));

        TQSignal* tqs = nullptr;
        CHECK_CUDA(cudaMallocAsync(&tqs, sizeof(TQSignal) * processors, stream));
        CHECK_CUDA(cudaMemsetAsync(tqs, 0, sizeof(TQSignal) * processors, stream));

        uint* dispatchSync = nullptr;
        CHECK_CUDA(cudaMallocAsync(&dispatchSync, sizeof(uint) * args.numExperts, stream));
        CHECK_CUDA(cudaMemsetAsync(dispatchSync, 0, sizeof(uint) * args.numExperts, stream));

        uint* gtqHeads = nullptr;
        // ~= tiles(S)
        const size_t gtqHeadsLength = args.epWorld * args.numLocalExperts * ecTilesM;
        CHECK_CUDA(cudaMallocAsync(&gtqHeads, sizeof(uint) * gtqHeadsLength, stream));
        CHECK_CUDA(cudaMemsetAsync(gtqHeads, 0, sizeof(uint) * gtqHeadsLength, stream));

        uint* tileSync = nullptr;
        CHECK_CUDA(cudaMallocAsync(&tileSync, sizeof(uint) * gtqHeadsLength, stream));
        CHECK_CUDA(cudaMemsetAsync(tileSync, 0, sizeof(uint) * gtqHeadsLength, stream));

        uint* statusQ = nullptr;
        CHECK_CUDA(cudaMallocAsync(&statusQ, sizeof(uint) * processors, stream));
        static_assert(ReadySignal::observed == 0);
        CHECK_CUDA(cudaMemsetAsync(statusQ, 0, sizeof(uint) * processors, stream));

        CHECK_CUDA(cudaPeekAtLastError());
        expertParallelBookkeeping(expertToEpRank, epRankToGlobalRank, args.epWorld, args.myPE,
            args.numExperts, args.numLocalExperts, sHeap, signals, pel, pli, eli, lxi, stream);

        return Context{
            .symHeap = sHeap,
            .signals = signals,
            .tQ = tQ,
            .pTq = pTq,
            .GEMM0Staging =  GEMM0Staging,
            .consumerCombineBitMap = consumerBitMap,
            .producerCombineBitMap = producerBitMap,
            .pel = pel,
            .pli = pli,
            .eli = eli, .lxi = lxi, .tqs = tqs,
            .dispatchSync = dispatchSync,
            .gTqHeads = gtqHeads,
            .tileSync = tileSync,
            .statusQueue = statusQ,
            .tokenIndices = tps,
            .processors_v = cuda::fast_mod_div(processors),
            .blocks = args.blocks,
            .S = args.sequenceLength,
            .H = args.tokenDim,
            .I = args.ffnIntermediateSize,
            .EC = args.EC,
            .bM = static_cast<uint16_t>(args.bM),
            .bN0 = static_cast<uint16_t>(args.bN0),
            .bN1 = static_cast<uint16_t>(args.bN1),
            .nLx = args.numLocalExperts,
            .E = args.numExperts,
            .world = args.epWorld,
            .epRank = args.epRank,
            .myPE = args.myPE,
            .initialized = true,
            .topo = args.topo,
            .stateNumber = SignalConstants::sequenceStart
        };
    }

    // profiling purposes
    __host__ __forceinline__ size_t getWorkspaceBytes(const MoEArgs& args) {
        const auto roundEC = cute::ceil_div(args.EC, args.bM) * args.bM;
        const auto ecTilesM = cute::ceil_div(roundEC, args.bM);
        const auto tilesN0 = cute::ceil_div(args.ffnIntermediateSize, args.bN0);
        const auto tilesN1 = cute::ceil_div(args.tokenDim, args.bN1);
        const auto subscriberCount = args.threads - WARP_SIZE;
        const auto processors = args.blocks - 1;

        size_t bytes = 0;
        bytes += sizeof(uint64_t) * (args.epWorld * args.numLocalExperts) + (args.numExperts * ecTilesM * tilesN1);
        bytes += args.elementBytes * HEAP_STAGES * HEAP_CELLS * args.epWorld * args.numLocalExperts * roundEC * args.tokenDim;
        const size_t tQLength = subscriberTQLength<WARP_SIZE>(args.epWorld, args.numLocalExperts, ecTilesM, args.numExperts, tilesN0,
            tilesN1, args.threads - WARP_SIZE);
        const size_t secondaryTQL = secondaryTQLength(args.epWorld, args.numLocalExperts, ecTilesM, tilesN1);
        bytes += sizeof(Task) * (tQLength + secondaryTQL);
        bytes += args.elementBytes * static_cast<size_t>(args.epWorld * args.numLocalExperts * roundEC) * args.ffnIntermediateSize;
        bytes += sizeof(BitSet) * nSI(args.numExperts * ecTilesM * tilesN1, subscriberCount);
        bytes += sizeof(uint8_t) * args.epWorld * args.numLocalExperts * ecTilesM * tilesN1;
        bytes += sizeof(PEL) * args.numExperts;
        bytes += sizeof(PLI) * args.epWorld;
        bytes += sizeof(ELI) * args.numExperts;
        bytes += sizeof(LXI) * args.numLocalExperts;
        bytes += sizeof(TPS) * args.numExperts * roundEC;
        bytes += sizeof(TQSignal) * processors;
        bytes += sizeof(uint) * args.numExperts;
        bytes += sizeof(uint) * args.epWorld * args.numLocalExperts * ecTilesM;
        bytes += sizeof(uint) * args.epWorld * args.numLocalExperts * ecTilesM;
        bytes += sizeof(uint) * processors;
        return bytes;
    }

    __host__ __forceinline__
    GateContext initializeGate(const uint& bNGate, const uint& numExperts, const uint& S, cudaStream_t stream) {
#if defined(FLASHMOE_NVTX) && FLASHMOE_NVTX
        const flashmoeRange range{"FlashMoE::initializeGate"};
#endif
        int* ecGuards = nullptr;
        CHECK_CUDA(cudaMallocAsync(&ecGuards, sizeof(int) * numExperts, stream));
        CHECK_CUDA(cudaMemsetAsync(ecGuards, flashmoe::STALE_AS_BYTE, sizeof(int) * numExperts, stream));
        SoftmaxStatePacked* ssp = nullptr;
        RingTopKPayload* rtp = nullptr;
        if (numExperts > bNGate) {
            const auto tE = cute::ceil_div(numExperts, bNGate);
            CHECK_CUDA(cudaMallocAsync(&ssp, sizeof(SoftmaxStatePacked) * S * tE, stream));
            CHECK_CUDA(cudaMemsetAsync(ssp, 0, sizeof(SoftmaxStatePacked) * S * tE, stream));

            CHECK_CUDA(cudaMallocAsync(&rtp, 2 * sizeof(RingTopKPayload) * S * tE, stream));
            CHECK_CUDA(cudaMemsetAsync(rtp, 0, 2 * sizeof(RingTopKPayload) * S * tE, stream));
        }
        return GateContext{ecGuards, ssp, rtp};
    }

    __host__ __forceinline__
    void finalizeGate(const GateContext& gCtx, cudaStream_t stream) {
#if defined(FLASHMOE_NVTX) && FLASHMOE_NVTX
        const flashmoeRange range{"FlashMoE::finalizeGate"};
#endif
        cudaFreeAsync(gCtx.ecGuards, stream);
        if (gCtx.ssp != nullptr) {
            cudaFreeAsync(gCtx.ssp, stream);
        }
        if (gCtx.rtp != nullptr) {
            cudaFreeAsync(gCtx.rtp, stream);
        }
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    __host__ __forceinline__
    void finalize(const Context& ctx, cudaStream_t stream) {
#if defined(FLASHMOE_NVTX) && FLASHMOE_NVTX
        const flashmoeRange range{"FlashMoE::finalize"};
#endif
        if (ctx.initialized) {
            // free workspace memory
            cudaFreeAsync(ctx.tQ, stream);
            cudaFreeAsync(ctx.GEMM0Staging, stream);
            cudaFreeAsync(ctx.pel, stream);
            cudaFreeAsync(ctx.pli, stream);
            cudaFreeAsync(ctx.eli, stream);
            cudaFreeAsync(ctx.lxi, stream);
            cudaFreeAsync(ctx.consumerCombineBitMap, stream);
            cudaFreeAsync(ctx.producerCombineBitMap, stream);
            cudaFreeAsync(ctx.tokenIndices, stream);
            cudaFreeAsync(ctx.tqs, stream);
            cudaFreeAsync(ctx.dispatchSync, stream);
            cudaFreeAsync(ctx.gTqHeads, stream);
            cudaFreeAsync(ctx.tileSync, stream);
            cudaFreeAsync(ctx.statusQueue, stream);
            CHECK_CUDA(cudaStreamSynchronize(stream));
            nvshmem_free(ctx.symHeap);
            nvshmem_free(ctx.signals);
        }
        CHECK_CUDA(cudaPeekAtLastError());
    }
}
#endif //FLASHMOE_BOOTSTRAP_CUH