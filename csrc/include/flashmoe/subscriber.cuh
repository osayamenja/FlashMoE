/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by Jonathan on 7/4/24.
//

#ifndef FLASHMOE_QUEUE_CUH
#define FLASHMOE_QUEUE_CUH
#include <nvshmem.h>

#include "bitset.cuh"
#include "dq.cuh"
#include "heap.cuh"
#include "signal.cuh"
#include "task.cuh"

namespace flashmoe::subscriber{
    struct Args {
        uint64_t* const signals; // symmetric global
        Task* const tQ; // global
        const int* tileIndices; // global
        cuda::std::byte* const GEMM0Staging;
        cuda::std::byte* const expertsUp; // global
        cuda::std::byte* const expertsDown; // global
        cuda::std::byte* const biasUp; // global
        cuda::std::byte* const biasDown; // global
        BitSet* const bitSet; // shared
        int* const interrupt; // shared
        int* const tQHead; // shared
        const PLI* const pL; // shared
        const LXI* const lX; // shared
        const ELI* const eL; // shared
        int* const status; // shared
        int* const taskCount; // shared
        const int ssfC; // second stage flag count
        const int gfSfC; // global first stage flag count -> global expert slots * epWorld
        const int world; // ep world
        const int nLx; // number of local experts
        const int epRank;
        const int E; // number of experts
        const int tilesN0;
        const int tilesN1;
        const int ecTilesM; // ceil_div(EC, tileM)
        const uint16_t tIdx;
        const uint16_t seqNumber;

        Args(uint64_t* const& _signals, Task* const& tq,
            const int* const& _tileIndices,
            cuda::std::byte* const& gemm0Staging,
            cuda::std::byte* const& _expertsUp,
            cuda::std::byte* const& _expertsDown,
            cuda::std::byte* const& _biasUp,
            cuda::std::byte* const& _biasDown,
            BitSet* const& _bitSet,
            int* const& _interrupt,
            int* const& _tQHead,
            const PLI* const& _pL,
            const LXI* const& _lX,
            const ELI* const& _eL,
            int* const& _status,
            int* const& _taskCount,
            const int& _ssfC,
            const int& _gfSfC, const int& _world,
            const int& nLx,
            const int& _epRank, const int& _experts,
            const uint threadIdx, const int& _tilesN0, const int& _tilesN1, const int& _eCTilesM,
            const uint16_t seqNo):
        signals(_signals),
        tQ(tq + threadIdx),
        tileIndices(_tileIndices),
        GEMM0Staging(gemm0Staging),
        expertsUp(_expertsUp),
        expertsDown(_expertsDown),
        biasUp(_biasUp),
        biasDown(_biasDown),
        bitSet(_bitSet),
        interrupt(_interrupt),
        tQHead(_tQHead),
        pL(_pL),
        lX(_lX),
        eL(_eL),
        status(_status),
        taskCount(_taskCount),
        ssfC(_ssfC),
        gfSfC(_gfSfC),
        world(_world),
        nLx(nLx),
        epRank(_epRank),
        E(_experts),
        tilesN0(_tilesN0), tilesN1(_tilesN1), ecTilesM(_eCTilesM),
        tIdx(threadIdx),
        seqNumber(seqNo) {}
    };
    /// Decodes a single packet from the initial stage
    template<
        int subscriberCount,
        PacketStage s,
        PeerConnectivity p,
        typename Element = void
    >
    struct Decoder {
        static_assert(flashmoe::TensorValueType<Element>);
        static_assert(s == PacketStage::initial);
        __device__ __forceinline__
        void operator()(const DecoderArg& dA,
            cuda::std::byte* const& sHeap,
            flagsType* const& flags,
            const cuda::std::byte* const& packet,
            uint const& routedTokens,
            unsigned int const& localExpertIdx,
            cuda::std::byte* __restrict__ const& pGB, //postGEMM buffer
            const cuda::std::array<const cuda::std::byte*, GEMMs>& weights,
            const cuda::std::array<const cuda::std::byte*, GEMMs>& bias,
            unsigned int const& peer, // relative to the EP group
            unsigned int const& gPeer, // relative to the global group, needed for network operations
            const uint& laneId,
            int& lTQHead,
            int* __restrict__ const& tQHead) const {
            constexpr auto tN = ACC::TN::value;
            const auto qIdx = DQ::sNext<subscriberCount>(lTQHead);
            const auto fTilesM = routedTokens / BLOCK_M;
            // pad here to meet tile requirements
            const auto padM = Bookkeeping::pad<BLOCK_M>(routedTokens);
            // expert, peer offset
            const auto sO = ACC::TCM::value * (peer * dA.nLx + localExpertIdx);
            cuda::std::array<cuda::std::byte*, GEMMs> taskResults{};
            // Staging buffer for results of preGEMM
            taskResults[0] = pGB + (peer * dA.nLx * ACC::pEC::value * ACC::P::value * sizeof(Element));
            // Egress packet buffer
            auto* rcData = heap::advance<1, 1>(sHeap, dA.epRank, localExpertIdx);
            taskResults[1] = p == PeerConnectivity::remote ?
                heap::advance<1, 0>(sHeap, peer, localExpertIdx) : rcData;
            const auto wT = fTilesM * tN;
            const auto fS = wT / WARP_SIZE + (laneId < wT % WARP_SIZE);
            constexpr auto rT = tN % WARP_SIZE;
            const auto lS = tN / WARP_SIZE + (rT > 0 ? laneId < rT : 0);
            const auto tSlice = fS + (routedTokens % BLOCK_M == 0 ? 0 : lS);

            for (uint i = 0; i < fS; ++i) {
                const auto tileIdx = laneId + i * WARP_SIZE;
                const auto rowIdx = tileIdx / tN;
                dA.tQ[DQ::next<subscriberCount>(qIdx, i)] = Task{
                    TaskType::GEMM0,
                    packet,
                    weights,
                    taskResults,
                    bias,
                    rcData,
                    flags,
                    sO + rowIdx,
                    tileIdx,
                    padM,
                    static_cast<uint16_t>(BLOCK_M),
                    gPeer,
                    rowIdx,
                    p == PeerConnectivity::remote
                };
            }

            // residue tile
            if (const auto residue = routedTokens - fTilesM * BLOCK_M; residue) {
                for (uint j = 0; j < lS; j++) {
                    const auto tileIdx = fTilesM * tN + laneId + j * WARP_SIZE;
                    dA.tQ[DQ::next<subscriberCount>(qIdx, fS + j)] = Task{
                        TaskType::GEMM0,
                        packet,
                        weights,
                        taskResults,
                        bias,
                        rcData,
                        flags,
                        sO + fTilesM,
                        tileIdx,
                        padM,
                        static_cast<uint16_t>(residue),
                        gPeer,
                        fTilesM,
                        p == PeerConnectivity::remote
                    };
                }
            }

            if (tSlice) {
                lTQHead += tSlice;
                cuda::atomic_ref<int, cuda::thread_scope_block> tqh{*tQHead};
                cuda::std::ignore = tqh.fetch_add(tSlice, cuda::memory_order_release);
            }
        }
    };


    template<int subscriberCount>
    struct Decoder<subscriberCount, PacketStage::last, PeerConnectivity::p2p> {
        __device__ __forceinline__
        void operator()(Task* __restrict__ const& tQ,
            unsigned int& lTQHead,
            const cuda::std::byte* const& packet,
            const cuda::std::byte* const& tokenIndices,
            const unsigned int& nTokens,
            const unsigned int& tileIdx,
            int* __restrict__ const& tQHead,
            const unsigned int& expertIdx) const {
            // now let's decode this single tile
            tQ[DQ::sNext<subscriberCount>(lTQHead++)] = Task{
                TaskType::combine,
                tokenIndices,
                cuda::std::array<const cuda::std::byte*, GEMMs>{packet},
                nTokens,
                tileIdx,
                expertIdx
            };
            cuda::atomic_ref<int, cuda::thread_scope_block> tqh{*tQHead};
            // notifies scheduler of work
            cuda::std::ignore = tqh.fetch_add(1, cuda::memory_order_release);
        }
    };

    template<int subscriberCount>
    struct Decoder<subscriberCount, PacketStage::last, PeerConnectivity::remote> {
        __device__ __forceinline__
        void operator()(const DecoderArg& dA,
            const cuda::std::byte* const& packet,
            const cuda::std::byte* const& tokenIndices,
            const unsigned int& nTokens,
            unsigned int& lTQHead,
            int* __restrict__ const& tQHead,
            const unsigned int& expertIdx) const {
            const auto qIdx = DQ::sNext<subscriberCount>(lTQHead);
            for (uint i = 0; i < tNx; ++i) {
                dA.tQ[DQ::next<subscriberCount>(qIdx, i)] = Task{
                    TaskType::combine,
                    tokenIndices,
                    cuda::std::array<const cuda::std::byte*, GEMMs>{packet},
                    nTokens,
                    i,
                    expertIdx
                };
            }
            lTQHead += tNx;
            cuda::atomic_ref<int, cuda::thread_scope_block> tqh{*tQHead};
            cuda::std::ignore = tqh.fetch_add(tNx, cuda::memory_order_release);
        }
    };
    __device__
    enum class SubscriberStage {
        initial,
        final
    };

    // Self-correct Termination Bound
    __device__ __forceinline__
    void sTB(int* __restrict__ const& taskCount,
        int* __restrict__ const& status,
        const int& peer, const int& nLx,
        const int& tilesN1, const int& tilesN2, const int& tcm,
        const int& peerTaskTiles = 0U) {
        if (!atomicTAS<cuda::thread_scope_block>(status + peer)) {
            const auto superfluous = (tilesN1 + tilesN2) * ((nLx * tcm) - peerTaskTiles);
            atomicSub_block(taskCount, superfluous);
        }
    }

    template<
        SubscriberStage s,
        typename Element,
        int subscriberCount,
        int sNW = subscriberCount / WARP_SIZE
    >
    struct Subscriber {
        static_assert(s == SubscriberStage::initial);
        __device__ __forceinline__
        void operator()(const Heap& symHeap,
            const Args& args,
            uint64_t* __restrict__ const& flags,
            BitSet* __restrict__ const& bitSet,
            const int& stageLength,
            int& stagePending, int& ltQHead) const {
            /// Flags has dimension [W, L], where W is expert parallel world and L is number of local experts
            constexpr Decoder<subscriberCount, PacketStage::initial, PeerConnectivity::p2p, Element> fPd{};
            constexpr Decoder<subscriberCount, PacketStage::initial, PeerConnectivity::remote, Element> fRd{};
            constexpr int bSw = sizeof(uint) * 8U;
            const auto laneId = args.tIdx % WARP_SIZE;
            const auto warpId = args.tIdx / WARP_SIZE;
            #pragma unroll 2
            for (int i = 0; i < stageLength; ++i) {
                const auto vSIdx = i / bSw;
                const auto vIdx = i % bSw;
                const auto flagIdx = warpId + i * sNW;
                const auto peerIdx = flagIdx / args.nLx;
                const auto pLI = args.pL[peerIdx];
                uint64_t signal = SignalConstants::ground;
                if (!laneId) {
                    auto visitedSet = bitSet[warpId + vSIdx * sNW];
                    if (!visitedSet.get(vIdx)) {
                        if (pLI.isRemote) {
                            // RDMA peer
                            signal = nvshmem_signal_fetch(flags + flagIdx);
                            const auto sigPayload = cuda::std::bit_cast<SignalPayload<PacketStage::initial>>(signal);
                            if (sigPayload.seqNumber == args.seqNumber) {
                                // set visited bit
                                // self-correct the termination bound
                                sTB(args.taskCount, args.status, peerIdx, args.nLx,
                                    args.tilesN0, args.tilesN1,
                                    args.ecTilesM, sigPayload.totalTilesM);
                                visitedSet.set(vIdx);

                                // enforce memory consistency of expected packet
                                const bool isPacketHere = nvshmem_uint64_test(flags + flagIdx,
                                    NVSHMEM_CMP_EQ, signal);
                                if (!isPacketHere) {
                                    // this scenario means that this peer sent another packet in between us
                                    // observing the signal and testing the signal's presence.
                                    // This is fine, specifically within our protocol,
                                    // this occurs if the producer previously sent a noop,
                                    // advanced to the next epoch and sent another packet.
                                    // we simply do nothing in this scenario as
                                    // we will process the new packet in the next epoch.
                                    if (__builtin_expect(sigPayload.routedTokens > 0, 0)) {
                                        __trap(); // protocol violation
                                    }
                                }
                            }
                            else if (sbs::ahead(sigPayload.seqNumber, args.seqNumber)) {
                                /*
                                Their sequence number is ahead of ours,
                                meaning that we missed processing a preceding packet
                                of theirs before they sent this current packet.
                                In short, they overrode those prior sequence bits before we observed them.
                                This occurrence is fine and more importantly,
                                only happens if the preceding, overridden packet was a noop.
                                Thus, as we catch up to them, we self-correct
                                our termination bound to avoid a deadlock.
                                */
                                sTB(args.taskCount, args.status, peerIdx, args.nLx,
                                    args.tilesN0, args.tilesN1,
                                    args.ecTilesM);
                                // set visited bit
                                visitedSet.set(vIdx);
                            }
                        }
                        else {
                            // NVLink peer
                            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> f{*(flags + flagIdx)};
                            signal = f.load(cuda::memory_order::acquire);
                            const auto sigPayload = cuda::std::bit_cast<SignalPayload<PacketStage::initial>>(signal);
                            if (sigPayload.seqNumber == args.seqNumber) {
                                sTB(args.taskCount, args.status, peerIdx, args.nLx,
                                    args.tilesN0, args.tilesN1, args.ecTilesM, sigPayload.totalTilesM);
                                visitedSet.set(vIdx);
                            }
                            else if (sbs::ahead(sigPayload.seqNumber, args.seqNumber)) {
                                sTB(args.taskCount, args.status, peerIdx, args.nLx,
                                    args.tilesN0, args.tilesN1,
                                    args.ecTilesM);
                                // set visited bit
                                visitedSet.set(vIdx);
                            }
                        }
                        // update state
                        bitSet[warpId + vSIdx * sNW] = visitedSet;
                    }
                }
                __syncwarp();
                // broadcast received signal from leader to others
                signal = __shfl_sync(0xffffffff, signal, 0);
                const auto sigPayload = cuda::std::bit_cast<SignalPayload<PacketStage::initial>>(signal);
                if (sigPayload.seqNumber == args.seqNumber && sigPayload.routedTokens > 0) {
                    stagePending -= 1;
                    const auto myLocalExIdx = flagIdx % args.nLx;
                    const auto lXI = args.lX[myLocalExIdx];
                    const auto* packet = symHeap.advance<0, 1>(peerIdx, myLocalExIdx);
                    if (!pLI.isRemote) {
                        auto* nFlags = pLI.remoteSFlags + args.gfSfC + lXI.expertIndex * (args.ecTilesM * args.tilesN1);
                        fPd(dA, pLI.remoteSHeap, nFlags, packet, sigPayload.routedTokens,
                                myLocalExIdx, pGB, weights, bias, peerIdx, pLI.pe,
                                laneId, ltQHead, tQHead);
                    }
                    else {
                        auto* nFlags = dA.sFlags + args.gfSfC + lXI.expertIndex * (args.ecTilesM * args.tilesN1);
                        fRd(dA, dA.sHeap, nFlags, packet, sigPayload.routedTokens,
                                myLocalExIdx, pGB, weights, bias, peerIdx, pLI.pe, laneId, ltQHead, tQHead);
                    }
                }
            }
        }
    };

    template<int subscriberCount, typename Element>
    struct Subscriber<SubscriberStage::final, Element, subscriberCount> {
        template<
            typename WorkSet,
            typename TokenIds,
            //unsigned int TN = ACC::TNx::value,
            //unsigned int CS = ACC::TCM::value * TN
        >
        requires(isRegisterV<WorkSet>)
        __device__ __forceinline__
        void operator()(
            WorkSet& workSet,
            BitSet* __restrict__ const& bitSet,
            const DecoderArg& dA,
            /// Task Arguments
            TokenIds const& tokenIds,
            /// Data Structures
            const uint* __restrict__ const& tileIndices,
            /// Lookup Table
            const ELI* __restrict__ const& eL,
            /// State
            uint* __restrict__ const& scratch,
            flagsType* __restrict__ const& flags,
            BookType* __restrict__ tQHead,
            uint& ltQHead,
            /// Constants
            const uint& stageLength,
            const uint& stageTrips,
            const uint& tIdx,
            const uint16_t& localSeqBit,
            uint16_t* __restrict__ const& sSeqBit) const {
            constexpr auto bSw = sizeof(uint) * 8U;
            static_assert(WorkSet::kElements == 16 || WorkSet::kElements % bSw == 0);
            constexpr Decoder<PacketStage::last, PeerConnectivity::p2p> lPd{};
            constexpr Decoder<PacketStage::last, PeerConnectivity::remote> lRd{};
            // prefetch
            if (stageTrips) {
                // global -> shared
                #pragma unroll
                for (uint j = 0; j < WorkSet::kElements; ++j) {
                    scratch[tIdx + j * subscriberCount] = tileIndices[tIdx + j * subscriberCount];
                }
            }
            for (uint i = 0; i < stageTrips; ++i) {
                const uint sBIdx = tIdx + (i * WorkSet::kElements / bSw) * subscriberCount;
                auto sBS = bitSet[sBIdx];
                // shared -> registers
                #pragma unroll
                for (uint j = 0; j < WorkSet::kElements; ++j) {
                    workSet[j] = scratch[tIdx + j * subscriberCount];
                    if (i + 1 < stageTrips) {
                        // Eagerly initiate global memory loads
                        scratch[tIdx + j * subscriberCount] =
                            tileIndices[tIdx + (j + (i + 1) * WorkSet::kElements) * subscriberCount];
                    }
                }
                #pragma unroll
                for (uint j = 0; j < WorkSet::kElements; ++j) {
                    const uint bIdx = (i * WorkSet::kElements + j) % bSw;
                    const auto flagIdx = workSet[j];
                    if (const auto isVisited = sBS.get(bIdx); !isVisited) {
                        const auto signal = atomicExch_system(CAST_TO(ull_t, flags + flagIdx),
                            SignalConstants::ground);
                        const auto* __restrict__ sP = CONST_CAST_TO(SignalPayload<PacketStage::last>, &signal);
                        if (sP->seqNumber == localSeqBit) {
                            // let's decode this packet
                            // set visited bit
                            sBS.set(bIdx);
                            const auto expertIdx = flagIdx / CS;
                            const auto lookup = eL[expertIdx];
                            const auto tokenIdx = sP->batchIdx * BLOCK_M;
                            const auto* tI = &tokenIds(expertIdx, tokenIdx);
                            const auto* packet = heap::advance<1, 1>(dA.sHeap, lookup.epRank,
                                    lookup.localExpertIndex,tokenIdx);
                            if (lookup.isRemote) {
                                // enforce memory consistency
                                eMC(sSeqBit, localSeqBit);
                                lRd(dA, packet, CONST_CAST_TO(cuda::std::byte, tI), sP->tokensM,
                                    ltQHead, tQHead, expertIdx);
                            }
                            else {
                                // enforce memory consistency
                                __threadfence_system();
                                lPd(dA.tQ, ltQHead, packet, CONST_CAST_TO(cuda::std::byte, tI),
                                    sP->tokensM, flagIdx % TN, tQHead, expertIdx);
                            }
                        }
                    }
                }
                // update checkpoint state
                bitSet[sBIdx] = sBS;
            }
            if (const auto residue = stageLength - stageTrips * WorkSet::kElements; residue) {
                for (uint j = 0; j < residue; ++j) {
                    scratch[tIdx + j * subscriberCount] = tileIndices[tIdx +
                        (j + stageTrips * WorkSet::kElements) * subscriberCount];
                }
                const uint sBIdx = tIdx + (stageTrips * WorkSet::kElements / bSw) * subscriberCount;
                auto sBS = bitSet[sBIdx];
                #pragma unroll
                for (uint j = 0; j < WorkSet::kElements; ++j) {
                    if (j < residue) {
                        workSet[j] = scratch[tIdx + j * subscriberCount];
                    }
                }
                #pragma unroll
                for (uint j = 0; j < WorkSet::kElements; ++j) {
                    if (j < residue) {
                        const uint bIdx = (stageTrips * WorkSet::kElements + j) % bSw;
                        const auto flagIdx = workSet[j];
                        if (const auto isVisited = sBS.get(bIdx); !isVisited) {
                            const auto signal = atomicExch_system(CAST_TO(ull_t, flags + flagIdx),
                                SignalConstants::ground);
                            const auto* __restrict__ sP = CONST_CAST_TO(SignalPayload<PacketStage::last>, &signal);
                            if (sP->seqNumber == localSeqBit) {
                                // set visited bit
                                sBS.set(bIdx);
                                // let's decode this packet
                                const auto expertIdx = flagIdx / CS;
                                const auto lookup = eL[expertIdx];
                                const auto tokenIdx = sP->batchIdx * BLOCK_M;
                                const auto* tI = &tokenIds(expertIdx, tokenIdx);
                                const auto* packet = heap::advance<1, 1>(dA.sHeap, lookup.epRank,
                                        lookup.localExpertIndex, tokenIdx);
                                if (lookup.isRemote) {
                                    // enforce memory consistency
                                    eMC(sSeqBit, localSeqBit);
                                    lRd(dA, packet, CONST_CAST_TO(cuda::std::byte, tI), sP->tokensM,
                                        ltQHead, tQHead, expertIdx);
                                }
                                else {
                                    // enforce memory consistency
                                    __threadfence_system();
                                    lPd(dA.tQ, ltQHead, packet,
                                        CONST_CAST_TO(cuda::std::byte, tI),
                                        sP->tokensM, flagIdx % TN, tQHead, expertIdx);
                                }
                            }
                        }
                    }
                }
                // update checkpoint state
                bitSet[sBIdx] = sBS;
            }
        }
    };

    /// Decode packets deposited
    template<
        unsigned int wSet,
        unsigned int subscriberCount,
        typename Element
    >
    requires(subscriberCount % WARP_SIZE == 0 && wSet <= sizeof(uint) * 8U)
    __device__ __forceinline__
    void start(const Heap& symHeap, const Args& args){
        cutlass::AlignedArray<uint, wSet> rWSet{};

        // lookup tables
        const auto tokenIds = make_tensor(cute::make_gmem_ptr(bookkeeping.tP()),
            cute::make_layout(cute::make_shape(args.E, symHeap.EC), cute::LayoutRight{}));

        auto ltQHead = 0U; // local tQ Head

        // first stage
        constexpr auto sNW = subscriberCount / WARP_SIZE;
        const auto fSfC = args.world * args.nLx; // first stage flag count
        const auto fSl = fSfC / sNW + (args.tIdx / WARP_SIZE < fSfC % sNW);
        auto fSp = fSl; // first stage pending

        // second stage
        const auto ssL = args.ssfC / subscriberCount + (args.tIdx < args.ssfC % subscriberCount);
        const auto ssT = ssL / wSet;

        constexpr Subscriber<SubscriberStage::initial, Element, subscriberCount> initialSubscriber{};
        constexpr Subscriber<SubscriberStage::final, Element, subscriberCount> finalSubscriber{};

        const auto pSI = nSI<subscriberCount>(args.ssfC);

        cuda::atomic_ref<int, cuda::thread_scope_block> interrupt{*args.interrupt};
        while (!interrupt.load(cuda::memory_order_relaxed)) {
            auto* __restrict__ flags = args.signals;
            // sweep through flags by stages
            // start with the first stage
            if (fSp) {
                initialSubscriber(symHeap, args, flags,
                    args.bitSet + pSI, fSl, fSp, ltQHead);
            }
            flags += gfSfC;
            finalSubscriber(rWSet,
                bitSet,
                dA,
                tokenIds,
                tileIndices,
                eL,
                workspace,
                flags,
                tQHead,
                ltQHead,
                ssL,
                ssT,
                tIdx,
                lSeqBit);
        }
    }
}
#endif //FLASHMOE_QUEUE_CUH
