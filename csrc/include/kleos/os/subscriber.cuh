/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by Jonathan on 7/4/24.
//

#ifndef KLEOS_QUEUE_CUH
#define KLEOS_QUEUE_CUH
#include <nvshmem.h>

#include "../types.cuh"
#include "packet.cuh"

namespace kleos::subscriber{
    __device__
    enum class SubscriberStage {
        initial,
        final
    };

    // Self-correct Termination Bound
    template<
        unsigned int TN = ACC::TN::value,
        unsigned int TNx = ACC::TNx::value,
        unsigned int TCM = ACC::TCM::value
    >
    __device__ __forceinline__
    void sTB(unsigned int* __restrict__ const& taskCount,
        unsigned int* __restrict__ const& status,
        const unsigned int& peer, const unsigned int& nLx,
        const unsigned int& peerTaskTiles = 0U) {
        if (!atomicTAS<cuda::thread_scope_block>(status + peer)) {
            const auto superfluous = (TN + TNx) * (nLx * TCM - peerTaskTiles);
            atomicSub_block(taskCount, superfluous);
        }
    }

    // Below enforces consistency
    // We cannot decouple the API, unfortunately,
    // as the memory ordering mechanism is internal.
    __device__ __forceinline__
    void eMC(uint16_t* __restrict__ const& sSeqBit, const uint16_t& localSeqBit) {
        nvshmem_ushort_test(sSeqBit, NVSHMEM_CMP_EQ, localSeqBit);
    }

    template<
        SubscriberStage s,
        unsigned int subscriberCount
    >
    struct Subscribe {
        static_assert(s == SubscriberStage::initial);
        template<
            typename ExpertsUp,
            typename ExpertsDown,
            typename BiasUp,
            typename BiasDown,
            typename Element = ACC::Element,
            unsigned int sNW = subscriberCount / WARP_SIZE
        >
        __device__ __forceinline__
        void operator()(
            const packet::DecoderArg& dA,
            ExpertsUp const& expertsUp,
            ExpertsDown const& expertsDown,
            BiasUp const& biasUp,
            BiasDown const& biasDown,
            cuda::std::byte* __restrict__ const& pGB, /*post GEMM buffer*/
            /// Lookup Table
            const PLI* __restrict__ const& pL,
            const LXI* __restrict__ const& lX,
            /// State
            BitSet* __restrict__ const& bitSet,
            uint* __restrict__ const& status,
            uint* __restrict__ const& taskCount,
            flagsType* __restrict__ const& flags,
            BookType* __restrict__ tQHead,
            uint& stagePending,
            uint& ltQHead,
            /// Constants
            const unsigned int& gfSfC,
            const uint& stageLength,
            const uint &nLx,
            const uint &tIdx,
            const uint16_t& localSeqBit,
            uint16_t* __restrict__ const& sSeqBit) const {
            /// Flags has dimension [W, L], where W is expert parallel world and L is number of local experts
            constexpr packet::Decoder<PacketStage::initial, PeerConnectivity::p2p, Element> fPd{};
            constexpr packet::Decoder<PacketStage::initial, PeerConnectivity::remote, Element> fRd{};
            constexpr auto bSw = sizeof(uint) * 8U;
            const auto laneId = tIdx % WARP_SIZE;
            const auto warpId = tIdx / WARP_SIZE;
            #pragma unroll 2
            for (uint i = 0; i < stageLength; ++i) {
                const auto vSIdx = i / bSw;
                const auto vIdx = i % bSw;
                const auto flagIdx = warpId + i * sNW;
                ull_t signal = SignalConstants::ground;
                if (!laneId) {
                    auto visitedSet = bitSet[warpId + vSIdx * sNW];
                    if (!visitedSet.get(vIdx)) {
                        signal = atomicExch_system(CAST_TO(ull_t, flags + flagIdx),
                            SignalConstants::ground);
                        const auto* __restrict__ sP = CONST_CAST_TO(SignalPayload<PacketStage::initial>, &signal);
                        if (sP->seqBit == localSeqBit) {
                            // set visited bit
                            visitedSet.set(vIdx);
                        }
                        else if (sbs::ahead(sP->seqBit, localSeqBit)) {
                            /*
                            This is an exotic scenario.
                            Their sequence bit is ahead of ours,
                            meaning that we missed processing some preceding packets
                            of theirs before they sent this current packet.
                            In short, they overrode those prior sequence bits before we observed them.
                            This occurrence is fine and more importantly,
                            only happens if the preceding,
                            overridden packets were noops or the sender timed out.
                            Thus, as we catch up to them, we self-correct our termination bound to avoid a deadlock.
                            Also, we have to restore the signal for self-correction in subsequent rounds,
                            until we are fully caught up.
                            Potentially, we may have received a signal in the meantime, so we only swap if the current
                            value is the ground state, which we previously stored.
                            */
                            atomicCAS_system(CAST_TO(ull_t, flags + flagIdx), SignalConstants::ground, signal);
                            const auto peer = flagIdx / nLx;
                            sTB(taskCount, status, peer, nLx);
                            // set visited bit
                            visitedSet.set(vIdx);
                        }
                        // update state
                        bitSet[warpId + vSIdx * sNW] = visitedSet;
                    }
                }
                // broadcast received signal from leader to others
                signal = __shfl_sync(0xffffffff, signal, 0);
                const auto* __restrict__ sP = CONST_CAST_TO(SignalPayload<PacketStage::initial>, &signal);
                if (sP->seqBit == localSeqBit) {
                    stagePending -= 1;
                    const auto myLocalExIdx = flagIdx % nLx;
                    const auto peerIdx = flagIdx / nLx;
                    const auto pLI = pL[peerIdx];
                    const auto lXI = lX[myLocalExIdx];
                    cuda::std::array weights{
                        CONST_CAST_TO(cuda::std::byte, &expertsUp(myLocalExIdx)),
                        CONST_CAST_TO(cuda::std::byte, &expertsDown(myLocalExIdx))
                    };
                    cuda::std::array bias{
                        CONST_CAST_TO(cuda::std::byte, &biasUp(myLocalExIdx)),
                        CONST_CAST_TO(cuda::std::byte, &biasDown(myLocalExIdx))
                    };
                    const auto* packet = heap::advance<0, 1>(dA.sHeap, peerIdx, myLocalExIdx);
                    if (!pLI.isRemote) {
                        if (!laneId) {
                            // self-correct the termination bound
                            sTB(taskCount, status, peerIdx, nLx, sP->totalTilesM);
                            __threadfence_system();
                        }
                        __syncwarp();
                        auto* nFlags = pLI.remoteSFlags + gfSfC +
                            lXI.expertIndex * (ACC::TCM::value * ACC::TNx::value);
                        fPd(dA, pLI.remoteSHeap, nFlags, packet, sP->routedTokens,
                                myLocalExIdx, pGB, weights, bias, peerIdx, pLI.pe,
                                laneId, ltQHead, tQHead);
                    }
                    else {
                        if (!laneId) {
                            sTB(taskCount, status, peerIdx, nLx, sP->totalTilesM);
                            eMC(sSeqBit, localSeqBit);
                        }
                        __syncwarp();
                        auto* nFlags = dA.sFlags + gfSfC +
                                lXI.expertIndex * (ACC::TCM::value * ACC::TNx::value);
                        fRd(dA, dA.sHeap, nFlags, packet, sP->routedTokens,
                                myLocalExIdx, pGB, weights, bias, peerIdx, pLI.pe, laneId, ltQHead, tQHead);
                    }
                }
            }
        }
    };

    template<unsigned int subscriberCount>
    struct Subscribe<SubscriberStage::final, subscriberCount> {
        template<
            typename WorkSet,
            typename TokenIds,
            unsigned int TN = ACC::TNx::value,
            unsigned int CS = ACC::TCM::value * TN
        >
        requires(isRegisterV<WorkSet>)
        __device__ __forceinline__
        void operator()(
            WorkSet& workSet,
            BitSet* __restrict__ const& bitSet,
            const packet::DecoderArg& dA,
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
            constexpr packet::Decoder<PacketStage::last, PeerConnectivity::p2p> lPd{};
            constexpr packet::Decoder<PacketStage::last, PeerConnectivity::remote> lRd{};
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
                        if (sP->seqBit == localSeqBit) {
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
                            if (sP->seqBit == localSeqBit) {
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
        unsigned int subscriberCount = SUBSCRIBERS,
        typename ExpertsUp,
        typename ExpertsDown,
        typename BiasUp,
        typename BiasDown
    >
    requires(subscriberCount % WARP_SIZE == 0 && wSet <= sizeof(uint) * 8U)
    __device__ __forceinline__
    void start(BitSet* __restrict__ const& bitSet,
        uint* __restrict__ const& workspace,
        uint16_t* __restrict__ const& sSeqBit,
        unsigned int* __restrict__ const& interrupt,
        unsigned int* __restrict__ const& tQHead,
        const PLI* __restrict__ const& pL,
        const LXI* __restrict__ const& lX,
        const ELI* __restrict__ const& eL,
        const unsigned int& ssfC,
        unsigned int* __restrict__ const& status, // shared
        unsigned int* __restrict__ const& taskCount,
        ExpertsUp const& expertsUp,
        ExpertsDown const& expertsDown,
        BiasUp const& biasUp,
        BiasDown const& biasDown,
        const uint16_t& lSeqBit,
        const uint& tIdx){
        // offset due to warp specialization for the scheduler
        static_assert(sizeof(unsigned long long int) == sizeof(flagsType));
        static_assert(sizeof(SignalPayload<>) == sizeof(uint64_t));
        static_assert(sizeof(SignalPayload<PacketStage::last>) == sizeof(uint64_t));

        cutlass::AlignedArray<uint, wSet> rWSet{};

        // lookup tables
        const auto tokenIds = make_tensor(cute::make_gmem_ptr(bookkeeping.tP()),
            cute::Layout<cute::Shape<cute::Int<ACC::E::value>, cute::Int<ACC::pEC::value>>,
                cute::Stride<cute::Int<ACC::pEC::value>, cute::_1>>{});
        const auto* __restrict__ tileIndices = bookkeeping.tIx();

        // tQ things
        auto ltQHead = 0U; // local tQ Head

        // pointers
        auto* __restrict__ sFlags = bookkeeping.flags;
        auto* __restrict__ pGB = CAST_TO(cuda::std::byte, bookkeeping.xM()); // post GEMM buffer

        // Constants
        const auto nLx = bookkeeping.nLx;

        // first stage
        constexpr auto sNW = subscriberCount / WARP_SIZE;
        const auto fSfC = bookkeeping.world * nLx; // first stage flag count
        const auto fSl = fSfC / sNW + (tIdx / WARP_SIZE < fSfC % sNW);
        auto fSp = fSl; // first stage pending

        // second stage
        const auto ssL = ssfC / subscriberCount + (tIdx < ssfC % subscriberCount);
        const auto ssT = ssL / wSet;

        constexpr Subscribe<SubscriberStage::initial, subscriberCount> initialSubscriber{};
        constexpr Subscribe<SubscriberStage::final, subscriberCount> finalSubscriber{};

        const auto pSI = nSI<subscriberCount>(ssfC);

        // Register allocation
        const auto gfSfC = bookkeeping.gfSfC;
        const auto dA = packet::DecoderArg{
            bookkeeping.sHeap,
            bookkeeping.tQ() + tIdx, // coalesced accessing
            bookkeeping.flags,
        };

        while (!atomicLoad<cuda::thread_scope_block>(interrupt)) {
            auto* __restrict__ flags = sFlags;
            // sweep through flags by stages
            // start with the first stage
            if (fSp) {
                initialSubscriber(
                    dA,
                    expertsUp,
                    expertsDown,
                    biasUp,
                    biasDown,
                    pGB,
                    pL,
                    lX,
                    bitSet + pSI,
                    status,
                    taskCount,
                    flags,
                    tQHead,
                    fSp,
                    ltQHead,
                    gfSfC,
                    fSl,
                    nLx,
                    tIdx,
                    lSeqBit, sSeqBit
                );
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
                lSeqBit, sSeqBit);
        }
    }
}
#endif //KLEOS_QUEUE_CUH
