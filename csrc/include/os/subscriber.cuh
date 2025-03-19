//
// Created by Jonathan on 7/4/24.
//

#ifndef ARISTOS_QUEUE_CUH
#define ARISTOS_QUEUE_CUH
#include <nvshmem.h>

#include "../types.cuh"
#include "packet.cuh"

namespace aristos::subscriber{
    __device__
    enum class SubscriberStage {
        initial,
        final
    };

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
            typename Element = ACC::Element
        >
        __device__ __forceinline__
        void operator()(
            packet::DecoderArg& dA,
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
            const uint& stageLength,
            const uint &nLx,
            const uint &tIdx,
            const uint16_t& localSeqBit) const {
            /// Flags has dimension [W, L], where W is expert parallel world and L is number of local experts
            constexpr packet::Decoder<PacketStage::initial, PeerConnectivity::p2p, Element> fPd{};
            constexpr packet::Decoder<PacketStage::initial, PeerConnectivity::remote, Element> fRd{};
            constexpr auto bSw = sizeof(uint) * 8U;
            for (uint i = 0; i < stageLength; ++i) {
                const auto vSIdx = i / bSw;
                const auto vIdx = i % bSw;
                // no need to batch reads from shared memory here as stageLength is very small, most likely <= 1
                auto visitedSet = bitSet[tIdx + vSIdx * subscriberCount];
                const auto flagIdx = tIdx + i * subscriberCount;
                auto signal = atomicLoad<cuda::thread_scope_system>(
                    CAST_TO(ull_t, flags + flagIdx));
                const auto sP = CAST_TO(SignalPayload<PacketStage::initial>, &signal);
                if (sP->seqBit > localSeqBit) {
                    // This is an exotic scenario.
                    // Their sequence bit is ahead of ours, meaning that we missed processing some preceding packets
                    // of theirs before they sent this current packet.
                    // In short, they overrode those prior sequence bits before we observed them.
                    // This occurrence is fine and more importantly,
                    // only happens if the preceding, overridden n packets were noops,
                    // where n = seqBit - localSeqBit.
                    // Thus, as we catch up to them, we self-correct our termination bound to avoid a deadlock.
                    const auto peer = flagIdx / nLx;
                    packet::sTB(taskCount, status, peer, nLx);
                }
                const auto received = sP->seqBit == localSeqBit;
                stagePending -= received;
                if (!visitedSet.get(vIdx) && received) {
                    // set visited bit
                    visitedSet.set(vIdx);
                    #if ARISTOS_DEBUG
                    printf("{rt: %u, ttm: %u, sb: %u}, vs: %u\n", sP->routedTokens, sP->totalTilesM, sP->seqBit,
                        visitedSet.get(vIdx));
                    #endif
                    // decode the received packet
                    const auto myLocalExIdx = flagIdx % nLx;
                    const auto peerIdx = flagIdx / nLx;
                    printf("peerIdx: %u, nLx: %u\n", peerIdx, nLx);
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
                    const auto* packet = heap::advance<0, 1>(dA.rSHeap, peerIdx, myLocalExIdx);
                    #if ARISTOS_DEBUG
                    printf("Subscriber %u received a packet from peer %u to expert %u\n",
                        tIdx, peerIdx, myLocalExIdx);
                    #endif
                    if (!pLI.isRemote) {
                        // P2P peer
                        // Use DMA pointers over UVA
                        dA.rSHeap = pLI.remoteSHeap;
                        dA.rFlags = pLI.remoteSFlags;
                        // Enforce consistency
                        __threadfence_system();
                        fPd(dA, packet, status, taskCount, sP->routedTokens, sP->totalTilesM,
                            myLocalExIdx, lXI.expertIndex, pGB, weights, bias, peerIdx, pLI.pe, nLx, ltQHead, tQHead);
                    }
                    else {
                        // Remote peer
                        // Below enforces consistency
                        // We cannot decouple the API, unfortunately,
                        // as the memory ordering mechanism is internal.
                        nvshmem_ushort_test(&sP->seqBit, NVSHMEM_CMP_EQ, localSeqBit);
                        fRd(dA, packet, status, taskCount, sP->routedTokens, sP->totalTilesM,
                            myLocalExIdx, lXI.expertIndex, pGB, weights, bias, peerIdx, pLI.pe, nLx, ltQHead, tQHead);
                    }
                }
                // update state
                bitSet[tIdx + vSIdx * subscriberCount] = visitedSet;
            }
        }
    };

    template<unsigned int subscriberCount>
    struct Subscribe<SubscriberStage::final, subscriberCount> {
        template<
            typename WorkSet,
            typename RBitSet,
            unsigned int EC = ACC::EC::value,
            unsigned int TN = ACC::TNx::value,
            unsigned int CS = ACC::TCM::value * TN
        >
        requires(isRegisterV<WorkSet> && isRegisterV<RBitSet>)
        __device__ __forceinline__
        void operator()(
            WorkSet& workSet,
            RBitSet& rBitSet,
            BitSet* __restrict__ const& bitSet,
            const packet::DecoderArg& dA,
            /// Task Arguments
            const TPS* const& tokenIds,
            cuda::std::byte* const& mO,
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
            const uint16_t& localSeqBit
            ) const {
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
                #pragma unroll
                for (uint j = 0; j < RBitSet::kElements; ++j) {
                    rBitSet[j] = bitSet[tIdx + (j + (i * WorkSet::kElements) / bSw) * subscriberCount];
                }
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
                    const auto vSIdx = j / bSw;
                    const auto vIdx = j % bSw;
                    auto visitedSet = rBitSet[vSIdx];
                    const auto flagIdx = workSet[j];
                    auto signal = atomicLoad<cuda::thread_scope_system>(
                        CAST_TO(ull_t, flags + flagIdx));
                    const auto sP = CAST_TO(SignalPayload<PacketStage::last>, &signal);
                    if (!visitedSet.get(vIdx) && sP->seqBit == localSeqBit) {
                        // let's decode this packet
                        // set visited bit
                        visitedSet.set(vIdx);
                        const auto expertIdx = flagIdx / CS;
                        const ELI lookup = eL[expertIdx];
                        const auto* tI = tokenIds + (expertIdx * EC + sP->batchIdx * BLOCK_M);
                        const auto* packet = heap::advance<1, 1>(dA.rSHeap, lookup.epRank,
                                lookup.localExpertIndex,sP->batchIdx * BLOCK_M);
                        if (lookup.isRemote) {
                            // enforce memory consistency
                            nvshmem_ushort_test(&sP->seqBit, NVSHMEM_CMP_EQ, localSeqBit);
                            lRd(dA, packet, CONST_CAST_TO(cuda::std::byte, tI), mO, sP->tokensM,
                                ltQHead, tQHead, expertIdx);
                        }
                        else {
                            // enforce memory consistency
                            __threadfence_system();
                            lPd(dA.tQ, ltQHead, packet,
                                CONST_CAST_TO(cuda::std::byte, tI),
                                mO, sP->tokensM, flagIdx % TN, tQHead, expertIdx);
                        }
                    }
                    rBitSet[vSIdx] = visitedSet;
                }
                // update checkpoint state
                #pragma unroll
                for (uint j = 0; j < RBitSet::kElements; ++j) {
                    bitSet[tIdx + (j + (i * WorkSet::kElements) / bSw) * subscriberCount] = rBitSet[j];
                }
            }
            if (const auto residue = stageLength - stageTrips * WorkSet::kElements; residue) {
                for (uint j = 0; j < residue; ++j) {
                    scratch[tIdx + j * subscriberCount] = tileIndices[tIdx +
                        (j + stageTrips * WorkSet::kElements) * subscriberCount];
                }
                #pragma unroll
                for (uint j = 0; j < WorkSet::kElements; ++j) {
                    if (j < residue) {
                        workSet[j] = scratch[tIdx + j * subscriberCount];
                    }
                }
                #pragma unroll
                for (uint j = 0; j < RBitSet::kElements; ++j) {
                    if (j < residue) {
                        rBitSet[j] = bitSet[tIdx + (j + (stageTrips * WorkSet::kElements) / bSw) * subscriberCount];
                    }
                }
                #pragma unroll
                for (uint j = 0; j < WorkSet::kElements; ++j) {
                    if (j < residue) {
                        const auto vSIdx = j / bSw;
                        const auto vIdx = j % bSw;
                        auto visitedSet = rBitSet[vSIdx];
                        const auto flagIdx = workSet[j];
                        auto signal = atomicLoad<cuda::thread_scope_system>(
                            CAST_TO(ull_t, flags + flagIdx));
                        const auto sP = CAST_TO(SignalPayload<PacketStage::last>, &signal);
                        if (!visitedSet.get(vIdx) && sP->seqBit == localSeqBit) {
                            // set visited bit
                            #if ARISTOS_DEBUG
                            printf("{Thread %u, tM: %u, bIdx: %u, sb: %u}, vs: %u\n",
                                tIdx, sP->tokensM, sP->batchIdx, sP->seqBit,
                                visitedSet.get(vIdx));
                            #endif
                            visitedSet.set(vIdx);
                            // let's decode this packet
                            const auto expertIdx = flagIdx / CS;
                            const ELI lookup = eL[expertIdx];
                            const auto* tI = tokenIds + (expertIdx * EC + (sP->batchIdx * BLOCK_M));
                            const auto* packet = heap::advance<1, 1>(dA.rSHeap, lookup.epRank,
                                    lookup.localExpertIndex, sP->batchIdx * BLOCK_M);
                            if (lookup.isRemote) {
                                // enforce memory consistency
                                nvshmem_ushort_test(&sP->seqBit, NVSHMEM_CMP_EQ, localSeqBit);
                                lRd(dA, packet, CONST_CAST_TO(cuda::std::byte, tI), mO, sP->tokensM,
                                    ltQHead, tQHead, expertIdx);
                            }
                            else {
                                // enforce memory consistency
                                __threadfence_system();
                                lPd(dA.tQ, ltQHead, packet,
                                    CONST_CAST_TO(cuda::std::byte, tI),
                                    mO, sP->tokensM, flagIdx % TN, tQHead, expertIdx);
                            }
                        }
                        rBitSet[vSIdx] = visitedSet;
                    }
                }

                // update checkpoint state
                #pragma unroll
                for (uint j = 0; j < RBitSet::kElements; ++j) {
                    bitSet[tIdx + (j + (stageTrips * WorkSet::kElements) / bSw) * subscriberCount] = rBitSet[j];
                }
            }
        }
    };
    /// Decode packets deposited
    template<
        unsigned int bSzPs,
        unsigned int wSet = 16U,
        unsigned int subscriberCount = SUBSCRIBERS,
        typename Output,
        typename ExpertsUp,
        typename ExpertsDown,
        typename BiasUp,
        typename BiasDown
    >
    requires(wSet == 16 || wSet % 32 == 0)
    __device__ __forceinline__
    void start(BitSet* __restrict__ const& bitSet,
        cuda::std::byte* __restrict__ const& workspace,
        unsigned int* __restrict__ const& interrupt,
        unsigned int* __restrict__ const& tQHead,
        const PLI* __restrict__ const& pL,
        const LXI* __restrict__ const& lX,
        const ELI* __restrict__ const& eL,
        const unsigned int& ssfC,
        unsigned int* __restrict__ const& status, // shared
        unsigned int* __restrict__ const& taskCount,
        Output const& moeOutput,
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
        cutlass::AlignedArray<BitSet, bSzPs> rBitSet{};

        // lookup tables
        const auto* tokenIds = bookkeeping.tP();
        const auto* __restrict__ tileIndices = bookkeeping.tIx();

        // tQ things
        auto ltQHead = 0U; // local tQ Head

        // pointers
        auto* __restrict__ sharedSpace = CAST_TO(unsigned int, workspace);
        auto* __restrict__ sFlags = bookkeeping.flags;
        auto* __restrict__ pGB = bookkeeping.xM(); // post GEMM buffer

        // Constants
        const auto nLx = bookkeeping.nLx;

        // first stage
        const auto fSfC = bookkeeping.world * nLx; // first stage flag count
        const auto fSl = fSfC / subscriberCount + (tIdx < fSfC % subscriberCount);
        auto fSp = fSl; // first stage pending

        // second stage
        const auto ssL = ssfC / subscriberCount + (tIdx < fSfC % subscriberCount);
        const auto ssT = ssL / wSet;

        constexpr Subscribe<SubscriberStage::initial, subscriberCount> initialSubscriber{};
        constexpr Subscribe<SubscriberStage::final, subscriberCount> finalSubscriber{};

        const auto pSI = nSI<subscriberCount>(ssfC);

        // Register allocation
        const auto gfSfC = bookkeeping.world * bookkeeping.xs;
        auto dA = packet::DecoderArg{
            bookkeeping.sHeap,
            bookkeeping.tQ() + tIdx, // coalesced accessing
            bookkeeping.flags + gfSfC,
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
                    fSl,
                    nLx,
                    tIdx,
                    lSeqBit
                );
            }
            flags += gfSfC;
            finalSubscriber(rWSet,
                rBitSet,
                bitSet,
                dA,
                tokenIds,
                CAST_TO(cuda::std::byte, moeOutput.data().get()),
                tileIndices,
                eL,
                sharedSpace,
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
#endif //ARISTOS_QUEUE_CUH
