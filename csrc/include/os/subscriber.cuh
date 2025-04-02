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
            typename Element = ACC::Element
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
            for (uint i = 0; i < stageLength; ++i) {
                const auto vSIdx = i / bSw;
                const auto vIdx = i % bSw;
                // no need to batch reads from shared memory here as stageLength is very small, most likely <= 1
                auto visitedSet = bitSet[tIdx + vSIdx * subscriberCount];
                if (!visitedSet.get(vIdx)) {
                    const auto flagIdx = tIdx + i * subscriberCount;
                    const auto signal = atomicExch_system(CAST_TO(ull_t, flags + flagIdx),
                        SignalConstants::ground);
                    const auto* __restrict__ sP = CONST_CAST_TO(SignalPayload<PacketStage::initial>, &signal);
                    if (sP->seqBit == localSeqBit) {
                        stagePending -= 1;
                        // set visited bit
                        visitedSet.set(vIdx);
                        // decode the received packet
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
                            // P2P peer
                            // Use DMA pointers over UVA space
                            // Enforce consistency
                            auto* nFlags = pLI.remoteSFlags + gfSfC +
                                lXI.expertIndex * (ACC::TCM::value * ACC::TNx::value);
                            __threadfence_system();
                            fPd(dA, pLI.remoteSHeap, nFlags, packet, status, taskCount, sP->routedTokens,
                                sP->totalTilesM, myLocalExIdx, pGB, weights, bias, peerIdx, pLI.pe,
                                nLx, ltQHead, tQHead);
                        }
                        else {
                            // Remote peer
                            auto* nFlags = dA.sFlags + gfSfC +
                                lXI.expertIndex * (ACC::TCM::value * ACC::TNx::value);
                            eMC(sSeqBit, localSeqBit);
                            fRd(dA, dA.sHeap, nFlags, packet, status, taskCount, sP->routedTokens, sP->totalTilesM,
                                myLocalExIdx, pGB, weights, bias, peerIdx, pLI.pe, nLx, ltQHead, tQHead);
                        }
                    }
                    else if (sbs::ahead(sP->seqBit, localSeqBit)) {
                        /*
                        This is an exotic scenario.
                        Their sequence bit is ahead of ours, meaning that we missed processing some preceding packets
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
                        packet::sTB(taskCount, status, peer, nLx);
                        // set visited bit
                        visitedSet.set(vIdx);
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
            typename TokenIds,
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
            TokenIds const& tokenIds,
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
                    if (!visitedSet.get(vIdx)) {
                        const auto signal = atomicExch_system(CAST_TO(ull_t, flags + flagIdx),
                            SignalConstants::ground);
                        const auto* __restrict__ sP = CONST_CAST_TO(SignalPayload<PacketStage::last>, &signal);
                        if (sP->seqBit == localSeqBit) {
                            // let's decode this packet
                            // set visited bit
                            visitedSet.set(vIdx);
                            const auto expertIdx = flagIdx / CS;
                            const ELI lookup = eL[expertIdx];
                            const auto tokenIdx = sP->batchIdx * BLOCK_M;
                            const auto* tI = &tokenIds(expertIdx, tokenIdx);
                            const auto* packet = heap::advance<1, 1>(dA.sHeap, lookup.epRank,
                                    lookup.localExpertIndex,tokenIdx);
                            if (lookup.isRemote) {
                                // enforce memory consistency
                                eMC(sSeqBit, localSeqBit);
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
                const auto bR = cute::ceil_div(residue, bSw);
                #pragma unroll
                for (uint j = 0; j < RBitSet::kElements; ++j) {
                    if (j < bR) {
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
                        if (!visitedSet.get(vIdx)) {
                            const auto signal = atomicExch_system(CAST_TO(ull_t, flags + flagIdx),
                                SignalConstants::ground);
                            const auto* __restrict__ sP = CONST_CAST_TO(SignalPayload<PacketStage::last>, &signal);
                            if (sP->seqBit == localSeqBit) {
                                // set visited bit
                                visitedSet.set(vIdx);
                                // let's decode this packet
                                const auto expertIdx = flagIdx / CS;
                                const ELI lookup = eL[expertIdx];
                                const auto tokenIdx = sP->batchIdx * BLOCK_M;
                                const auto* tI = &tokenIds(expertIdx, tokenIdx);
                                const auto* packet = heap::advance<1, 1>(dA.sHeap, lookup.epRank,
                                        lookup.localExpertIndex, tokenIdx);
                                if (lookup.isRemote) {
                                    // enforce memory consistency
                                    eMC(sSeqBit, localSeqBit);
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
        const auto tokenIds = make_tensor(cute::make_gmem_ptr(bookkeeping.tP()),
            cute::Layout<cute::Shape<cute::Int<ACC::E::value>, cute::Int<ACC::pEC::value>>,
                cute::Stride<cute::Int<ACC::pEC::value>, cute::_1>>{});
        const auto* __restrict__ tileIndices = bookkeeping.tIx();

        // tQ things
        auto ltQHead = 0U; // local tQ Head

        // pointers
        auto* __restrict__ sFlags = bookkeeping.flags;
        auto* __restrict__ pGB = bookkeeping.xM(); // post GEMM buffer

        // Constants
        const auto nLx = bookkeeping.nLx;

        // first stage
        const auto fSfC = bookkeeping.world * nLx; // first stage flag count
        const auto fSl = fSfC / subscriberCount + (tIdx < fSfC % subscriberCount);
        auto fSp = fSl; // first stage pending

        // second stage
        const auto ssL = ssfC / subscriberCount + (tIdx < ssfC % subscriberCount);
        const auto ssT = ssL / wSet;

        constexpr Subscribe<SubscriberStage::initial, subscriberCount> initialSubscriber{};
        constexpr Subscribe<SubscriberStage::final, subscriberCount> finalSubscriber{};

        const auto pSI = nSI<subscriberCount>(ssfC);

        // Register allocation
        const auto gfSfC = bookkeeping.world * bookkeeping.xs;
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
                rBitSet,
                bitSet,
                dA,
                tokenIds,
                CAST_TO(cuda::std::byte, moeOutput.data().get()),
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
#endif //ARISTOS_QUEUE_CUH
