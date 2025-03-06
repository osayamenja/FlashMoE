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
            const packet::DecoderArg& dA,
            ExpertsUp const& expertsUp,
            ExpertsDown const& expertsDown,
            BiasUp const& biasUp,
            BiasDown const& biasDown,
            cuda::std::byte* __restrict__ pGB, /*post GEMM buffer*/
            /// Lookup Table
            const PLI* __restrict__ const& pL,
            /// State
            uint* __restrict__ const& bitSet,
            uint* __restrict__ const& status,
            uint *__restrict__ const&taskCount,
            flagsType* __restrict__ const& flags,
            BookType* __restrict__ tQHead,
            uint& stagePending,
            uint& lTQHead,
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
                auto visitedSet = bitSet[vSIdx];
                const auto flagIdx = tIdx + i * subscriberCount;
                auto signal = atomicLoad<cuda::thread_scope_system>(
                    CAST_TO(ull_t, flags + flagIdx));
                const auto sP = CAST_TO(SignalPayload<>, &signal);
                const auto received = sP->seqBit == localSeqBit;
                stagePending -= received;
                if (const uint hasVisited = visitedSet >> vIdx & 1; !hasVisited && received) {
                    // set visited bit
                    visitedSet |= 1 << vIdx;
                    // decode the received packet
                    const auto expertIdx = flagIdx % nLx;
                    const auto peerIdx = flagIdx / nLx;
                    const auto pLI = pL[peerIdx];
                    cuda::std::array weights{
                        CONST_CAST_TO(cuda::std::byte, &expertsUp(expertIdx)),
                        CONST_CAST_TO(cuda::std::byte, &expertsDown(expertIdx))
                    };
                    cuda::std::array bias{
                        CONST_CAST_TO(cuda::std::byte, &biasUp(expertIdx)),
                        CONST_CAST_TO(cuda::std::byte, &biasDown(expertIdx))
                    };
                    const auto* packet = heap::advance<0, 1>(dA.sHeap, peerIdx, expertIdx);
                    if (!pLI.isRemote) {
                        // P2P peer
                        // Enforce consistency
                        __threadfence_system();
                        fPd(dA, packet, status, taskCount, sP->routedTokens, sP->totalTilesM,
                            expertIdx, pGB, weights, bias, peerIdx, pLI.pe, lTQHead, tQHead);
                    }
                    else {
                        // Remote peer
                        // Below enforces consistency
                        // We cannot decouple the API, unfortunately,
                        // as the memory ordering mechanism is internal.
                        nvshmem_ushort_test(&sP->seqBit, NVSHMEM_CMP_EQ, localSeqBit);
                        fRd(dA, packet, status, taskCount, sP->routedTokens, sP->totalTilesM,
                            expertIdx, pGB, weights, bias, peerIdx, pLI.pe, lTQHead, tQHead);
                    }
                }
                // update state
                bitSet[vSIdx] = visitedSet;
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
            uint* __restrict__ bitSet,
            const packet::DecoderArg& dA,
            /// Task Arguments
            const TokenIdxTuple* const& tokenIds,
            cuda::std::byte* const& mO,
            /// Data Structures
            const uint* __restrict__ const& tileIndices,
            /// Lookup Table
            const ELI* __restrict__ const& eL,
            /// State
            uint* __restrict__ const& scratch,
            flagsType* __restrict__ const& flags,
            BookType* __restrict__ tQHead,
            uint& lTQHead,
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
                    if (i + 1 < stageLength) {
                        // Eagerly start global memory loads
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
                    if (const uint hasVisited = visitedSet >> vIdx & 1; !hasVisited && sP->seqBit == localSeqBit) {
                        // let's decode this packet
                        // set visited bit
                        visitedSet |= 1 << vIdx;
                        const auto expertIdx = flagIdx / CS;
                        const ELI lookup = eL[expertIdx];
                        const auto* tI = tokenIds + (expertIdx * EC + sP->batchIdx * BLOCK_M);
                        if (lookup.isRemote) {
                            // enforce memory consistency
                            nvshmem_ushort_test(&sP->seqBit, NVSHMEM_CMP_EQ, localSeqBit);
                            const auto* packet = heap::advance<1, 1>(dA.sHeap, lookup.epRank,
                                lookup.localExpertIndex,sP->batchIdx * BLOCK_M);

                            lRd(dA, packet, CONST_CAST_TO(cuda::std::byte, tI), mO, sP->tokensM,
                                lTQHead, tQHead, expertIdx);
                        }
                        else {
                            // enforce memory consistency
                            __threadfence_system();
                            const auto* packet = heap::advance<1, 1>(dA.sHeap, lookup.epRank,
                                lookup.localExpertIndex, sP->batchIdx * BLOCK_M);
                            lPd(dA.tQ + (tIdx * dA.tPs + lTQHead++), packet,
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
                        if (const uint hasVisited = visitedSet >> vIdx & 1; !hasVisited && sP->seqBit == localSeqBit) {
                            // set visited bit
                            visitedSet |= 1 << vIdx;
                            // let's decode this packet
                            const auto expertIdx = flagIdx / CS;
                            const ELI lookup = eL[expertIdx];
                            const auto* tI = tokenIds + (expertIdx * EC + (sP->batchIdx * BLOCK_M));
                            if (lookup.isRemote) {
                                // enforce memory consistency
                                nvshmem_ushort_test(&sP->seqBit, NVSHMEM_CMP_EQ, localSeqBit);
                                const auto* packet = heap::advance<1, 1>(dA.sHeap, lookup.epRank,
                                    lookup.localExpertIndex,sP->batchIdx * BLOCK_M);

                                lRd(dA, packet, CONST_CAST_TO(cuda::std::byte, tI), mO, sP->tokensM,
                                    lTQHead, tQHead, expertIdx);
                            }
                            else {
                                // enforce memory consistency
                                __threadfence_system();
                                const auto* packet = heap::advance<1, 1>(dA.sHeap, lookup.epRank,
                                    lookup.localExpertIndex, sP->batchIdx * BLOCK_M);
                                lPd(dA.tQ + (tIdx * dA.tPs + lTQHead++), packet,
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
    void start(uint* __restrict__ const& bitSet,
        cuda::std::byte* __restrict__ const& workspace,
        unsigned int* __restrict__ const& interrupt,
        const PLI* __restrict__ const& pL,
        const ELI* __restrict__ const& eL,
        const unsigned int& rE, // number of remote experts
        unsigned int* __restrict__ const& status, // shared
        unsigned int* __restrict__ const& taskCount,
        Output const& moeOutput,
        ExpertsUp const& expertsUp,
        ExpertsDown const& expertsDown,
        BiasUp const& biasUp,
        BiasDown const& biasDown,
        const uint16_t& lSeqBit){
        // offset due to warp specialization for the scheduler
        const auto tIdx = threadIdx.x - WARP_SIZE;
        static_assert(sizeof(unsigned long long int) == sizeof(flagsType));
        static_assert(sizeof(SignalPayload<>) == sizeof(uint64_t));
        static_assert(sizeof(SignalPayload<PacketStage::last>) == sizeof(uint64_t));
        // Register allocation
        const auto dA = packet::DecoderArg{
            bookkeeping.sHeap,
            bookkeeping.tQ(),
            bookkeeping.tPs,
        };

        cutlass::AlignedArray<uint, wSet> rWSet{};
        cutlass::AlignedArray<uint, bSzPs> rBitSet{};

        // lookup tables
        const auto* tokenIds = bookkeeping.tP();
        const auto* __restrict__ tileIndices = bookkeeping.tIx();

        // tQ things
        auto* tQHead = bookkeeping.tQH() + tIdx;
        auto lTQHead = 0U; // local tQ Head

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
        const auto ssfC = ACC::TCM::value * (ACC::TNx::value * (ACC::E::value - rE) + rE);
        const auto ssL = ssfC / subscriberCount + (tIdx < fSfC % subscriberCount);
        const auto ssT = ssL / wSet;

        constexpr Subscribe<SubscriberStage::initial, subscriberCount> initialSubscriber{};
        constexpr Subscribe<SubscriberStage::final, subscriberCount> finalSubscriber{};

        const auto pSI = nSI<subscriberCount>(ssfC);

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
                    bitSet + pSI,
                    status,
                    taskCount,
                    flags,
                    tQHead,
                    fSp,
                    lTQHead,
                    fSl,
                    nLx,
                    tIdx,
                    lSeqBit
                );
            }
            flags += fSfC;
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
                lTQHead,
                ssL,
                ssT,
                tIdx,
                lSeqBit);
        }
    }
}
#endif //ARISTOS_QUEUE_CUH
