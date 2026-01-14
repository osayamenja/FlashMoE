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

#ifndef FLASHMOE_SUBSCRIBER_CUH
#define FLASHMOE_SUBSCRIBER_CUH
#include <cuda/atomic>
#include <nvshmem.h>

#include "infra/bitset.cuh"
#include "infra/dq.cuh"
#include "infra/heap.cuh"
#include "infra/signal.cuh"
#include "infra/structures.cuh"
#include "infra/task.cuh"

namespace flashmoe::subscriber{
    constexpr int WARP_SIZE = 32;
    struct __align__(16) Args {
        uint64_t* const flags; // symmetric global
        Task* const tQ; // global
        const uint* tileIndices; // global
        cuda::std::byte* const GEMM0Staging;
        BitSet* const bitSet; // shared
        int* const interrupt; // shared
        unsigned int* const tQHead; // shared
        const PLI* const pL; // shared
        const LXI* const lX; // shared
        const ELI* const eL; // shared
        int* const status; // shared
        unsigned int* const taskCount; // shared
        const int ssfC; // second stage flag count
        const int gfSfC; // global first stage flag count -> global expert slots * epWorld
        const int world; // ep world
        const int nLx; // number of local experts
        const int epRank;
        const int E; // number of experts
        const int P; // FFN intermediate size
        const int tilesN0; // tiles across FFN intermediate size
        const int tilesN1; // tiles across hidden dim
        const int ecTilesM; // ceil_div(EC, tileM)
        const int ecSignalCount;
        const uint16_t tIdx;
        const uint16_t seqNumber;

        Args(uint64_t* const& _signals, Task* const& tq,
            const uint* const& _tileIndices,
            cuda::std::byte* const& gemm0Staging,
            BitSet* const& _bitSet,
            int* const& _interrupt,
            unsigned int* const& _tQHead,
            const PLI* const& _pL,
            const LXI* const& _lX,
            const ELI* const& _eL,
            int* const& _status,
            unsigned int* const& _taskCount,
            const int& _ssfC,
            const int& _gfSfC, const int& _world,
            const int& nLx,
            const int& _epRank, const int& _experts, const int& _p,
            const uint threadIdx, const int& _tilesN0, const int& _tilesN1, const int& _eCTilesM,
            const uint16_t seqNo):
        flags(_signals),
        tQ(tq + threadIdx),
        tileIndices(_tileIndices),
        GEMM0Staging(gemm0Staging),
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
        E(_experts), P(_p),
        tilesN0(_tilesN0), tilesN1(_tilesN1), ecTilesM(_eCTilesM), ecSignalCount(ecTilesM * tilesN1),
        tIdx(threadIdx),
        seqNumber(seqNo) {}
    };
    /// Decodes a single packet from the initial stage
    template<
        int subscriberCount,
        PacketStage s,
        PeerConnectivity p,
        int bM = -1,
        typename Element = void
    >
    struct Decoder {
        static_assert(s == PacketStage::initial);
        __device__ __forceinline__
        void operator()(const Args& args,
            const Heap& heap,
            cuda::std::byte* __restrict__ sHeap,
            Ingredients& ingredients,
            uint64_t* const& flags,
            const cuda::std::byte* const& packet,
            unsigned int const& routedTokens,
            int const& peer, // relative to the EP group
            const uint& laneId,
            int& lTQHead) const {
            static_assert(bM > 0);
            const auto qIdx = DQ::sNext<DQType::stride, subscriberCount>(lTQHead);
            const auto fTilesM = routedTokens / bM;
            // expert, peer offset
            const auto sO = args.ecTilesM * (peer * args.nLx + ingredients.localExpertIdx);
            cuda::std::array<cuda::std::byte*, GEMMs> taskResults{};
            // Staging buffer for results of preGEMM
            taskResults[0] = args.GEMM0Staging + (peer * args.nLx * heap.EC * args.P * sizeof(Element));
            // Egress packet buffer
            auto* rcData = sHeap + heap.advanceOffset<1, 1>(args.epRank, ingredients.localExpertIdx);
            auto* intraStaging = sHeap + heap.advanceOffset<1, 0>(peer, ingredients.localExpertIdx);
            taskResults[1] = p == PeerConnectivity::remote ? intraStaging : rcData;
            const auto wT = fTilesM * args.tilesN0;
            const auto fS = wT / WARP_SIZE + (laneId < wT % WARP_SIZE);
            const auto rT = args.tilesN0 % WARP_SIZE;
            const auto lS = args.tilesN0 / WARP_SIZE + (rT > 0 ? laneId < rT : 0);
            const auto tSlice = fS + (routedTokens % bM == 0 ? 0 : lS);

            for (int i = 0; i < fS; ++i) {
                const auto tileIdx = laneId + i * WARP_SIZE;
                const auto rowIdx = tileIdx / args.tilesN0;
                const auto syncIdx = sO + rowIdx;
                ingredients.stash = rowIdx;
                ingredients.tileSize = bM;
                args.tQ[DQ::next<DQType::stride, subscriberCount>(qIdx, i)] = Task{
                    ingredients, packet, taskResults, rcData, flags, syncIdx, tileIdx
                };
            }

            // residue tile
            if (const auto residue = routedTokens - fTilesM * bM; residue) {
                for (uint j = 0; j < lS; j++) {
                    const auto tileIdx = fTilesM * args.tilesN0 + laneId + j * WARP_SIZE;
                    const auto syncIdx = sO + fTilesM;
                    const auto rowIdx = fTilesM;
                    ingredients.stash = rowIdx;
                    ingredients.tileSize = static_cast<uint16_t>(residue);
                    args.tQ[DQ::next<DQType::stride, subscriberCount>(qIdx, fS + j)] = Task{
                        ingredients, packet, taskResults, rcData, flags, syncIdx, tileIdx
                    };
                }
            }

            if (tSlice) {
                lTQHead += tSlice;
                cuda::atomic_ref<unsigned int, cuda::thread_scope_block> tqh{*args.tQHead};
                cuda::std::ignore = tqh.fetch_add(tSlice, cuda::memory_order_release);
            }
        }
    };


    template<int subscriberCount>
    struct Decoder<subscriberCount, PacketStage::last, PeerConnectivity::p2p> {
        __device__ __forceinline__
        void operator()(const Args& args, const Ingredients& ingredients, unsigned int& lTQHead) const {
            // now let's decode this single tile
            // Note: we intentionally modeled the Task struct so that the below compiles to a single 128B
            // instruction rather than 4 of them, which would have been the case if we updated the entire object.
            args.tQ[DQ::sNext<DQType::stride, subscriberCount>(lTQHead++)].ingredients = ingredients;
            cuda::atomic_ref<unsigned int, cuda::thread_scope_block> tqh{*args.tQHead};
            // notifies scheduler of work
            cuda::std::ignore = tqh.fetch_add(1, cuda::memory_order_release);
        }
    };

    template<int subscriberCount>
    struct Decoder<subscriberCount, PacketStage::last, PeerConnectivity::remote> {
        __device__ __forceinline__
            void operator()(const Args& args, Ingredients& ingredients, unsigned int& lTQHead) const {
            const auto qIdx = DQ::sNext<DQType::stride, subscriberCount>(lTQHead);
            for (uint i = 0; i < args.tilesN1; ++i) {
                ingredients.stash = i;
                args.tQ[DQ::next<DQType::stride, subscriberCount>(qIdx, i)].ingredients = ingredients;
            }
            lTQHead += args.tilesN1;
            cuda::atomic_ref<unsigned int, cuda::thread_scope_block> tqh{*args.tQHead};
            cuda::std::ignore = tqh.fetch_add(args.tilesN1, cuda::memory_order_release);
        }
    };
    __device__
    enum class SubscriberStage {
        initial,
        final
    };

    // Self-correct Termination Bound
    __device__ __forceinline__
    void sTB(const Args& args, const int& peer, const int& peerTaskTiles = 0) {
        if (!atomicTAS<cuda::thread_scope_block>(args.status + peer)) {
            const auto superfluous = (args.tilesN0 + args.tilesN1) * ((args.nLx * args.ecTilesM) - peerTaskTiles);
            atomicSub_block(args.taskCount, superfluous);
        }
    }

    template<
        SubscriberStage s,
        typename Element,
        int subscriberCount,
        int bM,
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
            int& pending, int& ltQHead) const {
            /// Flags has dimension [W, L], where W is expert parallel world and L is number of local experts
            constexpr Decoder<subscriberCount, PacketStage::initial, PeerConnectivity::p2p, bM, Element> fPd{};
            constexpr Decoder<subscriberCount, PacketStage::initial, PeerConnectivity::remote, bM, Element> fRd{};
            constexpr int bSw = sizeof(uint) * 8U;
            const auto laneId = args.tIdx % WARP_SIZE;
            const auto warpId = args.tIdx / WARP_SIZE;
            for (int i = 0; i < stageLength; ++i) {
                const auto vSIdx = i / bSw;
                const auto vIdx = i % bSw;
                const auto flagIdx = warpId + i * sNW;
                const auto peerIdx = flagIdx / args.nLx;
                const auto pLI = args.pL[peerIdx];
                uint64_t signal = SignalConstants::ground;
                if (laneId == 0) {
                    auto visitedSet = bitSet[warpId + vSIdx * sNW];
                    if (!visitedSet.get(vIdx)) {
                        if (pLI.isRemote) {
                            // RDMA peer
                            signal = nvshmem_signal_fetch(flags + flagIdx);
                            const auto sigPayload = cuda::std::bit_cast<SignalPayload<PacketStage::initial>>(signal);
                            if (sigPayload.seqNumber == args.seqNumber) {
                                // set visited bit
                                // self-correct the termination bound
                                sTB(args, peerIdx, sigPayload.totalTilesM);
                                visitedSet.set(vIdx);

                                // enforce memory consistency of expected packet
                                const bool isPacketHere = nvshmem_uint64_test(flags + flagIdx,
                                    NVSHMEM_CMP_EQ, signal);
                                if (!isPacketHere) {
                                    /*this scenario means that this peer sent another packet in between us
                                    observing the signal and testing the signal's presence.
                                    This is fine, specifically within our protocol,
                                    this occurs if the producer previously sent a noop,
                                    advanced to the next epoch and sent another packet.
                                    we simply do nothing in this scenario as
                                    we will process the new packet in the next epoch.
                                    */
                                    if (__builtin_expect(sigPayload.routedTokens > 0, 0)) {
                                        // protocol violation
                                        __trap();
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
                                sTB(args, peerIdx);
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
                                sTB(args, peerIdx, sigPayload.totalTilesM);
                                visitedSet.set(vIdx);
                            }
                            else if (sbs::ahead(sigPayload.seqNumber, args.seqNumber)) {
                                sTB(args, peerIdx);
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
                    pending -= 1;
                    const auto myLocalExIdx = flagIdx % args.nLx;
                    const auto lXI = args.lX[myLocalExIdx];
                    const auto* packet = symHeap.advance<0, 1>(peerIdx, myLocalExIdx);

                    Ingredients ingredients{};
                    ingredients.localExpertIdx = myLocalExIdx;
                    ingredients.taskType = TaskType::GEMM0;
                    ingredients.peerIdx = pLI.pe;
                    // pad here for compatibility
                    ingredients.M = cute::ceil_div(sigPayload.routedTokens, bM) * bM;
                    if (!pLI.isRemote) {
                        auto* nFlags = pLI.remoteSFlags + args.gfSfC + lXI.expertIndex * (args.ecTilesM * args.tilesN1);
                        // fpd
                        ingredients.isPeerRemote = 0;
                        fPd(args, symHeap, pLI.remoteSHeap, ingredients, nFlags, packet,
                            sigPayload.routedTokens, peerIdx, laneId, ltQHead);
                    }
                    else {
                        auto* nFlags = args.flags + args.gfSfC + lXI.expertIndex * (args.ecTilesM * args.tilesN1);
                        // frd
                        ingredients.isPeerRemote = 1;
                        fRd(args, symHeap, symHeap.sHeap, ingredients, nFlags, packet,
                            sigPayload.routedTokens, peerIdx, laneId, ltQHead);
                    }
                }
            }
        }
    };

    template<int subscriberCount, int bM, typename Element>
    struct Subscriber<SubscriberStage::final, Element, subscriberCount, bM> {
        __device__ __forceinline__
        void operator()(const Args& args,
            BitSet* __restrict__ const& bitSet, uint64_t* __restrict__ const& flags,
            int& ltQHead, const int& stageLength) const {
            constexpr int wSet = 16;
            constexpr int bSw = sizeof(uint) * 8U;
            static_assert(wSet == 16 || wSet == 32);
            const int stageTrips = stageLength / wSet;
            cutlass::AlignedArray<uint, wSet> workSet{};
            constexpr Decoder<subscriberCount, PacketStage::last, PeerConnectivity::p2p> lPd{};
            constexpr Decoder<subscriberCount, PacketStage::last, PeerConnectivity::remote> lRd{};
            // prefetch
            for (int i = 0; i < stageTrips; ++i) {
                const uint sBIdx = args.tIdx + (i * wSet / bSw) * subscriberCount;
                auto sBS = bitSet[sBIdx];
                // gmem -> registers
                #pragma unroll
                for (uint j = 0; j < wSet; ++j) {
                    workSet[j] = args.tileIndices[args.tIdx + j * subscriberCount];
                }
                #pragma unroll
                for (uint j = 0; j < wSet; ++j) {
                    const uint bIdx = (i * wSet + j) % bSw;
                    const auto flagIdx = workSet[j];
                    const auto expertIdx = flagIdx / args.ecSignalCount;
                    const auto lookup = args.eL[expertIdx];
                    if (!sBS.get(bIdx)) {
                        if (lookup.isRemote) {
                            // RDMA peer
                            const auto signal = nvshmem_signal_fetch(flags + flagIdx);
                            const auto sigPayload = cuda::std::bit_cast<SignalPayload<PacketStage::last>>(signal);
                            if (sigPayload.seqNumber == args.seqNumber) {
                                // set visited bit
                                sBS.set(bIdx);
                                // enforce memory consistency
                                const bool isPacketHere = nvshmem_uint64_test(flags + flagIdx, NVSHMEM_CMP_EQ,
                                    signal);
                                if (__builtin_expect(!isPacketHere, 0)) {
                                    // protocol violation, this should be impossible
                                    // if we are here, something insanely wrong has happened external to our program
                                    __trap(); 
                                }
                                // construct combine ingredients
                                Ingredients ingredients{};
                                const auto tokenIdx = sigPayload.batchIdx * bM;
                                ingredients.expertIdx = expertIdx;
                                ingredients.M = tokenIdx;
                                ingredients.localExpertIdx = lookup.localExpertIndex;
                                ingredients.peerIdx = args.epRank;
                                ingredients.tileSize = sigPayload.tokensM;
                                ingredients.taskType = TaskType::combine;
                                ingredients.isPeerRemote = 1;
                                lRd(args, ingredients, ltQHead);
                            }
                        }
                        else {
                            // NVLink peer
                            cuda::atomic_ref<uint64_t, cuda::thread_scope_system> f{*(flags + flagIdx)};
                            const auto sigPayload = cuda::std::bit_cast<SignalPayload<PacketStage::last>>(
                                f.load(cuda::memory_order_acquire));
                            if (sigPayload.seqNumber == args.seqNumber) {
                                sBS.set(bIdx);
                                const auto tokenIdx = sigPayload.batchIdx * bM;
                                // construct combine ingredients
                                Ingredients ingredients{};
                                ingredients.expertIdx = expertIdx;
                                ingredients.M = tokenIdx;
                                ingredients.localExpertIdx = lookup.localExpertIndex;
                                ingredients.peerIdx = args.epRank;
                                ingredients.tileSize = sigPayload.tokensM;
                                ingredients.stash = flagIdx % args.tilesN1;
                                ingredients.taskType = TaskType::combine;
                                ingredients.isPeerRemote = 0;
                                lPd(args, ingredients, ltQHead);
                            }
                        }
                    }
                }
                bitSet[sBIdx] = sBS;
            }
            if (const auto residue = stageLength - stageTrips * wSet; residue) {
                const uint sBIdx = args.tIdx + (stageTrips * wSet / bSw) * subscriberCount;
                auto sBS = bitSet[sBIdx];
                #pragma unroll
                for (uint j = 0; j < wSet; ++j) {
                    if (j < residue) {
                        workSet[j] = args.tileIndices[args.tIdx + (j + stageTrips * wSet) * subscriberCount];
                    }
                }
                #pragma unroll
                for (uint j = 0; j < wSet; ++j) {
                    if (j < residue) {
                        const uint bIdx = (stageTrips * wSet + j) % bSw;
                        const auto flagIdx = workSet[j];
                        const auto expertIdx = flagIdx / args.ecSignalCount;
                        const auto lookup = args.eL[expertIdx];
                        if (!sBS.get(bIdx)) {
                            if (lookup.isRemote) {
                                const auto signal = nvshmem_signal_fetch(flags + flagIdx);
                                const auto sigPayload = cuda::std::bit_cast<SignalPayload<PacketStage::last>>(signal);
                                if (sigPayload.seqNumber == args.seqNumber) {
                                    sBS.set(bIdx);
                                    const bool isPacketHere = nvshmem_uint64_test(flags + flagIdx, NVSHMEM_CMP_EQ,
                                        signal);
                                    if (__builtin_expect(!isPacketHere, 0)) {
                                        __trap();
                                    }
                                    Ingredients ingredients{};
                                    const auto tokenIdx = sigPayload.batchIdx * bM;
                                    ingredients.M = tokenIdx;
                                    ingredients.localExpertIdx = lookup.localExpertIndex;
                                    ingredients.peerIdx = args.epRank;
                                    ingredients.tileSize = sigPayload.tokensM;
                                    ingredients.taskType = TaskType::combine;
                                    ingredients.isPeerRemote = 1;
                                    lRd(args, ingredients, ltQHead);
                                }
                            }
                            else {
                                cuda::atomic_ref<uint64_t, cuda::thread_scope_system> f{*(flags + flagIdx)};
                                const auto sigPayload = cuda::std::bit_cast<SignalPayload<PacketStage::last>>(
                                    f.load(cuda::memory_order_acquire));
                                if (sigPayload.seqNumber == args.seqNumber) {
                                    sBS.set(bIdx);
                                    const auto tokenIdx = sigPayload.batchIdx * bM;
                                    // construct combine ingredients
                                    Ingredients ingredients{};
                                    ingredients.M = tokenIdx;
                                    ingredients.localExpertIdx = lookup.localExpertIndex;
                                    ingredients.peerIdx = args.epRank;
                                    ingredients.tileSize = sigPayload.tokensM;
                                    ingredients.stash = flagIdx % args.tilesN1;
                                    ingredients.taskType = TaskType::combine;
                                    ingredients.isPeerRemote = 0;
                                    lPd(args, ingredients, ltQHead);
                                }
                            }
                        }
                    }
                }
                // update bitset
                bitSet[sBIdx] = sBS;
            }
        }
    };

    template<
        unsigned int subscriberCount,
        int bM,
        typename Element
    >
    requires(subscriberCount % WARP_SIZE == 0)
    __device__ __forceinline__
    void start(const Heap& symHeap, const Args& args){
        int ltQHead = 0; // local tQ Head

        // first stage
        constexpr auto sNW = subscriberCount / WARP_SIZE;
        const auto fSfC = args.world * args.nLx; // first stage flag count
        const auto fSl = fSfC / sNW + (args.tIdx / WARP_SIZE < fSfC % sNW);
        auto fSp = fSl; // first stage pending

        // second stage
        const auto ssL = args.ssfC / subscriberCount + (args.tIdx < args.ssfC % subscriberCount);
        constexpr Subscriber<SubscriberStage::initial, Element, subscriberCount, bM> initialSubscriber{};
        constexpr Subscriber<SubscriberStage::final, Element, subscriberCount, bM> finalSubscriber{};
        const auto pSI = nSI<subscriberCount>(args.ssfC);

        cuda::atomic_ref<int, cuda::thread_scope_block> interrupt{*args.interrupt};
        while (!interrupt.load(cuda::memory_order_relaxed)) {
            auto* __restrict__ flags = args.flags;
            // sweep through flags by stages
            // start with the first stage
            if (fSp) {
                auto* __restrict bitset = args.bitSet + pSI;
                initialSubscriber(symHeap, args, flags, bitset, fSl, fSp, ltQHead);
            }
            flags += args.gfSfC;
            finalSubscriber(args, args.bitSet, flags, ltQHead, ssL);
        }
    }
}
#endif //FLASHMOE_SUBSCRIBER_CUH
