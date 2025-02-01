//
// Created by Jonathan on 7/4/24.
//

#ifndef ARISTOS_QUEUE_CUH
#define ARISTOS_QUEUE_CUH

#include <nvshmemx.h>
#include <nvshmem.h>
#include <host/nvshmemx_api.h>

#include "../types.cuh"
#include "packet.cuh"

namespace aristos::subscriber{
    ///Receive and decode packets deposited
    template<
        unsigned int wSet = 16U,
        unsigned int subscriberCount = SUBSCRIBERS,
        typename ExpertsTensor,
        typename BiasTensor,
        typename Activations,
        typename Element = typename ExpertsTensor::value_type
    >
    requires(cuda::std::is_same_v<typename ExpertsTensor::value_type, typename BiasTensor::value_type>
        && aristos::Tensor<ExpertsTensor>
        && aristos::Matrix<BiasTensor> && aristos::Matrix<Activations> && cutlass::ispow2(wSet)
        && wSet > 1 && wSet <= 32)
    __device__ __forceinline__
    void start(cuda::std::byte* __restrict__ const& workspace,
        unsigned int* __restrict__ const& interrupt,
        const unsigned int* __restrict__& peerTranslation, // shared
        // remote experts: {actual & local expert idx, peer idx}
        const cuda::std::tuple<uint, uint, uint>* __restrict__& rE, // shared
        const cuda::std::tuple<uint, uint, uint>* __restrict__& nRe, // p2p experts: like above
        const unsigned int& rEl, // number of remote peers
        unsigned int* __restrict__ const& status, // shared
        unsigned int* __restrict__ const& taskCount,
        Activations const& activations,
        ExpertsTensor const& experts,
        BiasTensor const& biasT,
        const uint16_t& lSeqBit){
        // offset due to warp specialization for the scheduler
        const auto tIdx = threadIdx.x - WARP_SIZE;
        static_assert(sizeof(unsigned long long int) == sizeof(flagsType));
        static_assert(sizeof(SignalPayload<>) == sizeof(uint64_t));
        static_assert(sizeof(SignalPayload<PacketStage::last>) == sizeof(uint64_t));
        /*assert(__isShared(workspace) &&
            __isShared(&mC) && // faster to retrieve from shared than constant memory due to strewn accesses
            __isShared(&sC) &&
            __isShared(interrupt) &&
            __isShared(status) && __isShared(taskCount));*/

        const auto dA = packet::DecoderArg{
            moeConfig.sHeap,
            bookkeeping.tQ(),
            moeConfig.cellSize,
            bookkeeping.eCap,
            moeConfig.embedDim,
            bookkeeping.tPs,
            bookkeeping.tN,
            Config::tiles<BLOCK_N>(bookkeeping.px),
            moeConfig.expertSlots,
            bookkeeping.nx
        };

        // each thread gets 64 bytes of workspace
        cutlass::AlignedArray<unsigned int, wSet> rWSet{};

        // token indices
        auto* __restrict__ tokenIds = bookkeeping.tP();

        // tQ things
        auto* __restrict__ tQHead = bookkeeping.tQH() + tIdx;
        auto lTQHead = 0U; // local tQ Head

        // pointers
        auto* __restrict__ sharedSpace = CAST_TO(unsigned int, workspace);
        auto* __restrict__ sFlags = moeConfig.flags;
        auto* __restrict__ pGB = bookkeeping.xM<Element>(); // post GEMM buffer

        // Constants
        const auto nLx = bookkeeping.nLx;

        // first stage
        const auto fSfC = bookkeeping.world * nLx; // first stage flag count
        const auto fSl = fSfC / subscriberCount + (tIdx < fSfC % subscriberCount);
        const auto fSt = fSl / wSet;
        auto fSp = fSl; // first stage pending

        // second stage: remote
        const auto tilesMc = Config::tiles<BLOCK_M>(dA.eCap);
        const auto sRfC = rEl * tilesMc * dA.tN;
        const auto sRl = sRfC / subscriberCount + (tIdx < sRfC % subscriberCount);
        const auto sRt = sRl / wSet;

        // second stage: p2p
        const auto sPfC = (dA.nx - rEl) * tilesMc * dA.tN;
        const auto sPl = sPfC / subscriberCount + (tIdx < sPfC % subscriberCount);
        const auto sPt = sPl / wSet;
        const auto iPfS = cute::make_shape(tilesMc, dA.tN);
        const auto fS = make_shape(dA.nx - rEl, iPfS);
        const auto fStride = make_stride(size(iPfS), cute::make_stride(dA.tN, 1));

        auto* __restrict__ ffC = bookkeeping.fC(); // flags checkpoint
        auto* __restrict__ rfC = ffC + fSfC; // second stage: remote flags checkpoint
        auto* __restrict__ pfC = rfC + sRfC; // second stage: p2p flags checkpoint

        // Decoder stuff
        packet::Decoder<PacketStage::initial, PeerConnectivity::p2p, Element> fPd{};
        packet::Decoder<PacketStage::initial, PeerConnectivity::remote, Element> fRd{};
        packet::Decoder<PacketStage::last, PeerConnectivity::p2p> lPd{};
        packet::Decoder<PacketStage::last, PeerConnectivity::remote> lRd{};

        while (!atomicLoad<cuda::thread_scope_block>(interrupt)) {
            auto* __restrict__ flags = sFlags;
            // sweep through flags by stages
            // start with the first stage
            // Apply loop unroll with residue
            if (fSp) {
                for (uint i = 0; i < fSt; ++i) {
                    #pragma unroll
                    for (uint j = 0; j < wSet; ++j) {
                        // global to shared memory
                        const auto flagIdx = tIdx + (j + i * wSet) * subscriberCount;
                        sharedSpace[tIdx + j * subscriberCount] = ffC[flagIdx];
                    }
                    #pragma unroll
                    for (uint j = 0; j < wSet; ++j) {
                        rWSet[j] = sharedSpace[tIdx + j * subscriberCount];
                    }
                    // no need for overlapping the next stage, as most likely, only one trip will occur
                    #pragma unroll
                    for (uint j = 0; j < wSet; ++j) {
                        const auto flagIdx = tIdx + (j + i * wSet) * subscriberCount;
                        // main loop
                        if (!rWSet[j]) {
                            // we need to check this flag
                            auto signal = atomicLoad<cuda::thread_scope_system>(
                                CAST_TO(unsigned long long int, flags + flagIdx));
                            const auto sP = CAST_TO(SignalPayload<>, &signal);
                            rWSet[j] = sP->seqBit == lSeqBit;
                            fSp -= rWSet[j];
                            if (rWSet[j]) {
                                // decode the received packet
                                auto expertIdx = flagIdx % nLx;
                                auto peerIdx = flagIdx / nLx;
                                cuda::std::array weights{
                                    CAST_TO(cuda::std::byte, &experts(expertIdx, 0)),
                                    CAST_TO(cuda::std::byte, &experts(expertIdx, 1))
                                };
                                cuda::std::array bias{
                                    CAST_TO(cuda::std::byte, &biasT(expertIdx, 0)),
                                    CAST_TO(cuda::std::byte, &biasT(expertIdx, 1))
                                };
                                auto* __restrict__ packet = heap::advance<0, 1, sizeof(Element)>(dA.sHeap, dA.cellSize,
                                    dA.expertSlots, dA.tokenSize, peerIdx, expertIdx);
                                if (nvshmem_ptr(packet, peerTranslation[peerIdx]) != nullptr) {
                                    // P2P peer
                                    // Enforce consistency
                                    // before decoding the packet
                                    __threadfence_system();
                                    fPd(dA, packet, status, taskCount, sP->routedTokens, sP->totalTilesM,
                                        expertIdx, pGB, weights, bias, peerIdx, lTQHead, tQHead);
                                }
                                else {
                                    // Remote peer
                                    // Below enforces consistency
                                    // before reading the packet
                                    // we cannot decouple the API, unfortunately,
                                    // as the consistency function is internal.
                                    nvshmem_ushort_test(&sP->seqBit, NVSHMEM_CMP_EQ, lSeqBit);
                                    fRd(dA, packet, status, taskCount, sP->routedTokens, sP->totalTilesM,
                                        expertIdx, pGB, weights, bias, peerIdx, lTQHead, tQHead);
                                }
                            }
                        }
                    }
                    // update checkpoints
                    #pragma unroll
                    for (uint j = 0; j < wSet; ++j) {
                        const auto flagIdx = tIdx + (j + i * wSet) * subscriberCount;
                        ffC[flagIdx] = rWSet[j];
                    }
                }
                // residue
                if (const auto residue = fSl - fSt * wSet) {
                    for (uint j = 0; j < residue; ++j) {
                        // global to shared memory
                        const auto flagIdx = tIdx + (j + fSt * wSet) * subscriberCount;
                        sharedSpace[tIdx + j * subscriberCount] = ffC[flagIdx];
                    }
                    for (uint j = 0; j < residue; ++j) {
                        rWSet[j] = sharedSpace[tIdx + j * subscriberCount];
                    }
                    for (uint j = 0; j < residue; ++j) {
                        const auto flagIdx = tIdx + (j + fSt * wSet) * subscriberCount;
                        // main loop
                        if (!rWSet[j]) {
                            // we need to check this flag
                            auto signal = atomicLoad<cuda::thread_scope_system>(
                                    CAST_TO(unsigned long long int, flags + flagIdx));
                            const auto sP = CAST_TO(SignalPayload<>, &signal);
                            rWSet[j] = sP->seqBit == lSeqBit;
                            fSp -= rWSet[j];
                            if (rWSet[j]) {
                                // decode the received packet
                                auto expertIdx = flagIdx % nLx;
                                auto peerIdx = flagIdx / nLx;
                                cuda::std::array weights{
                                    CAST_TO(cuda::std::byte, &experts(expertIdx, 0)),
                                    CAST_TO(cuda::std::byte, &experts(expertIdx, 1))
                                };
                                cuda::std::array bias{
                                    CAST_TO(cuda::std::byte, &biasT(expertIdx, 0)),
                                    CAST_TO(cuda::std::byte, &biasT(expertIdx, 1))
                                };

                                if (auto* packet = heap::advance<0, 1, sizeof(Element)>(dA.sHeap, dA.cellSize,
                                    dA.expertSlots, dA.tokenSize, peerIdx, expertIdx);
                                    nvshmem_ptr(packet, peerTranslation[peerIdx]) != nullptr) {
                                    // Enforce consistency before decoding the packet
                                    __threadfence_system();
                                    fPd(packet, status, taskCount, sP->routedTokens, sP->totalTilesM,
                                        expertIdx, pGB, weights, bias, peerIdx, lTQHead, tQHead);
                                }
                                else {
                                    nvshmem_ushort_test(&sP->seqBit, NVSHMEM_CMP_EQ, lSeqBit);
                                    fRd(dA, packet, status, taskCount, sP->routedTokens, sP->totalTilesM,
                                        expertIdx, pGB, weights, bias, peerIdx, lTQHead, tQHead);
                                }
                            }
                        }
                    }
                    for (uint j = 0; j < residue; ++j) {
                        const auto flagIdx = tIdx + (j + fSt * wSet) * subscriberCount;
                        rWSet[j] = ffC[flagIdx];
                    }
                }
            }

            flags += fSfC;

            // second stage, where flag dimension is (E, C)
            // remote
            if (sRt) {
                // prefetch to shared memory
                #pragma unroll
                for (uint j = 0; j < wSet; ++j) {
                    const auto flagIdx = tIdx + j * subscriberCount;
                    sharedSpace[tIdx + j * subscriberCount] = rfC[flagIdx];
                }
            }
            for (uint i = 0; i < sRt; ++i) {
                #pragma unroll
                for (uint j = 0; j < wSet; ++j) {
                    rWSet[j] = sharedSpace[tIdx + j * subscriberCount];
                    if (i + 1 < sRt) {
                        const auto flagIdx = tIdx + (j + (i + 1) * wSet) * subscriberCount;
                        sharedSpace[tIdx + j * subscriberCount] = rfC[flagIdx];
                    }
                }
                #pragma unroll
                for (uint j = 0; j < wSet; ++j) {
                    const auto flagIdx = tIdx + (j + i * wSet) * subscriberCount;
                    if (!rWSet[j]) {
                        auto signal = atomicLoad<cuda::thread_scope_system>(
                                CAST_TO(unsigned long long int, flags + flagIdx));
                        // SignalPayload -> {batchIdx, {seqNo, M}}, where M <= BLOCK_M
                        // we do not necessarily need
                        // to transmit batchIdx as we can deduce it locally
                        const auto sP = CAST_TO(SignalPayload<PacketStage::last>, &signal);
                        rWSet[j] = sP->seqBit == lSeqBit;
                        fSp -= rWSet[j];
                        if (rWSet[j]) {
                            // enforce remote memory consistency
                            nvshmem_ushort_test(&sP->seqBit, NVSHMEM_CMP_EQ, lSeqBit);
                            const auto [aEIdx, lEIdx, pIdx] = rE[flagIdx / tilesMc];
                            auto* __restrict__ packet = heap::advance<1, 1, sizeof(Element)>(dA.sHeap, dA.cellSize,
                                dA.expertSlots, dA.tokenSize, pIdx, lEIdx, sP->batchIdx * BLOCK_M);
                            lRd(dA, packet, CAST_TO(cuda::std::byte, tokenIds + (aEIdx * dA.eCap + sP->batchIdx * BLOCK_M)),
                                CAST_TO(cuda::std::byte, activations.data()), sP->tokensM, lTQHead, tQHead, aEIdx);
                        }
                    }
                }
                // update checkpoints
                #pragma unroll
                for (uint j = 0; j < wSet; ++j) {
                    const auto flagIdx = tIdx + (j + i * wSet) * subscriberCount;
                    rfC[flagIdx] = rWSet[j];
                }
            }
            if (const auto residue = sRl - sRt * wSet) {
                for (uint j = 0; j < residue; ++j) {
                    const auto flagIdx = tIdx + (j + sRt * wSet) * subscriberCount;
                    sharedSpace[tIdx + j * subscriberCount] = ffC[flagIdx];
                }
                for (uint j = 0; j < residue; ++j) {
                    rWSet[j] = sharedSpace[tIdx + j * subscriberCount];
                }
                for (uint j = 0; j < residue; ++j) {
                    const auto flagIdx = tIdx + (j + sRt * wSet) * subscriberCount;
                    if (!rWSet[j]) {
                        auto signal = atomicLoad<cuda::thread_scope_system>(
                                CAST_TO(unsigned long long int, flags + flagIdx));
                        // SignalPayload -> {batchIdx, {seqNo, M}}, where M <= BLOCK_M
                        const auto sP = CAST_TO(SignalPayload<PacketStage::last>, &signal);
                        rWSet[j] = sP->seqBit == lSeqBit;
                        fSp -= rWSet[j];
                        if (rWSet[j]) {
                            // enforce remote memory consistency
                            nvshmem_ushort_test(&sP->seqBit, NVSHMEM_CMP_EQ, lSeqBit);
                            const auto [aEIdx, lEIdx, pIdx] = rE[flagIdx / tilesMc];
                            auto* __restrict__ packet = heap::advance<1, 1, sizeof(Element)>(dA.sHeap, dA.cellSize,
                                dA.expertSlots, dA.tokenSize,
                                pIdx, lEIdx, sP->batchIdx * BLOCK_M);
                            lRd(dA, packet, CAST_TO(cuda::std::byte, tokenIds + (aEIdx * dA.eCap + sP->batchIdx * BLOCK_M)),
                                CAST_TO(cuda::std::byte, activations.data()),
                                sP->tokensM, lTQHead, tQHead, aEIdx);
                        }
                    }
                }
                for (uint j = 0; j < residue; ++j) {
                    const auto flagIdx = tIdx + (j + sRt * wSet) * subscriberCount;
                    rfC[flagIdx] = rWSet[j];
                }
            }

            flags += sRfC;
            // second stage
            // p2p
            if (sPt) {
                // prefetch to shared memory
                #pragma unroll
                for (uint j = 0; j < wSet; ++j) {
                    const auto flagIdx = tIdx + j * subscriberCount;
                    sharedSpace[tIdx + j * subscriberCount] = pfC[flagIdx];
                }
            }
            for (uint i = 0; i < sPt; ++i) {
                #pragma unroll
                for (uint j = 0; j < wSet; ++j) {
                    rWSet[j] = sharedSpace[tIdx + j * subscriberCount];
                    if (i + 1 < sPt) {
                        const auto flagIdx = tIdx + (j + (i + 1) * wSet) * subscriberCount;
                        sharedSpace[tIdx + j * subscriberCount] = pfC[flagIdx];
                    }
                }
                #pragma unroll
                for (uint j = 0; j < wSet; ++j) {
                    const auto flagIdx = tIdx + (j + i * wSet) * subscriberCount;
                    if (!rWSet[j]) {
                        auto signal = atomicLoad<cuda::thread_scope_system>(
                                CAST_TO(unsigned long long int, flags + flagIdx));
                        // SignalPayload -> {batchIdx, {seqNo, M}}, where M <= BLOCK_M
                        const auto sP = CAST_TO(SignalPayload<PacketStage::last>, &signal);
                        rWSet[j] = sP->seqBit == lSeqBit;
                        fSp -= rWSet[j];
                        if (rWSet[j]) {
                            // enforce memory consistency
                            __threadfence_system();
                            // [index to nRe, batchIdx, tileIdx]
                            const auto coord = idx2crd(flagIdx, fS, fStride);
                            const auto [aEIdx, lEIdx, pIdx] = nRe[cute::get<0>(coord)];
                            auto* __restrict__ packet = heap::advance<1, 1, sizeof(Element)>(dA.sHeap, dA.cellSize,
                                dA.expertSlots, dA.tokenSize,
                                pIdx, lEIdx, sP->batchIdx * BLOCK_M);
                            lPd(dA.tQ + (tIdx * dA.tPs + lTQHead++), packet, CAST_TO(cuda::std::byte, tokenIds + (aEIdx * dA.eCap + sP->batchIdx * BLOCK_M)),
                                CAST_TO(cuda::std::byte, activations.data()), sP->tokensM,
                                cute::get<2>(coord), tQHead, aEIdx);
                        }
                    }
                }
                #pragma unroll
                for (uint j = 0; j < wSet; ++j) {
                    const auto flagIdx = tIdx + (j + i * wSet) * subscriberCount;
                    pfC[flagIdx] = rWSet[j];
                }
            }
            if (const auto residue = sPl - sPt * wSet) {
                for (uint j = 0; j < residue; ++j) {
                    const auto flagIdx = tIdx + (j + sPt * wSet) * subscriberCount;
                    sharedSpace[tIdx + j * subscriberCount] = pfC[flagIdx];
                }
                for (uint j = 0; j < residue; ++j) {
                    rWSet[j] = sharedSpace[tIdx + j * subscriberCount];
                }
                for (uint j = 0; j < residue; ++j) {
                    const auto flagIdx = tIdx + (j + sPt * wSet) * subscriberCount;
                    if (!rWSet[j]) {
                        auto signal = atomicLoad<cuda::thread_scope_system>(
                                CAST_TO(unsigned long long int, flags + flagIdx));
                        // SignalPayload -> {batchIdx, {seqNo, M}}, where M <= BLOCK_M
                        const auto sP = CAST_TO(SignalPayload<PacketStage::last>, &signal);
                        rWSet[j] = sP->seqBit == lSeqBit;
                        fSp -= rWSet[j];
                        if (rWSet[j]) {
                            // enforce memory consistency
                            __threadfence_system();
                            // [index to nRe, batchIdx, tileIdx]
                            const auto coord = idx2crd(flagIdx, fS, fStride);
                            const auto [aEIdx, lEIdx, pIdx] = nRe[cute::get<0>(coord)];
                            auto* __restrict__ packet = heap::advance<1, 1, sizeof(Element)>(pIdx, lEIdx,
                                sP->batchIdx * BLOCK_M);
                            lPd(dA.tQ + (tIdx * dA.tPs + lTQHead++), packet, CAST_TO(cuda::std::byte, tokenIds + (aEIdx * dA.eCap + sP->batchIdx * BLOCK_M)),
                                CAST_TO(cuda::std::byte, activations.data()), sP->tokensM,
                                cute::get<2>(coord), tQHead, aEIdx);
                        }
                    }
                }
                for (uint j = 0; j < residue; ++j) {
                    const auto flagIdx = tIdx + (j + sPt * wSet) * subscriberCount;
                    pfC[flagIdx] = rWSet[j];
                }
            }
        }
    }
}
#endif //ARISTOS_QUEUE_CUH
