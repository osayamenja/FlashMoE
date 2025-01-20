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
        unsigned int subscriberCount = THREADS - 2,
        typename ExpertsTensor,
        typename BiasTensor,
        typename Activations,
        typename Element = typename ExpertsTensor::value_type
    >
    requires(cuda::std::is_same_v<typename ExpertsTensor::value_type, typename BiasTensor::value_type>
        && aristos::Tensor<ExpertsTensor>
        && aristos::Matrix<BiasTensor> && aristos::Matrix<Activations>)
    __device__ __forceinline__
    void start(cuda::std::byte* __restrict__ const& workspace,
        unsigned int* __restrict__ const& interrupt,
        const unsigned int* __restrict__& peerTranslation,
        // remote experts: {actual & local expert idx, peer idx}
        const cuda::std::tuple<uint, uint, uint>* __restrict__& rE,
        const cuda::std::tuple<uint, uint, uint>* __restrict__& nRe, // p2p experts: like above
        const unsigned int& rEl, // number of remote peers
        unsigned int* __restrict__ const& status,
        unsigned int* __restrict__ const& taskCount,
        const Config& mC, const SchedulerConfig& sC,
        Activations const& activations,
        ExpertsTensor const& experts,
        BiasTensor const& biasT){
        static_assert(sizeof(unsigned long long int) == sizeof(flagsType));
        static_assert(sizeof(SignalPayload<>) == sizeof(uint64_t));
        static_assert(sizeof(SignalPayload<PacketStage::last>) == sizeof(uint64_t));
        /*assert(__isShared(workspace) &&
            __isShared(&mC) && // faster to retrieve from shared than constant memory due to strewn accesses
            __isShared(&sC) &&
            __isShared(interrupt) &&
            __isShared(status) && __isShared(taskCount));*/
        
        // each thread gets 64 bytes of workspace
        constexpr auto wSet = 16U; // working set size
        const auto eC = mC.expertCapacity;
        cutlass::AlignedArray<unsigned int, wSet> rWSet{};

        // token indices
        auto* __restrict__ tokenIds = mC.tIdx();

        // tQ things
        auto* __restrict__ gTQHead = sC.tQHeads + threadIdx.x;
        auto lTQHead = 0U; // local tQ Head

        // pointers
        auto* __restrict__ sharedSpace = CAST_TO(unsigned int, workspace);
        auto* __restrict__ sFlags = mC.flags;
        auto* __restrict__ pGB = mC.xMid<Element>(); // post GEMM buffer

        // Constants
        const auto sequenceNumber = seqNo;
        const auto nX = mC.numLocalExperts;

        // first stage
        const auto fSfC = mC.worldSize * nX; // first stage flag count
        const auto fSl = fSfC / subscriberCount + (threadIdx.x < fSfC % subscriberCount);
        const auto fSt = fSl / wSet;
        auto fSp = fSl; // first stage pending

        // second stage: remote
        const auto tilesMc = Config::tiles<BLOCK_M>(mC.expertCapacity);
        const auto sRfC = rEl * tilesMc;
        const auto sRl = sRfC / subscriberCount + (threadIdx.x < sRfC % subscriberCount);
        const auto sRt = sRl / wSet;

        // second stage: p2p
        const auto tilesN = mC.tilesN;
        const auto sPfC = (mC.numExperts - rEl) * tilesMc * tilesN;
        const auto sPl = sPfC / subscriberCount + (threadIdx.x < sPfC % subscriberCount);
        const auto sPt = sPl / wSet;
        const auto iPfS = cute::make_shape(tilesMc, tilesN);
        const auto fS = make_shape(mC.numExperts - rEl, iPfS);
        const auto fStride = make_stride(size(iPfS), cute::make_stride(tilesN, 1));

        auto* __restrict__ ffC = mC.fCheck(); // flags checkpoint
        auto* __restrict__ rfC = ffC + fSfC; // second stage: remote flags checkpoint
        auto* __restrict__ pfC = rfC + sRfC; // second stage: p2p flags checkpoint

        // Decoders
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
                        const auto flagIdx = threadIdx.x + (j + i * wSet) * subscriberCount;
                        sharedSpace[threadIdx.x + j * subscriberCount] = ffC[flagIdx];
                    }
                    #pragma unroll
                    for (uint j = 0; j < wSet; ++j) {
                        rWSet[j] = sharedSpace[threadIdx.x + j * subscriberCount];
                    }
                    // no need for overlapping the next stage, as most likely, only one trip will occur
                    #pragma unroll
                    for (uint j = 0; j < wSet; ++j) {
                        const auto flagIdx = threadIdx.x + (j + i * wSet) * subscriberCount;
                        // main loop
                        if (!rWSet[j]) {
                            // we need to check this flag
                            auto signal = atomicLoad<cuda::thread_scope_system>(
                                CAST_TO(unsigned long long int, flags + flagIdx));
                            const auto sP = CAST_TO(SignalPayload<PacketStage::initial>, &signal);
                            rWSet[j] = sP->seqNo == sequenceNumber;
                            fSp -= rWSet[j];
                            if (rWSet[j]) {
                                // decode the received packet
                                auto expertIdx = flagIdx % nX;
                                auto peerIdx = flagIdx / nX;
                                cuda::std::array weights{
                                    CAST_TO(cuda::std::byte, &experts(expertIdx, 0)),
                                    CAST_TO(cuda::std::byte, &experts(expertIdx, 1))
                                };
                                cuda::std::array bias{
                                    CAST_TO(cuda::std::byte, &biasT(expertIdx, 0)),
                                    CAST_TO(cuda::std::byte, &biasT(expertIdx, 1))
                                };
                                auto* __restrict__ packet = mC.advanceHeap<0, 1, sizeof(Element)>(peerIdx, expertIdx);
                                if (nvshmem_ptr(packet, peerTranslation[peerIdx]) != nullptr) {
                                    // P2P peer
                                    // Enforce consistency
                                    // before decoding the packet
                                    __threadfence_system();
                                    fPd(packet, status, taskCount, expertIdx, pGB, weights, bias,
                                        peerIdx, lTQHead, gTQHead);
                                }
                                else {
                                    // Remote peer
                                    // Below enforces consistency
                                    // before reading the packet
                                    // we cannot decouple the API, unfortunately,
                                    // as the consistency function is internal.
                                    nvshmem_ushort_test(&sP->seqNo, NVSHMEM_CMP_EQ, sequenceNumber);
                                    fRd(packet, status, taskCount, expertIdx, pGB, weights, bias,
                                        peerIdx, lTQHead, gTQHead);
                                }
                            }
                        }
                    }
                    // update checkpoints
                    #pragma unroll
                    for (uint j = 0; j < wSet; ++j) {
                        const auto flagIdx = threadIdx.x + (j + i * wSet) * subscriberCount;
                        ffC[flagIdx] = rWSet[j];
                    }
                }
                // residue
                if (const auto residue = fSl - fSt * wSet) {
                    for (uint j = 0; j < residue; ++j) {
                        // global to shared memory
                        const auto flagIdx = threadIdx.x + (j + fSt * wSet) * subscriberCount;
                        sharedSpace[threadIdx.x + j * subscriberCount] = ffC[flagIdx];
                    }
                    for (uint j = 0; j < residue; ++j) {
                        rWSet[j] = sharedSpace[threadIdx.x + j * subscriberCount];
                    }
                    for (uint j = 0; j < residue; ++j) {
                        const auto flagIdx = threadIdx.x + (j + fSt * wSet) * subscriberCount;
                        // main loop
                        if (!rWSet[j]) {
                            // we need to check this flag
                            auto signal = atomicLoad<cuda::thread_scope_system>(
                                    CAST_TO(unsigned long long int, flags + flagIdx));
                            const auto sP = CAST_TO(SignalPayload<PacketStage::initial>, &signal);
                            rWSet[j] = sP->seqNo == sequenceNumber;
                            fSp -= rWSet[j];
                            if (rWSet[j]) {
                                // decode the received packet
                                auto expertIdx = flagIdx % nX;
                                auto peerIdx = flagIdx / nX;
                                cuda::std::array weights{
                                    CAST_TO(cuda::std::byte, &experts(expertIdx, 0)),
                                    CAST_TO(cuda::std::byte, &experts(expertIdx, 1))
                                };
                                cuda::std::array bias{
                                    CAST_TO(cuda::std::byte, &biasT(expertIdx, 0)),
                                    CAST_TO(cuda::std::byte, &biasT(expertIdx, 1))
                                };

                                if (auto* packet = mC.advanceHeap<0, 1, sizeof(Element)>(peerIdx, expertIdx);
                                    nvshmem_ptr(packet, peerTranslation[peerIdx]) != nullptr) {
                                    // Enforce consistency before decoding the packet
                                    __threadfence_system();
                                    fPd(packet, status, taskCount, expertIdx, pGB, weights, bias,
                                        peerIdx, lTQHead, gTQHead);
                                }
                                else {
                                    nvshmem_ushort_test(&sP->seqNo, NVSHMEM_CMP_EQ, sequenceNumber);
                                    fRd(packet, status, taskCount, expertIdx, pGB, weights, bias,
                                        peerIdx, lTQHead, gTQHead);
                                }
                            }
                        }
                    }
                    for (uint j = 0; j < residue; ++j) {
                        const auto flagIdx = threadIdx.x + (j + fSt * wSet) * subscriberCount;
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
                    const auto flagIdx = threadIdx.x + j * subscriberCount;
                    sharedSpace[threadIdx.x + j * subscriberCount] = rfC[flagIdx];
                }
            }
            for (uint i = 0; i < sRt; ++i) {
                #pragma unroll
                for (uint j = 0; j < wSet; ++j) {
                    rWSet[j] = sharedSpace[threadIdx.x + j * subscriberCount];
                    if (i + 1 < sRt) {
                        const auto flagIdx = threadIdx.x + (j + (i + 1) * wSet) * subscriberCount;
                        sharedSpace[threadIdx.x + j * subscriberCount] = rfC[flagIdx];
                    }
                }
                #pragma unroll
                for (uint j = 0; j < wSet; ++j) {
                    const auto flagIdx = threadIdx.x + (j + i * wSet) * subscriberCount;
                    if (!rWSet[j]) {
                        auto signal = atomicLoad<cuda::thread_scope_system>(
                                CAST_TO(unsigned long long int, flags + flagIdx));
                        // SignalPayload -> {batchIdx, {seqNo, M}}, where M <= BLOCK_M
                        // we do not necessarily need
                        // to transmit batchIdx as we can deduce it locally
                        const auto sP = CAST_TO(SignalPayload<PacketStage::last>, &signal);
                        rWSet[j] = sP->seqNo == sequenceNumber;
                        fSp -= rWSet[j];
                        if (rWSet[j]) {
                            // enforce remote memory consistency
                            nvshmem_ushort_test(&sP->seqNo, NVSHMEM_CMP_EQ, sequenceNumber);
                            const auto [aEIdx, lEIdx, pIdx] = rE[flagIdx / tilesMc];
                            auto* __restrict__ packet = mC.advanceHeap<1, 1, sizeof(Element)>(pIdx, lEIdx,
                                sP->batchIdx * BLOCK_M);
                            lRd(packet, CAST_TO(cuda::std::byte, tokenIds + (aEIdx * eC + sP->batchIdx * BLOCK_M)),
                                CAST_TO(cuda::std::byte, activations.data()), sP->tokensM, lTQHead, gTQHead);
                        }
                    }
                }
                // update checkpoints
                #pragma unroll
                for (uint j = 0; j < wSet; ++j) {
                    const auto flagIdx = threadIdx.x + (j + i * wSet) * subscriberCount;
                    rfC[flagIdx] = rWSet[j];
                }
            }
            if (const auto residue = sRl - sRt * wSet) {
                for (uint j = 0; j < residue; ++j) {
                    const auto flagIdx = threadIdx.x + (j + sRt * wSet) * subscriberCount;
                    sharedSpace[threadIdx.x + j * subscriberCount] = ffC[flagIdx];
                }
                for (uint j = 0; j < residue; ++j) {
                    rWSet[j] = sharedSpace[threadIdx.x + j * subscriberCount];
                }
                for (uint j = 0; j < residue; ++j) {
                    const auto flagIdx = threadIdx.x + (j + sRt * wSet) * subscriberCount;
                    if (!rWSet[j]) {
                        auto signal = atomicLoad<cuda::thread_scope_system>(
                                CAST_TO(unsigned long long int, flags + flagIdx));
                        // SignalPayload -> {batchIdx, {seqNo, M}}, where M <= BLOCK_M
                        const auto sP = CAST_TO(SignalPayload<PacketStage::last>, &signal);
                        rWSet[j] = sP->seqNo == sequenceNumber;
                        fSp -= rWSet[j];
                        if (rWSet[j]) {
                            // enforce remote memory consistency
                            nvshmem_ushort_test(&sP->seqNo, NVSHMEM_CMP_EQ, sequenceNumber);
                            const auto [aEIdx, lEIdx, pIdx] = rE[flagIdx / tilesMc];
                            auto* __restrict__ packet = mC.advanceHeap<1, 1, sizeof(Element)>(pIdx, lEIdx,
                                sP->batchIdx * BLOCK_M);
                            lRd(packet, CAST_TO(cuda::std::byte, tokenIds + (aEIdx * eC + sP->batchIdx * BLOCK_M)),
                                CAST_TO(cuda::std::byte, activations.data()),
                                sP->tokensM, lTQHead, gTQHead);
                        }
                    }
                }
                for (uint j = 0; j < residue; ++j) {
                    const auto flagIdx = threadIdx.x + (j + sRt * wSet) * subscriberCount;
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
                    const auto flagIdx = threadIdx.x + j * subscriberCount;
                    sharedSpace[threadIdx.x + j * subscriberCount] = pfC[flagIdx];
                }
            }
            for (uint i = 0; i < sPt; ++i) {
                #pragma unroll
                for (uint j = 0; j < wSet; ++j) {
                    rWSet[j] = sharedSpace[threadIdx.x + j * subscriberCount];
                    if (i + 1 < sPt) {
                        const auto flagIdx = threadIdx.x + (j + (i + 1) * wSet) * subscriberCount;
                        sharedSpace[threadIdx.x + j * subscriberCount] = pfC[flagIdx];
                    }
                }
                #pragma unroll
                for (uint j = 0; j < wSet; ++j) {
                    const auto flagIdx = threadIdx.x + (j + i * wSet) * subscriberCount;
                    if (!rWSet[j]) {
                        auto signal = atomicLoad<cuda::thread_scope_system>(
                                CAST_TO(unsigned long long int, flags + flagIdx));
                        // SignalPayload -> {batchIdx, {seqNo, M}}, where M <= BLOCK_M
                        const auto sP = CAST_TO(SignalPayload<PacketStage::last>, &signal);
                        rWSet[j] = sP->seqNo == sequenceNumber;
                        fSp -= rWSet[j];
                        if (rWSet[j]) {
                            // enforce memory consistency
                            __threadfence_system();
                            // [index to nRe, batchIdx, tileIdx]
                            const auto coord = idx2crd(flagIdx, fS, fStride);
                            const auto [aEIdx, lEIdx, pIdx] = nRe[cute::get<0>(coord)];
                            auto* __restrict__ packet = mC.advanceHeap<1, 1, sizeof(Element)>(pIdx, lEIdx,
                                sP->batchIdx * BLOCK_M);
                            lPd(packet, CAST_TO(cuda::std::byte, tokenIds + (aEIdx * eC + sP->batchIdx * BLOCK_M)),
                                CAST_TO(cuda::std::byte, activations.data()), sP->tokensM, lTQHead,
                                cute::get<2>(coord), gTQHead);
                        }
                    }
                }
                #pragma unroll
                for (uint j = 0; j < wSet; ++j) {
                    const auto flagIdx = threadIdx.x + (j + i * wSet) * subscriberCount;
                    pfC[flagIdx] = rWSet[j];
                }
            }
            if (const auto residue = sPl - sPt * wSet) {
                for (uint j = 0; j < residue; ++j) {
                    const auto flagIdx = threadIdx.x + (j + sPt * wSet) * subscriberCount;
                    sharedSpace[threadIdx.x + j * subscriberCount] = pfC[flagIdx];
                }
                for (uint j = 0; j < residue; ++j) {
                    rWSet[j] = sharedSpace[threadIdx.x + j * subscriberCount];
                }
                for (uint j = 0; j < residue; ++j) {
                    const auto flagIdx = threadIdx.x + (j + sPt * wSet) * subscriberCount;
                    if (!rWSet[j]) {
                        auto signal = atomicLoad<cuda::thread_scope_system>(
                                CAST_TO(unsigned long long int, flags + flagIdx));
                        // SignalPayload -> {batchIdx, {seqNo, M}}, where M <= BLOCK_M
                        const auto sP = CAST_TO(SignalPayload<PacketStage::last>, &signal);
                        rWSet[j] = sP->seqNo == sequenceNumber;
                        fSp -= rWSet[j];
                        if (rWSet[j]) {
                            // enforce memory consistency
                            __threadfence_system();
                            // [index to nRe, batchIdx, tileIdx]
                            const auto coord = idx2crd(flagIdx, fS, fStride);
                            const auto [aEIdx, lEIdx, pIdx] = nRe[cute::get<0>(coord)];
                            auto* __restrict__ packet = mC.advanceHeap<1, 1, sizeof(Element)>(pIdx, lEIdx,
                                sP->batchIdx * BLOCK_M);
                            lPd(packet, CAST_TO(cuda::std::byte, tokenIds + (aEIdx * eC + sP->batchIdx * BLOCK_M)),
                                CAST_TO(cuda::std::byte, activations.data()), sP->tokensM, lTQHead,
                                cute::get<2>(coord), gTQHead);
                        }
                    }
                }
                for (uint j = 0; j < residue; ++j) {
                    const auto flagIdx = threadIdx.x + (j + sPt * wSet) * subscriberCount;
                    pfC[flagIdx] = rWSet[j];
                }
            }
        }
    }
}
#endif //ARISTOS_QUEUE_CUH
