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
        typename Element = typename ExpertsTensor::value_type
    >
    requires(cuda::std::is_same_v<typename ExpertsTensor::value_type, typename BiasTensor::value_type>
        && aristos::Tensor<ExpertsTensor> && aristos::Tensor<BiasTensor>)
    __device__ __forceinline__
    void start(cuda::std::byte* __restrict__ const& workspace,
        unsigned int* __restrict__ const& interrupt,
        const unsigned int* __restrict__& peerTranslation,
        unsigned int* __restrict__ const& status,
        unsigned int* __restrict__ const& taskCount,
        const Config& mC, const SchedulerConfig& sC,
        ExpertsTensor const& experts,
        BiasTensor const& biasT){
        static_assert(sizeof(unsigned long long int) == sizeof(flagsType));
        /*assert(__isShared(workspace) &&
            __isShared(&mC) && // faster to retrieve from shared than constant memory due to strewn accesses
            __isShared(&sC) &&
            __isShared(interrupt) &&
            __isShared(status) && __isShared(taskCount));*/
        
        // each thread gets 64 bytes of workspace
        constexpr auto wSet = 16U;
        cutlass::AlignedArray<unsigned int, wSet> rWSet{};

        // tQ things
        auto* __restrict__ gTQHead = sC.tQHeads + threadIdx.x;
        auto lTQHead = 0U;

        // pointers
        auto* __restrict__ sharedSpace = CAST_TO(unsigned int, workspace);
        auto* __restrict__ flags = mC.flags;
        auto* __restrict__ pGB = mC.xMid<Element>();

        // Constants
        const auto sequenceNumber = seqNo;
        const auto nX = mC.numLocalExperts;

        // first stage
        const auto fSfC = mC.worldSize * nX; // first stage flag count
        const auto fSl = fSfC / subscriberCount + (threadIdx.x < fSfC % subscriberCount);
        const auto fSt = fSl / wSet;
        auto fSp = fSl; // first stage pending

        //second stage
        const auto sSfC = mC.tilesM * mC.tilesN;
        const auto sl = sSfC / subscriberCount + (threadIdx.x < sSfC % subscriberCount);
        const auto st = sl / wSet;

        const auto* __restrict__ ffC = mC.fCheck(); // flags checkpoint
        const auto* __restrict__ sfC = mC.fCheck() + fSfC; // second stage flags checkpoint

        // Decoders
        packet::Decoder<PacketStage::initial, PeerConnectivity::p2p, Element> fPd{};
        packet::Decoder<PacketStage::initial, PeerConnectivity::remote, Element> fRd{};
        packet::Decoder<PacketStage::final, PeerConnectivity::p2p> lPd{};
        packet::Decoder<PacketStage::final, PeerConnectivity::remote> lRd{};

        while (!atomicLoad<cuda::thread_scope_block>(interrupt)) {
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
                            const auto sP = CAST_TO(SignalPayload, &signal);
                            rWSet[j] = sP->first == sequenceNumber;
                            fSp -= rWSet[j];
                            if (rWSet[j]) {
                                // below enforces consistency before reading the packet
                                // we cannot decouple the API, unfortunately,
                                // as the consistency function is internal.
                                nvshmem_uint_test(CAST_TO(unsigned int, flags + flagIdx),
                                    NVSHMEM_CMP_EQ, sequenceNumber);
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
                                    fPd(packet, status, taskCount, expertIdx, pGB, weights, bias,
                                        peerIdx, lTQHead, gTQHead);
                                }
                                else {
                                    // Remote peer
                                    fRd(packet, status, taskCount, expertIdx, pGB, weights, bias,
                                        peerIdx, lTQHead, gTQHead);
                                }
                            }
                        }
                    }
                    // restore state
                    #pragma unroll
                    for (uint j = 0; j < wSet; ++j) {
                        const auto flagIdx = threadIdx.x + (j + i * wSet) * subscriberCount;
                        rWSet[j] = ffC[flagIdx];
                    }
                }
                // residue
                for (uint j = 0; j < wSet; ++j) {
                    // global to shared memory
                    const auto flagIdx = threadIdx.x + (j + fSt * wSet) * subscriberCount;
                    sharedSpace[threadIdx.x + j * subscriberCount] = ffC[flagIdx];
                }
                for (uint j = 0; j < wSet; ++j) {
                    rWSet[j] = sharedSpace[threadIdx.x + j * subscriberCount];
                }
                for (uint j = 0; j < wSet; ++j) {
                    const auto flagIdx = threadIdx.x + (j + fSt * wSet) * subscriberCount;
                    // main loop
                    if (!rWSet[j]) {
                        // we need to check this flag
                        auto signal = atomicLoad<cuda::thread_scope_system>(
                                CAST_TO(unsigned long long int, flags + flagIdx));
                        const auto sP = CAST_TO(SignalPayload, &signal);
                        rWSet[j] = sP->first == sequenceNumber;
                        fSp -= rWSet[j];
                        if (rWSet[j]) {
                            nvshmem_uint_test(CAST_TO(unsigned int, flags + flagIdx),
                            NVSHMEM_CMP_EQ, sequenceNumber);
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
                                fPd(packet, status, taskCount, expertIdx, pGB, weights, bias,
                                    peerIdx, lTQHead, gTQHead);
                            }
                            else {
                                fRd(packet, status, taskCount, expertIdx, pGB, weights, bias,
                                    peerIdx, lTQHead, gTQHead);
                            }
                        }
                    }
                }
                for (uint j = 0; j < wSet; ++j) {
                    const auto flagIdx = threadIdx.x + (j + fSt * wSet) * subscriberCount;
                    rWSet[j] = ffC[flagIdx];
                }
            }

            // second stage
            // flags spanning each tile
            for (uint i = 0; i < st; ++i) {
                #pragma unroll
                for (uint j = 0; j < wSet; ++j) {
                    sharedSpace[threadIdx.x + j * subscriberCount] = sfC[threadIdx.x + (j + i * wSet) * subscriberCount];
                }
                #pragma unroll
                for (uint j = 0; j < wSet; ++j) {
                    rWSet[j] = sharedSpace[threadIdx.x + j * subscriberCount];
                }
                #pragma unroll
                for (uint j = 0; j < wSet; ++j) {
                    const auto flagIdx = threadIdx.x + (j + i * wSet) * subscriberCount;
                    if (!rWSet[j]) {
                        auto signal = atomicLoad<cuda::thread_scope_system>(
                                CAST_TO(unsigned long long int, flags + flagIdx));
                        const auto sP = CAST_TO(SignalPayload, &signal);
                        rWSet[j] = sP->first == sequenceNumber;
                        fSp -= rWSet[j];
                    }
                }
            }

        }
    }
}
#endif //ARISTOS_QUEUE_CUH
