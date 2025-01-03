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
        unsigned int* __restrict__ const& status,
        unsigned int* __restrict__ const& taskCount,
        const Config& mC, const SchedulerConfig& sC,
        ExpertsTensor const& experts){
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
        const auto* __restrict__ fC = mC.fCheck(); // flags checkpoint
        auto* __restrict__ pGB = mC.xMid<Element>();

        // Constants
        const auto sequenceNumber = seqNo;
        const auto nX = mC.numLocalExperts;

        // first stage
        const auto fSfC = mC.worldSize * nX; // first stage flag count
        const auto fSl = fSfC / subscriberCount + (threadIdx.x < fSfC % subscriberCount);
        const auto fSt = fSl / wSet;
        auto fSp = fSl; // first stage pending

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
                        sharedSpace[threadIdx.x + j * subscriberCount] = fC[flagIdx];
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
                            rWSet[j] = nvshmem_uint_test(CAST_TO(unsigned int, flags + flagIdx),
                                NVSHMEM_CMP_EQ, sequenceNumber);
                            fSp -= rWSet[j];
                        }
                    }

                    // restore state
                    #pragma unroll
                    for (uint j = 0; j < wSet; ++j) {
                        const auto flagIdx = threadIdx.x + (j + i * wSet) * subscriberCount;
                        rWSet[j] = fC[flagIdx];
                    }
                }

                // residue
                for (uint j = 0; j < wSet; ++j) {
                    // global to shared memory
                    const auto flagIdx = threadIdx.x + (j + fSt * wSet) * subscriberCount;
                    sharedSpace[threadIdx.x + j * subscriberCount] = fC[flagIdx];
                }
                for (uint j = 0; j < wSet; ++j) {
                    rWSet[j] = sharedSpace[threadIdx.x + j * subscriberCount];
                }
                for (uint j = 0; j < wSet; ++j) {
                    const auto flagIdx = threadIdx.x + (j + fSt * wSet) * subscriberCount;
                    // main loop
                    if (!rWSet[j]) {
                        // we need to check this flag
                        rWSet[j] = nvshmem_uint_test(CAST_TO(unsigned int, flags + flagIdx),
                            NVSHMEM_CMP_EQ, sequenceNumber);
                        fSp -= rWSet[j];
                        if (rWSet[j]) {
                            // decode the received packet
                            auto expertIdx = flagIdx % nX;
                            auto peerIdx = flagIdx / nX;
                            cuda::std::array<cuda::std::byte*, GEMMs> weights{};
                            cuda::std::array<cuda::std::byte*, GEMMs> bias{};
                        }
                    }
                }
                for (uint j = 0; j < wSet; ++j) {
                    const auto flagIdx = threadIdx.x + (j + fSt * wSet) * subscriberCount;
                    rWSet[j] = fC[flagIdx];
                }
            }


        }
    }
}
#endif //ARISTOS_QUEUE_CUH
