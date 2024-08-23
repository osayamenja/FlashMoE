//
// Created by Jonathan on 7/18/24.
//

#ifndef ARISTOS_COMMUNICATOR_CUH
#define ARISTOS_COMMUNICATOR_CUH

#include "../util/atomics.cuh"
#include "../definition/memory_layout.cuh"
#include "../definition/values.cuh"

namespace aristos{
    __device__ unsigned int doorbell = 0U;
    __device__ unsigned int blockade = 1U;
    __device__ unsigned int queueHead = 0U;
    __device__ unsigned int queueTail = 0U;
    __device__ unsigned int queueTag = 0U;

    CUTE_DEVICE
    void tryUntilSignal(){
        while(atomicLoad(&doorbell) == 0 && atomicLoad(&stillExecuting)){
            /// Mitigate frivolous consumption of prized memory bandwidth
            __nanosleep(2);
        }
    }

    CUTE_DEVICE
    void startSingularPublisher(){
        __shared__ PublisherConfig publisherConfig;
        __shared__ unsigned int warpsWidth;
        extern __shared__ unsigned int scratchpad[];
        auto isRemote = scratchpad;
        if(!aristos::block::threadID()){
            warpsWidth = blockSizeWarp * moeConfig.numPublisherBlocks;
            publisherConfig = PublisherConfig(moeConfig);
            CUTE_UNROLL
            for(unsigned int i = 0; i < moeConfig.worldSize; ++i){
                isRemote[i] = nvshmem_ptr(moeConfig.sHeap, moeConfig.peerTranslation[i]) == NULL;
            }
        }
        __threadfence_block();

        while(atomicLoad(&stillExecuting)){
            tryUntilSignal();
            /// Optimistically complete data transfers without interruption.
            while(atomicLoad(&doorbell) > 0){
                /// Use warp instead of thread below due to warp divergence
                if(atomicLoad(&queueTag) == aristos::block::warpID()){
                    /// I am responsible for making this transfer.
                    /// Technically, only a single thread is needed;
                    /// however, predicated instruction execution due to warp divergence allegedly
                    /// nullifies any purported performance gains of thread-level parallelism, thus we use a warp.
                    auto peer = atomicLoad((moeConfig.pubQueue + queueHead));
                    auto numTokensBytes = moeConfig.pubQueue[queueHead + 1] * moeConfig.tokenStride;
                    if(isRemote[peer]){
                        if(!aristos::warp::laneID()){
                            atomicDec(&doorbell, 1U);
                            atomicAdd(&queueHead, 1U);
                            /// Advance token pointer
                            atomicAdd(&publisherConfig.checkpoints[peer], moeConfig.pubQueue[queueHead + 1]);
                            atomicCAS(&queueTag, aristos::block::warpID(), ((aristos::block::warpID() + 1) % warpsWidth));
                        }
                        nvshmemx_putmem_signal_nbi_warp(getTokenPointer(moeConfig.rank, 1, receiveCell, publisherConfig.checkpoints[peer]),
                                                        getTokenPointer(peer, 1, sendCell, publisherConfig.checkpoints[peer]),
                                                        numTokensBytes, moeConfig.flags + moeConfig.rank,
                                                        constructSignal(moeConfig.sequenceNumber, header::processed),
                                                        NVSHMEM_SIGNAL_SET, peer);
                    }
                    else{
                        /// All warps must cooperate for efficient completion of this NVLink transfer
                        if(!aristos::warp::laneID()){
                            atomicCAS(&queueTag, aristos::block::warpID(), ((aristos::block::warpID() + 1) % warpsWidth));
                        }
                        numTokensBytes = cute::ceil_div(numTokensBytes, moeConfig.numPublisherBlocks);
                        if(publisherConfig.localBlockID == moeConfig.numPublisherBlocks - 1 && numTokensBytes % moeConfig.numPublisherBlocks != 0){
                            /// Residual chunk
                            nvshmemx_putmem_block((getTokenPointer(moeConfig.rank, 1, receiveCell, publisherConfig.checkpoints[peer]) + (publisherConfig.localBlockID * numTokensBytes)),
                                                      (getTokenPointer(peer, 1, sendCell, publisherConfig.checkpoints[peer]) + (publisherConfig.localBlockID * numTokensBytes)),
                                                      numTokensBytes % moeConfig.numPublisherBlocks, peer);
                        }
                        else{
                            nvshmemx_putmem_block((getTokenPointer(moeConfig.rank, 1, receiveCell, publisherConfig.checkpoints[peer]) + (publisherConfig.localBlockID * numTokensBytes)),
                                                      (getTokenPointer(peer, 1, sendCell, publisherConfig.checkpoints[peer]) + (publisherConfig.localBlockID * numTokensBytes)),
                                                      numTokensBytes, peer);
                        }
                        if(!aristos::block::threadID()){
                            if(atomicAdd(publisherConfig.syncGrid + peer, 1U) % moeConfig.numPublisherBlocks == 0){
                                // I am the last
                                atomicDec(&doorbell, 1U);
                                atomicAdd(&queueHead, 1U);
                                atomicAdd(publisherConfig.syncGrid + peer, 1U);
                                atomicAdd(&publisherConfig.checkpoints[peer], moeConfig.pubQueue[queueHead + 1]);
                                nvshmemx_signal_op(moeConfig.flags + moeConfig.rank,
                                                   constructSignal(moeConfig.sequenceNumber, aristos::processed),
                                                   NVSHMEM_SIGNAL_SET,
                                                   peer);

                            }
                            else{
                                atomicAdd(publisherConfig.pubBarrier + peer, 1U);
                                /// Await the last publishing block to update queue head and doorbell
                                while(atomicLoad(publisherConfig.pubBarrier) % moeConfig.numPublisherBlocks == 0){
                                    __nanosleep(2);
                                }
                            }
                        }
                        else{
                            __syncthreads();
                        }
                    }
                }
                else{
                    /// Note that this sleep is unlike backoff as there is no contention for access due to the
                    /// static round-robin allocation.
                    /// Instead, the below delay mitigates frivolous consumption of scarce memory bandwidth
                    __nanosleep(2);
                }
            }
        }
    }

    CUTE_DEVICE
    void startPublisher(){
        // broadcast()
        if(floor(blockSize * (moeConfig.numPublisherBlocks - 1)) == 0){
            /// number of threads is insufficient for remote specialization
            startSingularPublisher();
        }
        else{
            /// Remote specialization
            while(atomicLoad(&stillExecuting)){
                tryUntilSignal();
                while(atomicLoad(&stillExecuting) && atomicLoad(&doorbell) > 0){

                }
            }
        }
    }

    CUTE_DEVICE
    void batchNotifyPublisher(){
        if(atomicAdd(&blockade, 1U) % moeConfig.numResultChunks == 0){
            //TODO enqueue index using queueTail
            atomicAdd(&doorbell, 1U);
        }
    }


}
#endif //ARISTOS_COMMUNICATOR_CUH
