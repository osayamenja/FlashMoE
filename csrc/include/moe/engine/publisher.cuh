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

    CUTE_DEVICE
    void tryUntilSignal(){
        while(atomicLoad(&doorbell) == 0 && atomicLoad(&stillExecuting)){
            /// Mitigate frivolous consumption of prized memory bandwidth
            __nanosleep(2);
        }
    }

    CUTE_DEVICE
    void startSingularPublisher(){
        __shared__ SenderConfig senderConfig;
        __shared__ unsigned int queueHead;
        __shared__ unsigned int queueTag;
        __shared__ unsigned int warpsWidth;
        __shared__ unsigned int normalizedBID;
        extern __shared__ unsigned int scratchpad[];
        auto checkpoints = scratchpad + moeConfig.worldSize;
        auto isRemote = scratchpad;
        if(!aristos::block_tid()){
            normalizedBID = blockIdx.x - (gridDim.x - moeConfig.numPublisherBlocks);
            warpsWidth = blockSizeWarp * (moeConfig.numPublisherBlocks - 1); // excluding subscriber block
            queueHead = 0;
            queueTag = 0;
            senderConfig = SenderConfig(moeConfig);
            CUTE_UNROLL
            for(unsigned int i = 0; i < moeConfig.worldSize; ++i){
                isRemote[i] = nvshmem_ptr(moeConfig.sHeap, moeConfig.peerTranslation[i]) == NULL;
                checkpoints[i] = 0;
            }
        }
        __threadfence_block();

        while(atomicLoad(&stillExecuting)){
            tryUntilSignal();
            /// Optimistically complete data transfers without interruption.
            while(atomicLoad(&doorbell) > 0){
                /// Use warp instead of thread below due to warp divergence
                if(atomicLoad(&queueTag) == aristos::block_wid()){
                    /// I am responsible for making this transfer.
                    /// Technically, only a single thread is needed;
                    /// however, predicated instruction execution due to warp divergence allegedly
                    /// nullifies any purported performance gains of thread-level parallelism, thus we use a warp.
                    auto peer = atomicLoad((senderConfig.pubQueue + queueHead));
                    auto numTokensBytes = senderConfig.pubQueue[queueHead + 1] * moeConfig.tokenStride;
                    if(isRemote[peer]){
                        if(!aristos::block_lid()){
                            atomicDec(&doorbell, 1U);
                            atomicAdd(&queueHead, 1U);
                            atomicCAS(&queueTag, aristos::block_wid(), ((aristos::block_wid() + 1) % warpsWidth));
                        }
                        nvshmemx_putmem_signal_nbi_warp(getTokenPointer(moeConfig.rank, 1, receiveCell, checkpoints[peer]),
                                                        getTokenPointer(peer, 1, sendCell, checkpoints[peer]),
                                                        numTokensBytes, moeConfig.flags + moeConfig.rank,
                                                        constructSignal(moeConfig.sequenceNumber, header::processed),
                                                        NVSHMEM_SIGNAL_SET, peer);
                    }
                    else{
                        /// All warps must cooperate
                        if(!aristos::block_lid()){
                            atomicCAS(&queueTag, aristos::block_wid(), ((aristos::block_wid() + 1) % warpsWidth));
                        }
                        numTokensBytes = cute::ceil_div(numTokensBytes, moeConfig.numPublisherBlocks);
                        if(normalizedBID == moeConfig.numPublisherBlocks - 1 && numTokensBytes % moeConfig.numPublisherBlocks != 0){
                            nvshmemx_putmem_block((getTokenPointer(moeConfig.rank, 1, receiveCell, checkpoints[peer]) + (normalizedBID * numTokensBytes)),
                                                      (getTokenPointer(peer, 1, sendCell, checkpoints[peer]) + (normalizedBID * numTokensBytes)),
                                                      numTokensBytes % moeConfig.numPublisherBlocks, peer);
                        }
                        else{
                            nvshmemx_putmem_block((getTokenPointer(moeConfig.rank, 1, receiveCell, checkpoints[peer]) + (normalizedBID * numTokensBytes)),
                                                      (getTokenPointer(peer, 1, sendCell, checkpoints[peer]) + (normalizedBID * numTokensBytes)),
                                                      numTokensBytes, peer);
                        }
                        if(!aristos::block_tid()){
                            if(atomicAdd(moeConfig.syncGrid + peer, 1U) % moeConfig.numPublisherBlocks == 0){
                                // I am the last
                                atomicDec(&doorbell, 1U);
                                atomicAdd(&queueHead, 1U);
                                nvshmemx_signal_op(moeConfig.flags + moeConfig.rank,
                                                   constructSignal(moeConfig.sequenceNumber, aristos::processed),
                                                   NVSHMEM_SIGNAL_SET,
                                                   peer);

                            }
                        }
                        if(!aristos::block_wid()){
                            __syncwarp();
                        }
                        /// Other warps are free to go without synchronization!
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
            atomicAdd(&doorbell, 1U);
        }
    }


}
#endif //ARISTOS_COMMUNICATOR_CUH
