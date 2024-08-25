//
// Created by Jonathan on 7/18/24.
//

#ifndef ARISTOS_COMMUNICATOR_CUH
#define ARISTOS_COMMUNICATOR_CUH

#include "../util/atomics.cuh"
#include "../definition/memory_layout.cuh"
#include "../definition/values.cuh"

namespace aristos::publisher{
    __device__ unsigned int doorbell = 0U;
    __device__ unsigned int blockade = 1U;
    __device__ unsigned int logHead = 0U;
    __device__ unsigned int logTail = 0U;
    /// Nifty gadget enabling round-robin queue access
    __device__ unsigned int baton = 0U;
    __device__ unsigned int syncStages = 1U;

    CUTE_DEVICE
    void tryUntilSignal(){
        while(atomicLoad(&stillExecuting) && atomicLoad(&doorbell) == 0){
            /// Mitigate frivolous consumption of prized memory bandwidth
            __nanosleep(2);
        }
    }

    CUTE_DEVICE
    void awaitBaton(unsigned int const& tag){
        while(atomicLoad(&baton) != tag){
            __nanosleep(2);
        }
    }

    CUTE_DEVICE
    void startSingularPublisher(){
        //TODO optimize this to mitigate synchronization via append-only log replay
        __shared__ PublisherConfig publisherConfig;
        __shared__ unsigned int warpsWidth;
        __shared__ extern bool isRemote[];
        if(!aristos::block::threadID()){
            warpsWidth = blockSizeWarp * moeConfig.numPublisherBlocks;
            publisherConfig = PublisherConfig(moeConfig);
            CUTE_UNROLL
            for(unsigned int i = 0; i < moeConfig.worldSize; ++i){
                isRemote[i] = nvshmem_ptr(moeConfig.sHeap, moeConfig.peerTranslation[i]) == NULL;
            }
        }
        __threadfence_block();
        __syncthreads();

        while(atomicLoad(&stillExecuting)){
            tryUntilSignal();
            /// Optimistically complete data transfers without interruption.
            while(atomicLoad(&doorbell) > 0){
                /// Use warp instead of thread below due to warp divergence
                if(atomicLoad(&baton) == aristos::block::warpID()){
                    /// I am responsible for making this transfer.
                    /// Technically, only a single thread is needed;
                    /// however, predicated instruction execution due to warp divergence allegedly
                    /// nullifies any purported performance gains of thread-level parallelism, thus we use a warp.
                    auto peer = atomicLoad((moeConfig.publisherLog + logHead));
                    auto numTokensBytes = moeConfig.publisherLog[logHead + 1] * moeConfig.tokenStride;
                    if(isRemote[peer]){
                        if(!aristos::warp::laneID()){
                            atomicDec(&doorbell, 1U);
                            atomicAdd(&logHead, 2U);
                            /// Advance token pointer
                            atomicAdd(&publisherConfig.checkpoints[peer], moeConfig.publisherLog[logHead + 1]);
                            atomicExch(&baton, ((aristos::block::warpID() + 1) % warpsWidth));
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
                            atomicCAS(&baton, aristos::block::warpID(), ((aristos::block::warpID() + 1) % warpsWidth));
                        }
                        auto splitNumTokensBytes = cute::ceil_div(numTokensBytes, moeConfig.numPublisherBlocks);
                        if(publisherConfig.localBlockID == moeConfig.numPublisherBlocks - 1 && numTokensBytes % moeConfig.numPublisherBlocks != 0){
                            /// Residual chunk
                            nvshmemx_putmem_block((getTokenPointer(moeConfig.rank, 1, receiveCell, publisherConfig.checkpoints[peer]) + (publisherConfig.localBlockID * splitNumTokensBytes)),
                                                      (getTokenPointer(peer, 1, sendCell, publisherConfig.checkpoints[peer]) + (publisherConfig.localBlockID * splitNumTokensBytes)),
                                                      splitNumTokensBytes % moeConfig.numPublisherBlocks, peer);
                        }
                        else{
                            nvshmemx_putmem_block((getTokenPointer(moeConfig.rank, 1, receiveCell, publisherConfig.checkpoints[peer]) + (publisherConfig.localBlockID * splitNumTokensBytes)),
                                                      (getTokenPointer(peer, 1, sendCell, publisherConfig.checkpoints[peer]) + (publisherConfig.localBlockID * splitNumTokensBytes)),
                                                      splitNumTokensBytes, peer);
                        }
                        if(!aristos::block::threadID()){
                            if(atomicAdd(publisherConfig.syncGrid + peer, 1U) == (moeConfig.numPublisherBlocks * (syncStages + 1))){
                                // I am the last
                                atomicDec(&doorbell, 1U);
                                atomicAdd(&logHead, 2U);
                                atomicAdd(&publisherConfig.checkpoints[peer], moeConfig.publisherLog[logHead + 1]);
                                atomicAdd(&syncStages, 2U);
                                /// unblock other blocks
                                atomicAdd(publisherConfig.syncGrid + peer, 1U);
                                nvshmemx_signal_op(moeConfig.flags + moeConfig.rank,
                                                   constructSignal(moeConfig.sequenceNumber, aristos::processed),
                                                   NVSHMEM_SIGNAL_SET,
                                                   peer);

                            }
                            else{
                                /// Await the last publishing block to update queue head and doorbell
                                atomicAdd(publisherConfig.syncGrid + peer, 1U);
                                while(atomicLoad(publisherConfig.syncGrid + peer) != (moeConfig.numPublisherBlocks * (syncStages + 2))){
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
    void startPluralRemotePublisher() {
        __shared__ unsigned int remoteLogHead;
        __shared__ PublisherConfig publisherConfig;
        __shared__ extern bool isRemote[];
        if(!aristos::grid::blockID()){
            remoteLogHead = 0;
            publisherConfig = PublisherConfig(moeConfig);
            for(unsigned int i = 0; i < moeConfig.worldSize; ++i){
                isRemote[i] = nvshmem_ptr(moeConfig.sHeap, moeConfig.peerTranslation[i]) == NULL;
            }
        }
        __threadfence_block();
        __syncthreads();
        while(atomicLoad(&stillExecuting)){
            awaitBaton(aristos::block::warpID());
            /// Optimistically complete data transfers without interruption.
            while(atomicLoad(&doorbell) > 0){
                auto peer = atomicLoad((moeConfig.publisherLog + logHead));
                if(isRemote[peer]){
                    auto numTokensBytes = moeConfig.publisherLog[logHead + 1] * moeConfig.tokenStride;
                    if(!aristos::warp::laneID()){
                        atomicDec(&doorbell, 1U);
                        atomicAdd(&remoteLogHead, 2U);
                        /// Advance token pointer
                        atomicAdd(&publisherConfig.checkpoints[peer], moeConfig.publisherLog[logHead + 1]);
                        /// Eager baton exchange
                        atomicExch(&baton, ((aristos::block::warpID() + 1) % blockSizeWarp));
                    }
                    nvshmemx_putmem_signal_nbi_warp(getTokenPointer(moeConfig.rank, 1, receiveCell, publisherConfig.checkpoints[peer]),
                                                    getTokenPointer(peer, 1, sendCell, publisherConfig.checkpoints[peer]),
                                                    numTokensBytes, moeConfig.flags + moeConfig.rank,
                                                    constructSignal(moeConfig.sequenceNumber, header::processed),
                                                    NVSHMEM_SIGNAL_SET, moeConfig.peerTranslation[peer]);
                }
                else{
                    atomicAdd(&remoteLogHead, 2U);
                }
            }
        }
    }

    CUTE_DEVICE
    void startPluralP2PPublisher() {
        //TODO migrate some variables into config
        __shared__ unsigned int numSuperBlocks;
        __shared__ unsigned int numSubBlocks;
        __shared__ unsigned int p2pLogHead;
        __shared__ extern bool isP2P[];
        __shared__ unsigned int superBlockID;
        __shared__ bool isLastSubBlock;
        __shared__ PublisherConfig publisherConfig;
        __shared__ unsigned int lastSubBlockID;
        if(!aristos::grid::blockID()){
            numSuperBlocks = (moeConfig.numPublisherBlocks - 1) / superBlockSize;
            publisherConfig = PublisherConfig(moeConfig);
            p2pLogHead = 0;
            for(unsigned int i = 0; i < moeConfig.worldSize; ++i){
                isP2P[i] = nvshmem_ptr(moeConfig.sHeap, moeConfig.peerTranslation[i]) != NULL;
            }
            superBlockID = publisherConfig.localBlockID / superBlockSize;
            numSubBlocks = superBlockSize;
            lastSubBlockID = ((superBlockID + 1) * superBlockSize) - 1;
            if(publisherConfig.localBlockID >= ((numSuperBlocks - 1)*superBlockSize)){
                superBlockID = (numSuperBlocks - 1);
                numSubBlocks = (moeConfig.numPublisherBlocks - 1) - ((numSuperBlocks - 1) * superBlockSize);
                lastSubBlockID = PublisherConfig::getLastLocalBlockID();
            }
            isLastSubBlock = publisherConfig.localBlockID == lastSubBlockID;
        }
        __threadfence_block();
        __syncthreads();
        while(atomicLoad(&stillExecuting)){
            awaitBaton(superBlockID);
            while(atomicLoad(&doorbell) > 0){
                auto peer = atomicLoad((moeConfig.publisherLog + p2pLogHead));
                if(isP2P[peer]){
                    auto numTokensBytes = moeConfig.publisherLog[p2pLogHead + 1] * moeConfig.tokenStride;
                    auto splitNumTokensBytes = cute::ceil_div(numTokensBytes, numSubBlocks);
                    /// Leader thread of the super block takes care of business
                    if(!publisherConfig.localBlockID && !aristos::block::threadID()){
                        atomicDec(&doorbell, 1U);
                        atomicAdd(&p2pLogHead, 2U);
                        /// Advance token pointer
                        atomicAdd(&publisherConfig.checkpoints[peer], moeConfig.publisherLog[p2pLogHead + 1]);
                        /// Eager baton exchange
                        atomicExch(&baton, ((superBlockID + 1) % numSuperBlocks));
                    }
                    if(numTokensBytes % numSubBlocks != 0 && isLastSubBlock){
                        /// Residual chunk
                        nvshmemx_putmem_block((getTokenPointer(moeConfig.rank, 1,receiveCell,
                                                               publisherConfig.checkpoints[peer])
                                                               + (publisherConfig.localBlockID * splitNumTokensBytes)),
                                              (getTokenPointer(peer, 1, sendCell,
                                                               publisherConfig.checkpoints[peer])
                                                               + (publisherConfig.localBlockID * splitNumTokensBytes)),
                                              (numTokensBytes - (splitNumTokensBytes * (numSubBlocks - 1))),
                                              moeConfig.peerTranslation[peer]);
                    }
                    else{
                        nvshmemx_putmem_block((getTokenPointer(moeConfig.rank, 1,receiveCell,
                                                               publisherConfig.checkpoints[peer])
                                                               + (publisherConfig.localBlockID * splitNumTokensBytes)),
                                              (getTokenPointer(peer, 1, sendCell,
                                                               publisherConfig.checkpoints[peer])
                                                               + (publisherConfig.localBlockID * splitNumTokensBytes)),
                                              splitNumTokensBytes, moeConfig.peerTranslation[peer]);
                    }
                    if(!aristos::block::threadID()){
                        if(atomicAdd(publisherConfig.syncGrid + peer, 1U) == (moeConfig.numPublisherBlocks * (syncStages + 1))){
                            atomicAdd(&syncStages, 1U);
                            nvshmemx_signal_op(moeConfig.flags + moeConfig.rank,
                                               constructSignal(moeConfig.sequenceNumber, aristos::processed),
                                               NVSHMEM_SIGNAL_SET,
                                               moeConfig.peerTranslation[peer]);
                        }
                    }
                    else{
                        __syncthreads();
                    }
                }
                else{
                    atomicAdd(&p2pLogHead, 2U);
                }
            }
        }
    }

    CUTE_DEVICE
    void start(){
        // broadcast()
        if(moeConfig.numPublisherBlocks < ((NVLinkThreshold / blockSize) + 1)){
            /// number of threads is insufficient for remote specialization
            startSingularPublisher();
        }
        else{
            /// local publisher block 0 will service remote communication requests
            /// while other blocks will further specialize to serve parallel P2P, à la NVLink, transfers
            if(!PublisherConfig::getLocalBlockID(moeConfig.numPublisherBlocks)){
                /// remote publisher
                startPluralRemotePublisher();
            }
            else{
                /// P2P publisher
                startPluralP2PPublisher();
            }
        }
    }

    CUTE_DEVICE
    void batchNotify(){
        if(atomicAdd(&blockade, 1U) % moeConfig.numResultChunks == 0){
            //TODO enqueue index using logTail and add num tokens as well
            atomicAdd(&doorbell, 1U);
        }
    }
}
#endif //ARISTOS_COMMUNICATOR_CUH
