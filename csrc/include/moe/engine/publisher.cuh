//
// Created by Jonathan on 7/18/24.
//

#ifndef ARISTOS_COMMUNICATOR_CUH
#define ARISTOS_COMMUNICATOR_CUH

#include "../util/atomics.cuh"
#include "../definition/memory_layout.cuh"
#include "../definition/values.cuh"

#define GRAND_MASTER 0U

namespace aristos::publisher{
    //TODO use nvstd::function to abstract nvshmem communication operations
    __device__ __inline__ unsigned int doorbell = 0U;
    __device__ __inline__ unsigned int blockade = 0U;
    __device__ __inline__ unsigned int logHead = 0U;
    __device__ __inline__ unsigned int logTail = 0U;
    /// Nifty gadget enabling round-robin queue access
    __device__ __inline__ unsigned int baton = GRAND_MASTER;
    __device__ __inline__ unsigned int syncStages = 0U;
    __device__ __inline__ unsigned int logTailSyncStages = 0U;

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
            warpsWidth = ARISTOS_BLOCK_SIZE_WARP * moeConfig.numP2PPublisherBlocks;
            publisherConfig = PublisherConfig(moeConfig);
            CUTE_UNROLL
            for(unsigned int i = 0; i < moeConfig.worldSize; ++i){
                isRemote[i] = nvshmem_ptr(moeConfig.sHeap, moeConfig.peerTranslation[i]) == nullptr;
            }
        }
        __threadfence_block();
        __syncthreads();

        while(atomicLoad(&stillExecuting)){
            awaitBaton(aristos::block::warpID());
            /// Optimistically complete data transfers without interruption.
            while(atomicLoad(&doorbell) > 0){
                /// I am responsible for making this transfer.
                /// Technically, only a single thread is needed;
                /// however, predicated instruction execution due to warp divergence allegedly
                /// nullifies any purported performance gains of thread-level parallelism, thus we use a warp.
                auto peer = atomicLoad((moeConfig.publisherLog + logHead));
                auto numTokensBytes = moeConfig.publisherLog[logHead + 1] * (moeConfig.embedDim + moeConfig.k + 2);
                if(isRemote[peer]){
                    if(!aristos::warp::laneID()){
                        atomicDec(&doorbell, 1U);
                        atomicAdd(&logHead, 2U);
                        /// Advance token pointer
                        atomicAdd(&publisherConfig.checkpoints[peer], moeConfig.publisherLog[logHead + 1]);
                        atomicExch(&baton, ((aristos::block::warpID() + 1) % warpsWidth));
                    }
                    nvshmemx_putmem_signal_nbi_warp(getTokenPointer(moeConfig.rank, 1, RECEIVE_CELL, publisherConfig.checkpoints[peer]),
                                                    getTokenPointer(peer, 1, SEND_CELL, publisherConfig.checkpoints[peer]),
                                                    numTokensBytes, moeConfig.flags + moeConfig.rank,
                                                    constructSignal(moeConfig.sequenceNumber, header::processed),
                                                    NVSHMEM_SIGNAL_SET, peer);
                }
                else{
                    /// All warps must cooperate for efficient completion of this NVLink transfer
                    if(!aristos::warp::laneID()){
                        atomicExch(&baton, ((aristos::block::warpID() + 1) % warpsWidth));
                    }
                    auto splitNumTokensBytes = cute::ceil_div(numTokensBytes, moeConfig.numP2PPublisherBlocks);
                    if(publisherConfig.localBlockID == moeConfig.numP2PPublisherBlocks - 1 && numTokensBytes % moeConfig.numP2PPublisherBlocks != 0){
                        /// Residual chunk
                        nvshmemx_putmem_block((getTokenPointer(moeConfig.rank, 1, RECEIVE_CELL, publisherConfig.checkpoints[peer]) + (publisherConfig.localBlockID * splitNumTokensBytes)),
                                              (getTokenPointer(peer, 1, SEND_CELL, publisherConfig.checkpoints[peer]) + (publisherConfig.localBlockID * splitNumTokensBytes)),
                                              (numTokensBytes - (splitNumTokensBytes * (publisherConfig.numSubBlocks - 1))), moeConfig.peerTranslation[peer]);
                    }
                    else{
                        nvshmemx_putmem_block((getTokenPointer(moeConfig.rank, 1, RECEIVE_CELL, publisherConfig.checkpoints[peer]) + (publisherConfig.localBlockID * splitNumTokensBytes)),
                                              (getTokenPointer(peer, 1, SEND_CELL, publisherConfig.checkpoints[peer]) + (publisherConfig.localBlockID * splitNumTokensBytes)),
                                              splitNumTokensBytes, moeConfig.peerTranslation[peer]);
                    }
                    if(!aristos::block::threadID()){
                        auto nextStage = atomicLoad(&syncStages) + 1;
                        if((atomicAdd(publisherConfig.syncGrid + peer, 1U) + 1) == (moeConfig.numP2PPublisherBlocks * nextStage)){
                            // I am the last
                            atomicDec(&doorbell, 1U);
                            atomicAdd(&logHead, 2U);
                            atomicAdd(&publisherConfig.checkpoints[peer], moeConfig.publisherLog[logHead + 1]);
                            /// unblock other blocks
                            atomicAdd(&syncStages, 1U);
                            nvshmemx_signal_op(moeConfig.flags + moeConfig.rank,
                                               constructSignal(moeConfig.sequenceNumber, aristos::processed),
                                               NVSHMEM_SIGNAL_SET,
                                               peer);
                        }
                        else{
                            /// Await the last publishing block to update queue head and doorbell
                            while(atomicLoad(&syncStages) != nextStage){
                                __nanosleep(2);
                            }
                        }
                    }
                    __syncwarp();
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
            CUTE_UNROLL
            for(unsigned int i = 0; i < moeConfig.worldSize; ++i){
                isRemote[i] = nvshmem_ptr(moeConfig.sHeap, moeConfig.peerTranslation[i]) == nullptr;
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
                    auto numTokensBytes = moeConfig.publisherLog[logHead + 1] * (moeConfig.embedDim + moeConfig.k + 2);
                    if(!aristos::warp::laneID()){
                        atomicDec(&doorbell, 1U);
                        atomicAdd(&remoteLogHead, 2U);
                        /// Advance token pointer
                        atomicAdd(&publisherConfig.checkpoints[peer], moeConfig.publisherLog[logHead + 1]);
                        /// Eager baton exchange
                        atomicExch(&baton, ((aristos::block::warpID() + 1) % ARISTOS_BLOCK_SIZE_WARP));
                    }
                    nvshmemx_putmem_signal_nbi_warp(getTokenPointer(moeConfig.rank, 1, RECEIVE_CELL, publisherConfig.checkpoints[peer]),
                                                    getTokenPointer(peer, 1, SEND_CELL, publisherConfig.checkpoints[peer]),
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
        __shared__ extern bool isP2P[];
        __shared__ PublisherConfig publisherConfig;
        if(!aristos::grid::blockID()){
            publisherConfig = PublisherConfig(moeConfig);
            CUTE_UNROLL
            for(unsigned int i = 0; i < moeConfig.worldSize; ++i){
                isP2P[i] = nvshmem_ptr(moeConfig.sHeap, moeConfig.peerTranslation[i]) != nullptr;
            }
        }
        __threadfence_block();
        __syncthreads();
        while(atomicLoad(&stillExecuting)){
            awaitBaton(publisherConfig.superBlockID);
            while(atomicLoad(&doorbell) > 0){
                auto logHeadSnapshot = atomicLoad(&publisherConfig.p2pLogHead);
                auto peer = atomicLoad((moeConfig.publisherLog + logHeadSnapshot));
                if(isP2P[peer]){
                    auto numTokensBytes = moeConfig.publisherLog[logHeadSnapshot + 1] * (moeConfig.embedDim + moeConfig.k + 2);
                    auto splitNumTokensBytes = cute::ceil_div(numTokensBytes, publisherConfig.numSubBlocks);
                    /// Leader thread of the super block takes care of business
                    if(publisherConfig.isFirstSubBlock && !aristos::block::threadID()){
                        atomicDec(&doorbell, 1U);
                        atomicAdd(&publisherConfig.p2pLogHead, 2U);
                        /// Advance token pointer
                        atomicAdd(&publisherConfig.checkpoints[peer], moeConfig.publisherLog[publisherConfig.p2pLogHead + 1]);
                        /// Eager baton exchange
                        atomicExch(&baton, ((publisherConfig.superBlockID + 1) % publisherConfig.numSuperBlocks));
                    }
                    if(numTokensBytes % publisherConfig.numSubBlocks != 0 && publisherConfig.isLastSubBlock){
                        /// Residual chunk
                        nvshmemx_putmem_block((getTokenPointer(moeConfig.rank, 1,RECEIVE_CELL,
                                                               publisherConfig.checkpoints[peer])
                                                               + (publisherConfig.localBlockID * splitNumTokensBytes)),
                                              (getTokenPointer(peer, 1, SEND_CELL,
                                                               publisherConfig.checkpoints[peer])
                                                               + (publisherConfig.localBlockID * splitNumTokensBytes)),
                                              (numTokensBytes - (splitNumTokensBytes * (publisherConfig.numSubBlocks - 1))),
                                              moeConfig.peerTranslation[peer]);
                    }
                    else{
                        nvshmemx_putmem_block((getTokenPointer(moeConfig.rank, 1,RECEIVE_CELL,
                                                               publisherConfig.checkpoints[peer])
                                                               + (publisherConfig.localBlockID * splitNumTokensBytes)),
                                              (getTokenPointer(peer, 1, SEND_CELL,
                                                               publisherConfig.checkpoints[peer])
                                                               + (publisherConfig.localBlockID * splitNumTokensBytes)),
                                              splitNumTokensBytes, moeConfig.peerTranslation[peer]);
                    }
                    if(!aristos::block::threadID()){
                        auto nextStage = atomicLoad(&syncStages) + 1;
                        if((atomicAdd(publisherConfig.syncGrid + peer, 1U) + 1) == (moeConfig.numP2PPublisherBlocks * nextStage)){
                            atomicAdd(&syncStages, 1U);
                            nvshmemx_signal_op(moeConfig.flags + moeConfig.rank,
                                               constructSignal(moeConfig.sequenceNumber, aristos::processed),
                                               NVSHMEM_SIGNAL_SET,
                                               moeConfig.peerTranslation[peer]);
                        }
                        else{
                            while(atomicLoad(&syncStages) != nextStage){
                                __nanosleep(2);
                            }
                        }
                    }
                    /// Empirical benchmarks show that using below gives 1.5x better performance
                    /// than without or even sync warp. It remains a mystery why this is the case.
                    __syncthreads();
                }
                else{
                    atomicAdd(&publisherConfig.p2pLogHead, 2U);
                }
            }
        }
    }

    CUTE_DEVICE
    void start(){
        // broadcast()
        if(moeConfig.numP2PPublisherBlocks < ((ARISTOS_NVLINK_THRESHOLD / ARISTOS_BLOCK_SIZE) + 1)){
            /// number of threads is insufficient for remote specialization
            startSingularPublisher();
        }
        else{
            /// local publisher block 0 will service remote communication requests
            /// while other blocks will further specialize to serve parallel P2P, à la NVLink, transfers
            if(!PublisherConfig::getLocalBlockID(moeConfig.numP2PPublisherBlocks)){
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
    void batchNotify(const specType& peerIndex, const specType& numPackets){
        auto nextStage = atomicLoad(&logTailSyncStages) + 1;
        if((atomicAdd(&blockade, 1U) + 1) == moeConfig.numResultChunks * nextStage){
            moeConfig.publisherLog[logTail++] = peerIndex;
            moeConfig.publisherLog[logTail++] = numPackets;
            /// Eagerly notify doorbell
            atomicAdd(&doorbell, 1U);
            atomicAdd(&logTailSyncStages, 1U);
        }
        else{
            while(atomicLoad(&logTailSyncStages) != nextStage){
                __nanosleep(2);
            }
        }
    }
}
#endif //ARISTOS_COMMUNICATOR_CUH
