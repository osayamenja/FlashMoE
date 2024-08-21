//
// Created by Jonathan on 7/5/24.
//

#ifndef ARISTOS_MOE_CUH
#define ARISTOS_MOE_CUH

#include "definition/tensor.cuh"
#include "util/indexing.cuh"
#include "definition/types.cuh"
#include "algorithm/algorithm.cuh"
#include "definition/packet.cuh"
#include "engine/publisher.cuh"
#include "engine/subscriber.cuh"
#include "engine/processor.cuh"
#include "definition/values.cuh"
#include <cuda/atomic>
#include <cuda/cmath>
#include <cuda/annotated_ptr>

namespace aristos{
    CUTE_DEVICE
    void persistHotPointers(){
        /// Persist global stillExecuting flag
        cuda::associate_access_property(&stillExecuting, cuda::access_property::persisting{});
        /// Persist sequence number
        cuda::associate_access_property(&sequenceNumber, cuda::access_property::persisting{});
        /// Persist symmetric heap flags
        cuda::associate_access_property(moeConfig.flags,
                                        cuda::access_property(moeConfig.flags,
                                                              moeConfig.worldSize*sizeof(flagsType),
                                                              moeConfig.worldSize*sizeof(flagsType),
                                                              cuda::access_property::persisting{}));
        /// Persist book keeping state
        cuda::associate_access_property(moeConfig.pubQueue,
                                        cuda::access_property(moeConfig.pubQueue,
                                                              sizeof(specType)*(moeConfig.worldSize * (moeConfig.numLocalExperts + 1)),
                                                              sizeof(specType)*(moeConfig.worldSize * (moeConfig.numLocalExperts + 1)),
                                                              cuda::access_property::persisting{}));
    }

    template<Matrix M>
    CUTE_DEVICE
    void gate(M activations, M weights, M routing){
        // 1. TODO Compute Fused GEMM and Softmax output routing
        //2. Apply top_idx and return result
        topKMask(activations);
    }

    //TODO add launch bounds
    template<Matrix M, Tensor T>
    __global__ void forward(M activations, T expertsWeights, M gateWeights,
                            M gateOutput, M mappingTensor, M sharedSpec) {
        persistHotPointers();
        gate(activations, gateWeights, gateOutput);
        tokenToPeers(gateOutput, sharedSpec, mappingTensor);
        /// mappingTensor (S, D)

        if (blockIdx.x >= (gridDim.x - moeConfig.numCommBlocks)) {
            /// Exclusive Subscribers get only one block
            if(blockIdx.x == gridDim.x - moeConfig.numCommBlocks){
                // We are Subscribers explicitly and Publishers semantically
                startSubscriber();
            }
            else{
                /// Exclusive Publishers get the remainder
                // We are Publishers explicitly and Subscribers semantically
                startPublisher();
            }
        }
        // We are Processors explicitly and semantic Subscribers and Publishers
        startProcessor();
    }

    __global__ void backward(){

    }
}
#endif //ARISTOS_MOE_CUH
