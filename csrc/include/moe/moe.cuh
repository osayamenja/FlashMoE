//
// Created by Jonathan on 7/5/24.
//

#ifndef ARISTOS_MOE_CUH
#define ARISTOS_MOE_CUH

#include "definition/tensor.cuh"
#include "util/indexing.cuh"
#include "definition/types.cuh"
#include "algorithm/algorithm.cuh"
#include "engine/publisher.cuh"
#include "engine/subscriber.cuh"
#include "engine/processor.cuh"
#include "definition/values.cuh"
#include <cuda/annotated_ptr>

namespace aristos{
    CUTE_DEVICE
    void persistHotPointers(){
        /// Persist global interrupt
        cuda::associate_access_property(&stillExecuting, cuda::access_property::persisting{});
        /// Persist sequence number
        cuda::associate_access_property(&sequenceNumber, cuda::access_property::persisting{});
        /// Persist publisher doorbell
        cuda::associate_access_property(&doorbell, cuda::access_property::persisting{});
        cuda::associate_access_property(&blockade, cuda::access_property::persisting{});

        /// Persist symmetric heap flags
        cuda::associate_access_property(moeConfig.flags,
                                        cuda::access_property(moeConfig.flags,
                                                              moeConfig.worldSize*sizeof(flagsType),
                                                              moeConfig.worldSize*sizeof(flagsType),
                                                              cuda::access_property::persisting{}));
        /// Persist book keeping state
        cuda::associate_access_property(moeConfig.bookKeeping,
                                        cuda::access_property(moeConfig.bookKeeping,
                                                              sizeof(specType)*((moeConfig.worldSize * moeConfig.numExperts * 2) + (moeConfig.worldSize * 2) + (moeConfig.numExperts)),
                                                              sizeof(specType)*((moeConfig.worldSize * moeConfig.numExperts * 2) + (moeConfig.worldSize * 2) + (moeConfig.numExperts)),
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
    __global__ void forward(M const& activations, T const& expertsWeights, M const& gateWeights,
                            M gateOutput, M mappingTensor, M sharedSpec) {
        persistHotPointers();
        gate(activations, gateWeights, gateOutput);
        tokenToPeers(gateOutput, sharedSpec, mappingTensor);
        /// mappingTensor (S, D)

        if (blockIdx.x >= (gridDim.x - (moeConfig.numPublisherBlocks + 1))) {
            /// Exclusive Subscribers get only one block
            if(blockIdx.x == gridDim.x - (moeConfig.numPublisherBlocks + 1)){
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
