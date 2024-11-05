/******************************************************************************
 * Copyright (c) 2024, Osayamen Jonathan Aimuyo.
 ******************************************************************************/

#ifndef ARISTOS_MOE_CUH
#define ARISTOS_MOE_CUH

#include "definition/tensor.cuh"
#include "util/indexing.cuh"
#include "definition/types.cuh"
#include "algorithm/algorithm.cuh"
#include "engine/publisher.cuh"
#include "engine/subscriber.cuh"
#include "engine/processor/processor.cuh"
#include "definition/values.cuh"
#include <cuda/annotated_ptr>

namespace aristos{
    template<Matrix M>
    CUTE_DEVICE
    void gate(M const& activations, M const& weights, M routing){
        // 1. TODO Compute Fused GEMM and Softmax output routing
        //2. Apply top_idx and return result
        topKMask(activations);
    }

    template<Matrix M, Tensor T>
    __global__ void forward(M const& activations, T const& expertsWeights, M const& gateWeights,
                            M gateOutput, M mappingTensor, M sharedSpec) {
        gate(activations, gateWeights, gateOutput);
        tokenToPeers(gateOutput, sharedSpec, mappingTensor);
        /// mappingTensor (S, D)

        if (blockIdx.x >= (gridDim.x - (moeConfig.numP2PPublisherBlocks + 1))) {
            /// Exclusive Subscribers get only one block
            if(blockIdx.x == gridDim.x - (moeConfig.numP2PPublisherBlocks + 1)){
                // We are Subscribers explicitly and Publishers semantically
                subscriber::start();
            }
            else{
                /// Exclusive Publishers get the remainder
                // We are Publishers explicitly and Subscribers semantically
                publisher::start();
            }
        }
        else{
            // We are Processors explicitly and semantic Subscribers and Publishers
            processor::start();
        }
    }

    __global__ void backward(){

    }
}
#endif //ARISTOS_MOE_CUH
