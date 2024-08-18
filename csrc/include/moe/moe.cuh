//
// Created by Jonathan on 7/5/24.
//

#ifndef ARISTOS_MOE_CUH
#define ARISTOS_MOE_CUH

#include "definition/tensor.cuh"
#include "util/indexing.cuh"
#include "definition/types.cuh"
#include "algorithm/algorithm.cuh"
#include "util/nifty.cuh"
#include <cuda/atomic>
#include <cuda/cmath>

namespace aristos{
    __constant__ Config moeConfig{};
    /// len = |E|
    __constant__ specType* expertParallelSpec;
    /// len <= |D|
    __constant__ specType* peerTranslation;

    __device__ bool stop = false;
    __device__ unsigned long sequenceNumber = 3;
    //TODO gate, expert fusion and control plane
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
                // We are Subscribers exclusively and Publishers partially
                __syncthreads();
            }
            else{
                /// Exclusive Publishers get the remainder
                // We are Publishers exclusively and Subscribers partially
                __shared__ SenderConfig s;
                if(aristos::block_tid() == 0){
                    s = SenderConfig(moeConfig);
                }
                __threadfence_block();
                __syncthreads();
            }
        }
        // We are Processors exclusively and partial Subscribers and Publishers
        __syncthreads();
    }

    __global__ void backward(){

    }
}
#endif //ARISTOS_MOE_CUH
