//
// Created by Jonathan on 7/5/24.
//

#ifndef ARISTOS_MOE_CUH
#define ARISTOS_MOE_CUH

#include "definition/tensor.cuh"
#include "util/indexing.cuh"
#include "definition/types.cuh"
#include "algorithm/algorithm.cuh"
#include <cuda/atomic>
#include <cuda/cmath>

__constant__ aristos::Config moeConfig{};
/// len = |E|
__constant__ int* expertParallelSpec;
/// len <= |D|
__constant__ int* peerTranslation;

namespace aristos{
    __device__ cuda::atomic<bool, cuda::thread_scope_device> stop{};
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
        gate(activations, gateWeights, gateOutput);
        tokenToPeers(gateOutput, sharedSpec, mappingTensor);
        /// mappingTensor (S, D)

        if (blockIdx.x >= (gridDim.x - moeConfig.numCommBlocks)) {
            /// Receivers get only one block
            if(blockIdx.x == gridDim.x - moeConfig.numCommBlocks){
                // We are Receivers
                __syncthreads();
            }
            else{
                /// Senders get the remainder r, where 1 <= r <= numEPPeers
                // We are Senders
                __shared__ SenderConfig s;
                if(aristos::block_tid() == 0){
                    s = SenderConfig(moeConfig);
                }
                __threadfence_block();
                __syncthreads();
            }
        }
        // We are Calculators
        __syncthreads();
    }

    __global__ void backward(){

    }
}
#endif //ARISTOS_MOE_CUH
