//
// Created by azureuser on 1/5/26.
//

#ifndef FLASHMOE_BITSET_CUH
#define FLASHMOE_BITSET_CUH
namespace flashmoe {
    struct __align__(4) BitSet {
        uint storage = 0U;
        __device__ __forceinline__
        uint8_t get(const uint idx) const {
            return storage >> idx & 1U;
        }
        __device__ __forceinline__
        void set(const uint idx) {
            storage |= 1U << idx;
        }
        __device__ __forceinline__
        void clear(const uint idx) {
            storage &= ~(1U << idx);
        }
        __device__ __forceinline__
        void flip(const uint idx) {
            storage ^= (1U << idx);
        }
    };
}
#endif //FLASHMOE_BITSET_CUH