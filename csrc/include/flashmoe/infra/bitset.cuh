//
// Created by Osayamen on 1/5/26.
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

    /// Computes precise number of integers needed to represent a consecutive set of bits
    /// each of T threads has stride ownership of a single bit
    /// and requires an integer to store 32 of such bits.
    template<
        unsigned int T
    >
    __device__ __forceinline__
    constexpr uint nSI(const unsigned int& numBits) {
        constexpr unsigned int integerBitWidth = 32U;
        constexpr auto width = integerBitWidth * T;
        return (numBits / width) * T + cuda::std::min(numBits % width, T);
    }

    __host__ __forceinline__
    constexpr uint nSI(const unsigned int& numBits, const uint& T) {
        constexpr unsigned int integerBitWidth = 32U;
        const auto width = integerBitWidth * T;
        return (numBits / width) * T + cuda::std::min(numBits % width, T);
    }
}
#endif //FLASHMOE_BITSET_CUH