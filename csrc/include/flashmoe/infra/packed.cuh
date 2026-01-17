//
// Created by azureuser on 1/5/26.
//

#ifndef FLASHMOE_PACKED_CUH
#define FLASHMOE_PACKED_CUH
namespace flashmoe {
    struct __align__(8) TPS {
        uint32_t tokenIdx;
        float probability;
    };
    struct __align__(8) SoftmaxStatePacked {
        uint32_t m_bits;   // raw bits of mI (fp32)
        uint32_t d_bits;   // raw bits of dI (fp32), sign bit used as signal
    };
    __device__ __forceinline__
    auto to_softmax_state(const unsigned long long int& raw) {
        SoftmaxStatePacked s{};
        s.m_bits = static_cast<uint32_t>(raw & 0xFFFFFFFFull);
        s.d_bits = static_cast<uint32_t>(raw >> 32);
        return s;
    }
    // Pack: signal is implicitly always 1
    __device__ __forceinline__
    SoftmaxStatePacked pack_state(const float& m, const float& d) {
        SoftmaxStatePacked p{};
        p.m_bits = __float_as_uint(m);

        uint32_t db = __float_as_uint(d);
        db |= 0x80000000u;          // force signal bit (sign) = 1
        p.d_bits = db;

        return p;
    }
    // Unpack: returns (m, d)
    __device__ __forceinline__
    void unpack_state(const SoftmaxStatePacked &p, float &m, float &d) {
        m = __uint_as_float(p.m_bits);
        d = __uint_as_float(p.d_bits & 0x7FFFFFFFu);  // clear sign bit
    }
    __device__ __forceinline__
    bool has_payload_arrived(const SoftmaxStatePacked &p) {
        return (p.d_bits >> 31) != 0;
    }

    struct __align__(8) RingTopKPayload {
        uint32_t sV = 0U; // raw fp32 bits
        uint32_t sIdx = 0U; // raw index bits, sign bit used as signal
    };
    __device__ __forceinline__
    auto to_tk_payload(const unsigned long long int& raw) {
        RingTopKPayload r{};
        r.sV = static_cast<uint32_t>(raw & 0xFFFFFFFFull);
        r.sIdx = static_cast<uint32_t>(raw >> 32);
        return r;
    }
    // Pack: signal is implicitly always 1
    __device__ __forceinline__
    auto pack_tk_payload(const float& sV, const uint32_t& sIdx) {
        RingTopKPayload r{};
        r.sV = __float_as_uint(sV);

        r.sIdx = sIdx;
        r.sIdx = (sIdx & 0x7FFFFFFFu) | 0x80000000u;  // keep 31 bits, set signal bit
        return r;
    }
    // Unpack: returns (m, d)
    __device__ __forceinline__
    void unpack_tk_payload(const RingTopKPayload &p, float &sV, uint32_t &sIdx) {
        sV = __uint_as_float(p.sV);
        sIdx = p.sIdx & 0x7FFFFFFFu; // clear sign bit
    }
    __device__ __forceinline__
    bool has_payload_arrived(const RingTopKPayload &p) {
        return (p.sIdx >> 31) != 0;
    }
}
#endif //FLASHMOE_PACKED_CUH