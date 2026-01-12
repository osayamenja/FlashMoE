//
// Created by azureuser on 1/5/26.
//

#ifndef FLASHMOE_DQ_CUH
#define FLASHMOE_DQ_CUH
namespace flashmoe {
    enum class DQType {
        stride,
        block
    };
}
namespace flashmoe::DQ {
    /// Decoupled Queue, comprises tail and doorbell
    template<
        DQType dqt = DQType::stride,
        int nQ = 0
    >
        __device__ __forceinline__
    constexpr auto next(const int& prev, const int& slot) {
        if constexpr (dqt == DQType::stride) {
            return prev + slot * nQ;
        }
        return prev + slot;
    }
    template<
        DQType dqt = DQType::stride,
        int nQ
    >
    __device__ __forceinline__
    constexpr auto sNext(const int& slot) {
        return next<dqt, nQ>(0, slot);
    }
}
#endif //FLASHMOE_DQ_CUH