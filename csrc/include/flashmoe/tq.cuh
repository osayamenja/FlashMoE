//
// Created by osayamen on 1/7/26.
//

#ifndef FLASHMOE_TQ_CUH
#define FLASHMOE_TQ_CUH
namespace flashmoe
{
    __device__
    enum SchedulerConstants : uint {
        interruptSignal = 0,
        tQHeadGroundState = 0
    };

    // rq
    __device__
    enum ReadySignal : uint {
        observed,
        ready,
        COUNT = 2
    };
    __device__
    struct __align__(8) TQState {
        uint tQTail;
        uint tasks;
    };

    __device__
    struct __align__(8) TQSignal{
        uint signal; // one ahead
        uint interrupt;

        __device__ __forceinline__
        void encodeSig(const uint& sig) {
            signal = sig + 1;
        }
        __device__ __forceinline__
        auto decodeSig() const {
            return signal - 1;
        }
    };
}
#endif //FLASHMOE_TQ_CUH