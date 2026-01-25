//
// Created by Osayamen on 1/4/26.
//

#ifndef FLASHMOE_SIGNAL_CUH
#define FLASHMOE_SIGNAL_CUH
namespace flashmoe {
    enum class Topology: uint16_t {
        NVLINK_ONLY,
        MIXED // NVLink + RDMA
    };
    enum class PeerConnectivity {
        remote,
        p2p
    };
    // Captures transitory states of a finite state machine
    enum SignalConstants {
        ground = 0U,
        sequenceStart = 1U,
     };

    enum class PacketStage: uint {
        initial,
        last,
    };

    template<PacketStage p = PacketStage::initial>
    struct __align__(8) SignalPayload {
        static_assert(p == PacketStage::initial);
        uint routedTokens;
        uint16_t totalTilesM;
        uint16_t stateNumber;

        __device__ __forceinline__
        void dump() const {
            printf("{\n\t"
                   "routedTokens: %u,\n\t"
                   "totalTilesM: %u,\n\t"
                   "state: %u"
                   "\n}\n",
                   routedTokens, totalTilesM, stateNumber);
        }
    };

    template<>
    struct __align__(8) SignalPayload<PacketStage::last> {
        uint batchIdx;
        uint16_t tokensM; // <= BLOCK_M
        uint8_t senseBit;
        uint8_t stateNumber;

        __device__ __forceinline__
        void dump() const {
            printf("{\n\t"
                   "batchIdx: %u,\n\t"
                   "tokensM: %u,\n\t"
                   "stateNumber: %u"
                   "\n}\n",
                   batchIdx, tokensM, stateNumber);
        }
    };
}

namespace flashmoe::sbs {
    constexpr int AEE = 0; // allowable early exits
    constexpr int IDZ = AEE + 1;
    // TODO document a proof for the below
    // sequence bit states necessary to break symmetry in forward or backward detection
    // includes ground state
    constexpr int SNS = (2 * (2 + AEE));
    __forceinline__ __host__
    constexpr uint16_t next(const uint16_t& current) {
        return current + 1 == SNS ?
            static_cast<decltype(current)>(sequenceStart) : current + 1;
    }

    __forceinline__ __device__
    constexpr auto ahead(const uint16_t& receivedState, const uint16_t& localState) {
        if (receivedState < sequenceStart) {
            // this is the case, when we observe the ground state
            return false;
        }
        const auto wD =  (SNS - localState) + (receivedState -
            static_cast<decltype(receivedState)>(sequenceStart));
        return (receivedState > localState && ((receivedState - localState) <= IDZ)) ||
            (receivedState < localState && wD <= IDZ);
    }
}
#endif //FLASHMOE_SIGNAL_CUH