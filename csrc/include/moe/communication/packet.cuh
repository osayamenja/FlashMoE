//
// Created by Jonathan on 7/4/24.
//

#ifndef ARISTOS_PACKET_CUH
#define ARISTOS_PACKET_CUH
namespace aristos{
    enum header: unsigned int {
        NOOP = 0,
        processed = 0,
        shouldProcess = 1,
        begin = 2
    };

    CUTE_HOST_DEVICE
    uint64_t constructSignal(unsigned long seqNo, header tag){
        return seqNo + tag;
    }
}
#endif //ARISTOS_PACKET_CUH
