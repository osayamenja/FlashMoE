//
// Created by Jonathan on 7/18/24.
//

#ifndef ARISTOS_TYPES_CUH
#define ARISTOS_TYPES_CUH

#include "memory_layout.cuh"
#include <cuda/cmath>

namespace aristos{
    /// 2 Bytes
    using medium_int = unsigned short;
    namespace cg = cooperative_groups;

    /// Types can be much smaller than int
    struct SenderConfig{
        uint64_t* flags;
        void* symmetricHeap;
        std::byte* symmetricHeap_Byte;
        unsigned int* syncGrid;
        const unsigned int worldSize;
        const unsigned long peerOffset;
        const unsigned int rank;
        const unsigned long sequenceNumber;
        const unsigned long blocksToPeers;
        const unsigned int peerStripeLength;
        const unsigned int intraPeerIndex;
        const unsigned int numExperts;
        const unsigned int firstPeer;
        const unsigned long singularPacketBytes;
        CUTE_HOST_DEVICE
        SenderConfig(uint64_t* _flags,
                     void* _symmetricHeap,
                     unsigned int* _syncGrid,
                     const unsigned int _worldSize,
                     const unsigned int _rank,
                     const unsigned long _sequenceNumber,
                     const unsigned int capacity,
                     const unsigned int k,
                     const unsigned int embedDim,
                     const unsigned int embedPrecision,
                     const unsigned int _numExperts):
        flags(_flags),
        symmetricHeap(_symmetricHeap),
        symmetricHeap_Byte(static_cast<std::byte*>(_symmetricHeap)),
        syncGrid(_syncGrid),
        worldSize(_worldSize),
        peerOffset(symmetric_heap_peer_offset(capacity, k,
                                              (embedDim * embedPrecision)) / micro_header_bytes),
        rank(_rank),
        sequenceNumber(_sequenceNumber),
        blocksToPeers(max((gridDim.z - 1) * (gridDim.x * gridDim.y) / worldSize, 1)),
        peerStripeLength(cuda::ceil_div(1UL, blocksToPeers)),
        intraPeerIndex(aristos::bid() - ((aristos::bid() / blocksToPeers) * blocksToPeers)),
        numExperts(_numExperts),
        firstPeer(static_cast<int>(aristos::bid() / blocksToPeers)),
        singularPacketBytes(payload_bytes((embedDim * embedPrecision), 1)){}
    };

}
#endif //ARISTOS_TYPES_CUH
