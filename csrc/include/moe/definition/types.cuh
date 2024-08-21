//
// Created by Jonathan on 7/18/24.
//

#ifndef ARISTOS_TYPES_CUH
#define ARISTOS_TYPES_CUH

#include "memory_layout.cuh"

namespace aristos{
    using maxPrecision = float;
    using specType = unsigned int;
    using flagsType = uint64_t;
    /// Okay, I know the type name is relatively misleading, but we had no other choice due to
    /// NVIDIA native atomics only accepting int types.
    using AtomicBoolType = unsigned int;

    /// GEMM Block spec
    extern constexpr unsigned int bM = 128;
    extern constexpr unsigned int bN = 128;
    extern constexpr unsigned int bK = 8;
    extern constexpr unsigned int bP = 1; // pipeline stages, have to use 1 due to shared mem constraints
    extern constexpr unsigned int blockSize = 128; // 256 is too high, since an SM can only hold <= 2048 threads
    /// empirical threshold of threads to saturate NVLink bandwidth
    extern constexpr unsigned int NVLinkBlockReq = 4096;

    namespace cg = cooperative_groups;

    struct Config{
        void* sHeap;
        void* flags;
        void* pubQueue;
        // Expert parallel World Size
        unsigned int worldSize;
        unsigned int capacity;
        unsigned long peerOffset;
        unsigned int rank;
        unsigned long sequenceNumber;
        unsigned int seqLen;
        unsigned int numExperts;
        unsigned int numLocalExperts;
        unsigned int k;
        unsigned int embedDim;
        unsigned int embedPrecision;
        unsigned int numCommBlocks;
        unsigned int numResultChunks;

        CUTE_HOST_DEVICE
        Config() = default;

        CUTE_HOST_DEVICE
        Config(void* _symmetricHeap,
               void* _flags,
               const unsigned int _worldSize,
               const unsigned int _rank,
               const unsigned long _sequenceNumber,
               const unsigned int _capacityFactor,
               const unsigned int _k,
               const unsigned int _embedDim,
               const unsigned int _embedPrecision,
               const unsigned int _numExperts,
               const unsigned int _numLocalExperts,
               const unsigned int _seqLen):
                sHeap(_symmetricHeap),
                flags(_flags),
                worldSize(_worldSize),
                capacity(getCapacity(_seqLen, _numExperts, _capacityFactor, _k)),
                peerOffset(symmetric_heap_peer_offset(capacity, _k,
                                              (_embedDim * _embedPrecision)) / micro_header_bytes),
                rank(_rank),
                sequenceNumber(_sequenceNumber),
                seqLen(_seqLen),
                numExperts(_numExperts),
                numLocalExperts(_numLocalExperts),
                k(_k), embedDim(_embedDim),
                embedPrecision(_embedPrecision),
                numResultChunks(cute::ceil_div(embedDim, bN))
                {};

        CUTE_HOST_DEVICE
        static unsigned int getCapacity(const unsigned int _seqLen, const unsigned int _numExperts,
                                        const unsigned int _capacityFactor, const unsigned int _k){
            return cute::ceil_div(_seqLen, _numExperts) * _capacityFactor * _k;
        }

        CUTE_HOST_DEVICE
        void dump(){
            printf("{\n\t"
                   "\"Capacity\": %u,\n\t"
                   "\"E\": %u,\n\t"
                   "\"localE\": %u,\n\t"
                   "\"H\": %u,\n\t"
                   "\"PeerOffset\": %lu,\n\t"
                   "\"Rank\": %u,\n\t"
                   "\"SB\": %u,\n\t"
                   "\"SequenceNumber\": %lu,\n\t"
                   "\"WorldSize\": %u,\n\t"
                   "\"k\": %u\n}\n",
                   capacity, numExperts, numLocalExperts, embedDim, peerOffset,
                   rank, seqLen, sequenceNumber, worldSize, k);
        }
    };

    /// Owned by a block
    struct SenderConfig : Config{
        unsigned int bid;
        unsigned long blocksToPeers;
        unsigned int peerStripeLength;
        unsigned int intraPeerIndex;
        unsigned int firstPeer;

        CUTE_DEVICE
        SenderConfig() = default; // Circumvents warning about 'initializing shared variables'

        CUTE_DEVICE
        SenderConfig(const Config c){
            /// Old Stuff
            sHeap = c.sHeap;
            worldSize = c.worldSize;
            capacity = c.capacity;
            peerOffset = c.peerOffset;
            rank = c.rank;
            sequenceNumber = c.sequenceNumber;
            seqLen = c.seqLen;
            k = c.k;
            embedDim = c.embedDim;
            embedPrecision = c.embedPrecision;
            numLocalExperts = c.numLocalExperts;

            /// New stuff
            bid = aristos::bid();
            blocksToPeers = max((gridDim.z - 1) * (gridDim.x * gridDim.y) / worldSize, 1);
            peerStripeLength = cute::ceil_div(1UL, blocksToPeers);
            intraPeerIndex = bid - ((bid / blocksToPeers) * blocksToPeers);
            firstPeer = static_cast<int>(bid / blocksToPeers);
        };

        CUTE_DEVICE
        void dump(){
            printf("{\n\t"
                   "\"BlockID\": %u,\n\t"
                   "\"BlocksToPeers\": %lu,\n\t"
                   "\"Capacity\": %u,\n\t"
                   "\"E\": %u,\n\t"
                   "\"localE\": %u,\n\t"
                   "\"H\": %u,\n\t"
                   "\"FirstPeer\": %u,\n\t"
                   "\"IntraPeerIndex\": %u,\n\t"
                   "\"PeerOffset\": %lu,\n\t"
                   "\"PeerStripeLength\": %u,\n\t"
                   "\"Rank\": %u,\n\t"
                   "\"SB\": %u,\n\t"
                   "\"SequenceNumber\": %lu,\n\t"
                   "\"WorldSize\": %u,\n\t"
                   "\"k\": %u\n}\n",
                   bid, blocksToPeers, capacity, numExperts, numLocalExperts,
                   embedDim, firstPeer, intraPeerIndex, peerOffset,
                   peerStripeLength, rank, seqLen, sequenceNumber, worldSize, k);
        }
    };

    enum header : unsigned int {
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
#endif //ARISTOS_TYPES_CUH
