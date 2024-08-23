//
// Created by Jonathan on 7/18/24.
//

#ifndef ARISTOS_TYPES_CUH
#define ARISTOS_TYPES_CUH

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
    extern constexpr unsigned int blockSizeWarp = 4; // 128 / 32
    /// empirical threshold of threads to saturate NVLink bandwidth for one transfer
    extern constexpr unsigned int NVLinkBlockReq = 2048;

    namespace cg = cooperative_groups;

    struct Config{
        cuda::std::byte* sHeap;
        flagsType* flags;
        specType* bookKeeping;
        specType* syncGrid;
        /// len = |E|
        specType* expertParallelSpec;
        /// len <= |D|
        specType* peerTranslation;
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
        unsigned int numPublisherBlocks;
        unsigned int numResultChunks;
        unsigned int peerStride;
        unsigned int stageStride;
        unsigned int cellStride;
        unsigned int tokenStride;

        CUTE_HOST_DEVICE
        Config() = default;

        CUTE_HOST_DEVICE
        Config(cuda::std::byte* _symmetricHeap,
               flagsType* _flags,
               specType* _bookKeeping,
               const unsigned int _worldSize,
               const unsigned int _rank,
               const unsigned long _sequenceNumber,
               const unsigned int _capacityFactor,
               const unsigned int _k,
               const unsigned int _embedDim,
               const unsigned int _numExperts,
               const unsigned int _numLocalExperts,
               const unsigned int _seqLen):
                sHeap(_symmetricHeap),
                flags(_flags),
                bookKeeping(_bookKeeping),
                worldSize(_worldSize),
                capacity(getCapacity(_seqLen, _numExperts, _capacityFactor, _k)),
                rank(_rank),
                sequenceNumber(_sequenceNumber),
                seqLen(_seqLen),
                numExperts(_numExperts),
                numLocalExperts(_numLocalExperts),
                k(_k), embedDim(_embedDim),
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
    struct SenderConfig{
        unsigned int bid;
        unsigned long blocksToPeers;
        unsigned int peerStripeLength;
        unsigned int intraPeerIndex;
        unsigned int firstPeer;
        specType* pubQueue;

        CUTE_DEVICE
        SenderConfig() = default; // Circumvents warning about 'initializing shared variables'

        CUTE_DEVICE
        SenderConfig(const Config& c){
            bid = aristos::bid();
            blocksToPeers = max((gridDim.z - 1) * (gridDim.x * gridDim.y) / c.worldSize, 1);
            peerStripeLength = cute::ceil_div(1UL, blocksToPeers);
            intraPeerIndex = bid - ((bid / blocksToPeers) * blocksToPeers);
            firstPeer = static_cast<int>(bid / blocksToPeers);
        };

        CUTE_DEVICE
        void dump(){
            printf("{\n\t"
                   "\"BlockID\": %u,\n\t"
                   "\"BlocksToPeers\": %lu,\n\t"
                   "\"FirstPeer\": %u,\n\t"
                   "\"IntraPeerIndex\": %u,\n\t"
                   "\"PeerStripeLength\": %u,\n\t",
                   bid, blocksToPeers, firstPeer, intraPeerIndex,
                   peerStripeLength);
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
