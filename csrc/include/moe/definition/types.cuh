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
    extern constexpr unsigned int bP = 1; // pipeline stages; have to use 1 due to shared mem constraints
    extern constexpr unsigned int blockSize = 128; // 256 is too high, since SM can only hold at most 2048 threads
    extern constexpr unsigned int blockSizeWarp = 4; // 128 / 32
    extern constexpr unsigned int maxRegsPerThread = 32; // Enforces that we approach max active blocks per SM
    /// empirical threshold of threads to saturate NVLink bandwidth for one transfer
    extern constexpr unsigned int NVLinkThreshold = 4096;
    extern constexpr unsigned int superBlockSize = NVLinkThreshold / blockSize;

    struct Config{
        cuda::std::byte* sHeap;
        flagsType* flags;
        specType* bookKeeping;
        /// Append-only log, allowing replays
        specType* publisherLog;
        /// len = |E|
        specType* expertParallelSpec;
        /// len <= |D|
        specType* peerTranslation;
        /// Expert parallel World Size
        unsigned int worldSize;
        unsigned long peerOffset;
        /// Expert parallel group rank
        unsigned int rank;
        unsigned long sequenceNumber;
        unsigned int seqLen;
        unsigned int numExperts;
        unsigned int numLocalExperts;
        unsigned int k;
        unsigned int embedDim;
        unsigned int numPublisherBlocks;
        unsigned int numResultChunks;
        unsigned int* capacity;
        unsigned int peerStride;
        unsigned int stageStride;
        unsigned int cellStride;
        unsigned int tokenStride;
        unsigned int bookKeepingLen;
        /// Decider's deps
        unsigned int numLayers;
        unsigned int globalBatch;
        unsigned int redAmount;
        unsigned int miniBatch;
        unsigned int moeFreq;
        unsigned int p2pBuffer;
        unsigned int gradBuffer;

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
                rank(_rank),
                sequenceNumber(_sequenceNumber),
                seqLen(_seqLen),
                numExperts(_numExperts),
                numLocalExperts(_numLocalExperts),
                k(_k), embedDim(_embedDim),
                numResultChunks(cute::ceil_div(embedDim, bN))
                {};

        CUTE_HOST_DEVICE
        static unsigned int getCapacity(const unsigned int& _seqLen, const unsigned int& _numExperts,
                                        const unsigned int& _capacityFactor, const unsigned int& _k){
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
    struct PublisherConfig{
        unsigned int localBlockID;
        unsigned long blocksToPeers;
        unsigned int peerStripeLength;
        unsigned int intraPeerIndex;
        unsigned int firstPeer;
        unsigned int numSuperBlocks;
        unsigned int numSubBlocks;
        unsigned int p2pLogHead;
        unsigned int superBlockID;
        bool isLastSubBlock;
        bool isFirstSubBlock;
        unsigned int lastSubBlockID;
        specType* syncGrid;
        specType* checkpoints;

        CUTE_DEVICE
        static decltype(auto) getLocalBlockID(auto const& numPublisherBlocks){
            return blockIdx.x - (gridDim.x - numPublisherBlocks);
        }

        CUTE_DEVICE
        static decltype(auto) getLastLocalBlockID(){
            return gridDim.x - 1;
        }

        CUTE_DEVICE
        PublisherConfig() = default; // Circumvents warning about 'initializing shared variables'

        CUTE_DEVICE
        PublisherConfig(const Config& c){
            localBlockID = PublisherConfig::getLocalBlockID(c.numPublisherBlocks);
            blocksToPeers = max((gridDim.z - 1) * (gridDim.x * gridDim.y) / c.worldSize, 1);
            peerStripeLength = cute::ceil_div(1UL, blocksToPeers);
            intraPeerIndex = aristos::grid::blockID() - ((aristos::grid::blockID() / blocksToPeers) * blocksToPeers);
            firstPeer = static_cast<int>(aristos::grid::blockID() / blocksToPeers);
            numSuperBlocks = (c.numPublisherBlocks - 1) / superBlockSize;
            p2pLogHead = 0;
            superBlockID = localBlockID / superBlockSize;
            numSubBlocks = superBlockSize;
            lastSubBlockID = ((superBlockID + 1) * superBlockSize) - 1;
            if(localBlockID >= ((numSuperBlocks - 1)*superBlockSize)){
                superBlockID = (numSuperBlocks - 1);
                numSubBlocks = (c.numPublisherBlocks - 1) - ((numSuperBlocks - 1) * superBlockSize);
                lastSubBlockID = PublisherConfig::getLastLocalBlockID();
            }
            isLastSubBlock = localBlockID == lastSubBlockID;
            isFirstSubBlock = localBlockID == (superBlockID*numSubBlocks);
        };

        CUTE_DEVICE
        void dump(){
            printf("{\n\t"
                   "\"BlockID\": %u,\n\t"
                   "\"LocalBlockID\": %u,\n\t"
                   "\"BlocksToPeers\": %lu,\n\t"
                   "\"FirstPeer\": %u,\n\t"
                   "\"IntraPeerIndex\": %u,\n\t"
                   "\"PeerStripeLength\": %u,\n\t",
                   aristos::grid::blockID(), localBlockID, blocksToPeers, firstPeer, intraPeerIndex,
                   peerStripeLength);
        }
    };


    CUTE_DEVICE
    enum header : unsigned short {
        NOOP = 0,
        processed = 0,
        shouldProcess = 1,
        begin = 2
    };

    CUTE_DEVICE
    enum putSignal : unsigned short {
        sent = 1
    };

    template<typename E = header> requires cuda::std::is_integral_v<cuda::std::underlying_type_t<E>>
    CUTE_DEVICE
    uint64_t constructSignal(E const& signal, unsigned long const& tagAlong){
        return tagAlong + signal;
    }
}
#endif //ARISTOS_TYPES_CUH
