//
// Created by Jonathan on 7/18/24.
//

#ifndef ARISTOS_TYPES_CUH
#define ARISTOS_TYPES_CUH

#include <ostream>

#define ARISTOS_BLOCK_SIZE 128
#define ARISTOS_BLOCK_SIZE_WARP (128 / 32)
#if 0
#define REGS_PER_THREAD 32
#endif
/// empirical threshold of threads to saturate NVLink bandwidth for one transfer
#define ARISTOS_NVLINK_THRESHOLD 4096
#define ARISTOS_SUPER_BLOCK_SIZE (ARISTOS_NVLINK_THRESHOLD / ARISTOS_BLOCK_SIZE)

// GEMM stuff
#define ARISTOS_M_BATCH 128
#define ARISTOS_N_BATCH 128
#define ARISTOS_K_BATCH 8
#define ARISTOS_PIPELINE_STAGES 2

namespace aristos{
    using maxPrecision = float;
    using specType = unsigned int;
    using flagsType = uint64_t;
    /// Okay, I know the type name is relatively misleading, but we had no other choice due to
    /// NVIDIA native atomics only accepting int types.
    using AtomicBoolType = unsigned int;

    struct ModelConfig{
        unsigned int numLayers;
        unsigned int globalBatch;
        unsigned int redAmount;
        unsigned int miniBatch;
        unsigned int moeFreq;
        unsigned int p2pBuffer;
        unsigned int gradBuffer;
        ModelConfig() = default;
        ModelConfig(const unsigned int& numLayers, const unsigned int& redAmount, const unsigned int& globalBatch,
                    const unsigned int& miniBatch, const unsigned int& moeFreq,
                    const unsigned int& p2PBuffer, const unsigned int& gradBuffer) :
                    numLayers(numLayers), globalBatch(globalBatch),
                    redAmount(redAmount), miniBatch(miniBatch), moeFreq(moeFreq),
                    p2pBuffer(p2PBuffer), gradBuffer(gradBuffer) {}

        friend std::ostream &operator<<(std::ostream &os, const ModelConfig &config) {
            os << "numLayers: " << config.numLayers << " globalBatch: " << config.globalBatch << " redAmount: "
               << config.redAmount << " miniBatch: " << config.miniBatch << " moeFreq: " << config.moeFreq
               << " p2pBuffer: " << config.p2pBuffer << " gradBuffer: " << config.gradBuffer;
            return os;
        }
    };

    struct __align__(16) Config{
        cuda::std::byte* sHeap;
        flagsType* flags;
        specType* bookKeeping;
        /// Append-only log, allowing replays
        specType* publisherLog;
        /// len = |E|
        specType* expertParallelSpec;
        /// len <= |D|
        specType* peerTranslation;
        /// Expert parallel group rank
        unsigned int rank;
        unsigned long sequenceNumber;
        unsigned int seqLen;
        unsigned int numExperts;
        unsigned int numLocalExperts;
        unsigned int k;
        unsigned int worldSize;
        unsigned int embedDim;
        unsigned int numP2PPublisherBlocks;
        unsigned int numResultChunks;
        unsigned int capacity;
        unsigned int bookKeepingLen;

        CUTE_HOST_DEVICE
        Config() = default;

        CUTE_HOST_DEVICE
        Config(cuda::std::byte* _symmetricHeap,
               flagsType* _flags,
               specType* _bookKeeping,
               const unsigned int _rank,
               const unsigned long _sequenceNumber,
               const unsigned int _k,
               const unsigned int _embedDim,
               const unsigned int _numExperts,
               const unsigned int _numLocalExperts,
               const unsigned int _seqLen):
                sHeap(_symmetricHeap),
                flags(_flags),
                bookKeeping(_bookKeeping),
                rank(_rank),
                sequenceNumber(_sequenceNumber),
                seqLen(_seqLen),
                numExperts(_numExperts),
                numLocalExperts(_numLocalExperts),
                k(_k), embedDim(_embedDim),
                numResultChunks(cute::ceil_div(embedDim, ARISTOS_N_BATCH))
                {};

        CUTE_HOST_DEVICE
        static unsigned int getCapacity(const unsigned int& _seqLen, const unsigned int& _numExperts,
                                        const unsigned int& _capacityFactor, const unsigned int& _k){
            return cute::ceil_div(_seqLen, _numExperts) * _capacityFactor * _k;
        }


        CUTE_HOST_DEVICE
        void dump() const {
            printf("{\n\t"
                   "\"Capacity\": %u,\n\t"
                   "\"E\": %u,\n\t"
                   "\"localE\": %u,\n\t"
                   "\"H\": %u,\n\t"
                   "\"World\": %u,\n\t"
                   "\"Rank\": %u,\n\t"
                   "\"SB\": %u,\n\t"
                   "\"SequenceNumber\": %lu,\n\t"
                   "\"k\": %u\n}\n",
                   capacity, numExperts, numLocalExperts, embedDim, worldSize,
                   rank, seqLen, sequenceNumber, k);
        }
    };

    /// Owned by a block
    struct PublisherConfig{
        unsigned int localBlockID;
        unsigned int numSuperBlocks;
        unsigned int numSubBlocks;
        unsigned int p2pLogHead;
        unsigned int superBlockID;
        bool isLastSubBlock;
        bool isFirstSubBlock;
        specType* syncGrid;
        specType* checkpoints;

        CUTE_DEVICE
        static decltype(auto) getLocalBlockID(auto const& numP2PBlocks){
            return blockIdx.x - (gridDim.x - numP2PBlocks);
        }

        CUTE_DEVICE
        static decltype(auto) getLastLocalBlockID(){
            return gridDim.x - 1;
        }

        CUTE_DEVICE
        PublisherConfig() = default; // Circumvents warning about 'initializing shared variables'

        CUTE_DEVICE
        explicit PublisherConfig(const Config& c){
            localBlockID = PublisherConfig::getLocalBlockID(c.numP2PPublisherBlocks);
            numSuperBlocks = (c.numP2PPublisherBlocks - 1) / ARISTOS_SUPER_BLOCK_SIZE;
            p2pLogHead = 0;
            superBlockID = localBlockID / ARISTOS_SUPER_BLOCK_SIZE;
            numSubBlocks = ARISTOS_SUPER_BLOCK_SIZE;
            unsigned int lastSubBlockID = ((superBlockID + 1) * ARISTOS_SUPER_BLOCK_SIZE) - 1;
            // last super block with more sub-blocks
            if(localBlockID >= ((numSuperBlocks - 1)*ARISTOS_SUPER_BLOCK_SIZE)){
                superBlockID = (numSuperBlocks - 1);
                numSubBlocks = (c.numP2PPublisherBlocks - 1) - ((numSuperBlocks - 1) * ARISTOS_SUPER_BLOCK_SIZE);
                lastSubBlockID = PublisherConfig::getLastLocalBlockID();
            }
            isLastSubBlock = localBlockID == lastSubBlockID;
            isFirstSubBlock = localBlockID == (superBlockID*numSubBlocks);
            syncGrid = c.bookKeeping;
            checkpoints = c.bookKeeping;
        };

        CUTE_DEVICE
        void dump() const{
            printf("{\n\t"
                   "\"BlockID\": %u,\n\t"
                   "\"LocalBlockID\": %u,\n\t",
                   aristos::grid::blockID(), localBlockID);
        }
    };


    __device__
    enum header : unsigned short {
        NOOP = 0,
        processed = 0,
        shouldProcess = 1,
        begin = 2
    };

    __device__
    enum putSignal : uint64_t {
        sent = 1
    };

    __constant__ uint64_t seqNo;
    template<typename E = header> requires cuda::std::is_integral_v<cuda::std::underlying_type_t<E>>
    CUTE_DEVICE
    uint64_t constructSignal(uint64_t const& tagAlong, E const& signal){
        return tagAlong + signal + seqNo;
    }
}
#endif //ARISTOS_TYPES_CUH
