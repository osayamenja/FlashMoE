//
// Created by Jonathan on 7/18/24.
//

#ifndef ARISTOS_TYPES_CUH
#define ARISTOS_TYPES_CUH

#define ARISTOS_BLOCK_SIZE 128
#define ARISTOS_BLOCK_SIZE_WARP (128 / 32)
// number of blocks
#define ARISTOS_SUPER_BLOCK_SIZE 32

#define CAST_TO(T, p) static_cast<T*>(static_cast<void*>(p))

#define N_READY_Q_SIGNALS 1 // head
#define N_TASK_Q_SIGNALS 3

/// The symmetric heap is a 4-D tensor (P, S, C, T)
/// where P, S, C, and T denote dimensions for peers, communication stages,
/// cells and tokens.
/// Number of communication stages S
#define STAGES 2

/// Per stage, there is one cell for sending and another for reception
#define CELLS 2
#define SEND_CELL 0
#define RECEIVE_CELL 1

#define HEAP_ALIGNMENT 16

// GEMM configuration constants
#define MIN_ARCH 700
#define THREADS 128
#define BLOCK_M 128
#define BLOCK_M_EXP 64
#define BLOCK_N 64
#define BLOCK_K_HALF 16
#define BLOCK_K_FULL 8
#define MAX_REGS (BLOCK_M * BLOCK_N) / THREADS
#define PIPELINE_STAGES 2
#define SHARED_SIZE 16 * 1024U
#include "tensor.cuh"

namespace aristos{
    using maxPrecision = float; // no support for double, unfortunately
    using specType = unsigned int;
    using flagsType = uint64_t;

    __device__
    enum class PacketStage {
        initial,
        final,
    };

    enum class PacketWrapDistribution {
        equidistant,
        variant
    };
    enum class GateReductionLevel {
        singleBlock,
        multiBlock
    };

    enum class HeapType {
        remote,
        p2p
    };

    enum class DropTokens {
        yes,
        no
    };

    __device__
    enum putSignal : uint64_t {
        sent = 1
    };

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
    };

    struct __align__(16) Config{
        using HeapTuple = cuda::std::pair<maxPrecision, unsigned int>;
        cuda::std::byte* sHeap;
        flagsType* flags;
        /// Needed for free
        cuda::std::byte* bookKeeping;
        /// EP rank -> global rank
        unsigned int* peerTranslation;
        unsigned int* parallelismSpec;
        unsigned long sequenceNumber;
        /// Expert parallel group rank
        unsigned int rank;
        unsigned int seqLen;
        unsigned int numExperts;
        unsigned int k;
        unsigned int worldSize;
        unsigned int embedDim;
        unsigned int upProjection;
        // per GPU
        unsigned int capacity;
        unsigned int expertCapacity;
        unsigned int nTiles;
        unsigned int tilesN;
        unsigned int tilesM;
        unsigned int tilesNx;
        cuda::barrier<cuda::thread_scope_device>* deviceBlockade;

        __device__ __forceinline__
        Config() = default;

        __host__ __device__ __forceinline__
        Config(cuda::std::byte* _symmetricHeap,
               flagsType* _flags,
               cuda::std::byte* _bk,
               const unsigned int& _rank,
               const unsigned long& _sequenceNumber,
               const unsigned int& _k,
               const unsigned int& _embedDim,
               const unsigned int& _numExperts,
               const unsigned int& _seqLen,
               const unsigned int& _world,
               const unsigned int& _proj,
               const unsigned int& _tilesN,
               const unsigned int& _tilesM,
               const unsigned int& _tilesNx,
               cuda::barrier<cuda::thread_scope_device>* _blockade,
               const unsigned int& _capFactor = 1):
                sHeap(_symmetricHeap),
                flags(_flags),
                bookKeeping(_bk),
                peerTranslation(CAST_TO(unsigned int, _bk)),
                parallelismSpec(CAST_TO(unsigned int, _bk) + _world),
                sequenceNumber(_sequenceNumber),
                rank(_rank),
                seqLen(_seqLen),
                numExperts(_numExperts),
                k(_k), worldSize(_world),
                embedDim(_embedDim),
                upProjection(_proj),
                capacity(cute::ceil_div(_seqLen, _world)),
                expertCapacity(cute::ceil_div(_seqLen, _numExperts) * _capFactor),
                nTiles(_tilesM * (_tilesN + _tilesNx)),
                tilesN(_tilesN), tilesM(_tilesM),
                tilesNx(_tilesNx), deviceBlockade(_blockade){}

        template<typename Element>
        __forceinline__ __device__
        constexpr size_t finalPacketSize(const unsigned int& numTokens) const {
            return sizeof(unsigned int) + numTokens * (sizeof(unsigned int) + sizeof(Element) * embedDim);
        }
        template<unsigned int tileDimension>
        __host__ __device__ __forceinline__
        static constexpr unsigned int pad(const unsigned int& dimension) {
            // find next multiple of dimension.
            return cute::ceil_div(dimension, tileDimension) * tileDimension;
        }

        template<unsigned int tileDimension>
        __host__ __device__ __forceinline__
        static constexpr unsigned int tiles(const unsigned int& dimension) {
            // find next multiple of dimension.
            return cute::ceil_div(dimension, tileDimension);
        }

        template<typename Element>
        requires aristos::TensorValueType<Element>
        __device__ __forceinline__
        auto* xMid() const {
            return CAST_TO(Element, bookKeeping);
        }

        template<GateReductionLevel g = GateReductionLevel::multiBlock>
        __device__ __forceinline__
        unsigned int* getBRSFlags() const {
            static_assert(g == GateReductionLevel::multiBlock);
            return CAST_TO(unsigned int, xMid<maxPrecision>() + pad<BLOCK_M>(seqLen) * embedDim);
        }

        template<GateReductionLevel g = GateReductionLevel::multiBlock>
        __device__ __forceinline__
        float2* getBRSValues() const {
            static_assert(g == GateReductionLevel::multiBlock);
            return CAST_TO(float2, getBRSFlags() + (seqLen * pad<BLOCK_N>(numExperts)));
        }

        template<GateReductionLevel g = GateReductionLevel::multiBlock>
        __device__ __forceinline__
        HeapTuple* getBRSHeap() const {
            static_assert(g == GateReductionLevel::multiBlock);
            return static_cast<HeapTuple*>(static_cast<void*>(getBRSValues() + pad<BLOCK_M>(seqLen)));
        }

        template<GateReductionLevel g = GateReductionLevel::singleBlock>
        unsigned int* tIdxFlag() const {
            if constexpr (g == GateReductionLevel::multiBlock) {
                return CAST_TO(unsigned int, getBRSHeap() + k * pad<BLOCK_M>(seqLen));
            }
            return CAST_TO(unsigned int, bookKeeping);
        }

        template<GateReductionLevel g = GateReductionLevel::singleBlock>
        unsigned int* tIdxVal() const {
            if constexpr (g == GateReductionLevel::multiBlock) {
                return CAST_TO(unsigned int, tIdxFlag() + tilesM * pad<BLOCK_N>(numExperts));
            }
            return CAST_TO(unsigned int, tIdxFlag() + tilesM * BLOCK_N);
        }
        template<GateReductionLevel g = GateReductionLevel::singleBlock>
        unsigned int* tIdx() const {
            if constexpr (g == GateReductionLevel::multiBlock) {
                return CAST_TO(unsigned int, tIdxVal() + pad<BLOCK_N>(numExperts));
            }
            return CAST_TO(unsigned int, tIdxVal() + BLOCK_N);
        }

        // Gate loss
        __device__ __forceinline__
        maxPrecision* getGateMeanLogits() const {
            return CAST_TO(maxPrecision, tIdx() + seqLen);
        }

        __device__ __forceinline__
        maxPrecision* getMeanExpertCounts() const {
            return CAST_TO(maxPrecision, getGateMeanLogits() + pad<BLOCK_N>(numExperts));
        }

        __device__ __forceinline__
        maxPrecision* getGateLoss() const {
            return CAST_TO(maxPrecision, getMeanExpertCounts() + pad<BLOCK_N>(numExperts));
        }

        __device__ __forceinline__
        unsigned int* getExpertCounts() const {
            return CAST_TO(unsigned int, getGateLoss() + 1);
        }

        __device__ __forceinline__
        unsigned int* getPeerXLookup() const {
            return CAST_TO(unsigned int, getExpertCounts() + numExperts);
        }

        __device__ __forceinline__
        unsigned int* xSync() const {
            return CAST_TO(unsigned int, getPeerXLookup() + numExperts + worldSize + 1);
        }

        // Packet stuff
        template<typename Element> requires aristos::TensorValueType<Element>
        __forceinline__
        static auto frameSize(const unsigned int& length, const unsigned int& dim){
            // Header and payload
            return  sizeof(maxPrecision) + (dim + 1) * Config::pad<BLOCK_M>(length) * sizeof(Element);
        }

        __host__ __device__ __forceinline__
        void dump() const {
            printf("{\n\t"
                   "\"Capacity\": %u,\n\t"
                   "\"E\": %u,\n\t"
                   "\"H\": %u,\n\t"
                   "\"World\": %u,\n\t"
                   "\"Rank\": %u,\n\t"
                   "\"SB\": %u,\n\t"
                   "\"SequenceNumber\": %lu,\n\t"
                   "\"k\": %u\n}\n",
                   capacity, numExperts, embedDim, worldSize,
                   rank, seqLen, sequenceNumber, k);
        }
    };

    enum class TaskType {
        Interrupt,
        preGEMM,
        postGEMM,
        GateScale
    };

    struct __align__(16) Task {
        // sensible sentinel values
        cuda::std::byte* aData = nullptr;
        cuda::std::byte** bData = nullptr;
        cuda::std::byte** cData = nullptr;
        cuda::std::byte** dData = nullptr;
        cuda::std::byte* scale = nullptr;
        // crd2Idx(peer, expertIdx, offset)
        unsigned int syncIdx = 0UL;
        unsigned int tileIdx = 0U;
        unsigned int M = 0U;
        unsigned int tileSize = 0U;
        unsigned int peerIdx = 0U;
        TaskType taskType = TaskType::Interrupt;

        __forceinline__ __device__
        Task() = default;

        __device__ __forceinline__
        Task(const TaskType& _taskType,
            cuda::std::byte* _aData,
            cuda::std::byte** _bData,
            cuda::std::byte** _cData,
            cuda::std::byte** _dData,
            cuda::std::byte* _scale,
            const unsigned int& _syncIdx,
            const unsigned int& _tile,
            const unsigned int& _M,
            const unsigned int& _size,
            const unsigned int& _peerIdx):
        aData(_aData), bData(_bData),
        cData(_cData), dData(_dData), scale(_scale),
        syncIdx(_syncIdx), tileIdx(_tile), M(_M), tileSize(_size), peerIdx(_peerIdx),
        taskType(_taskType){}

        __device__ __forceinline__
        explicit Task(const TaskType& _taskType):
        taskType(_taskType) {}
    };

    struct __align__(16) SchedulerConfig{
        unsigned int* readyQ;
        /// rQS[0] -> head
        /// rQS[1] -> tail
        unsigned int* readyQHead;
        unsigned int* taskSignal;
        unsigned int* taskSync;
        Task* taskQ;
        unsigned int* taskQSignals;

        __forceinline__ __device__
        SchedulerConfig() = default;

        __forceinline__ __device__ __host__
        SchedulerConfig(cuda::std::byte* _bk,
               const unsigned int& numberBlocks,
               const unsigned int& _syncTasksBound) {
            readyQ = CAST_TO(unsigned int, _bk);
            readyQHead = CAST_TO(unsigned int, readyQ + numberBlocks);
            taskSignal = CAST_TO(unsigned int, readyQHead + N_READY_Q_SIGNALS);
            taskSync = CAST_TO(unsigned int, taskSignal + numberBlocks);
            taskQSignals = CAST_TO(unsigned int, taskSync + _syncTasksBound);
            taskQ = CAST_TO(Task, taskQSignals + N_TASK_Q_SIGNALS);
        }
    };

    __constant__ __inline__ uint64_t seqNo;
    __constant__ __inline__ Config moeConfig{};
    __constant__ __inline__ SchedulerConfig schedulerState{};
    __inline__ Config hostMoEConfig;

    template<typename E = PacketStage> requires cuda::std::is_integral_v<cuda::std::underlying_type_t<E>>
    __device__ __forceinline__
    uint64_t constructSignal(E const& signal, uint64_t const& tagAlong = 0U){
        return tagAlong + signal + seqNo;
    }
}
#endif //ARISTOS_TYPES_CUH
