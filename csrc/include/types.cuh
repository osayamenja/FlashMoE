//
// Created by Jonathan on 7/18/24.
//

#ifndef ARISTOS_TYPES_CUH
#define ARISTOS_TYPES_CUH

#define ARISTOS_BLOCK_SIZE 128U
#define ARISTOS_BLOCK_SIZE_WARP (128U / 32)
// number of blocks
#define ARISTOS_SUPER_BLOCK_SIZE 32U

#define CAST_TO(T, p) static_cast<T*>(static_cast<void*>(p))
/// Number of communication stages S
#define STAGES 2U

/// Per stage, there is one cell for sending and another for reception
#define CELLS 2U
#define SEND_CELL 0U
#define RECEIVE_CELL 1U

#define HEAP_ALIGNMENT 16U

// Hardware description
#define MIN_ARCH 700U
#define THREADS 128U
#define SUBSCRIBERS (THREADS - 2U)
// GEMM configuration constants
#define BLOCK_M 128U
#define BLOCK_M_EXP 64U
#define BLOCK_N 64U
#define BLOCK_K_HALF 16U
#define BLOCK_K_FULL 8U
#define MAX_REGS (BLOCK_M * BLOCK_N) / THREADS
#define PIPELINE_STAGES 2U
#define SHARED_SIZE 16 * 1024U
#define GEMMs 2U // per expert

#define TOPO_LOOP_TRIP 4U // this may be too much
#define BETA_BUFFER (1024UL * 1024UL) // 1MB
#define ALPHA_BUFFER 1024UL // 1KB
#define NANO_TO_MILLI (cuda::std::nano::den / cuda::std::milli::den)
#define NANO_TO_MICRO (cuda::std::nano::den / cuda::std::micro::den)
#define BYTE_MAX cuda::std::numeric_limits<cuda::std::underlying_type_t<cuda::std::byte>>::max()
#define TO_MB(b) (static_cast<float>(b) / (1024.0f*1024.0f))
#define BETA_MB 1024.0f // 1GB

#include <cuda/std/type_traits>
#include <cuda/barrier>
#include <cuda/std/array>
#include <cute/tensor.hpp>

namespace aristos{
    template<typename V>
        concept TensorValueType = cuda::std::is_same_v<V, cute::half_t> ||
            cuda::std::is_same_v<V, cute::bfloat16_t> ||
            cuda::std::is_same_v<V, cute::tfloat32_t> ||
            cuda::std::is_same_v<V, float> ||
            cuda::std::is_same_v<V, cute::float_e4m3_t> ||
            cuda::std::is_same_v<V, cute::float_e5m2_t>;

    template<typename T>
    concept Tensor = cute::is_tensor<T>::value && TensorValueType<typename T::value_type>;

    template<typename M>
    concept Matrix = requires(M m){
        requires Tensor<M> && rank(m) == 2;
    };
    using maxPrecision = float; // no support for double, unfortunately
    using specType = unsigned int;
    using flagsType = uint64_t;

    using Nano = cuda::std::chrono::duration<float, cuda::std::nano>;
    using Milli = cuda::std::chrono::duration<float, cuda::std::milli>;
    using ull_t = unsigned long long int;

    struct floatPair {
        float alpha;
        float beta;

        __device__ __forceinline__
        friend bool operator<(const floatPair &lhs, const floatPair &rhs) {
            return fmaf(lhs.beta, BETA_MB, lhs.alpha) < fmaf(rhs.beta, BETA_MB, rhs.alpha);
        }

        __device__ __forceinline__
        friend bool operator<=(const floatPair &lhs, const floatPair &rhs) {
            return rhs >= lhs;
        }

        __device__ __forceinline__
        friend bool operator>(const floatPair &lhs, const floatPair &rhs) {
            return rhs < lhs;
        }

        __device__ __forceinline__
        friend bool operator>=(const floatPair &lhs, const floatPair &rhs) {
            return !(lhs < rhs);
        }
    };

    __device__
    enum class PacketStage {
        initial,
        final,
    };

    __device__
    enum class GateReductionLevel {
        singleBlock,
        multiBlock
    };

    __device__
    enum class PeerConnectivity {
        remote,
        p2p
    };

    __device__
    enum class DropTokens {
        yes,
        no
    };

    __device__
    enum PutSignal : uint8_t {
        sent = 1
    };

    __device__
    enum ReadySignal : unsigned int {
        observed,
        ready
    };

    template<PacketStage p = PacketStage::initial>
    struct SignalPayload {
        static_assert(p == PacketStage::initial);
        uint routedTokens;
        uint16_t seqNo;
        uint16_t totalTilesM;
    };

    template<>
    struct SignalPayload<PacketStage::final> {
        uint batchIdx;
        uint16_t seqNo;
        uint16_t tokensM; // <= BLOCK_M
    };

    /// A more apropos name would be "static storage" rather than registers.
    template<class T>
    struct isRegister : cuda::std::false_type {};

    template<class T, int N, int Alignment>
    struct isRegister<cutlass::AlignedArray<T, N, Alignment>> : cuda::std::true_type {};

    template<class T, int N, bool RegisterSized>
    struct isRegister<cutlass::Array<T, N, RegisterSized>> : cuda::std::true_type {};

    template <class T>
    constexpr bool isRegisterV = isRegister<T>::value;

    // Needed for decider
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

    // Index and gate combine weight
    using TokenIdxTuple = cuda::std::pair<unsigned int, maxPrecision>;
    using HeapTuple = cuda::std::pair<maxPrecision, unsigned int>;
    struct __align__(16) Config{
        cuda::std::byte* sHeap;
        flagsType* flags;
        /// Needed for free
        cuda::std::byte* bookKeeping;
        /// EP rank -> global rank
        unsigned int* peerTranslation;
        /// Expert index -> EP rank
        unsigned int* parallelismSpec;
        /// Expert parallel group rank
        unsigned int rank;
        unsigned int seqLen;
        unsigned int numExperts;
        unsigned int numLocalExperts;
        unsigned int k;
        unsigned int worldSize;
        unsigned int embedDim;
        unsigned int upProjection;
        // per GPU
        unsigned int expertSlots;
        unsigned int expertCapacity;
        unsigned int cellSize;
        unsigned int nTiles;
        unsigned int tilesN;
        unsigned int tilesM;
        unsigned int tilesNx;

        __device__ __forceinline__
        Config() = default;

        __host__ __device__ __forceinline__
        Config(cuda::std::byte* _symmetricHeap,
               flagsType* _flags,
               cuda::std::byte* _bk,
               const unsigned int& _rank,
               const unsigned int& _k,
               const unsigned int& _embedDim,
               const unsigned int& _numExperts,
               const unsigned int& _numLExperts,
               const unsigned int& _seqLen,
               const unsigned int& _world,
               const unsigned int& _proj,
               const unsigned int& _tilesN,
               const unsigned int& _tilesM,
               const unsigned int& _tilesNx,
               const unsigned int& _expertSlots,
               const unsigned int& _capFactor = 1):
                sHeap(_symmetricHeap),
                flags(_flags),
                bookKeeping(_bk),
                peerTranslation(CAST_TO(unsigned int, _bk)),
                parallelismSpec(CAST_TO(unsigned int, _bk) + _world),
                rank(_rank),
                seqLen(_seqLen),
                numExperts(_numExperts), numLocalExperts(_numLExperts),
                k(_k), worldSize(_world),
                embedDim(_embedDim),
                upProjection(_proj),
                expertSlots(_expertSlots),
                expertCapacity(cute::ceil_div(_seqLen, _numExperts) * _capFactor),
                cellSize(expertCapacity * (embedDim + 1)), // max packet frame size
                nTiles(_tilesM * (_tilesN + _tilesNx)),
                tilesN(_tilesN), tilesM(_tilesM),
                tilesNx(_tilesNx){}

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

        /// The symmetric heap is a 6-D tensor (P, S, C, E, M, H)
        /// where P, S, C, E, M, and H  denote dimensions for peers, communication stages,
        /// cells, experts, expert capacity, and token hidden dimension, respectively.
        template<unsigned int stage = 0, unsigned cell = 0, unsigned long int nBytes = 1>
        requires (stage < STAGES && cell < CELLS)
        __device__ __forceinline__
        auto* advanceHeap(unsigned int const& peer, unsigned int const& expert, const unsigned int& token = 0) const {
            return sHeap + (cellSize * (expertSlots * (CELLS * (peer * STAGES + stage) + cell) + expert) +
                token * embedDim) * nBytes;
        }

        __host__ __device__ __forceinline__
        void dump() const {
            printf("{\n\t"
                   "\"ExpertCapacity\": %u,\n\t"
                   "\"E\": %u,\n\t"
                   "\"H\": %u,\n\t"
                   "\"World\": %u,\n\t"
                   "\"Rank\": %u,\n\t"
                   "\"SB\": %u,\n\t"
                   "\"k\": %u\n}\n",
                   expertCapacity, numExperts, embedDim, worldSize,
                   rank, seqLen, k);
        }
    };

    enum class TaskType {
        preGEMM,
        postGEMM,
        combine,
        Interrupt,
    };

    struct __align__(16) Task {
        // D = A * B + C
        // sensible sentinel values
        cuda::std::byte* aData = nullptr;
        cuda::std::array<cuda::std::byte*, GEMMs> bData = {};
        cuda::std::array<cuda::std::byte*, GEMMs> cData = {};
        cuda::std::array<cuda::std::byte*, GEMMs> dData = {};
        cuda::std::byte* scale = nullptr;
        // crd2Idx(peer, expertIdx, offset)
        unsigned int syncIdx = 0UL;
        unsigned int tileIdx = 0U;
        unsigned int tileSize = 0U;
        unsigned int peerIdx = 0U;
        //padded
        unsigned int M = 0U;
        unsigned int flagIdx = 0U;
        unsigned int batchIdx = 0U;
        TaskType taskType = TaskType::Interrupt;

        __forceinline__ __device__
        Task() = default;

        // Stage 1
        __device__ __forceinline__
        Task(const TaskType& _taskType,
            cuda::std::byte*  const& _aData,
            const cuda::std::array<cuda::std::byte*, GEMMs>& _bData,
            const cuda::std::array<cuda::std::byte*, GEMMs>& _cData,
            const cuda::std::array<cuda::std::byte*, GEMMs>& _dData,
            cuda::std::byte*  const& _scale,
            const unsigned int& _syncIdx,
            const unsigned int& _tile,
            const unsigned int& _M,
            const unsigned int& _flagIdx,
            const unsigned int& _size,
            const unsigned int& _peerIdx,
            const unsigned int& _batchIdx):
        aData(_aData), bData(_bData),
        cData(_cData), dData(_dData), scale(_scale),
        syncIdx(_syncIdx), tileIdx(_tile), tileSize(_size), peerIdx(_peerIdx), M(_M), flagIdx(_flagIdx),
        batchIdx(_batchIdx), taskType(_taskType){}

        // Stage 2
        __device__ __forceinline__
        Task(const TaskType& _taskType,
        cuda::std::byte*  const& _aData,
        const cuda::std::array<cuda::std::byte*, GEMMs>& _bData,
        const cuda::std::array<cuda::std::byte*, GEMMs>& _cData,
        const unsigned int& _size,
        const unsigned int& _tile,
        const unsigned int& _M):
        aData(_aData), bData(_bData), cData(_cData), tileIdx(_tile), tileSize(_size), M(_M), taskType(_taskType){}


        __device__ __forceinline__
        explicit Task(const TaskType& _taskType):
        taskType(_taskType) {}
    };

    /// Information about auxiliary data structures comprising bookkeeping state
    /// Includes length of data structures (arrays) and pointer arithmetic functions
    struct __align__(16) Bookkeeping {
        /// default type for bookkeeping data structures
        using BookType = unsigned int;
        using HeapTuple = cuda::std::pair<maxPrecision, BookType>;
        cuda::std::byte* book;
        /// Note the below lengths are cumulative sums.
        /// Gate buffers in bytes
        unsigned long int gB = 0UL;
        /// EP group description and packet sync array in bytes
        unsigned long int eDsA = 0UL;
        /// Scheduler buffers and flag checkpoints in bytes
        unsigned long int sBfC = 0UL;
        /// Intermediate buffer and tQ length in bytes
        unsigned long int xMtQ = 0UL;
        /// Block Ring Softmax flags in bytes, non-cumulative
        unsigned int brs = 0UL;
        /// gate routing and loss vectors in bytes
        unsigned int gRl = 0U;
        /// gRl + gB + eDsA + sBfC + brs
        unsigned long int bookSize;

        /// Task Q maximum length
        unsigned int tQl;
        /// EP world
        unsigned int world;
        /// sequence length
        unsigned int sl;
        /// number of experts
        unsigned int nx;
        /// number of local experts
        unsigned int nLx;
        /// hidden projection dimension
        unsigned int pd;
        /// padded number of experts
        unsigned int px;
        /// tiles spanning sequence length
        unsigned int tM;
        /// tiles spanning embedding dimension
        unsigned int tN;
        /// tiles spanning capacity
        unsigned int tCM;
        /// processors
        unsigned int blocks;
        /// expert capacity
        unsigned int eCap;
        /// Global device barrier
        cuda::barrier<cuda::thread_scope_device>* deviceBlockade;

        Bookkeeping() = default;

        explicit Bookkeeping(
            cuda::std::byte* const& _book,
            const unsigned int& _sl,
            const unsigned int& _nx,
            const unsigned int& _nLx,
            const unsigned int& _pd,
            const unsigned int& _px,
            const unsigned int& _embedDim,
            const unsigned int& _eCapacity,
            const unsigned int& _blocks,
            const unsigned int& _world,
            const unsigned int& _k,
            cuda::barrier<cuda::thread_scope_device>* _blockade) :
        book(_book), world(_world), sl(_sl), nx(_nx), nLx(_nLx), pd(_pd), px(_px),
        tM(Config::tiles<BLOCK_M>(_sl)),
        tN(Config::tiles<BLOCK_N>(_embedDim)),
        tCM(Config::tiles<BLOCK_M>(_eCapacity)), blocks(_blocks), eCap(_eCapacity),
        deviceBlockade(_blockade) {
            if (_nx == 1)[[unlikely]] {
                // For this case, using any function other than xM yields undefined behavior
                xMtQ = sizeof(maxPrecision) * _world * _nLx * tCM * tN;
                bookSize = xMtQ;
            }
            else {
                gRl = sizeof(maxPrecision) * (_sl * px + (2 * px + 1));
                gB = gRl + sizeof(TokenIdxTuple) * (px * _eCapacity) + sizeof(BookType) * px * (tM + 3);
                eDsA = gB + sizeof(BookType) * (3 * nx + world);
                const unsigned int fCl = sizeof(bool) * (world * nLx + (nx * tCM * tN));
                sBfC = eDsA + sizeof(BookType) * (3 * blocks + SUBSCRIBERS + (world * nLx * tCM)) + fCl;
                // maximum gemm tiles/tasks scheduled by subscriber threads
                auto sT = world * nLx * tCM * tN + tCM * tN * nx;
                sT = sT / SUBSCRIBERS * SUBSCRIBERS;
                // maximum gemm tiles/tasks scheduled by processors
                const auto pT = world * nLx * tCM * Config::tiles<BLOCK_N>(pd);
                tQl = sizeof(Task) * (sT + pT);
                xMtQ = sBfC + tQl + sizeof(maxPrecision) * world * nLx * tCM * tN;
                brs = sizeof(BookType) * (tM * tN +  sl * Config::tiles<BLOCK_N>(px)) +
                    sizeof(maxPrecision) * (2 * sl) + sizeof(HeapTuple) * (_k * sl);
                bookSize = xMtQ + brs;
            }
        }

        /// Needed for malloc
        static unsigned long int BookLength(
            const unsigned int& _sl,
            const unsigned int& _nx,
            const unsigned int& _nLx,
            const unsigned int& _pd,
            const unsigned int& _px,
            const unsigned int& _embedDim,
            const unsigned int& _eCap,
            const unsigned int& _blocks,
            const unsigned int& _world,
            const unsigned int& _k){
            const auto tCM = Config::tiles<BLOCK_M>(_eCap);
            const auto tN = Config::tiles<BLOCK_N>(_embedDim);
            if (_nx == 1) {
                return sizeof(maxPrecision) * _world * _nLx * tCM * tN;
            }
            const auto tM = Config::tiles<BLOCK_M>(_sl);
            const auto gRl = sizeof(maxPrecision) * (_sl * _px + (2 * _px + 1));
            const auto gB = gRl + sizeof(TokenIdxTuple) * (_px * _eCap) + sizeof(BookType) * _px * (tM + 3);
            const auto eDsA = gB + sizeof(BookType) * (3 * _nx + _world);
            const auto fCl = sizeof(bool) * (_world * _nLx + (_nx * tCM * tN));
            const auto sBfC = eDsA + sizeof(BookType) * (3 * _blocks + THREADS - 2 + _world * _nLx * tCM) + fCl;
            // maximum gemm tiles/tasks scheduled by subscriber threads
            auto sT = _world * _nLx * tCM * tN + tCM * tN * _nx;
            sT = sT / SUBSCRIBERS * SUBSCRIBERS;
            // maximum gemm tiles/tasks scheduled by processors
            const auto pT = _world * _nLx * tCM * Config::tiles<BLOCK_N>(_pd);
            const auto tQl = sizeof(Task) * (sT + pT);

            const auto xMtQ = sBfC + tQl + sizeof(maxPrecision) * _world * _nLx * tCM * tN;
            const auto brs = sizeof(BookType) * (tM * tN +  _sl * Config::tiles<BLOCK_N>(_px)) +
                sizeof(maxPrecision) * (2 * _sl) + sizeof(HeapTuple) * (_k * _sl);
            return xMtQ + brs;
        }

        template<typename Element>
        __device__ __forceinline__
        auto* gRt() const {
            return CAST_TO(Element, book);
        }

        /// Gate mean logits
        __device__ __forceinline__
        auto* gML() const {
            return CAST_TO(maxPrecision, book + sizeof(maxPrecision) * (sl * px));
        }
        /// Gate mean expert counts
        __device__ __forceinline__
        auto* gMeC() const {
            return gL() + nx;
        }
        /// Gate loss
        __device__ __forceinline__
        auto* gL() const {
            return gMeC() + nx;
        }

        __device__ __forceinline__
        auto* tP() const {
            return CAST_TO(TokenIdxTuple, book + gRl);
        }
        __device__ __forceinline__
        auto* tF() const {
            return CAST_TO(BookType, tP() + px * eCap);
        }
        __device__ __forceinline__
        auto* tV() const {
            return tF() + tM * px;
        }
        __device__ __forceinline__
        auto* eC() const {
            return tV() + px;
        }

        /// Expert parallelism specification
        /// Expert index -> resident GPU EP rank
        __device__ __forceinline__
        auto* ePs() const {
            return CAST_TO(BookType, book + gB);
        }
        /// Inverse specification
        /// GPU EP rank -> start index of {Expert indices}
        __device__ __forceinline__
        auto* iEPs() const {
            return ePs() + nx;
        }
        /// Peer Translation
        /// EP rank -> PE rank
        __device__ __forceinline__
        auto* pT() const {
            return iEPs() + nx;
        }
        /// Packet Sync array
        __device__ __forceinline__
        auto* pSA() const {
            return pT() + world;
        }

        /// Scheduler buffers and flag checkpoints
        __device__ __forceinline__
        auto* rQ() const {
            return CAST_TO(BookType, book + eDsA);
        }
        __device__ __forceinline__
        auto* sQ() const {
            return rQ() + blocks;
        }
        __device__ __forceinline__
        auto* tQH() const {
            return sQ() + blocks;
        }
        /// tQ sync array
        __device__ __forceinline__
        auto* tQS() const {
            return tQH() + blocks + SUBSCRIBERS;
        }
        __device__ __forceinline__
        auto* fC() const {
            return CAST_TO(bool, tQS() + world * nLx * tCM);
        }

        __device__ __forceinline__
        auto* tQ() const {
            return CAST_TO(Task, book + sBfC);
        }

        // Intermediate buffer
        template<typename Element>
        __device__ __forceinline__
        auto* xM() const {
            return CAST_TO(Element, tQ() + tQl);
        }

        // These must be together and last due to
        // 1. Contiguity requirements as we erase this region of memory after every step.
        // 2. Dependency on GateReductionLevel
        __device__ __forceinline__
        auto* tPT() const {
            return CAST_TO()
        }
        __device__ __forceinline__
        auto* rAt() const {
            return CAST_TO(BookType, book + xMtQ);
        }

        __device__ __forceinline__
        auto* bRSync() const {
            return rAt() + tM * tN;
        }
        __device__ __forceinline__
        auto* bRSoftM() const {
            return CAST_TO(maxPrecision, bRSync() + sl * Config::tiles<BLOCK_N>(px));
        }
        __device__ __forceinline__
        auto* bRsH() const {
            return CAST_TO(HeapTuple, bRSoftM() + 2 * sl);
        }
    };

    struct __align__(16) SchedulerConfig{
        unsigned int* statusQ;
        unsigned int* taskSignal;
        unsigned int* taskSync;
        // at least the upper bound for number of tasks, excluding interrupt tasks.
        // ceiling rounded to allow uniform strides for each producer's slice
        unsigned int tUB;
        unsigned int* tQHeads;
        unsigned int* xTQHeads;
        Task* taskQ;
        Task* xTaskQ;

        __forceinline__ __device__
        SchedulerConfig() = default;

        __forceinline__ __device__ __host__
        SchedulerConfig(cuda::std::byte* _bk,
               const unsigned int& numberBlocks,
               const unsigned int& numberSubscribers,
               const unsigned int& gtQCL,
               const unsigned int& tQRl,
               const unsigned int& _syncTasksBound,
               const unsigned int& _tUB) {
            tUB = _tUB;
            statusQ = CAST_TO(unsigned int, _bk);
            taskSignal = statusQ + numberBlocks;
            taskSync = taskSignal + numberBlocks;
            tQHeads = taskSync + _syncTasksBound;
            xTQHeads = tQHeads + numberSubscribers;
            taskQ = CAST_TO(Task, xTQHeads + gtQCL);
            xTaskQ = taskQ + numberBlocks * tQRl;
        }
    };

    __device__ __inline__ uint16_t seqNo;
    __constant__ __inline__ Config moeConfig{};
    __constant__ __inline__ SchedulerConfig schedulerState{};
    __constant__ __inline__ Bookkeeping bookkeeping{};
    __inline__ Config hostMoEConfig;

    namespace heap {
        template<unsigned int stage = 0, unsigned cell = 0>
        requires (stage < STAGES && cell < CELLS)
        __device__ __forceinline__
        auto* advance(unsigned int const& peer, unsigned int const& expert) {
            return moeConfig.sHeap + moeConfig.expertCapacity * moeConfig.embedDim *
                (moeConfig.numExperts * (CELLS * (peer * STAGES + stage) + cell) + expert);
        }
    }

    template<typename E = PacketStage> requires cuda::std::is_integral_v<cuda::std::underlying_type_t<E>>
    __device__ __forceinline__
    uint64_t constructSignal(E const& signal, uint64_t const& tagAlong = 0U){
        return tagAlong + signal + seqNo;
    }
}
#endif //ARISTOS_TYPES_CUH
