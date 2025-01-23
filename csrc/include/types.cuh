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
#define WARP_SIZE 32U
#define SUBSCRIBERS (THREADS - WARP_SIZE)
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
    using mp_t = float; // no support for double, unfortunately
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
    struct __align__(4) TQState {
        uint16_t tQTail;
        uint16_t tasks;
    };

    __device__
    enum class PacketStage {
        initial,
        last,
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

    __device__
    struct __align__(8) RingSoftmaxPayload {
        mp_t mI;
        cute::half_t dI;
        uint16_t signal;
    };
    __device__
    struct __align__(8) RingTopKPayload {
        mp_t sV;
        uint16_t sIdx;
        uint16_t signal;
    };
    
    template<PacketStage p = PacketStage::initial>
    __device__
    struct __align__(8) SignalPayload {
        static_assert(p == PacketStage::initial);
        uint routedTokens;
        uint16_t seqBit;
        uint16_t totalTilesM;
    };

    template<>
    __device__
    struct __align__(8) SignalPayload<PacketStage::last> {
        uint batchIdx;
        uint16_t seqBit;
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
    using TokenIdxTuple = cuda::std::pair<unsigned int, mp_t>;
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
        // crd2Idx(peer, expertIdx, offset)
        unsigned int syncIdx = 0UL;
        unsigned int tileIdx = 0U;
        uint16_t tileSize = 0U; // <= BLOCK_M
        unsigned int peerIdx = 0U;
        //padded
        unsigned int M = 0U;
        unsigned int flagIdx = 0U;
        unsigned int batchIdx = 0U;
        unsigned int expertIdx = 0U;
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
            const unsigned int& _syncIdx,
            const unsigned int& _tile,
            const unsigned int& _M,
            const unsigned int& _flagIdx,
            const unsigned int& _size,
            const unsigned int& _peerIdx,
            const unsigned int& _batchIdx):
        aData(_aData), bData(_bData),
        cData(_cData), dData(_dData),
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
        const unsigned int& _M,
        const unsigned int& _expertIdx):
        aData(_aData), bData(_bData), cData(_cData), tileIdx(_tile), tileSize(_size), M(_M), expertIdx(_expertIdx),
        taskType(_taskType){}


        __device__ __forceinline__
        explicit Task(const TaskType& _taskType):
        taskType(_taskType) {}
    };

    /// Information about auxiliary data structures comprising bookkeeping state
    /// Includes length of data structures (arrays) and pointer arithmetic functions
    struct __align__(16) Bookkeeping {
        /// default type for bookkeeping data structures
        using BookType = unsigned int;
        using TKTuple = cuda::std::pair<BookType, mp_t>;
        using EDT = cuda::std::tuple<uint, uint, uint>;
        cuda::std::byte* book = nullptr;
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
        unsigned long int bookSize = 0UL;

        /// Task Q length per producer
        unsigned int tPs = 0U;
        /// Task Q maximum length
        unsigned int tQml = 0U;
        /// Task Q length
        unsigned int tQl = 0U;
        /// EP world
        unsigned int world = 0U;
        /// sequence length
        unsigned int sl = 0U;
        /// number of experts
        unsigned int nx = 0U;
        /// number of local experts
        unsigned int nLx = 0U;
        /// hidden projection dimension
        unsigned int pd = 0U;
        /// padded number of experts
        unsigned int px = 0U;
        /// tiles spanning sequence length
        unsigned int tM = 0U;
        /// tiles spanning embedding dimension
        unsigned int tN = 0U;
        /// tiles spanning capacity
        unsigned int tCM = 0U;
        /// processors
        unsigned int blocks = 0U;
        /// expert capacity
        unsigned int eCap = 0U;
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
            cuda::barrier<cuda::thread_scope_device>* _blockade) :
        book(_book), world(_world), sl(_sl), nx(_nx), nLx(_nLx), pd(_pd), px(_px),
        tM(Config::tiles<BLOCK_M>(_sl)),
        tN(Config::tiles<BLOCK_N>(_embedDim)),
        tCM(Config::tiles<BLOCK_M>(_eCapacity)), blocks(_blocks), eCap(_eCapacity),
        deviceBlockade(_blockade) {
            if (_nx == 1)[[unlikely]] {
                // For this case, using any function other than xM yields undefined behavior
                xMtQ = sizeof(mp_t) * _world * _nLx * tCM * tN;
                bookSize = xMtQ;
            }
            else {
                gRl = sizeof(mp_t) * (_sl * px + (2 * px + 1));
                gB = gRl + sizeof(TokenIdxTuple) * (px * _eCapacity) + sizeof(BookType) * px * (tM + 3);
                eDsA = gB + sizeof(BookType) * (4 * nx + world + 1);
                const unsigned int fCl = sizeof(bool) * (world * nLx + nx * tCM * tN);
                sBfC = eDsA + sizeof(BookType) * (3 * blocks + SUBSCRIBERS + world * nLx * tCM) + fCl;
                // maximum gemm tiles/tasks scheduled by subscriber threads
                auto sT = world * nLx * tCM * tN + tCM * tN * nx;
                tPs = cute::ceil_div(sT, SUBSCRIBERS);
                sT = tPs * SUBSCRIBERS;
                // maximum gemm tiles/tasks scheduled by processors
                const auto pT = world * nLx * tCM * Config::tiles<BLOCK_N>(pd);
                tQl = sizeof(Task) * (sT + pT);
                tQml = tQl + blocks; // interrupt tasks
                xMtQ = sBfC + tQml + sizeof(mp_t) * world * nLx * tCM * tN;
                brs = _sl * Config::tiles<BLOCK_N>(_px) * (sizeof(RingSoftmaxPayload) + 2 * sizeof(RingTopKPayload));
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
            const unsigned int& _world){
            const auto tCM = Config::tiles<BLOCK_M>(_eCap);
            const auto tN = Config::tiles<BLOCK_N>(_embedDim);
            if (_nx == 1) {
                return sizeof(mp_t) * _world * _nLx * tCM * tN;
            }
            const auto tM = Config::tiles<BLOCK_M>(_sl);
            const auto gRl = sizeof(mp_t) * (_sl * _px + (2 * _px + 1));
            const auto gB = gRl + sizeof(TokenIdxTuple) * (_px * _eCap) + sizeof(BookType) * _px * (tM + 3);
            const auto eDsA = gB + sizeof(BookType) * (4 * _nx + _world + 1);
            const auto fCl = sizeof(bool) * (_world * _nLx + _nx * tCM * tN);
            const auto sBfC = eDsA + sizeof(BookType) * (3 * _blocks + THREADS - 2 + _world * _nLx * tCM) + fCl;
            // maximum gemm tiles/tasks scheduled by subscriber threads
            auto sT = _world * _nLx * tCM * tN + tCM * tN * _nx;
            sT = cute::ceil_div(sT, SUBSCRIBERS) * SUBSCRIBERS;
            // maximum gemm tiles/tasks scheduled by processors
            const auto pT = _world * _nLx * tCM * Config::tiles<BLOCK_N>(_pd);
            const auto tQl = sizeof(Task) * (sT + pT);
            const auto tQml = tQl + _blocks; // interrupt tasks
            const auto xMtQ = sBfC + tQml + sizeof(mp_t) * _world * _nLx * tCM * tN;
            const auto brs = _sl * Config::tiles<BLOCK_N>(_px) * (sizeof(RingSoftmaxPayload) + 2 * sizeof(RingTopKPayload));
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
            return CAST_TO(mp_t, book + sizeof(mp_t) * (sl * px));
        }
        /// Gate mean expert counts
        __device__ __forceinline__
        auto* gMeC() const {
            return gML() + nx;
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

        /// Packet data structures
        __device__ __forceinline__
        auto* pDs() const {
            return tV() + px;
        }
        __device__ __forceinline__
        auto* eC() const {
            return pDs();
        }

        /// Expert parallelism specification
        /// Expert index -> resident GPU EP rank
        __device__ __forceinline__
        auto* ePs() const {
            return CAST_TO(BookType, book + gB);
        }
        /// Expert index to local expert index
        __device__ __forceinline__
        auto* eLs() const {
            return ePs() + nx;
        }
        /// Expert Data
        /// remote experts first: {actual & local expert idx, peer idx}
        __device__ __forceinline__
        auto* eD() const {
            return CAST_TO(EDT, ePs() + nx);
        }
        /// number of remote experts
        __device__ __forceinline__
        auto* nRx() const {
            return CAST_TO(BookType, eD() + nx);
        }
        /// Peer Translation
        /// EP rank -> PE rank
        __device__ __forceinline__
        auto* pT() const {
            return nRx() + 1;
        }
        /// Packet Sync array
        __device__ __forceinline__
        auto* pSA() const {
            return pT() + world;
        }

        /// Scheduler buffers and flag checkpoints
        /// processors' doorbell
        __device__ __forceinline__
        auto* pDB() const {
            return CAST_TO(BookType, book + eDsA);
        }
        __device__ __forceinline__
        auto* sQ() const {
            return pDB() + blocks;
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

        // processor interrupts
        __device__ __forceinline__
        auto* tQI() const {
            return tQ() + tQl;
        }

        // Intermediate buffer
        template<typename Element = cuda::std::byte>
        __device__ __forceinline__
        auto* xM() const {
            return CAST_TO(Element, tQ() + tQml);
        }

        // These must be together and last due to
        // 1. Contiguity requirements as we erase this region of memory after every step.
        // 2. Dependency on GateReductionLevel
        __device__ __forceinline__
        auto* gateBk() const {
            // Entrypoint for vectorized memory cleaning
            return CAST_TO(uint4, book + xMtQ);
        }
        __device__ __forceinline__
        auto* bRsP() const {
            return CAST_TO(RingSoftmaxPayload, gateBk());
        }
        /// Ring top k flags
        /// Two sets for pipelining termination phase of round i and initial phase of round i + 1
        __device__ __forceinline__
        auto* rTp() const {
            return CAST_TO(RingTopKPayload, bRsP() + sl * Config::tiles<BLOCK_N>(px));
        }
    };

    __device__ __inline__ uint16_t seqBit;
    __constant__ __inline__ Config moeConfig{};
    __constant__ __inline__ Bookkeeping bookkeeping{};
    __inline__ Config hostMoEConfig;

    template<typename E = PacketStage> requires cuda::std::is_integral_v<cuda::std::underlying_type_t<E>>
    __device__ __forceinline__
    uint64_t constructSignal(E const& signal, uint64_t const& tagAlong = 0U){
        return tagAlong + signal + seqBit;
    }

    namespace heap {
        // The symmetric heap is a 6-D tensor (P, S, C, E, M, H)
        /// where P, S, C, E, M, and H  denote dimensions for peers, communication stages,
        /// cells, experts, expert capacity, and token hidden dimension, respectively.
        template<unsigned int stage = 0, unsigned cell = 0, unsigned long int nBytes = 1>
        requires (stage < STAGES && cell < CELLS)
        __device__ __forceinline__
        auto* advance(cuda::std::byte* __restrict__ const& sHeap, const uint& cellSize, const uint& expertSlots,
            const uint& tokenDim, const uint& peer, const uint& expert, const uint& token = 0){
            return sHeap + (cellSize * (expertSlots * (CELLS * (peer * STAGES + stage) + cell) + expert) +
                token * tokenDim) * nBytes;
        }
    }
}
#endif //ARISTOS_TYPES_CUH
