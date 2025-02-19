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
#define CONST_CAST_TO(T, p) static_cast<const T*>(static_cast<const void*>(p))
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
#define GEMMs 2U // per expert
#define REGINALD 128 // max registers per thread

#define TOPO_LOOP_TRIP 4U // this may be too much
#define BETA_BUFFER (1024UL * 1024UL) // 1MB
#define ALPHA_BUFFER 1024UL // 1KB
#define NANO_TO_MILLI (cuda::std::nano::den / cuda::std::milli::den)
#define NANO_TO_MICRO (cuda::std::nano::den / cuda::std::micro::den)
#define BYTE_MAX cuda::std::numeric_limits<cuda::std::underlying_type_t<cuda::std::byte>>::max()
#define TO_MB(b) (static_cast<float>(b) / (1024.0f*1024.0f))
#define BETA_MB 1024.0f // 1GB

#include <cuda/barrier>
#include <cuda/std/tuple>
#include <cuda/std/array>
#include <cute/tensor.hpp>

namespace aristos{
    template<typename V>
        concept TensorValueType = cuda::std::is_same_v<V, cute::half_t> ||
            cuda::std::is_same_v<V, cute::bfloat16_t> ||
            cuda::std::is_same_v<V, cute::tfloat32_t> ||
            cuda::std::is_same_v<V, float> /*||
            cuda::std::is_same_v<V, cute::float_e4m3_t> ||
            cuda::std::is_same_v<V, cute::float_e5m2_t>*/;

    template<typename T>
    concept isTensor = cute::is_tensor<T>::value && TensorValueType<typename T::value_type>;
    template<typename T>
    concept isMatrix = isTensor<T> && rank(T{}) == 2;

    template<typename S>
    struct ToCute {
        using T = S;
        static_assert(aristos::TensorValueType<T>);
    };
    template<>
    struct ToCute<__half> {
        using T = cute::half_t;
    };
    template<>
    struct ToCute<__nv_bfloat16> {
        using T = cute::bfloat16_t;
    };

    template<typename S>
    requires(aristos::TensorValueType<S>)
    struct ToCDx {
        using T = S;
    };
    template<>
    struct ToCDx<cute::tfloat32_t> {
        using T = float;
    };
    template<>
    struct ToCDx<cute::half_t> {
        using T = __half;
    };
    template<>
    struct ToCDx<cute::bfloat16_t> {
        using T = __nv_bfloat16;
    };
    using mp_t = float; // or tf32
    using GEA = float;
    using specType = unsigned int;
    using flagsType = uint64_t;

    using Nano = cuda::std::chrono::duration<float, cuda::std::nano>;
    using Milli = cuda::std::chrono::duration<float, cuda::std::milli>;
    using ull_t = unsigned long long int;

    struct __align__(8) floatPair {
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

    // These could be much more, as supported by CUTLASS
    __host__ __device__
    enum ActivationFunction: uint8_t {
        ReLu,
        GeLU
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

    __host__ __device__
    enum class UseBarrier {
        yes,
        no
    };

    __device__
    enum class DropTokens {
        yes,
        no
    };

    __device__
    enum ReadySignal : unsigned int {
        observed,
        ready
    };

    __device__
    enum class CombineMode {
        single,
        multithreaded
    };

    __device__
    enum class JobType : uint8_t {
        training,
        inference
    };

    template<
        CombineMode c,
        typename T
    >
    requires(aristos::TensorValueType<T>)
    struct VCT {
        static_assert(c == CombineMode::single);
        using Element = T;
    };
    template<
        typename T
    >
    requires(aristos::TensorValueType<T>)
    struct VCT<CombineMode::multithreaded, T> {
        // tf32 does not have device intrinsics for atomic operations, so we use float instead
        using Element = cuda::std::conditional_t<cuda::std::is_same_v<T, cute::tfloat32_t>,
            float, T>;
    };

    struct __align__(4) WorkerAttribute{
        cute::half_t throughput; // ms for a single expert
        uint16_t memoryCapacity; // upper bound of experts that we can accommodate
    };
    struct __align__(8) TopologySignal{
        unsigned int signal;
        WorkerAttribute wA;
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

    template<class Engine, class Layout>
    struct isRegister<cute::Tensor<Engine, Layout>> :
    cuda::std::conditional_t<cute::is_rmem_v<cute::Tensor<Engine, Layout>>,
    cuda::std::true_type, cuda::std::false_type> {};

    template <class T>
    constexpr bool isRegisterV = isRegister<T>::value;

    struct __align__(16) InitialConfig {
        const uint vocabSize;
        const uint numLayers;
        const uint globalBatch;
        const uint miniBatch;
        const uint moeFrequency;
        const uint seqLen;
        const uint embedDim;
        const uint hiddenProjDim;
        const uint16_t k;
        const uint capacityFactor;
        const ulong numParameters;
        // logical elements
        const uint p2pBuffer; // in MB
        const uint gradBuffer; // in MB
        const uint16_t numExperts;
        const uint16_t redAmount;
        const bool shouldDrop;
        const ActivationFunction actF;
        const JobType jobType;

        __host__
        InitialConfig(const uint& _vocabSize,
            const uint& _numLayers,
            const uint& _globalBatch,
            const uint& _miniBatch,
            const uint& _moeFreq,
            const uint& _seqLen,
            const uint& _embedDim,
            const uint& _hiddenProjDim,
            const uint16_t& _k,
            const uint& _capacityFactor,
            const uint16_t& _numExperts,
            const bool& _shouldDrop,
            const ActivationFunction& _actF,
            const bool& _isTraining,
            const uint16_t& _redAmount = 1):
        vocabSize(_vocabSize),
        numLayers(_numLayers),
        globalBatch(_globalBatch),
        miniBatch(_miniBatch),
        moeFrequency(_moeFreq),
        seqLen(_seqLen * miniBatch), // s * b
        embedDim(_embedDim), // h
        hiddenProjDim(_hiddenProjDim),
        k(_k), capacityFactor(_capacityFactor),
        numParameters(embedDim * (numLayers * (12U * embedDim + 13U) + (vocabSize + embedDim))),
        p2pBuffer(cute::ceil_div(seqLen * miniBatch * embedDim, 1024U * 1024U)),
        gradBuffer(cute::ceil_div(numParameters, 1024U * 1024U)),
        numExperts(_numExperts), redAmount(_redAmount),
        shouldDrop(_shouldDrop), actF(_actF), jobType(_isTraining ? JobType::training : JobType::inference)
        {}

        // formula for total number of parameters
        // source: https://arxiv.org/abs/2401.14489
        __forceinline__
        static auto parameterCount(const uint& embedDim, const uint& numLayers, const uint& vocabSize) {
            return embedDim * (numLayers * (12U * embedDim + 13U) + (vocabSize + embedDim));
        }

        // source: https://arxiv.org/pdf/2201.11990
        template<typename Element>
        __forceinline__
        auto bytesPerParameter() const {
            return jobType == JobType::training ? 2 * sizeof(Element) + 12 : sizeof(Element);
        }

        __forceinline__
        auto expertCapacity(const uint& epWorld) const {
            return cute::ceil_div(seqLen * miniBatch * capacityFactor, epWorld);
        }
    };
    // Needed for decider
    struct __align__(16) ModelConfig{
        const unsigned int numLayers;
        const unsigned int globalBatch;
        const unsigned int redAmount;
        const unsigned int miniBatch;
        const unsigned int moeFreq;
        const unsigned int p2pBuffer;
        const unsigned int gradBuffer;
        __host__
        ModelConfig(const unsigned int& numLayers, const unsigned int& redAmount,
                    const unsigned int& globalBatch,
                    const unsigned int& miniBatch, const unsigned int& moeFreq,
                    const unsigned int& p2PBuffer, const unsigned int& gradBuffer) :
                    numLayers(numLayers), globalBatch(globalBatch),
                    redAmount(redAmount), miniBatch(miniBatch), moeFreq(moeFreq),
                    p2pBuffer(p2PBuffer), gradBuffer(gradBuffer) {}
    };

    // Index and gate combine weight
    using TokenIdxTuple = cuda::std::pair<unsigned int, mp_t>;

    enum class TaskType : uint8_t {
        preGEMM,
        postGEMM,
        combine,
        Interrupt,
    };

    struct __align__(16) Task {
        // D = A * B + C
        // sensible sentinel values
        const cuda::std::byte* aData = nullptr;
        cuda::std::array<const cuda::std::byte*, GEMMs> bData = {};
        cuda::std::array<cuda::std::byte*, GEMMs> cData = {};
        cuda::std::array<const cuda::std::byte*, GEMMs> dData = {};
        // crd2Idx(peer, expertIdx, offset)
        unsigned int syncIdx = 0UL;
        unsigned int tileIdx = 0U;
        unsigned int peerIdx = 0U;
        //padded
        unsigned int M = 0U;
        unsigned int flagIdx = 0U;
        unsigned int batchIdx = 0U;
        unsigned int expertIdx = 0U;
        uint16_t tileSize = 0U; // <= BLOCK_M
        TaskType taskType = TaskType::Interrupt;
        bool isPeerRemote = false;

        __forceinline__ __device__
        Task() = default;

        // Stage 1
        __device__ __forceinline__
        Task(const TaskType& _taskType,
            const cuda::std::byte* const& _aData,
            const cuda::std::array<const cuda::std::byte*, GEMMs>& _bData,
            const cuda::std::array<cuda::std::byte*, GEMMs>& _cData,
            const cuda::std::array<const cuda::std::byte*, GEMMs>& _dData,
            const unsigned int& _syncIdx,
            const unsigned int& _tile,
            const unsigned int& _M,
            const unsigned int& _flagIdx,
            const uint16_t& _size,
            const unsigned int& _peerIdx,
            const unsigned int& _batchIdx,
            const bool& _isPeerRemote):
        aData(_aData), bData(_bData),
        cData(_cData), dData(_dData),
        syncIdx(_syncIdx), tileIdx(_tile), peerIdx(_peerIdx), M(_M), flagIdx(_flagIdx),
        batchIdx(_batchIdx), tileSize(_size), taskType(_taskType), isPeerRemote(_isPeerRemote){}

        // Stage 2
        __device__ __forceinline__
        Task(const TaskType& _taskType,
        const cuda::std::byte*  const& _aData,
        const cuda::std::array<const cuda::std::byte*, GEMMs>& _bData,
        const cuda::std::array<cuda::std::byte*, GEMMs>& _cData,
        const unsigned int& _size,
        const unsigned int& _tile,
        const unsigned int& _M,
        const unsigned int& _expertIdx):
        aData(_aData), bData(_bData), cData(_cData), tileIdx(_tile), M(_M), expertIdx(_expertIdx),
        tileSize(_size), taskType(_taskType){}


        __device__ __forceinline__
        explicit Task(const TaskType& _taskType):
        taskType(_taskType) {}
    };

    /// Information about auxiliary data structures comprising bookkeeping state
    /// Includes length of data structures (arrays) and pointer arithmetic functions
    using BookType = unsigned int;
    using EDT = cuda::std::tuple<uint, uint, uint>;
    struct __align__(16) Bookkeeping {
        cuda::std::byte* sHeap = nullptr;
        flagsType* flags = nullptr;
        /// default type for bookkeeping data structures
        cuda::std::byte *book = nullptr;
        /// Note the below lengths are cumulative sums.
        unsigned long int eDsA = 0UL;
        /// Scheduler buffers and flag checkpoints in bytes
        unsigned long int sBfC = 0UL;
        /// gRl + gB + eDsA + sBfC + brs
        unsigned long int bookSize = 0UL;
        unsigned int tQXt = 0UL;
        /// length of gTQHeads
        unsigned int gtQCl = 0U;
        /// Block Ring Softmax flags in bytes, non-cumulative
        unsigned int brs = 0UL;
        /// gate routing and loss vectors in bytes
        unsigned int gRl = 0U;
        /// Task Q length per producer
        unsigned int tPs = 0U;
        /// Task Q maximum length
        unsigned int tQml = 0U;
        /// Task Q length
        unsigned int tQl = 0U;
        /// sequence length
        unsigned int sl = 0U;
        /// embedding dimension
        unsigned int ed = 0U;
        /// hidden projection dimension
        unsigned int pd = 0U;
        /// tiles spanning sequence length
        unsigned int tM = 0U;
        /// tiles spanning embedding dimension
        unsigned int tN = 0U;
        /// tiles spanning capacity
        unsigned int tCM = 0U;
        /// expert capacity
        unsigned int eCap = 0U;
        /// EP rank
        uint16_t rank = 0U;
        /// EP world
        uint16_t world = 0U;
        /// number of experts
        uint16_t nx = 0U;
        /// top k
        uint16_t k = 0U;
        /// padded number of experts
        uint16_t px = 0U;
        /// number of local experts
        uint16_t nLx = 0U;
        /// expert slots
        uint16_t xs = 0U;
        /// processors
        uint16_t blocks = 0U;
        /// Encoded result of template parameters
        uint8_t fId = 0U;
        /// Encoded result of template parameters
        uint8_t pfId = 0U;

        __device__ __forceinline__
        Bookkeeping() = default;

        __host__ __forceinline__
        explicit Bookkeeping(
            cuda::std::byte* _sHeap,
            flagsType* _flags,
            cuda::std::byte* const& _book,
            const unsigned int& _sl,
            const uint16_t& _nx,
            const uint16_t& _k,
            const uint16_t& _nLx,
            const uint16_t& _xS,
            const uint16_t& _rank,
            const unsigned int& _pd,
            const unsigned int& _embedDim,
            const unsigned int& _eCapacity,
            const uint16_t& _blocks,
            const uint16_t& _world,
            const uint8_t& _fId,
            const uint8_t& _pfId,
            const unsigned int& _eNb // number of bytes for the matrix element type
            ) :
        sHeap(_sHeap), flags(_flags),
        book(_book),
        sl(_sl), ed(_embedDim), pd(_pd),
        tM(tiles<BLOCK_M>(_sl)),
        tN(tiles<BLOCK_N>(_embedDim)),
        tCM(tiles<BLOCK_M>(_eCapacity)),
        eCap(_eCapacity), rank(_rank), world(_world), nx(_nx), k(_k),
        px(pad<BLOCK_N>(_nx)), nLx(_nLx), xs(_xS), blocks(_blocks), fId(_fId), pfId(_pfId){
            if (_nx > 1)[[likely]] {
                const bool isSingleBlockGate = _nx <= BLOCK_N;
                // maximum gemm tiles/tasks scheduled by processors
                const auto prT = world * nLx * tCM * tiles<BLOCK_N>(pd);
                // maximum gemm tiles/tasks scheduled by subscriber threads
                auto sT = world * nLx * tCM * tN + tCM * tN * nx;
                tPs = cute::ceil_div(sT, SUBSCRIBERS);
                sT = tPs * SUBSCRIBERS;
                tQl = sizeof(Task) * (sT + prT);
                tQml = tQl + blocks * sizeof(Task); // interrupt tasks
                brs = tQml + (isSingleBlockGate ? 0U : sl * tiles<BLOCK_N>(px) *
                    (sizeof(RingSoftmaxPayload) + 2 * sizeof(RingTopKPayload)));
                tQXt = brs + sizeof(EDT) * _nx + sizeof(TokenIdxTuple) * (px * _eCapacity) +
                    sizeof(cuda::barrier<cuda::thread_scope_device>);
                gRl = tQXt + sizeof(mp_t) * (2 * nx + 1);
                eDsA = gRl + sizeof(BookType) * (4 * nx + world + 1);
                const unsigned int fCl = sizeof(bool) * (world * nLx + nx * tCM * tN);
                sBfC = eDsA + sizeof(BookType) * 2 * (blocks + world * nLx * tCM) + fCl;
            }
            gtQCl = world * nLx * tCM;
            bookSize = sBfC + _eNb * (_world * _nLx * tCM * tN);
        }

        /// Needed for malloc
        __host__ __forceinline__
        static unsigned long int bookLength(
            const unsigned int& _sl,
            const unsigned int& _nx,
            const unsigned int& _nLx,
            const unsigned int& _pd,
            const unsigned int& _embedDim,
            const unsigned int& _eCap,
            const unsigned int& _blocks,
            const unsigned int& _world,
            const unsigned int& _eNb // number of bytes for the matrix element type
            ){
            const bool isSingleBlockGate = _nx <= BLOCK_N;
            const auto tCM = tiles<BLOCK_M>(_eCap);
            const auto tN = tiles<BLOCK_N>(_embedDim);
            auto sBfC = 0UL;
            const auto _px = pad<BLOCK_N>(_nx);
            if (_nx > 1)[[likely]] {
                // maximum gemm tiles/tasks scheduled by processors
                const auto prT = _world * _nLx * tCM * tiles<BLOCK_N>(_pd);
                // maximum gemm tiles/tasks scheduled by subscriber threads
                auto sT = _world * _nLx * tCM * tN + tCM * tN * _nx;
                const auto tPs = cute::ceil_div(sT, SUBSCRIBERS);
                sT = tPs * SUBSCRIBERS;
                const auto tQl = sizeof(Task) * (sT + prT);
                const auto tQml = tQl + _blocks * sizeof(Task); // interrupt tasks
                const auto brs = tQml + (isSingleBlockGate ? 0U : _sl * tiles<BLOCK_N>(_px) *
                                    (sizeof(RingSoftmaxPayload) + 2 * sizeof(RingTopKPayload)));
                const auto tQXt = brs + sizeof(EDT) * _nx + sizeof(TokenIdxTuple) * (_px * _eCap) +
                    sizeof(cuda::barrier<cuda::thread_scope_device>);
                const auto gRl = tQXt + sizeof(mp_t) * (2 * _nx + 1);
                const auto eDsA = gRl + sizeof(BookType) * (4 * _nx + _world + 1);
                const unsigned int fCl = sizeof(bool) * (_world * _nLx + _nx * tCM * tN);
                sBfC = eDsA + sizeof(BookType) * 2 * (_blocks + _world * _nLx * tCM) + fCl;
            }
            return sBfC + _eNb * (_world * _nLx * tCM * tN);
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

        __device__ __forceinline__
        auto cellSize() const {
            return eCap * ed;
        }

        /**********Salami slice Pointers!************/
        /// task queue
        __device__ __forceinline__
        auto* tQ() const {
            return CAST_TO(Task, book);
        }

        // processor interrupts
        __device__ __forceinline__
        auto* tQI() const {
            return tQ() + tQl;
        }

        static_assert(alignof(Task) % alignof(RingSoftmaxPayload) == 0
            && alignof(Task) % alignof(RingTopKPayload) == 0
            && alignof(RingSoftmaxPayload) % alignof(RingTopKPayload) == 0);
        /***********CONTIGUOUS**************/
        __device__ __forceinline__
        auto* gateBk() const {
            // Entrypoint for vectorized memory cleaning
            return CAST_TO(uint2, book + tQml);
        }
        __device__ __forceinline__
        auto* bRsP() const {
            return CAST_TO(RingSoftmaxPayload, gateBk());
        }
        /// Ring top k flags
        /// Two sets for pipelining termination phase of round i and initial phase of round i + 1
        __device__ __forceinline__
        auto* rTp() const {
            return CAST_TO(RingTopKPayload, bRsP() + sl * tiles<BLOCK_N>(px));
        }

        static_assert(alignof(RingTopKPayload) % alignof(cuda::barrier<cuda::thread_scope_device>) == 0);
        /// Device-wide barrier
        __host__ __device__ __forceinline__
        auto* dB() const {
            return CAST_TO(cuda::barrier<cuda::thread_scope_device>, book + brs);
        }
        /***********CONTIGUOUS**************/
        static_assert(alignof(cuda::barrier<cuda::thread_scope_device>) % alignof(EDT) == 0);
        /// Expert Data
        /// remote experts first: {actual & local expert idx, peer idx}
        __host__ __device__ __forceinline__
        auto* eD() const {
            return CAST_TO(EDT, dB() + 1);
        }

        static_assert(alignof(EDT) % alignof(TokenIdxTuple) == 0);
        __device__ __forceinline__
        auto* tP() const {
            return CAST_TO(TokenIdxTuple, eD() + nx);
        }

        static_assert(alignof(TokenIdxTuple) % alignof(mp_t) == 0);
        /// Gate mean logits
        __device__ __forceinline__
        auto* gML() const {
            return CAST_TO(mp_t, book + tQXt);
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

        static_assert(alignof(mp_t) % alignof(BookType) == 0);
        /***********CONTIGUOUS**************/
        /// Packet data structures
        __host__ __device__ __forceinline__
        auto* pDs() const {
            return CAST_TO(BookType, book + gRl);
        }
        __device__ __forceinline__
        auto* eC() const {
            return pDs();
        }
        /// Expert parallelism specification
        /// Expert index -> resident GPU EP rank
        __host__ __device__ __forceinline__
        auto* ePs() const {
            return CAST_TO(BookType, book + gRl) + nx;
        }
        /// Peer Translation
        /// EP rank -> PE rank
        __host__ __device__ __forceinline__
        auto* pT() const {
            return ePs() + nx;
        }
        /// Expert index to local expert index
        __host__ __device__ __forceinline__
        auto* eLs() const {
            return pT() + world;
        }

        /*************CONTIGUOUS************/

        /// number of remote experts
        __host__ __device__ __forceinline__
        auto* nRx() const {
            return eLs() + nx;
        }

        /// Packet Sync array
        __device__ __forceinline__
        auto* pSA() const {
            return nRx() + 1;
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
            return tQH() + world * nLx * tCM;
        }
        __device__ __forceinline__
        auto* fC() const {
            return CAST_TO(BookType, tQS() + world * nLx * tCM);
        }

        // Intermediate buffer
        static_assert(alignof(BookType) % alignof(cuda::std::byte) == 0);
        __device__ __forceinline__
        auto* xM() const {
            return book + sBfC;
        }
    };

    __device__ __inline__ uint16_t seqBit;
    __constant__ __inline__ Bookkeeping bookkeeping{};
    __inline__ Bookkeeping hostBookkeeping{};
    __inline__ bool isInitialized = false;
    __inline__ auto aristosStream = cudaStreamPerThread;

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
    __host__ __forceinline__
    uint8_t eTA(const torch::ScalarType& sT, const ActivationFunction& aF) {
        switch (sT) {
            case torch::kFloat32:
                return aF;
            case torch::kFloat16:
                return 2U + aF;
            case torch::kBFloat16:
                return 4U + aF;
            break;
            default:
                return 2U + aF; // default is half_t
        }
    }
}
#endif //ARISTOS_TYPES_CUH
