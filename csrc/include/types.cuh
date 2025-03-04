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
#define GEMMs 2U // per expert

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
#include <cute/numeric/integral_constant.hpp>
#include <cute/tensor.hpp>
#include <cutlass/epilogue/thread/activation.h>

#include "arch.cuh"

namespace aristos{
    template<typename V>
        concept TensorValueType = cuda::std::is_same_v<V, cute::half_t> ||
            cuda::std::is_same_v<V, cute::bfloat16_t> ||
            cuda::std::is_same_v<V, cute::tfloat32_t> ||
            cuda::std::is_same_v<V, float> /*||
            cuda::std::is_same_v<V, cute::float_e4m3_t> ||
            cuda::std::is_same_v<V, cute::float_e5m2_t>*/;

    template <class T>
    struct isTensor : cuda::std::false_type {};
    template <class Engine, class Layout>
    requires(TensorValueType<typename Engine::value_type>)
    struct isTensor<cute::Tensor<Engine,Layout>> : cuda::std::true_type {};
    template <class Engine, class Layout>
    requires(TensorValueType<typename Engine::value_type>)
    struct isTensor<const cute::Tensor<Engine,Layout>> : cuda::std::true_type {};

    template<typename T>
    concept isMatrix = isTensor<T>::value && cuda::std::is_same_v<decltype(rank(T{})), cute::Int<2>>;

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

    template<unsigned int dType>
    struct DType {
        static_assert(dType <= 3);
    };

    template<>
    struct DType<0U> {
        using DT = float;
    };

    template<>
    struct DType<1U> {
        using DT = cute::tfloat32_t;
    };

    template<>
    struct DType<2U> {
        using DT = cute::bfloat16_t;
    };

    template<>
    struct DType<3U> {
        using DT = cute::half_t;
    };

    template<
        unsigned int aFunction,
        typename Element
    > requires(TensorValueType<Element>)
    struct AFunction {
        static_assert(aFunction <= 2U);
    };

    template<typename Element>
    struct AFunction<0U, Element> {
        using DT = cutlass::epilogue::thread::ReLU<Element>;
    };

    template<typename Element>
    struct AFunction<1U, Element> {
        using DT = cutlass::epilogue::thread::GELU<Element>;
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

    __device__
    struct __align__(8) TQSignal{
        uint interrupt;
        uint signal; // one ahead
    };

    // These could be much more, as supported by CUTLASS
    __host__ __device__
    enum ActivationFunction: uint8_t {
        ReLu,
        GeLU
    };

    __device__
    enum class PacketStage: uint {
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
        cute::half_t throughput; // expert per ms; could be fractional
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
        uint16_t totalTilesM;
        uint16_t seqBit;
    };

    template<>
    __device__
    struct __align__(8) SignalPayload<PacketStage::last> {
        uint batchIdx;
        uint16_t tokensM; // <= BLOCK_M
        uint16_t seqBit;
    };

    /// Expert lookup info: key is global expert index
    __device__
    struct __align__(8) ELI {
        uint epRank; // host peer
        uint16_t localExpertIndex;
        uint16_t isRemote;
    };

    /// Peer lookup info: key is ep rank
    __device__
    struct __align__(4) PLI {
        uint16_t pe;
        uint16_t isRemote;
    };

    /// Packet Encoding Lookup info, retrievable in a single memory lookup
    /// Key is global expert index
    __device__
    struct __align__(16) PEL {
        cuda::std::byte* remoteSHeap;
        cuda::std::byte* remoteSFlags; //rank * expertSlots + xLIdx
        uint eC;
        uint pTT;
        uint16_t expertLocalIdx;
        uint16_t peer;
        uint16_t pe;
        uint16_t isRemote;
    };

    /// Aristos Compile-time Config
    struct ACC {
        using SYB = cute::C<1>;
        using GRL = cute::C<NUM_EXPERTS <= BLOCK_N ? GateReductionLevel::singleBlock :
            GateReductionLevel::multiBlock>;
        using TK = cute::C<E_TOP_K>;
        using CM = cute::C<(E_TOP_K > 1) ? CombineMode::multithreaded : CombineMode::single>;
        using Element = VCT<CM::value, DType<DTYPE>::DT>::Element;
        using DTK = cute::C<DROP_TOKENS? DropTokens::yes : DropTokens::no>;
        using ActivationOp = AFunction<HIDDEN_ACT, GEA>::DT;
        using ActivationOpX = cute::identity;
        using PeakHardware = aristos::Hardware<ARISTOS_ARCH, 255>;
        using JT = cute::C<IS_TRAINING? JobType::training : JobType::inference>;
        using S = cute::C<SEQ_LEN>;
        using P = cute::C<I_SIZE>;
        using H = cute::C<HIDDEN_SIZE>;
        using E = cute::C<NUM_EXPERTS>;
        // padded expert dimension
        using PX = cute::C<cute::ceil_div(E::value, BLOCK_N) * BLOCK_N>;
        using L = cute::C<NUM_LAYERS>;
        using F = cute::C<MOE_FREQ>;
        using GB = cute::C<GLOBAL_BATCH>;
        using MB = cute::C<MINI_BATCH>;
        // Global MoE Stages
        using GMS = cute::C<(JT::value == JobType::training ? 3 : 1) *
                (GB::value / MB::value) * (L::value / F::value)>;
        using BPP = cute::C<JT::value == JobType::training ? 2 * sizeof(Element) + 12 : sizeof(Element)>;
        // parameter count
        // source: https://arxiv.org/abs/2401.14489
        using PC = cute::C<H::value * (L::value * (12UL * H::value + 13U) + (VOCAB_SIZE + H::value))>;
        using GRB = cute::C<cute::ceil_div(PC::value, 1024 * 1024)>;
        using P2PB = cute::C<cute::ceil_div(S::value * MINI_BATCH * H::value, 1024 * 1024)>;
        using EC = cute::C<DTK::value == DropTokens::no ? S::value : cute::ceil_div(S::value, E::value)>;
        using SZ = cute::C<EC::value * H::value>;
        using TM = cute::C<S::value / BLOCK_M>;
        using TN = cute::C<P::value / BLOCK_N>;
        using TNx = cute::C<H::value / BLOCK_N>;
        using TCM = cute::C<EC::value / BLOCK_M>;
        using TPX = cute::C<PX::value / BLOCK_N>;
        using TSZ = cute::C<TM::value * cute::min(TNx::value, PeakHardware::blocks::value)>;

        // Scheduling state upper bound inside GEMM
        using TMU = cute::C<128>;
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

    // Index and gate combine weight
    using TokenIdxTuple = cuda::std::pair<unsigned int, mp_t>;

    enum class TaskType : uint8_t {
        preGEMM,
        postGEMM,
        combine
    };

    enum class EP {
        yes,
        no
    };

    __device__
    enum class FlagState : uint8_t {
        unidentified,
        identified,
        completed
    };

    struct __align__(16) Task {
        using TST = uint16_t;
        static_assert(BLOCK_M <= cuda::std::numeric_limits<TST>::max());
        // D = A * B + C
        // sensible sentinel values
        const cuda::std::byte* aData = nullptr;
        cuda::std::array<const cuda::std::byte*, GEMMs> bData = {};
        cuda::std::array<cuda::std::byte*, GEMMs> cData = {};
        cuda::std::array<const cuda::std::byte*, GEMMs> dData = {};
        // crd2Idx(peer, expertIdx, offset)
        unsigned int syncIdx = 0UL;
        unsigned int tileIdx = 0U;
        //padded
        unsigned int M = 0U;
        unsigned int flagIdx = 0U;
        unsigned int batchIdx = 0U;
        uint16_t peerIdx = 0U;
        uint16_t expertIdx = 0U;
        TST tileSize = 0U; // <= BLOCK_M
        TaskType taskType;
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
        syncIdx(_syncIdx), tileIdx(_tile),  M(_M), flagIdx(_flagIdx),
        batchIdx(_batchIdx), peerIdx(_peerIdx), tileSize(_size), taskType(_taskType), isPeerRemote(_isPeerRemote){}

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
    };

    /// Information about auxiliary data structures comprising bookkeeping state
    /// Includes length of data structures (arrays) and pointer arithmetic functions
    using BookType = unsigned int;
    using EDT = cuda::std::tuple<uint, uint, uint>;
    struct __align__(16) Bookkeeping {
        /// needed for free
        cuda::std::byte* symHeap = nullptr;
        flagsType* syncArray = nullptr;
        flagsType* syncCount = nullptr;
        flagsType* flags = nullptr;
        cuda::std::byte* sHeap = nullptr;
        /// default type for bookkeeping data structures
        cuda::std::byte *book = nullptr;
        /// gRl + gB + eDsA + sBfC + brs
        unsigned long int bookSize = 0UL;
        /// length of gTQHeads
        unsigned int gtQCl = 0U;
        unsigned int tPs = 0U;
        /// EP rank
        uint16_t rank = 0U;
        /// EP world
        uint16_t world = 0U;
        /// number of local experts
        uint16_t nLx = 0U;
        /// expert slots
        uint16_t xs = 0U;

        __host__ __device__ __forceinline__
        Bookkeeping() {
            // default for ffn with no experts
            sBfC = ACC::TSZ::value * sizeof(BookType) +
                sizeof(ACC::Element) * (ACC::S::value * ACC::P::value);
            bookSize = sBfC;
        }

        __host__ __forceinline__
        explicit Bookkeeping(cuda::std::byte* const& _symHeap,
            flagsType* const& _sA,
            flagsType* const& _flags,
            cuda::std::byte* const& _sHeap,
            cuda::std::byte* const& _book,
            const uint16_t& _nLx, // dynamically decided by an optimization algorithm
            const uint16_t& _rank,
            const uint16_t& _world,
            const uint16_t& _xS) : symHeap(_symHeap),
            syncArray(_sA), syncCount(_sA + _world), flags(_flags), sHeap(_sHeap), book(_book), rank(_rank),
            world(_world), nLx(_nLx), xs(_xS){
            constexpr auto TCM = ACC::TCM::value;
            constexpr auto TN = ACC::TN::value;
            constexpr auto blocks = ACC::PeakHardware::OS::processorBlocks::value;
            constexpr auto E = ACC::E::value;
            constexpr auto PX = ACC::PX::value;
            constexpr auto S = ACC::S::value;
            constexpr auto EC = ACC::EC::value;
            constexpr auto P = ACC::P::value;
            constexpr auto TPX = ACC::TPX::value;
            static_assert(E > 1);
            gtQCl = world * nLx * TCM;
            // maximum gemm tiles/tasks scheduled by processors
            const auto prT = world * nLx * TCM * tiles<BLOCK_N>(P);
            // maximum gemm tiles/tasks scheduled by subscriber threads
            auto sT = world * nLx * TCM * TN + TCM * TN * E;
            tPs = cute::ceil_div(sT, SUBSCRIBERS);
            sT = tPs * SUBSCRIBERS;
            tQl = sizeof(Task) * (sT + prT);
            tQml = tQl + blocks * sizeof(TQSignal) + E * sizeof(PEL); // processor doorbells
            brs = tQml + (ACC::GRL::value == GateReductionLevel::singleBlock ? 0U : S * TPX *
                (sizeof(RingSoftmaxPayload) + 2 * sizeof(RingTopKPayload)));
            tQXt = brs + sizeof(ELI) * E + sizeof(cuda::barrier<cuda::thread_scope_device>) + sizeof(PLI) * world;
            gRl = tQXt + sizeof(TokenIdxTuple) * (PX * EC) + (ACC::JT::value == JobType::inference ? 0U :
                sizeof(mp_t) * (2 * E + 1));
            eDsA = gRl + sizeof(BookType) * (2 * E + 1);
            sBfC = eDsA + sizeof(BookType) * 2 * (world * nLx * TCM) + blocks + TCM * TN * E;
            bookSize = sBfC + sizeof(ACC::Element) * (_world * _nLx * TCM * TN);
        }

        /// Needed for malloc
        __host__ __forceinline__
        constexpr static unsigned long int bookLength(const unsigned int& _nLx, const unsigned int& _world) {
            constexpr auto TCM = ACC::TCM::value;
            constexpr auto TN = ACC::TN::value;
            constexpr auto blocks = ACC::PeakHardware::OS::processorBlocks::value;
            constexpr auto E = ACC::E::value;
            constexpr auto PX = ACC::PX::value;
            constexpr auto S = ACC::S::value;
            constexpr auto EC = ACC::EC::value;
            constexpr auto P = ACC::P::value;
            constexpr auto TPX = ACC::TPX::value;
            // maximum gemm tiles/tasks scheduled by processors
            const auto prT = _world * _nLx * TCM * tiles<BLOCK_N>(P);
            // maximum gemm tiles/tasks scheduled by subscriber threads
            auto sT = _world * _nLx * TCM * TN + TCM * TN * E;
            const auto tPs = cute::ceil_div(sT, SUBSCRIBERS);
            sT = tPs * SUBSCRIBERS;
            const auto tQl = sizeof(Task) * (sT + prT);
            const auto tQml = tQl + blocks * sizeof(TQSignal) + E * sizeof(PEL); // processor doorbells
            const auto brs = tQml + (ACC::GRL::value == GateReductionLevel::singleBlock ? 0U :
                S * TPX * (sizeof(RingSoftmaxPayload) + 2 * sizeof(RingTopKPayload)));
            const auto tQXt = brs + sizeof(ELI) * E + sizeof(cuda::barrier<cuda::thread_scope_device>) +
                sizeof(PLI) * _world;
            const auto gRl = tQXt + sizeof(TokenIdxTuple) * (PX * EC) +
                (ACC::JT::value == JobType::inference ? 0U : sizeof(mp_t) * (2 * E + 1));
            const auto eDsA = gRl + sizeof(BookType) * (2 * E + 1);
            const auto sBfC = eDsA +
                sizeof(BookType) * 2 * (_world * _nLx * TCM) + blocks + TCM * TN * E;
            return sBfC + sizeof(ACC::Element) * (_world * _nLx * TCM * TN);
        }

        __host__ __forceinline__
        constexpr static unsigned long int bookLength() {
            return ACC::TSZ::value * sizeof(BookType) + sizeof(ACC::Element) * (ACC::S::value * ACC::P::value);
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

        /**********Salami slice Pointers!************/
        /// task queue
        __device__ __forceinline__
        auto* tQ() const {
            return CAST_TO(Task, book);
        }
        static_assert(alignof(Task) % alignof(PEL) == 0);
        __host__ __device__ __forceinline__
        auto* pEL() const {
            return CAST_TO(PEL, book + tQl);
        }
        static_assert(alignof(PEL) % alignof(TQSignal) == 0);
        /// processors' doorbell
        __device__ __forceinline__
        auto* pDB() const {
            return CAST_TO(TQSignal, book + tQl + sizeof(PEL) * ACC::E::value);
        }

        static_assert(alignof(ull_t) % alignof(RingSoftmaxPayload) == 0
            && alignof(ull_t) % alignof(RingTopKPayload) == 0
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
            return CAST_TO(RingTopKPayload, bRsP() + ACC::S::value * ACC::TPX::value);
        }
        /***********CONTIGUOUS**************/

        static_assert(alignof(RingTopKPayload) % alignof(cuda::barrier<cuda::thread_scope_device>) == 0);
        /// Device-wide barrier
        __host__ __device__ __forceinline__
        auto* dB() const {
            return CAST_TO(cuda::barrier<cuda::thread_scope_device>, book + brs);
        }
        static_assert(alignof(cuda::barrier<cuda::thread_scope_device>) % alignof(ELI) == 0);
        /// Expert Lookup
        /// expert index -> ELI
        __host__ __device__ __forceinline__
        auto* eL() const {
            return CAST_TO(ELI, dB() + 1);
        }

        static_assert(alignof(ELI) % alignof(PLI) == 0);
        __host__ __device__ __forceinline__
        auto* pLI() const {
            return CAST_TO(PLI, eL() + ACC::E::value);
        }

        static_assert(alignof(PLI) % alignof(TokenIdxTuple) == 0);
        __device__ __forceinline__
        auto* tP() const {
            return CAST_TO(TokenIdxTuple, book + tQXt);
        }

        static_assert(alignof(TokenIdxTuple) % alignof(mp_t) == 0);
        /// Gate mean logits
        __device__ __forceinline__
        auto* gML() const {
            return CAST_TO(mp_t, tP() + ACC::PX::value * ACC::EC::value);
        }
        /// Gate mean expert counts
        __device__ __forceinline__
        auto* gMeC() const {
            return gML() + ACC::E::value;
        }
        /// Gate loss
        __device__ __forceinline__
        auto* gL() const {
            return gMeC() + ACC::E::value;
        }

        static_assert(alignof(mp_t) % alignof(BookType) == 0);
        __device__ __forceinline__
        auto* eC() const {
            return CAST_TO(BookType, book + gRl);
        }

        /// number of remote experts
        __host__ __device__ __forceinline__
        auto* nRx() const {
            return eC() + ACC::E::value;
        }

        /// Packet Sync array
        __device__ __forceinline__
        auto* pSA() const {
            return eC() + ACC::E::value + 1;
        }

        /// Scheduler buffers and flag checkpoints
        __device__ __forceinline__
        auto* sQ() const {
            return CAST_TO(BookType, book + eDsA);
        }
        __device__ __forceinline__
        auto* tQH() const {
            return sQ() + ACC::PeakHardware::OS::processorBlocks::value;
        }
        /// tile sync array
        __device__ __forceinline__
        auto* tSA() const {
            return tQH() + world * nLx * ACC::TCM::value;
        }
        __device__ __forceinline__
        auto* tIx() const {
            return CAST_TO(BookType, tSA() + (ACC::E::value > 1 ? world * nLx * ACC::TCM::value :
                ACC::TSZ::value));
        }

        // Intermediate buffer
        static_assert(alignof(BookType) % alignof(ACC::Element) == 0);
        __device__ __forceinline__
        auto* xM() const {
            return book + sBfC;
        }

        private:
            /// Note the below lengths are cumulative sums.
            unsigned long int eDsA = 0UL;
            /// Scheduler buffers in bytes
            unsigned long int sBfC = 0UL;
            unsigned long int tQXt = 0UL;
            /// gate routing and loss vectors in bytes
            unsigned long int gRl = 0U;
            /// Block Ring Softmax flags in bytes, non-cumulative
            unsigned int brs = 0UL;
            /// Task Q maximum length
            unsigned int tQml = 0U;
            /// Task Q length
            unsigned int tQl = 0U;
    };

    // monotonically increasing integer
    // has to differ from the ground state of 0, so starts at 1U
    __inline__ uint16_t seqBit = 1U;
    __constant__ __inline__ Bookkeeping bookkeeping{};
    __inline__ Bookkeeping hostBookkeeping{};
    __inline__ bool isInitialized = false;
    __inline__ auto aristosStream = cudaStreamPerThread;

    namespace heap {
        // The symmetric heap is a 6-D tensor (P, S, C, E, EC) of tokens,
        /// where P, S, C, E, EC denote dimensions for peers, communication stages,
        /// cells, experts, expert capacity, respectively.
        template<
            unsigned int stage = 0,
            unsigned int cell = 0,
            /*The user should not specify the parameters below*/
            unsigned int tokenDim = ACC::H::value,
            unsigned int slotSize = ACC::SZ::value,
            unsigned int nBytes = sizeof(ACC::Element)
        >
        requires (stage < STAGES && cell < CELLS)
        __device__ __forceinline__
        auto* cAdvance(cuda::std::byte* __restrict__ const& sHeap, const uint& peer,
            const uint& expert, const uint& token = 0){
            return sHeap + (slotSize * (bookkeeping.xs * (CELLS * (peer * STAGES + stage) + cell) + expert) +
                token * tokenDim) * nBytes;
        }
    }
}
#endif //ARISTOS_TYPES_CUH
