/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Flashmoe Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by Jonathan on 7/18/24.
//

#ifndef FLASHMOE_TYPES_CUH
#define FLASHMOE_TYPES_CUH

#define FLASHMOE_BLOCK_SIZE_WARP (128U / 32)
#define FLASHMOE_STATIC_SBZ 32U

#define CAST_TO(T, p) static_cast<T*>(static_cast<void*>(p))
#define CONST_CAST_TO(T, p) static_cast<const T*>(static_cast<const void*>(p))

// Hardware description
#define FLASHMOE_DEBUG 1
#define NOOP_SIGNAL 0

#include <cuda/barrier>
#include <cuda/std/array>
#include <cute/numeric/integral_constant.hpp>
#include <cute/tensor.hpp>
#include <cutlass/epilogue/thread/activation.h>

namespace flashmoe{
    template<typename S>
    struct ToCute {
        using T = S;
        static_assert(flashmoe::TensorValueType<T>);
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
    requires(flashmoe::TensorValueType<S>)
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
    using specType = unsigned int;
    using flagsType = uint64_t;

    using Nano = cuda::std::chrono::duration<float, cuda::std::nano>;
    using Milli = cuda::std::chrono::duration<float, cuda::std::milli>;
    using ull_t = unsigned long long int;
    static_assert(sizeof(ull_t) == sizeof(flagsType) && alignof(ull_t) == alignof(flagsType));

    // These could be much more, as supported by CUTLASS
    __host__ __device__
    enum ActivationFunction: uint8_t {
        ReLu,
        GeLU
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
    enum class CombineMode {
        single,
        plural
    };

    // Index and gate combine weight

    __device__
    enum class FlagState {
        unidentified,
        identified,
        completed
    };

    // Also applies to shared memory banks
    template<typename Element>
    requires(128 % sizeof(Element) == 0)
    __device__ __forceinline__
    constexpr auto rTCL(uint const& zb) {
        return cute::ceil_div(zb, 128U / sizeof(Element)) * (128U / sizeof(Element));
    }

    // Expert Parallel Group details
    struct __align__(8) EPG {
        uint16_t epRank;
        uint16_t expertSlots;
        uint16_t nLx;
        uint16_t epWorld;
        uint epWorldM;

        EPG() = default;
        EPG(const uint16_t& _epR,
            const uint16_t& _eS,
            const uint16_t& _nLx,
            const uint16_t& _epW):
        epRank(_epR), expertSlots(_eS), nLx(_nLx), epWorld(_epW), epWorldM(_epW) {}

        void dump() const {
            printf("{\n\t"
                   "epRank: %u,\n\t"
                   "expertSlots: %u,\n\t"
                   "nLx: %u,\n\t"
                   "epWorld: %u"
                   "\n}\n",
                   epRank, expertSlots, nLx, epWorld);
        }

        void dump(const int& gRank) const {
            printf("{\n\t"
                   "gRank: %d,\n\t"
                   "epRank: %u,\n\t"
                   "expertSlots: %u,\n\t"
                   "nLx: %u,\n\t"
                   "epWorld: %u"
                   "\n}\n",
                   gRank,
                   epRank, expertSlots, nLx, epWorld);
        }
    };

    /// Information about auxiliary data structures comprising bookkeeping state
    /// Includes length of data structures (arrays) and pointer arithmetic functions
    using BookType = unsigned int;
    struct __align__(16) Bookkeeping {
        /// needed for free
        flagsType* flags = nullptr;
        cuda::std::byte* sHeap = nullptr;
        Task* bookTask = nullptr;
        PEL* bookPEL = nullptr;
        PLI* bookPLI = nullptr;
        TPS* bookTPS = nullptr;
        cuda::barrier<cuda::thread_scope_device>* bookDB = nullptr;
        TQSignal* bookTQS = nullptr;
        RingSoftmaxPayload* bookRSP = nullptr;
        RingTopKPayload* bookRTP = nullptr;
        ELI* bookELI = nullptr;
        BookType* book = nullptr;
        cuda::std::byte* bookElement = nullptr;
        unsigned long int ilt = 0U;
        unsigned int gtQCl = 0U;
        unsigned int sT = 0U;
        /// EP rank
        uint rank = 0U;
        /// EP world
        uint world = 0U;
        /// number of local experts
        uint nLx = 0U;
        /// expert slots
        uint xs = 0U;
        uint gfSfC = 0U;

        __host__ __device__ __forceinline__
        Bookkeeping() = default;

        __host__ __forceinline__
        explicit Bookkeeping(flagsType* const& _flags,
            cuda::std::byte* const& _sHeap,
            Task* const& _bookTask,
            PEL* const& _bookPEL,
            PLI* const& _bookPLI,
            TPS* const& _bookTPS,
            cuda::barrier<cuda::thread_scope_device>* const& _bookDB,
            TQSignal* const& _bookTQS,
            RingSoftmaxPayload* const& _bookRSP,
            RingTopKPayload* const& _bookRTP,
            ELI* const& _bookELI,
            BookType* const& _book,
            cuda::std::byte* const& _bookElement,
            const EPG& ePgD) :
                flags(_flags),
                sHeap(_sHeap),
                bookTask(_bookTask),
                bookPEL(_bookPEL),
                bookPLI(_bookPLI),
                bookTPS(_bookTPS),
                bookDB(_bookDB),
                bookTQS(_bookTQS),
                bookRSP(_bookRSP),
                bookRTP(_bookRTP),
                bookELI(_bookELI),
                book(_book),
                bookElement(_bookElement),
                rank(ePgD.epRank),
                world(ePgD.epWorld),
                nLx(ePgD.nLx),
                xs(ePgD.expertSlots),
                gfSfC(ePgD.epWorldM * ePgD.expertSlots){
            constexpr auto TCM = ACC::TCM::value;
            constexpr auto TN = ACC::TN::value;
            constexpr auto TNx = ACC::TNx::value;
            constexpr auto blocks = ACC::PeakHardware::OS::processorBlocks::value;
            constexpr auto E = ACC::E::value;
            if constexpr (E > 1) {
                gtQCl = world * nLx * TCM;
                // maximum gemm tiles/tasks scheduled by subscriber threads
                static_assert(SUBSCRIBERS % WARP_SIZE == 0);
                const auto tPS = cute::ceil_div(world * nLx, SUBSCRIBERS / WARP_SIZE) *
                        cute::ceil_div(TCM * TN, WARP_SIZE) +
                        cute::ceil_div(TCM * E, SUBSCRIBERS) * ACC::TNx::value;
                sT = tPS * SUBSCRIBERS;
                ilt = 1 + nLx + blocks + 2 * (gtQCl + E) + E * TCM * TNx;
            }
        }

        Bookkeeping(BookType* const& _book, cuda::std::byte* const& _bookElement) :
        book(_book), bookElement(_bookElement){}

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
        /// stride task queue
        __device__ __forceinline__
        auto* tQ() const {
            return bookTask;
        }
        /// blocked
        __device__ __forceinline__
        auto* ptQ() const {
            return bookTask + sT;
        }
        __host__ __forceinline__
        constexpr static auto tQlt(const unsigned int& _nLx, const unsigned int& _world) {
            // maximum gemm tiles/tasks scheduled by processors
            constexpr auto TCM = ACC::TCM::value;
            const auto prT = _world * _nLx * TCM * ACC::TNx::value;
            static_assert(SUBSCRIBERS % WARP_SIZE == 0);
            // maximum gemm tiles/tasks scheduled by subscriber threads
            const auto tPS = cute::ceil_div(_world * _nLx, SUBSCRIBERS / WARP_SIZE) *
                    cute::ceil_div(TCM * ACC::TN::value, WARP_SIZE) +
                    cute::ceil_div(TCM * ACC::E::value, SUBSCRIBERS) * ACC::TNx::value;
            const auto sT = tPS * SUBSCRIBERS;
            return sT + prT;
        }
        __host__ __device__ __forceinline__
        auto* pEL() const {
            return bookPEL;
        }
        __host__ __forceinline__
        constexpr static auto pELlt() {
            return ACC::E::value;
        }
        __host__ __device__ __forceinline__
        auto* pL() const {
            return bookPLI;
        }
        __host__ __forceinline__
        constexpr static auto pLlt(const unsigned int& _world) {
            return _world;
        }
        __device__ __forceinline__
        auto* tP() const {
            return bookTPS;
        }
        __host__ __forceinline__
        constexpr static auto tPlt() {
            return ACC::E::value * ACC::pEC::value;
        }
        /// Device-wide barrier
        __host__ __device__ __forceinline__
        auto* dB() const {
            return bookDB;
        }
        __host__ __forceinline__
        constexpr static auto dBlt() {
            return 1;
        }

        /// processors' doorbell
        __device__ __forceinline__
        auto* pDB() const {
            return bookTQS;
        }
        __host__ __forceinline__
        constexpr static auto pDBlt() {
            return ACC::PeakHardware::OS::processorBlocks::value;
        }

        __device__ __forceinline__
        auto* bRsP() const {
            return bookRSP;
        }
        __host__ __forceinline__
        constexpr static auto rSlt() {
            return ACC::GRL::value == GateReductionLevel::multiBlock ?
                ACC::S::value * ACC::TPX::value : 0U;
        }
        /// Ring top k flags
        /// Two sets for pipelining termination phase of round i with initial phase of round i + 1
        __device__ __forceinline__
        auto* rTp() const {
            return bookRTP;
        }
        __host__ __forceinline__
        constexpr static auto rTlt() {
            return ACC::GRL::value == GateReductionLevel::multiBlock ?
                2 * ACC::S::value * ACC::TPX::value : 0U;
        }
        /// Expert Lookup
        /// expert index -> ELI
        __host__ __device__ __forceinline__
        auto* eL() const {
            return bookELI;
        }
        __host__ __forceinline__
        constexpr static auto eLlt() {
            return ACC::E::value;
        }
        static_assert(sizeof(LXI) == sizeof(BookType) && alignof(LXI) == alignof(BookType));
        __host__ __device__ __forceinline__
        auto* tIx() const {
            return book;
        }
        /// second stage flag count
        __host__ __device__ __forceinline__
        auto* ssFc() const {
            return tIx() + ACC::E::value * ACC::TCM::value * ACC::TNx::value;
        }
        __host__ __device__ __forceinline__
        auto* lX() const {
            return CAST_TO(LXI, ssFc() + 1);
        }
        __device__ __forceinline__
        auto* eCSync() const {
            return CAST_TO(BookType, lX() + nLx);
        }
        __device__ __forceinline__
        auto* tQH() const {
            return eCSync() + 1;
        }
        /// tile sync array
        __device__ __forceinline__
        auto* tSA() const {
            return tQH() + gtQCl;
        }
        __device__ __forceinline__
        auto* sQ() const {
            return tSA() + gtQCl;
        }
        /// expert counts
        __device__ __forceinline__
        auto* eC() const {
            return sQ() + ACC::PeakHardware::OS::processorBlocks::value;
        }
        __device__ __forceinline__
        auto* pSA() const {
            return eC() + ACC::E::value;
        }
        static_assert(alignof(mp_t) == alignof(BookType) &&
            sizeof(BookType) == sizeof(mp_t));
        /// entrypoint for clearing
        __device__ __forceinline__
        auto* gBp() const {
            return CAST_TO(mp_t, book + ilt);
        }
        __device__ __forceinline__
        static constexpr auto gBz() {
            return 2 * ACC::E::value + 1;
        }
        /***********CONTIGUOUS**************/
        /// Gate mean logits
        __device__ __forceinline__
        auto* gML() const {
            return gBp();
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
        /***********CONTIGUOUS**************/
        constexpr static auto b4lt(const unsigned int& _nLx, const unsigned int& _world) {
            constexpr auto blocks = ACC::PeakHardware::OS::processorBlocks::value;
            const auto gtQCl = _world * _nLx * ACC::TCM::value;
            constexpr auto flt = 2 * ACC::E::value + 1;
            static_assert(sizeof(LXI) == sizeof(BookType) && alignof(LXI) == alignof(BookType));
            const auto ilt = 1 + 1 + _nLx + blocks + 2 * (gtQCl + ACC::E::value) +
                ACC::E::value * ACC::TCM::value * ACC::TNx::value;
            static_assert(sizeof(mp_t) == sizeof(BookType) && alignof(mp_t) == alignof(BookType));
            return flt + ilt;
        }
        constexpr static auto b4lt() {
            return ACC::TSZ::value;
        }

        // Intermediate buffer
        __device__ __forceinline__
        auto* xM() const {
            return bookElement;
        }
        constexpr static auto xMlt(const unsigned int& _nLx, const unsigned int& _world) {
            return _world * _nLx * ACC::pEC::value * ACC::P::value;
        }

        /// Expository purposes
        __host__ __forceinline__
        constexpr static unsigned long int bookLength(const unsigned int& _nLx, const unsigned int& _world) {
            return  sizeof(Task) * tQlt(_nLx, _world) +
                    sizeof(PEL) * pELlt() +
                    sizeof(PLI) * pLlt(_world) +
                    sizeof(TPS) * tPlt() +
                    sizeof(cuda::barrier<cuda::thread_scope_device>) * dBlt() +
                    sizeof(TQSignal) * pDBlt() +
                    sizeof(RingSoftmaxPayload) * rSlt() +
                    sizeof(RingTopKPayload) * rTlt() +
                    sizeof(ELI) * eLlt() +
                    sizeof(BookType) * b4lt(_nLx, _world) +
                    sizeof(ACC::Element) * xMlt(_nLx, _world);
        }
    };
}
#endif //FLASHMOE_TYPES_CUH
