//
// Created by osayamen on 1/5/26.
//

#ifndef FLASHMOE_TASK_CUH
#define FLASHMOE_TASK_CUH
namespace flashmoe {
#define GEMMs 2U // per expert
    enum class TaskType : uint8_t {
        GEMM0,
        GEMM1,
        combine,
        none
    };

    struct __align__(16) Task {
        const cuda::std::byte* const aData = nullptr;
        cuda::std::array<cuda::std::byte*, GEMMs> const cData = {};
        cuda::std::byte* const rcData = nullptr;
        flagsType* const flags = nullptr;
        const unsigned int syncIdx = 0U;
        const unsigned int tileIdx = 0U;
        const unsigned int M = 0U;
        const uint16_t batchIdx = 0U;
        const uint16_t localExpertIdx = 0U;
        const uint16_t expertIdx = 0U;
        const uint16_t peerIdx = 0U;
        const uint16_t tileSize = 0U; // <= BLOCK_M
        const uint8_t isPeerRemote = 0U;
        const TaskType taskType = TaskType::none;

        __forceinline__ __device__
        Task() = default;

        // GEMM0->GEMM1
        __device__ __forceinline__
        Task(const TaskType& _taskType,
            const cuda::std::byte* const& _aData,
            const cuda::std::array<cuda::std::byte*, GEMMs>& _cData,
            cuda::std::byte* const& _rcData,
            flagsType* const& _flags,
            const unsigned int& _syncIdx,
            const unsigned int& _tile,
            const unsigned int& m,
            const uint16_t& _size,
            const uint16_t& _peerIdx,
            const uint16_t& _batchIdx,
            const uint& _isPeerRemote):
        aData(_aData), cData(_cData), rcData(_rcData), flags(_flags),
        syncIdx(_syncIdx), tileIdx(_tile),  M(m),
        batchIdx(_batchIdx), peerIdx(_peerIdx), tileSize(_size), isPeerRemote(_isPeerRemote), taskType(_taskType){}

        // Combine
        __device__ __forceinline__
        Task(const TaskType& _taskType,
        const cuda::std::byte*  const& _aData,
        const unsigned int& _size,
        const unsigned int& _tile,
        const unsigned int& _expertIdx):
        aData(_aData), tileIdx(_tile), expertIdx(_expertIdx), taskType(_taskType),
        tileSize(_size){}

        __device__ __forceinline__
        void dump() const {
            printf("{\n\t"
                   "this: %p,\n\t"
                   "aData: %p,\n\t"
                   "cData[0]: %p,\n\t"
                   "cData[1]: %p,\n\t"
                   "rcData: %p,\n\t"
                   "flags: %p,\n\t"
                   "syncIdx: %u,\n\t"
                   "tileIdx: %u,\n\t"
                   "M: %u,\n\t"
                   "batchIdx: %u,\n\t"
                   "peerIdx: %u,\n\t"
                   "expertIdx: %u,\n\t"
                   "isPeerRemote: %s,\n\t"
                   "taskType: %u,\n\t"
                   "tileSize: %u"
                   "\n}\n",
                   this, aData, cData[0], cData[1],
                   rcData, flags, syncIdx, tileIdx, M,
                   batchIdx, peerIdx, expertIdx, isPeerRemote ? "True" : "False",
                   taskType, tileSize);
        }
    };
    static_assert(sizeof(Task) == 64);
}
#endif //FLASHMOE_TASK_CUH