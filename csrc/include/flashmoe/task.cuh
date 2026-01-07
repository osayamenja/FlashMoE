//
// Created by osayamen on 1/5/26.
//

#ifndef FLASHMOE_TASK_CUH
#define FLASHMOE_TASK_CUH
namespace flashmoe {
    constexpr int GEMMs = 2; // within
    enum class TaskType : uint8_t {
        GEMM0,
        GEMM1,
        combine,
    };
    static_assert(cuda::std::is_same_v<cuda::std::underlying_type_t<TaskType>, uint8_t>);

    struct __align__(16) Ingredients{
        unsigned int M = 0; // GEMM0->number of tokens or Combine->token
        uint16_t localExpertIdx = 0;
        uint16_t expertIdx = 0; // global
        uint16_t peerIdx = 0; // owner of the output
        uint16_t tileSize = 0; // <= BLOCK_M
        uint16_t stash = 0; // GEMM0->flagBatchIdx or Combine->tileIdx
        TaskType taskType;
        uint8_t isPeerRemote;

        Ingredients() = default;

        // GEMM0 and GEMM1
        __device__ __forceinline__
        Ingredients(const uint16_t& lei, const uint16_t& pei, const TaskType& tt, const uint8_t& ipr):
        localExpertIdx(lei), peerIdx(pei), taskType(tt), isPeerRemote(ipr){}

        __device__ __forceinline__
        Ingredients(const unsigned int& m, const uint16_t& lei, const uint16_t& ei,
            const uint16_t& pei, const uint16_t& ts, const uint16_t& sta,
            const TaskType& tt, const uint8_t& ipr):
        M(m), localExpertIdx(lei), expertIdx(ei), peerIdx(pei), tileSize(ts), stash(sta),
        taskType(tt), isPeerRemote(ipr){}
    };
    static_assert(sizeof(Ingredients) == 16);

    struct __align__(16) Task {
        Ingredients ingredients{};
        const cuda::std::byte* aData = nullptr;
        cuda::std::array<cuda::std::byte*, GEMMs> cData = {};
        cuda::std::byte* rcData = nullptr;
        flagsType* flags = nullptr;
        unsigned int syncIdx = 0U;
        unsigned int tileIdx = 0U;

        Task() = default;

        // GEMM0->GEMM1
        __device__ __forceinline__
        Task(const Ingredients& _ingredients,
            const cuda::std::byte* const& _aData,
            const cuda::std::array<cuda::std::byte*, GEMMs>& _cData,
            cuda::std::byte* const& _rcData,
            flagsType* const& _flags,
            const unsigned int& _syncIdx, const unsigned int& tile):
        ingredients(_ingredients), aData(_aData), cData(_cData), rcData(_rcData), flags(_flags),
        syncIdx(_syncIdx), tileIdx(tile){}

        // Combine
        __device__ __forceinline__
        explicit Task(const Ingredients& _ingredients):
        ingredients(_ingredients){}

    };
    static_assert(sizeof(Task) == 64);
}
#endif //FLASHMOE_TASK_CUH