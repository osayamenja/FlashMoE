//
// Created by oja7 on 11/17/24.
//

#ifndef SCHEDULER_CUH
#define SCHEDULER_CUH
namespace aristos {
    enum class TaskType {
        preGEMM,
        postGEMM,
        GateScale
    };

    struct Task {
        cuda::std::byte* aData;
        const cuda::std::byte* bData;
        cuda::std::byte* cData;
        const unsigned int expertIdx;
        // crd2Idx(peer, expertIdx, offset)
        const unsigned int syncIdx;

        __device__ __forceinline__
        Task(cuda::std::byte* _aData,
            const cuda::std::byte* _bData,
            cuda::std::byte* _cData,
            const unsigned int& _expertIdx,
            const unsigned int& _syncIdx):
        aData(_aData), bData(_bData), cData(_cData), expertIdx(_expertIdx),
        syncIdx(_syncIdx){}
    };

    __device__ __forceinline__
    void schedulerStart() {

    }
}
#endif //SCHEDULER_CUH
