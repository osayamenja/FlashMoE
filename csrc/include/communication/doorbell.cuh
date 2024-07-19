//
// Created by Jonathan on 7/18/24.
//

#ifndef ARISTOS_DOORBELL_CUH
#define ARISTOS_DOORBELL_CUH

#include <cuda/semaphore>
#include <cuda/atomic>
#include "../util/platform.cuh"

template<unsigned int Arch> requires MinArch<Arch>
class Doorbell{
    public:
        cuda::counting_semaphore<cuda::thread_scope_device> doorbell{0};
        cuda::atomic<bool, cuda::thread_scope_device> stop = false;
};
#endif //ARISTOS_DOORBELL_CUH
