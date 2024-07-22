//
// Created by Jonathan on 7/18/24.
//

#ifndef ARISTOS_PLATFORM_CUH
#define ARISTOS_PLATFORM_CUH

#define THREADS_PER_BLOCK 1024U
#define BLOCKS_PER_COMMUNICATOR 2U
namespace aristos{
    template<uint capability>
    concept MinArch = requires{
        // Due to unavailable support for atomics
        // See https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html#concurrency-restrictions
        // However, this may not be necessary when we upgrade to the pipeline construct.
        capability >= 6;
    };
}
#endif //ARISTOS_PLATFORM_CUH
