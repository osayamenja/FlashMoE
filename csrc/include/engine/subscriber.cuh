//
// Created by Jonathan on 7/4/24.
//

#ifndef ARISTOS_QUEUE_CUH
#define ARISTOS_QUEUE_CUH

namespace aristos::subscriber{
    enum class SubscriptionLevel {
        over, // world > THREADS - 1
        actual // world <= THREADS - 1
    };
    template<SubscriptionLevel s = SubscriptionLevel::actual>
    __device__ __forceinline__
    void start(cuda::std::byte* workspace){
        static_assert(s == SubscriptionLevel::actual);
        // each thread monitors a single peer

    }
}
#endif //ARISTOS_QUEUE_CUH
