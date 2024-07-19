//
// Created by Jonathan on 7/18/24.
//

#ifndef ARISTOS_COMMUNICATOR_CUH
#define ARISTOS_COMMUNICATOR_CUH

#include <cuda/semaphore>
#include <cuda/atomic>
#include "../definition/platform.cuh"
#include "../definition/tensor.cuh"
namespace aristos{
    template<Matrix M, uint Arch> requires MinArch<Arch>
    class Communicator{
    public:
        void* message_queue;
        cuda::atomic<uint_fast16_t, cuda::thread_scope_device> doorbell = 0;
        cuda::atomic<bool, cuda::thread_scope_device> stop = false;
        uint rank;


        explicit Communicator(void* _message_queue, uint _rank, int db_count = 0):
                message_queue(_message_queue),
                rank(_rank)
        {}

        void start(){
            while(!stop.load()){
                acquire_until_signal();

            }
        }
    private:
        void acquire_until_signal(){
            while(!doorbell.load() && !stop.load()){}
        }

    };

    class Metadata{

    };
}
#endif //ARISTOS_COMMUNICATOR_CUH
