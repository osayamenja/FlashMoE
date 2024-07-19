//
// Created by Jonathan on 7/18/24.
//

#ifndef ARISTOS_COMMUNICATOR_CUH
#define ARISTOS_COMMUNICATOR_CUH

#include <cuda/semaphore>
class Communicator{

    public:
        void* symmetric_heap;

        explicit Communicator(void* _symmetric_heap):
        symmetric_heap(_symmetric_heap){
        }

        void start(){
        }
    private:


};
#endif //ARISTOS_COMMUNICATOR_CUH
