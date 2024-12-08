//
// Created by oja7 on 12/2/24.
//

#ifndef PACKET_CUH
#define PACKET_CUH

namespace aristos::packet {
    template<typename MatrixA>
    requires aristos::Matrix<MatrixA>
    __forceinline__ __device__
    void constructSend(MatrixA gateOutput, cuda::std::byte* workspace) {

    }
}
#endif //PACKET_CUH
