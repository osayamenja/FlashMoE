//
// Created by oja7 on 12/2/24.
//

#ifndef PACKET_CUH
#define PACKET_CUH

namespace aristos::packet {
    template<unsigned int blocks, typename MatrixA>
    __forceinline__ __device__
    void constructSend(const MatrixA& gateOutput, cuda::std::byte* __restrict__ workspace) {
        auto* scratch = CAST_TO(unsigned int, workspace);
        for (unsigned int i = 0; i < moeConfig.numExperts + moeConfig.worldSize; ++i) {
            scratch[i] = moeConfig.getPeerXLookup()[i];
        }
        // Map 64 blocks to a peer and stride as thus
        constexpr auto residual = blocks / ARISTOS_SUPER_BLOCK_SIZE * ARISTOS_SUPER_BLOCK_SIZE;
        constexpr auto numSuperBlocks = blocks / ARISTOS_SUPER_BLOCK_SIZE + (blocks - residual > 32);
        const auto superBlockId = blockIdx.x / ARISTOS_SUPER_BLOCK_SIZE -
            (blockIdx.x >= residual && blocks - residual < 32);
        // Build and send a packet for this peer
        for (int i = superBlockId; i < moeConfig.worldSize; i += numSuperBlocks) {
            // Get all its experts and build the packet for all, then send
        }

    }
}
#endif //PACKET_CUH
