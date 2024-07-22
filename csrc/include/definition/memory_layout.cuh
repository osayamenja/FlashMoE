//
// Created by Jonathan on 7/21/24.
//

#ifndef ARISTOS_MEMORY_LAYOUT_CUH
#define ARISTOS_MEMORY_LAYOUT_CUH

namespace aristos{
    /// 4 bytes to represent number of bytes for a packet in memory
    constexpr int n_bytes_repr = 4;

    /// Number of communication stages
    constexpr int stages = 2;

    /// Per stage, there is one cell for send and another for receive
    constexpr int n_cells = 2 * stages;

    /// Per embedding vector
    /// k is top_k
    constexpr uint trailer_length_bytes(uint k){
        return (k + 1) * 4;
    }

    CUTE_HOST_DEVICE
    size_t symmetric_heap_index(uint capacity, uint k, uint embed_dim, uint embed_precision){
        auto trailer_len = trailer_length_bytes(k);
        auto embed_bytes = embed_dim * embed_precision;
        auto header_bytes = 8;
        auto packet_bytes = 4 * capacity * (embed_bytes + trailer_len);
        return packet_bytes + header_bytes;
    }
}
#endif //ARISTOS_MEMORY_LAYOUT_CUH
