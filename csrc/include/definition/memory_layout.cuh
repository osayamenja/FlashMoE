//
// Created by Jonathan on 7/21/24.
//

#ifndef ARISTOS_MEMORY_LAYOUT_CUH
#define ARISTOS_MEMORY_LAYOUT_CUH

namespace aristos{
    /// NVSHMEM uses size_t
    /// See https://docs.nvidia.com/nvshmem/api/gen/api/rma.html#nvshmem-put-nbi
    using n_bytes_repr = size_t;

    /// Number of communication stages
    constexpr int stages = 2;

    /// Per stage, there is one cell for send and another for receive
    constexpr int n_cells = 2 * stages;

    /// Per embedding vector
    /// k is top_k, +1 for index, and *4 for uint precision
    constexpr uint trailer_length_bytes(uint k){
        return (k + 1) * 4;
    }

    CUTE_HOST_DEVICE
    size_t symmetric_heap_index(uint capacity, uint k, uint embed_dim, uint embed_precision){
        auto trailer_len = trailer_length_bytes(k);
        auto embed_bytes = embed_dim * embed_precision;
        auto header_bytes = sizeof (n_bytes_repr);
        auto packet_bytes = n_cells * capacity * (embed_bytes + trailer_len);
        return packet_bytes + header_bytes;
    }
}
#endif //ARISTOS_MEMORY_LAYOUT_CUH
