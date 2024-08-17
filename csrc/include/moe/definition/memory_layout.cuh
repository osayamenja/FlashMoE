//
// Created by Jonathan on 7/21/24.
//

#ifndef ARISTOS_MEMORY_LAYOUT_CUH
#define ARISTOS_MEMORY_LAYOUT_CUH

namespace aristos{
    /// header
    /// NVSHMEM uses size_t
    /// See https://docs.nvidia.com/nvshmem/api/gen/api/rma.html#nvshmem-put-nbi
    /// But atomicAdd() requires ull
    using n_bytes_repr = unsigned long long int;

    /// Number of bytes for n_bytes_repr
    constexpr size_t header_bytes = sizeof(n_bytes_repr);

    /// Number of bytes for micro_header, namely k from top_k.
    /// Of course, 32 bits is overkill, since practical systems use k in [1,2].
    /// However, we desire encouraging much larger k, thus we adopt uint.
    constexpr size_t micro_header_bytes = sizeof(unsigned int);

    /// Number of communication stages
    constexpr unsigned int stages = 2;

    /// Per stage, there is one cell for send and another for receive
    constexpr unsigned int numCells = 2;

    /// Per embedding vector a la token.
    /// k is top_k, +1 for token index, and *4 for unsigned int precision
    CUTE_HOST_DEVICE
    constexpr unsigned int trailer_length_bytes(const unsigned int k){
        return (k + 1U) * 4U;
    }

    template<bool isPrimingStage=false>
    CUTE_HOST_DEVICE
    size_t cell_span(){
        return (stages - 1);
    }

    /// Special case
    template<>
    CUTE_HOST_DEVICE
    size_t cell_span<true>(){
        return 1;
    }

    CUTE_HOST_DEVICE
    size_t payload_bytes(const unsigned int embed_bytes,
                        const unsigned int k){
        return (embed_bytes + trailer_length_bytes(k));
    }

    CUTE_HOST_DEVICE
    size_t packet_bytes(const unsigned int capacity,
                               const unsigned int embed_bytes,
                               const unsigned int trailer_len){
        return (capacity * (embed_bytes + trailer_len));
    }


    CUTE_HOST_DEVICE
    size_t symmetric_heap_peer_offset(const unsigned int cap,
                                      const unsigned int k,
                                      size_t embed_bytes){
        return numCells * ((packet_bytes(cap, embed_bytes, trailer_length_bytes(k)) * cell_span<true>())
                           + (packet_bytes(cap,embed_bytes,
                                                   trailer_length_bytes(k)) * cell_span<false>()));
    }


    CUTE_HOST_DEVICE
    size_t packet_trailer_index(const unsigned int cell,
                                const unsigned int checkpoint,
                                const unsigned int capacity,
                                const size_t embed_bytes,
                                unsigned int k){
        return (packet_bytes(capacity, embed_bytes, trailer_length_bytes(k)) * cell) +
                (embed_bytes) +
                (checkpoint * (trailer_length_bytes(k) + embed_bytes));
    }

    CUTE_HOST_DEVICE
    unsigned int send_cell(unsigned int stage){
        return 2*stage;
    }

}
#endif //ARISTOS_MEMORY_LAYOUT_CUH
