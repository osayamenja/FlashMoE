//
// Created by Jonathan on 7/21/24.
//

#ifndef ARISTOS_MEMORY_LAYOUT_CUH
#define ARISTOS_MEMORY_LAYOUT_CUH

namespace aristos{
    /// Number of communication stages
    __constant__ constexpr unsigned int stages = 2;

    /// Per stage, there is one cell for send and another for receive
    __constant__ constexpr unsigned int numCells = 2;

    __constant__ constexpr unsigned int sendCell = 0;
    __constant__ constexpr unsigned int receiveCell = 1;

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


    CUTE_DEVICE
    cuda::std::byte* getTokenPointer(unsigned int const& peer, unsigned int const& stage, unsigned int const& cell, unsigned int const& token){
        return moeConfig.sHeap + ((peer * moeConfig.peerStride) + (stage * moeConfig.stageStride) + (cell * moeConfig.cellStride) + (token * moeConfig.tokenStride));
    }

}
#endif //ARISTOS_MEMORY_LAYOUT_CUH
