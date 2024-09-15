//
// Created by osayamen on 9/8/24.
//

#ifndef CSRC_WORKER_CUH
#define CSRC_WORKER_CUH

namespace aristos{
    struct Worker{
        unsigned int id;
        unsigned int processingRate;
        unsigned int memoryCapacity;

        Worker(unsigned int _id, unsigned int _processingRate, unsigned int memoryCapacity):
                id(_id), processingRate(_processingRate), memoryCapacity(memoryCapacity){};

        __forceinline__
        bool operator==(const Worker& other) const {
            return this->id == other.id;
        }

        __forceinline__
        bool operator!=(const Worker& other) const {
            return !(*this == other);
        }

        __forceinline__
        bool operator<(const Worker& other) const {
            return this->processingRate < other.processingRate;
        }

        __forceinline__
        bool operator<=(const Worker& other) const {
            return *this < other || *this == other;
        }

        __forceinline__
        bool operator>(const Worker& other) const {
            return this->processingRate > other.processingRate;
        }

        __forceinline__
        bool operator>=(const Worker& other) const {
            return *this > other || *this == other;
        }

        /// For debugging
        __forceinline__
        std::string toString() const {
            return "{\n\t"
                   "\"id\": " + std::to_string(id)
                   + ",\n\t\"rate\": " + std::to_string(processingRate) + ",\n\t}"
                   + ",\n\t\"cap\": " + std::to_string(memoryCapacity) + ",\n\t}";
        }
    };
}

template<>
struct std::hash<aristos::Worker>
{
    __forceinline__
    std::size_t operator()(const aristos::Worker& w) const noexcept
    {
        return w.id;
    }
};
#endif //CSRC_WORKER_CUH
