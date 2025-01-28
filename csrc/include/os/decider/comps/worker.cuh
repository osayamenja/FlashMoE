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

        Worker(const unsigned int& _id, const unsigned int& _processingRate, const unsigned int& memoryCapacity):
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
            return "{\"id\": " + std::to_string(id)
                   + ", \"rate\": " + std::to_string(processingRate)
                   + ", \"cap\": " + std::to_string(memoryCapacity) + "}";
        }
    };
}
#endif //CSRC_WORKER_CUH
