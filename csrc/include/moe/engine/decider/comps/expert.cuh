//
// Created by osayamen on 9/8/24.
//

#ifndef CSRC_EXPERT_CUH
#define CSRC_EXPERT_CUH
namespace aristos{
    struct Expert{
        unsigned int id;
        unsigned long cost;
        unsigned int memoryDemand = 1; // experimental feature for heterogeneous experts

        Expert(const unsigned int& _id, const unsigned long& _cost):
                id(_id), cost(_cost){};

        /// Sentinel
        explicit Expert(unsigned long _cost){
            cost = _cost;
            id = 0;
        }

        __forceinline__
        bool operator==(const Expert& other) const {
            return this->id == other.id;
        }

        __forceinline__
        bool operator!=(const Expert& other) const {
            return !(*this == other);
        }

        __forceinline__
        bool operator<(const Expert& other) const {
            return this->cost < other.cost;
        }

        __forceinline__
        bool operator<=(const Expert& other) const {
            return *this < other || *this == other;
        }

        __forceinline__
        bool operator>(const Expert& other) const {
            return this->cost > other.cost;
        }

        __forceinline__
        bool operator>=(const Expert& other) const {
            return *this > other || *this == other;
        }

        __forceinline__
        std::string toString() const {
            return "{\"id\": " + std::to_string(id)
                   + ", \"ComputeCost\": " + std::to_string(cost)
                   + ", \"MemoryDemand\": " + std::to_string(memoryDemand) + "}";
        }

        __forceinline__ static Expert closest(const Expert& left, const Expert& right, unsigned int val){
            return (std::abs(int(left.cost) - int(val)) <= std::abs(int(right.cost) - int(val))? left : right);
        }
    };
}
#endif //CSRC_EXPERT_CUH
