//
// Created by osayamen on 9/8/24.
//

#ifndef CSRC_EXPERT_CUH
#define CSRC_EXPERT_CUH
namespace aristos{
    struct Expert{
        unsigned int id;
        unsigned int cost;

        Expert(unsigned int _id, unsigned int _cost):
                id(_id), cost(_cost){};

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
        std::string toString(){
            return "{\n\t"
                   "\"id\": " + std::to_string(id)
                   + ",\n\t\"ComputeCost\": " + std::to_string(cost) + ",\n\t}";
        }

    };
}

template<>
struct std::hash<aristos::Expert>
{
    __forceinline__
    std::size_t operator()(const aristos::Expert& x) const noexcept
    {
        return x.id;
    }
};
#endif //CSRC_EXPERT_CUH
