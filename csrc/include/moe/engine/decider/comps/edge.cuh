//
// Created by osayamen on 9/8/24.
//

#ifndef CSRC_EDGE_CUH
#define CSRC_EDGE_CUH

namespace aristos{
    // Source for floatEqual: https://stackoverflow.com/a/253874
    __forceinline__
    bool floatEqual(float a, float b){
        return cuda::std::abs(a - b) <=
               ( (cuda::std::abs(a) > cuda::std::abs(b) ? cuda::std::abs(b) : cuda::std::abs(a)) * cuda::std::numeric_limits<float>::epsilon());
    }

    struct Edge{
        unsigned int node1;
        unsigned int node2;
        float weight;

        Edge(unsigned int _node1,unsigned int _node2, float _weight):
        node1(_node1), node2(_node2), weight(_weight){}

        __forceinline__
        unsigned int getNeighbor(const unsigned int& other){
            assert(node1 == other || node2 == other);
            return (node1 = other)? node2 : node1;
        }

        __forceinline__
        bool operator==(const aristos::Edge& other) const {
            return this->node1 == other.node1 && this->node2 == other.node2;
        }

        __forceinline__
        bool operator!=(const aristos::Edge& other) const {
            return !(*this == other);
        }

        __forceinline__
        bool operator<(const aristos::Edge& other) const {
            if(floatEqual(this->weight, other.weight)){
                if(this->node1 == other.node1){
                    return this->node2 < other.node2;
                }
                else{
                    return this->node1 < other.node1;
                }
            }
            return this->weight < other.weight;
        }

        __forceinline__
        bool operator<=(const aristos::Edge& other) const {
            return *this < other || *this == other;
        }

        __forceinline__
        bool operator>(const aristos::Edge& other) const {
            return !(*this <= other);
        }

        __forceinline__
        bool operator>=(const aristos::Edge& other) const {
            return *this > other || *this == other;
        }

        std::string toString(){
            return "{\n\t"
                   "\"weight\": " + std::to_string(weight)
                   + ",\n\t\"node1\": " + std::to_string(node1) + ",\n\t}"
                   + ",\n\t\"node2\": " + std::to_string(node2) + ",\n\t}";
        }
    };
}

template<>
struct std::hash<aristos::Edge>
{
    __forceinline__
    std::size_t operator()(const aristos::Edge& e) const noexcept
    {
        return std::hash<unsigned int>{}(e.node1) ^ (std::hash<unsigned int>{}(e.node2) << 1);
    }
};

#endif //CSRC_EDGE_CUH
