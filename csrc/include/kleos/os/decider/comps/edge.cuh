/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by osayamen on 9/8/24.
//

#ifndef CSRC_EDGE_CUH
#define CSRC_EDGE_CUH

#include <cstdlib>
namespace kleos{
    // Source for floatEqual: https://stackoverflow.com/a/253874
    template<typename F> requires cuda::std::is_floating_point_v<F>
    __forceinline__
    bool floatEqual(const F& a, const F& b){
        return fabs(a - b) <=
               ( (fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * cuda::std::numeric_limits<F>::epsilon());
    }

    struct Edge{
        unsigned int node1;
        unsigned int node2;
        float weight;

        Edge() = default;
        /// Order is important!
        Edge(const unsigned int& _node1, const unsigned int& _node2, const float& _weight):
        node1(_node1), node2(_node2), weight(_weight){}

        __forceinline__
        unsigned int getNeighbor(const unsigned int& other) const {
            assert(node1 == other || node2 == other);
            return (node1 == other)? node2 : node1;
        }

        __forceinline__
        bool operator==(const kleos::Edge& other) const {
            return this->node1 == other.node1 && this->node2 == other.node2;
        }

        __forceinline__
        bool operator!=(const kleos::Edge& other) const {
            return !(*this == other);
        }

        __forceinline__
        bool operator<(const kleos::Edge& other) const {
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
        bool operator<=(const kleos::Edge& other) const {
            return *this < other || *this == other;
        }

        __forceinline__
        bool operator>(const kleos::Edge& other) const {
            return !(*this <= other);
        }

        __forceinline__
        bool operator>=(const kleos::Edge& other) const {
            return *this > other || *this == other;
        }

        __forceinline__
        std::string toString() const {
            return "{"
                   "\"weight\": " + std::to_string(weight)
                   + ", \"node1\": " + std::to_string(node1)
                   + ", \"node2\": " + std::to_string(node2) + "}";
        }

        __forceinline__
        bool isLimboEdge() const{
            /// Define a self-edge with zero weight as limbo or null edge
            return node1 == node2 && floatEqual(weight, 0.0f);
        }

        __forceinline__
        static Edge limboEdge() {
            return {0,0,0.0f};
        }
    };
}

template<>
struct std::hash<kleos::Edge>
{
    __forceinline__
    std::size_t operator()(const kleos::Edge& e) const noexcept
    {
        return e.node1 ^ e.node2 << 1;
    }
};

#endif //CSRC_EDGE_CUH
