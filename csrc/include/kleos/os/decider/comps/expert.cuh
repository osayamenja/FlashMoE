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

#ifndef CSRC_EXPERT_CUH
#define CSRC_EXPERT_CUH
#include <ostream>

namespace kleos{
    struct __align__(8) Expert{
        unsigned int id;
        uint16_t cost;
        uint16_t memoryDemand = 1U; // experimental feature for heterogeneous experts

        Expert(const uint& _id, const uint16_t& _cost):
                id(_id), cost(_cost){}

        /// Sentinel
        explicit Expert(const uint16_t& _cost){
            cost = _cost;
            id = 0U;
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

        friend std::ostream & operator<<(std::ostream &os, const Expert &obj) {
            return os << obj.toString();
        }

        __forceinline__
        static Expert closest(const Expert& left, const Expert& right, const uint16_t& val){
            const uint16_t leftMargin = val > left.cost ? val - left.cost : left.cost - val;
            const uint16_t rightMargin = val > right.cost ? val - right.cost : right.cost - val;
            return leftMargin <= rightMargin? left : right;
        }
    };
}
#endif //CSRC_EXPERT_CUH
