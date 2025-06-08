/*
 * Copyright (c) 2025, Osayamen Jonathan Aimuyo
 * All rights reserved.
 *
 * This file is part of the Kleos Project and is licensed under the BSD 3-Clause License.
 * See the LICENSE file in the root directory for full terms.
 */

//
// Created by osayamen on 9/15/24.
//

#ifndef CSRC_NICHE_CUH
#define CSRC_NICHE_CUH

namespace kleos{
    template<typename  Element>
    requires(std::is_integral_v<Element>)
    __host__ __forceinline__
    auto subsets(uint* __restrict__ const& parents,
        Element* __restrict__ const& platoon,
        const unsigned int& world,
        const unsigned int& gID){
        uint16_t gSize = 0U;
        for(unsigned int i = 0; i < world; ++i){
            if (parents[i] == gID) {
                platoon[gSize++] = i;
            }
        }
        return gSize;
    }

    template<typename T> requires std::equality_comparable<T>
    __forceinline__
    bool dualSetCompare(const T& v00,
                                        const T& v01,
                                        const T& v10,
                                        const T& v11){
        return ((v00 == v10 && v01 == v11) || (v00 == v11 && v01 == v10));
    }

    enum Stability : uint8_t{
            STABLE = 0,
            EXPERIMENTAL = 1
    };
}
#endif //CSRC_NICHE_CUH
