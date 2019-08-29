/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <functional>
#include <vector>

#include "cudamapper/types.hpp"

namespace claragenomics {
    /// \brief given a vector of overlaps, combines all overlaps from the same read pair
    ///
    /// If two or more overlaps come from the same read pair they are combined into one large overlap:
    /// Example:
    /// Overlap 1:
    ///   Query ID = 18
    ///   Target ID = 42
    ///   Query start = 420
    ///   Query end = 520
    ///   Target start = 783
    ///   Target end = 883
    /// Overlap 2:
    ///   Query ID = 18
    ///   Target ID = 42
    ///   Query start = 900
    ///   Query end = 1200
    ///   Target start = 1200
    ///   Target end = 1500
    /// Fused overlap:
    ///   Query ID = 18
    ///   Target ID = 42
    ///   Query start = 420
    ///   Query end = 1200
    ///   Target start = 783
    ///   Target end = 1500
    ///
    /// \param unfused_overlaps vector of overlaps, sorted by (query_id, target_id) combination and query_start_position
    /// \return vector of overlaps
    std::vector<Overlap> fuse_overlaps(std::vector<Overlap> unfused_overlaps);

    /// \brief given a std::vector of two or more sorted std::vector objects and a function to compare them, returns one sorted std::vector
    ///
    /// \param src object of type std::vector<std::vector<T>> - vector of of two or more sorted vectors.
    /// \param dst reference to object of type std::vector<T> where result should be written
    /// \return vector of sorted elements
    // TODO this algorithm is not very performant - should be reimplemented to run in log2(N) time and use multithreading.
    template <class T>
    void merge_n_sorted_vectors(const std::vector<std::vector<T>> &src, std::vector<T> &dst,
                                std::function<bool(T, T)> comp) {
        if (src.size() < 1){
            return;
        }
        dst = src[0];
        for(size_t i=1; i<src.size(); i++) {
            std::vector<T> tmp (dst.size() + src[i].size());
            std::merge(dst.begin(), dst.end(), src[i].begin(), src[i].end(), tmp.begin(), comp);
            dst = tmp;
        }
    }
}
