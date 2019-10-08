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

#include <vector>
#include "claragenomics/cudamapper/index.hpp"

namespace claragenomics {
namespace cudamapper {

    /// Matcher - finds anchors
    ///
    /// For a given index, all reads equal to or less than query_target_division_idx are mapped to all other reads.
    /// If query_target_division_idx is 0 then all-vs-all mapping is performed.
    ///
    /// Anchor is a pair of two sketch elements with the same sketch element representation from different reads.
    /// Anchors are symmetrical, so only one anchor is generated for each pair (if sketch element from read 5 overlaps a sketch element from read 8
    /// then the same sketch element from read 8 overlaps the sketch element from read 5).
    /// 
    /// Anchors are grouped by query read id and within that by representation (both in increasing order).
    /// Assume q0p4t2p8 means anchor of read id 0 at position 4 and read id 2 at position 8.
    /// Assume read 0 has 30 sketch elements with certain representation, read 1 40 and read 2 50.
    /// Anchors for read 0 as query and that represtnation looks like this:
    /// q0p0t1p0, q0p0t1p1 .. q0p0t1p39, q0p0t2p0, q0p0t2p1 ... q0p0t2p49, q0p1t1p0, q0p1t1p1 ... q0p1t1p39, q0p1t2p0 .. q0p1t2p49, q0p2t1p0 ... q0p2t1p39, q0p2t2p0 ... q0p2t2p49, q0p3t1p0 ... q0p29t2p49
    class Matcher {
    public:

        /// \brief Construtor
        /// \param index index to generate anchors from
        /// \param query_target_division_idx the index after which all reads are target reads. If set to 0 then all-vs-all mapping is performed
        Matcher(const Index &index, uint32_t query_target_division_idx);

        /// \brief return anchors
        /// \return anchors
        std::vector<Anchor>& anchors();
    private:

        /// \biref list of anchors
        std::vector<Anchor> anchors_h_;
    };
}
}
