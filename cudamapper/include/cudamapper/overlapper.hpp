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

#include "index.hpp"
#include "types.hpp"

namespace claragenomics {
/// \addtogroup cudamapper
/// \{
    /// Overlapper - given anchors and a read index, calculates overlaps between reads
    class Overlapper {
    public:

        /// \brief Virtual destructor for Overlapper
        virtual ~Overlapper() = default;

        /// \brief returns overlaps for a set of reads
        /// \param anchors vector of anchor objects. Does not need to be ordered
        /// \param index representation index for reads
        /// \return vector of Overlap objects
        virtual const std::vector<Overlap> get_overlaps(std::vector<Anchor> &anchors, const Index &index) = 0;

        /// \brief prints overlaps to stdout in <a href="https://github.com/lh3/miniasm/blob/master/PAF.md">PAF format</a>
        static void print_paf(const std::vector<Overlap> &overlaps);

        /// \brief removes overlaps which are unlikely to be true overlaps
        /// \param overlaps vector of Overlap objects to be filtered
        /// \param min_residues smallest number of residues (anchors) for an overlap to be accepted
        /// \param min_overlap_len the smallest overlap distance which is accepted
        /// \return vector of filtered Overlap objects
        static std::vector<Overlap> filter_overlaps(const std::vector<Overlap> &overlaps, size_t min_residues=5,
                size_t min_overlap_len=0);
    };
//}
}
