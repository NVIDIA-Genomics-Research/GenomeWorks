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
        /// \brief returns overlaps for a set of reads
        /// \return vector of overlaps
        virtual const std::vector<Overlap> get_overlaps(const std::vector<claragenomics::Anchor> &, Index &) = 0;

        /// \brief prints overlaps to stdout in PAF format
        void print_paf(std::vector<Overlap> overlaps);
    };
//}
}