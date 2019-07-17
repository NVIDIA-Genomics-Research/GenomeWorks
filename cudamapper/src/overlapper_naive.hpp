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

#include "cudamapper/types.hpp"
#include "cudamapper/overlapper.hpp"
#include "matcher.hpp"

namespace claragenomics {

    /// OverlapperNaive - generates overlaps and displays them on screen. This overlapper uses a greedy approach where
    /// very simplet chaining is used - if two strings have more than one anchor this is considered to be an overlap.
    /// The extend of the overlap is the two anchors which are furthest apart.
    class OverlapperNaive: public Overlapper {

    public:
        /// \brief finds all overlaps
        ///
        /// \param anchors vector of anchors
        /// \param index Index
        /// \return vector of Overlap objects
        virtual const std::vector<Overlap> get_overlaps(const std::vector<Anchor>& anchors, const Index& index) override;

    };
}