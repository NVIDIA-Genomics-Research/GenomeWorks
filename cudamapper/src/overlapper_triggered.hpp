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

    /// OverlapperTriggered - generates overlaps and displays them on screen.
    /// Uses a dynamic programming approach where an overlap is "triggered" when a run of
    /// Anchors (e.g 3) with a score above a threshold is encountered and untriggerred
    /// when a single anchor with a threshold below the value is encountered.
    class OverlapperTriggered: public Overlapper {

    public:
        /// \brief finds all overlaps
        /// Uses a dynamic programming approach where an overlap is "triggered" when a run of
        /// Anchors (e.g 3) with a score above a threshold is encountered and untriggerred
        /// when a single anchor with a threshold below the value is encountered.
        /// \param anchors vector of anchors
        /// \param index Index
        /// \return vector of Overlap objects
        const std::vector<Overlap> get_overlaps(std::vector<Anchor> &anchors, const Index &index) override;
    };
}
