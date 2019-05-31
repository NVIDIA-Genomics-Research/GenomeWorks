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

#include "index_gpu.hpp"

namespace genomeworks {

    /// Matcher - Finds anchors
    /// For each representation in a sequence finds all other sequences that containt the same
    /// representation. Does this for each sequence.
    class Matcher {
    public:
        /// \brief construtor
        ///
        /// \param index
        Matcher(const IndexGPU& index);
    private:
    };
}
