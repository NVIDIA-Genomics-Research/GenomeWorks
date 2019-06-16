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

namespace genomeworks {

/// ArrayBlock - points to a part of an array
///
/// Contains the index of the first element in the block and the number of elements
struct ArrayBlock {
    /// index of the first element of the block
    size_t first_element_;
    /// number of elements of the block
    std::uint32_t block_size_;
};

/// position_in_read_t
using position_in_read_t = std::uint32_t;
/// representation_t
using representation_t = std::uint64_t; // this depends on kmer size, in some cases could also be 32-bit
/// read_id_t
using read_id_t = std::uint64_t; // can this be 32-bit?

}
