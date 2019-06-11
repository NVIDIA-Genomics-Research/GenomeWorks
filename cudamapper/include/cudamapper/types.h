/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

namespace genomeworks {

/// ArrayBlock - points to a part of an array
///
/// Contains the index of the first element in the block and the number of elements
struct ArrayBlock {
    size_t first_element_;
    std::uint32_t block_size_;
}

using position_in_read_t = std::uint32_t;
using representation_t = std::uint64_t; // this depends on kmer size, in some cases could also be 32-bit
using read_id_t = std::uint64_t; // can this be 32-bit?

}
