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
#include "matcher.hpp"

namespace claragenomics {
    typedef struct Overlap {
        read_id_t query_read_id_;
        read_id_t target_read_id_;
        position_in_read_t query_start_position_in_read_;
        position_in_read_t target_start_position_in_read_;
        position_in_read_t query_end_position_in_read_;
        position_in_read_t target_end_position_in_read_;
        uint32_t num_residues_ = 0;
        bool overlap_complete = false;
    } Overlap;

    std::vector<Overlap> get_overlaps(std::vector<Matcher::Anchor>);

    void print_paf(std::vector<Overlap>);
}