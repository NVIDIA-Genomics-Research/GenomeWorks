/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <vector>

#include "cudamapper/types.hpp"

namespace claragenomics {

    std::vector<Overlap> fuse_overlaps(std::vector<Overlap> unfused_overlaps) {
        std::vector<Overlap> fused_overlaps;

        if (unfused_overlaps.size() == 0) {
            return fused_overlaps;
        }

        Overlap fused_overlap = unfused_overlaps[0];

        for (size_t i = 0; i < unfused_overlaps.size() - 1; i++) {
            const Overlap &next_overlap = unfused_overlaps[i + 1];
            if ((fused_overlap.target_read_id_ == next_overlap.target_read_id_) &&
                (fused_overlap.query_read_id_ == next_overlap.query_read_id_)) {
                //need to fuse
                fused_overlap.num_residues_ += next_overlap.num_residues_;
                fused_overlap.query_end_position_in_read_ = next_overlap.query_end_position_in_read_;
                fused_overlap.target_end_position_in_read_ = next_overlap.target_end_position_in_read_;
            } else {
                fused_overlaps.push_back(fused_overlap);
                fused_overlap = unfused_overlaps[i + 1];
            }
        }
        fused_overlaps.push_back(fused_overlap);
        return fused_overlaps;
    }
}

