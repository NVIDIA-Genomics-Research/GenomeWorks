/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "cudamapper/overlapper.hpp"
#include "index_cpu.hpp"

namespace claragenomics {

    std::vector<Overlap> Overlapper::filter_overlaps(const std::vector<Overlap> &overlaps,size_t  min_residues, size_t  min_overlap_len) {
        std::vector<Overlap> filtered_overlaps;
        for(auto overlap: overlaps){
            if ((overlap.num_residues_ >= min_residues) &&
                ((overlap.query_end_position_in_read_ - overlap.query_start_position_in_read_) > min_overlap_len)){
                filtered_overlaps.push_back(overlap);
            }
        }
        return filtered_overlaps;
    }

    void Overlapper::print_paf(const std::vector<Overlap> &overlaps){
        std::vector<Overlap> filtered_overlaps = filter_overlaps(overlaps);

        std::string relative_strand = "+";
        for(const auto& overlap: filtered_overlaps){
            std::printf("%s\t%i\t%i\t%i\t%s\t%s\t%i\t%i\t%i\t%i\t%i\t%i\n",
                        overlap.query_read_name_.c_str(),
                        overlap.query_length_,
                        overlap.query_start_position_in_read_,
                        overlap.query_end_position_in_read_,
                        relative_strand.c_str(),
                        overlap.target_read_name_.c_str(),
                        overlap.target_length_,
                        overlap.target_start_position_in_read_,
                        overlap.target_end_position_in_read_,
                        overlap.num_residues_,
                        0,
                        255
            );
        }
    }
}
