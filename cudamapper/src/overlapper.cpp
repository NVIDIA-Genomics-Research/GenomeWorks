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
    void Overlapper::print_paf(std::vector<Overlap> overlaps){
        std::string relative_strand = "+";
        for(auto overlap: overlaps){
            std::printf("%s\t%i\t%i\t%i\t%s\t%s\t%i\t%i\t%i\t%i\t%i\t%i\n",
                        overlap.query_read_name_.c_str(),
                        0,
                        overlap.query_start_position_in_read_,
                        overlap.query_end_position_in_read_,
                        relative_strand.c_str(),
                        overlap.target_read_name_.c_str(),
                        0,
                        overlap.target_start_position_in_read_,
                        overlap.target_end_position_in_read_,
                        overlap.num_residues_,
                        0,
                        255
            );
        }
    }
}
