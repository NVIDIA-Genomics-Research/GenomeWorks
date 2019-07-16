/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>

#include "overlapper_naive.hpp"
#include "cudamapper/overlapper.hpp"
#include "matcher.hpp"


namespace claragenomics {
    void OverlapperNaive::print_paf(std::vector<Overlap> overlaps){
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

    std::vector<Overlap> const OverlapperNaive::get_overlaps(const std::vector<Anchor>& anchors, Index& index) {

        std::map<std::pair<int,int>, Overlap> reads_to_overlaps;

        auto read_names = index.read_id_to_read_name();

        for(auto anchor: anchors){
            std::pair<int,int> read_pair;
            read_pair.first= anchor.query_read_id_;
            read_pair.second = anchor.target_read_id_;

            //pair not seen yet
            if (reads_to_overlaps.find(read_pair) == reads_to_overlaps.end()){
                Overlap new_overlap;
                new_overlap.num_residues_++;
                new_overlap.query_read_id_ = anchor.query_read_id_;
                new_overlap.query_read_name_ = read_names[anchor.query_read_id_];

                new_overlap.target_read_id_ = anchor.target_read_id_;
                new_overlap.target_read_name_ = read_names[anchor.target_read_id_];

                new_overlap.query_start_position_in_read_ = anchor.query_position_in_read_;
                new_overlap.target_start_position_in_read_ = anchor.query_position_in_read_;
                reads_to_overlaps[read_pair] = new_overlap;
            } else {
                //Pair has been seen before
                Overlap& overlap = reads_to_overlaps[read_pair];

                if (overlap.num_residues_ == 1){
                    //need to complete the overlap
                    overlap.num_residues_++;
                    if (anchor.query_position_in_read_ < overlap.query_start_position_in_read_){
                        overlap.query_end_position_in_read_ = overlap.query_start_position_in_read_;
                        overlap.query_start_position_in_read_ = anchor.query_position_in_read_;
                    } else{
                        overlap.query_end_position_in_read_ = anchor.query_position_in_read_;
                    }
                    overlap.overlap_complete = true;
                } else {
                    overlap.num_residues_++;

                    if(anchor.query_position_in_read_ < overlap.query_start_position_in_read_){
                        overlap.query_start_position_in_read_ = anchor.query_position_in_read_;
                    }

                    if(anchor.query_position_in_read_ > overlap.query_end_position_in_read_){
                        overlap.query_end_position_in_read_ = anchor.query_position_in_read_;
                    }

                    if(anchor.target_position_in_read_ < overlap.target_start_position_in_read_){
                        overlap.target_start_position_in_read_ = anchor.target_position_in_read_;
                    }

                    if(anchor.target_position_in_read_ > overlap.target_end_position_in_read_){
                        overlap.target_end_position_in_read_ = anchor.target_position_in_read_;
                    }
                }
            }
        }


        std::vector<Overlap> overlaps;

        for( std::map<std::pair<int,int>, Overlap>::iterator it = reads_to_overlaps.begin(); it != reads_to_overlaps.end(); ++it ) {
            auto overlap = it->second;
            if (overlap.overlap_complete){
                overlaps.push_back(it->second);
            }
        }
        return overlaps;
    }

}
