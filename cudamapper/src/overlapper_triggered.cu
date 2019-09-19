/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>
#include <unordered_map>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "overlapper_triggered.hpp"
#include "cudamapper/overlapper.hpp"
#include "cudamapper_utils.hpp"
#include "matcher.hpp"

namespace claragenomics {
    std::vector<Overlap> const OverlapperTriggered::get_overlaps(const std::vector<Anchor> &anchors, const Index &index) {
        const auto& read_names = index.read_id_to_read_name();
        const auto& read_lengths = index.read_id_to_read_length();

	thrust::device_vector<Anchor> anchors_d(anchors.begin(), anchors.end());

	thrust::sort(anchors_d.begin(), anchors_d.end(),
		     [] __device__(Anchor i, Anchor j) -> bool {
			 return (i.query_read_id_ < j.query_read_id_) ||
			     ((i.query_read_id_ == j.query_read_id_) &&
			      (i.target_read_id_ < j.target_read_id_)) ||
			     ((i.query_read_id_ == j.query_read_id_) &&
			      (i.target_read_id_ == j.target_read_id_) &&
			      (i.query_position_in_read_ < j.query_position_in_read_)) ||
			     ((i.query_read_id_ == j.query_read_id_) &&
			      (i.target_read_id_ == j.target_read_id_) &&
			      (i.query_position_in_read_ == j.query_position_in_read_) &&
			      (i.target_position_in_read_ < j.target_position_in_read_));
		     }); // TODO: Matcher kernel should return sorted anchors, making this unnecessary

	thrust::host_vector<Anchor> sortedAnchors(anchors_d.begin(), anchors_d.end());

        //Loop through the overlaps, "trigger" when an overlap is detected and add it to vector of overlaps
        //when the overlap is left
        std::vector<Overlap> overlaps;
        bool in_chain = false;
        uint16_t tail_length = 0;
        uint16_t tail_length_for_chain = 3;
        uint16_t score_threshold = 1;
        Anchor overlap_start_anchor;
        Anchor prev_anchor;
        Anchor current_anchor;

        //Very simple scoring function to quantify quality of overlaps.
        auto anchor_score = [](Anchor a, Anchor b) {
            if ((b.query_position_in_read_ - a.query_position_in_read_) < 350){
                return 2;
            } else {
                return 1; //TODO change to a more sophisticated scoring method
            }

        };

        //Add an anchor to an overlap
        auto terminate_anchor = [&] () {
            Overlap new_overlap;
            new_overlap.query_read_id_ = prev_anchor.query_read_id_;
            new_overlap.query_read_name_ = read_names[prev_anchor.query_read_id_];
            new_overlap.target_read_id_ = prev_anchor.target_read_id_;
            new_overlap.target_read_name_ = read_names[prev_anchor.target_read_id_];
            new_overlap.query_length_ = read_lengths[prev_anchor.target_read_id_];
            new_overlap.target_length_ = read_lengths[prev_anchor.target_read_id_];
            new_overlap.num_residues_ = tail_length;
            new_overlap.target_end_position_in_read_ = prev_anchor.target_position_in_read_;
            new_overlap.target_start_position_in_read_ = overlap_start_anchor.target_position_in_read_;
            new_overlap.query_end_position_in_read_ = prev_anchor.query_position_in_read_;
            new_overlap.query_start_position_in_read_ = overlap_start_anchor.query_position_in_read_;
            new_overlap.overlap_complete = true;
            overlaps.push_back(new_overlap);
        };

        for(size_t i=0; i<sortedAnchors.size();i++){
            current_anchor = sortedAnchors[i];
            if ((current_anchor.query_read_id_ == prev_anchor.query_read_id_) && (current_anchor.target_read_id_ == prev_anchor.target_read_id_)){ //TODO: For first anchor where prev anchor is not initialised can give incorrect result
                //In the same read pairing as before
                int score = anchor_score(prev_anchor, current_anchor);
                if (score > score_threshold){
                    tail_length++;
                    if (tail_length == tail_length_for_chain) {//we enter a chain
                        in_chain = true;
                        overlap_start_anchor = sortedAnchors[i-tail_length + 1]; //TODO check
                    }
                } else {
                    if(in_chain){
                        terminate_anchor();
                    }

                    tail_length = 1;
                    in_chain = false;
                }
                prev_anchor = current_anchor;
            }
            else {
                //In a new read pairing
                if(in_chain){
                    terminate_anchor();
                }
                //Reinitialise all values
                tail_length = 1;
                in_chain = false;
                prev_anchor = current_anchor;
            }
        }

        //terminate any hanging anchors
        if(in_chain){
            terminate_anchor();
        }
        //Return fused overlaps
        return fuse_overlaps(overlaps);
    }
}
