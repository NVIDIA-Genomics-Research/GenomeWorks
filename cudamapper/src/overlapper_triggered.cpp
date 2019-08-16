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
#include <string>
#include <vector>
#include <unordered_map>

#include "overlapper_triggered.hpp"
#include "cudamapper/overlapper.hpp"
#include "matcher.hpp"

namespace claragenomics {
    std::vector<Overlap> OverlapperTriggered::fuse_overlaps(std::vector<Overlap> unfused_overlaps){
        std::vector<Overlap> fused_overlaps;
        Overlap fused_overlap = unfused_overlaps[0];
        for (size_t i=0; i< unfused_overlaps.size() - 1; i++){
            Overlap next_overlap = unfused_overlaps[i+1];
            if ((fused_overlap.target_read_id_ == next_overlap.target_read_id_) && (fused_overlap.query_read_id_ == next_overlap.query_read_id_)){
                //need to fuse
                fused_overlap.num_residues_ += next_overlap.num_residues_;
                fused_overlap.query_end_position_in_read_ = next_overlap.query_end_position_in_read_;
                fused_overlap.target_end_position_in_read_ = next_overlap.target_end_position_in_read_;
            } else {
                fused_overlaps.push_back(fused_overlap);
                fused_overlap = unfused_overlaps[i+1];
            }
        }
        return fused_overlaps;
    }

    std::vector<Overlap> const OverlapperTriggered::get_overlaps(const std::vector<Anchor> &anchors, const Index &index) {
        const auto& read_names = index.read_id_to_read_name();
        const auto& read_lengths = index.read_id_to_read_length();

        //Sort the anchors by two keys (read_id, query_start)
        std::vector<Anchor> sortedAnchors = anchors;
        std::sort(sortedAnchors.begin(), sortedAnchors.end(), [](Anchor i, Anchor j) -> bool {
                return (i.query_read_id_ < j.query_read_id_) ||
                ((i.query_read_id_ == j.query_read_id_) &&
                (i.target_read_id_ < j.target_read_id_)) ||
                ((i.query_read_id_ == j.query_read_id_) &&
                (i.target_read_id_ == j.target_read_id_) &&
                (i.query_position_in_read_ < j.query_position_in_read_));
        }); // TODO: Matcher kernel should return sorted anchors, making this unnecessary

        //Loop through the overlaps, "trigger" when an overlap is detected and add it to vector of overlaps
        //When the overlap is left
        std::vector<Overlap> overlaps;
        bool in_chain = false;
        uint16_t tail_length = 0;
        uint16_t tail_length_for_chain = 3;
        uint16_t score_threshold = 1;
        Anchor overlap_start_anchor;
        Anchor prev_anchor;

        //Very simple scoring function to quantify quality of overlaps.
        auto anchor_score = [](Anchor a, Anchor b) {
            if ((b.query_position_in_read_ - a.query_position_in_read_) < 350){
                return 2;
            } else {
                return 1; //TODO change
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
            Anchor current_anchor = sortedAnchors[i];

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

        //Stage 3: Return fused overlaps
        return fuse_overlaps(overlaps);
    }
}
