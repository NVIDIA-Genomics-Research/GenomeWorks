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
#include <iterator>
#include <unordered_map>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <claragenomics/utils/cudautils.hpp>
#include "overlapper_triggered.hpp"
#include "cudamapper/overlapper.hpp"
#include "cudamapper_utils.hpp"
#include "matcher.hpp"

namespace claragenomics {
    std::vector<Overlap> const OverlapperTriggered::get_overlaps(std::vector<Anchor> &anchors, const Index &index) {
        const auto& read_names = index.read_id_to_read_name();
        const auto& read_lengths = index.read_id_to_read_length();
	size_t total_anchors = anchors.size();

	// fetch memory info of the current device
	size_t total = 0, free = 0;
	CGA_CU_CHECK_ERR(cudaMemGetInfo(&free, &total));

	// Using 80% of available memory as heuristic since not all available memory can be used
	// due to fragmentation.
	size_t max_anchors_per_block = 0.8 * free/sizeof(Anchor);
    // The thurst sort function makes a local copy of the array, so we need twice
    // twice the device memory available for the sort to succeed.
    max_anchors_per_block /= 2;

	// comparison function object
	auto comp = [] __host__ __device__ (Anchor i, Anchor j) -> bool {
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
	};

	thrust::device_vector<Anchor> anchors_buf;

	// chunking anchors array to a size that fits in memory
	// sort the individual chunks and merge the sorted chunks into host array
	for(std::vector<Anchor>::iterator anchors_iter = anchors.begin();
	    anchors_iter < anchors.end();
	    anchors_iter += max_anchors_per_block){

	    auto curblock_start = anchors_iter;
	    auto curblock_end = anchors_iter + max_anchors_per_block;
	    if(curblock_end > anchors.end())
		curblock_end = anchors.end();

	    auto n_anchors_curblock = curblock_end - curblock_start;

	    // move current block to device
	    anchors_buf.resize(n_anchors_curblock);
	    thrust::copy(curblock_start, curblock_end, anchors_buf.begin());

	    // sort on device
	    thrust::sort(thrust::device, anchors_buf.begin(), anchors_buf.end(), comp);

	    // move sorted anchors in current block back to host
	    thrust::copy(anchors_buf.begin(), anchors_buf.end(), curblock_start);

	    // start merging the sorted anchor blocks from second iteration
	    if(anchors_iter != anchors.begin()){
		std::inplace_merge(anchors.begin(), curblock_start, curblock_end, comp);
	    }
	}

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
            new_overlap.query_length_ = read_lengths[prev_anchor.query_read_id_];
            new_overlap.target_length_ = read_lengths[prev_anchor.target_read_id_];
            new_overlap.num_residues_ = tail_length;
            new_overlap.target_end_position_in_read_ = prev_anchor.target_position_in_read_;
            new_overlap.target_start_position_in_read_ = overlap_start_anchor.target_position_in_read_;
            new_overlap.query_end_position_in_read_ = prev_anchor.query_position_in_read_;
            new_overlap.query_start_position_in_read_ = overlap_start_anchor.query_position_in_read_;
            new_overlap.overlap_complete = true;
            overlaps.push_back(new_overlap);
        };

        for(size_t i=0; i<anchors.size();i++){
            current_anchor = anchors[i];
            if ((current_anchor.query_read_id_ == prev_anchor.query_read_id_) && (current_anchor.target_read_id_ == prev_anchor.target_read_id_)){ //TODO: For first anchor where prev anchor is not initialised can give incorrect result
                //In the same read pairing as before
                int score = anchor_score(prev_anchor, current_anchor);
                if (score > score_threshold){
                    tail_length++;
                    if (tail_length == tail_length_for_chain) {//we enter a chain
                        in_chain = true;
                        overlap_start_anchor = anchors[i-tail_length + 1]; //TODO check
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
