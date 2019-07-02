/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "matcher.hpp"

namespace claragenomics {

/*    typedef struct Anchor{
        read_id_t query_read_id_;
        read_id_t target_read_id_;
        position_in_read_t query_position_in_read_;
        position_in_read_t target_position_in_read_;
    } Anchor;*/

/*    __global__ void matcher(const position_in_read_t* const positions_in_read_d,
                            const read_id_t* const read_ids_d,
                            const ArrayBlock* const blocks_belonging_to_reads_d,
                            const ArrayBlock* const sketch_elements_with_same_representation_to_check_d,
                            const ArrayBlock* const sketch_elements_with_same_representation_and_read_d,
                            Anchor** const anchors_for_read_d,
                            std:uint32_t* const total_anchors_for_read_d) {
        extern __shared__ position_in_read_t query_positions_in_read[]; // number of elements == blockDim.x

        const read_id_t query_read_id = blockIdx.x;
        const ArrayBlock block_for_query_sketch_elements = blocks_belonging_to_reads_d[query_read_id];

        __shared__ std::uint32_t anchors_written_so_far = 0;

        // go over all representations in this read one by one
        for (std::uint32_t representation_index_for_this_read = block_for_query_sketch_elements.first_element_;
             representation_index_for_this_read < block_for_query_sketch_elements.block_size_;
             ++representation_index_for_this_read) {
                 // load all position_in_read for this read and representation
                 ArrayBlock block_for_this_read_and_representation = sketch_elements_with_same_representation_and_read_d[representation_index_for_this_read];
                 for (std::uint32_t i = threadIdx.x; i < block_for_this_read_and_representation.block_size_; ++i) {
                     query_positions_in_read[i] = positions_in_read_d[block_for_this_read_and_representation.first + i];
                 }
                 std::uint32_t total_query_sketch_elements = block_for_this_read_and_representation.block_size_;
                 __syncthreads();

                 // load all other position_in_read which should be matched
                 block_for_this_read_and_representation = sketch_elements_with_same_representation_to_check_d[representation_index_for_this_read];
                 for (std::uint32_t i = threadIdx.x; i < block_for_this_read_and_representation.block_size_; ++i) {
                     const read_id_t target_read_id = read_ids_d[block_for_this_read_and_representation.first_element_ + i];
                     const position_in_read_t target_position_in_read = positions_in_read_d[block_for_this_read_and_representation.first_element_ + i];
                     for (int j = 0; j < total_query_sketch_elements; ++j) {
                         anchors_for_read_d[query_read_id][anchors_written_so_far + i*total_query_sketch_elements + j].query_read_id_ = query_read_id;
                         anchors_for_read_d[query_read_id][anchors_written_so_far + i*total_query_sketch_elements + j].taget_read_id_ = target_read_id;
                         anchors_for_read_d[query_read_id][anchors_written_so_far + i*total_query_sketch_elements + j].query_position_in_read_ = query_position_in_read_[i];
                         anchors_for_read_d[query_read_id][anchors_written_so_far + i*total_query_sketch_elements + j].target_position_in_read_ = target_position_in_read;
                     }
                 }
                 __syncthreads
                 (0 == threadIdx.x) anchors_written_so_far += block_for_this_read_and_representation.block_size_ * total_query_sketch_elements];
             }

        if (0 == threadIdx.x) total_anchors_for_read_d[query_read_id] = anchors_written_so_far;
    }*/

    Matcher::Matcher(const IndexCPU& index) {
/*        std::shared_ptr<position_in_read_t> positions_in_read_h = index.positions_in_read();
        // allocate std::unique<std::uint32_t> positions_in_read_d and move the data to the device
        // do the same for read_ids_d and directions_d

        // Each element points to blocks of sketch_elements_with_same_representation_to_check and sketch_elements_with_same_representation_and_read
        std::vector<ArrayBlock> blocks_belonging_to_reads(index.number_of_reads(), {0,0});

        // Each element points to one block of positions_in_read, read_ids and directions arrays that belong to sketch elements with the same representation
        // and whose read_id is larger than some read_id
        // Elements pointing to sketch elements with the same read are grouped together. Such groups are pointed to by the elements of blocks_belonging_to_reads
        std::vector<ArrayBlock> sketch_elements_with_same_representation_to_check;
        // Each element points to one block of positions_in_read, read_ids and directions arrays that belong to sketch elements with the same representation and read_id
        // Elements pointing to sketch elements with the same read are grouped together. Such groups are pointed to by the elements of blocks_belonging_to_reads
        std::vector<ArrayBlock> sketch_elements_with_same_representation_and_read;

        for (std::size_t read_id = 0; read_id < index.number_of_reads(); ++read_id) {
            if (read_id != 0) blocks_belonging_to_reads[read_id].first_element_ = blocks_belonging_to_reads[read_id - 1].first_element_ + blocks_belonging_to_reads[read_id - 1].block_size_;
            const std::map<representation_t, ArrayBlock>& array_blocks_for_representations = index.array_block_for_representations_and_reads()[read_id];
                for (const std::map<representation_t, ArrayBlock>::iterator& array_block_for_representation : array_blocks_for_representations) { // this is a map so we know that representations are sorted in ascending order
                    const representation_t& representation = array_block_for_representation.first;
                    const ArrayBlock& array_block_for_this_representation_and_read = iterator.second;
                    const ArrayBlock& array_block_for_this_representation_and_all_reads = index.representation_to_all_its_sketch_elements().find(representation);
                    // due to symmetry we only want to check reads with read_id greater than the current read_id
                    // within one representation all blocks are stored in ascending order, so we only want to add parts of arrays after array_block_for_this_representation_and_read
                    ArrayBlock array_block_to_check;
                    array_block_to_check.first_element_ = array_block_for_this_representation_and_read.first_element_ + array_block_for_this_representation_and_read.block_size_;
                    array_block_to_check.block_size_ = array_block_for_this_representation_and_all_reads.block_size_ - array_block_to_check.first_element_;

                    sketch_elements_with_same_representation_to_check.push_back(block_to_check);
                    sketch_elements_with_same_representation_and_read.push_back(array_block_for_this_representation_and_read);
                }
            ++blocks_belonging_to_reads[read_id].block_size_;
            }
        }

        // copy blocks_belonging_to_reads_d, sketch_elements_with_same_representation_to_check and sketch_elements_with_same_representation_and_read to device

        // call kernel

        // copy anchors_for_read_d and total_anchors_for_read_d to host

        // generate overlaps

        //convert read_id into real read identifier using const std::vector<string>& index.read_id_to_read_name()
*/    }

}
