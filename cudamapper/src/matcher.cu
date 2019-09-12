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
#include <memory>
#include "matcher.hpp"
#include <claragenomics/logging/logging.hpp>
#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/utils/device_buffer.cuh>

namespace claragenomics {

    /// \brief Generates anchors for all reads
    ///
    /// Each thread block works on one query read. read_id_to_pointer_arrays_section_d points sections of read_id_to_sketch_elements_d and read_id_to_sketch_elements_to_check_d.
    /// Kernel matches sketch elements that have the same representation. These sketch elements are pointed to by read_id_to_sketch_elements_d and read_id_to_sketch_elements_to_check_d.
    ///
    /// Kernel has an extern shared memory array which should be at least as large as the largest number of sketch elements with the same representation in any block
    ///
    /// Anchors are grouped by query read id and within that by representation (both in increasing order).
    /// Assume q0p4t2p8 means anchor of read id 0 at position 4 and read id 2 at position 8.
    /// Assume read 0 has 30 sketch elements with certain representation, read 1 40 and read 2 50.
    /// Anchors for read 0 as query and that represtnation looks like this:
    /// q0p0t1p0, q0p0t1p1 .. q0p0t1p39, q0p0t2p0, q0p0t2p1 ... q0p0t2p49, q0p1t1p0, q0p1t1p1 ... q0p1t1p39, q0p1t2p0 .. q0p1t2p49, q0p2t1p0 ... q0p2t1p39, q0p2t2p0 ... q0p2t2p49, q0p3t1p0 ... q0p29t2p49
    ///
    /// \param positions_in_reads_d positions of sketch elements in their reads first sorted by representation and then by read id
    /// \param read_ids_d read ids of reads sketch elements belong to first sorted by representation and then by read id (elements with the same index from positions_in_reads_d belong to the same sketch element)
    /// \param read_id_to_sketch_elements_d every element points to a section of positions_in_reads_d and read_ids_d that belong to sketch elements with the same read_id and representation
    /// \param read_id_to_sketch_elements_to_check_d every element points to a section of positions_in_reads_d and read_ids_d that belong to sketch elements with the same representation and read_id larger than some value
    /// \param read_id_to_pointer_arrays_section_d every element belongs to one read_id and points to its sections of read_id_to_sketch_elements_d and read_id_to_sketch_elements_to_check_d
    /// \param anchors_d pairs of sketch elements with the same representation that belong to different reads
    /// \param read_id_to_anchors_section_d points to parts of anchors_d in which all anchors have the same read_id
    __global__ void generate_anchors(const position_in_read_t *const positions_in_reads_d,
                                     const read_id_t *const read_ids_d,
            // const SketchElement::DirectionOfRepresentation* const directions_of_reads_d, // currently we don't use direction
                                     ArrayBlock *read_id_to_sketch_elements_d,
                                     ArrayBlock *read_id_to_sketch_elements_to_check_d,
                                     ArrayBlock *read_id_to_pointer_arrays_section_d,
                                     Anchor *const anchors_d,
                                     ArrayBlock *read_id_to_anchors_section_d) {

        extern __shared__ position_in_read_t query_positions[]; // size = largest value of block_size in read_id_to_sketch_elements_d

        const read_id_t query_read_id = blockIdx.x;
        const ArrayBlock pointer_to_arrays_section = read_id_to_pointer_arrays_section_d[query_read_id];

        __shared__ std::uint32_t anchors_written_so_far;
        if (0 == threadIdx.x) anchors_written_so_far = 0;

        // go over all representations in this read one by one
        for (auto representation_index = pointer_to_arrays_section.first_element_;
             representation_index < pointer_to_arrays_section.first_element_ + pointer_to_arrays_section.block_size_;
             ++representation_index) {

            // load all position_in_read for this read and representation (query)
            ArrayBlock query_sketch_elements_section = read_id_to_sketch_elements_d[representation_index];
            for (auto i = threadIdx.x; i < query_sketch_elements_section.block_size_; ++i) {
                query_positions[i] = positions_in_reads_d[query_sketch_elements_section.first_element_ + i];
            }
            __syncthreads();

            // section of sketch elements with that representation and read_id larger than query_read_id
            ArrayBlock target_sketch_elements_section = read_id_to_sketch_elements_to_check_d[representation_index];
            for (auto i = threadIdx.x; i < target_sketch_elements_section.block_size_; i += blockDim.x) {
                const read_id_t target_read_id = read_ids_d[target_sketch_elements_section.first_element_ + i];
                const position_in_read_t target_position_in_read = positions_in_reads_d[
                        target_sketch_elements_section.first_element_ + i];
                for (int j = 0; j < query_sketch_elements_section.block_size_; ++j) {
                    // writing anchors in form (q1t1,q1t2,q1t3...q2t1,q2t2,q3t3....) for coalescing
                    // TODO: split anchors_d into four arrays for better coalescing?
                    anchors_d[read_id_to_anchors_section_d[query_read_id].first_element_ + anchors_written_so_far +
                              j * target_sketch_elements_section.block_size_ + i].query_read_id_ = query_read_id;
                    anchors_d[read_id_to_anchors_section_d[query_read_id].first_element_ + anchors_written_so_far +
                              j * target_sketch_elements_section.block_size_ + i].target_read_id_ = target_read_id;
                    anchors_d[read_id_to_anchors_section_d[query_read_id].first_element_ + anchors_written_so_far +
                              j * target_sketch_elements_section.block_size_ +
                              i].query_position_in_read_ = query_positions[j];
                    anchors_d[read_id_to_anchors_section_d[query_read_id].first_element_ + anchors_written_so_far +
                              j * target_sketch_elements_section.block_size_ +
                              i].target_position_in_read_ = target_position_in_read;
                }
            }
            __syncthreads();
            if (0 == threadIdx.x) anchors_written_so_far += target_sketch_elements_section.block_size_ *
                                                            query_sketch_elements_section.block_size_;
            __syncthreads();
        }
    }

    Matcher::Matcher(const Index &index) {

        if (0 == index.number_of_reads()) {
            return;
        }


        //Now perform the matching in a loop

        size_t increment = index.maximum_representation();

        size_t max_representation = index.maximum_representation();
        size_t representation_min_range = index.minimum_representation();
        size_t representation_max_range = increment;
        size_t max_anchor_buffer_size_GB = 4;
        size_t max_anchor_buffer_size = max_anchor_buffer_size_GB * 1024 * 1024 *
                                        1024; //TODO: Make this dynamically chosen by available GPU memory
        size_t max_anchors = max_anchor_buffer_size / sizeof(Anchor);

        while (representation_min_range < max_representation) {

            const std::vector<position_in_read_t> &positions_in_reads_h = index.positions_in_reads();
            CGA_LOG_INFO("Allocating {} bytes for positions_in_reads_d",
                         positions_in_reads_h.size() * sizeof(position_in_read_t));
            device_buffer<position_in_read_t> positions_in_reads_d(positions_in_reads_h.size());
            CGA_CU_CHECK_ERR(cudaMemcpy(positions_in_reads_d.data(), positions_in_reads_h.data(),
                                        positions_in_reads_h.size() * sizeof(position_in_read_t),
                                        cudaMemcpyHostToDevice));

            const std::vector<read_id_t> &read_ids_h = index.read_ids();
            CGA_LOG_INFO("Allocating {} bytes for read_ids_d", read_ids_h.size() * sizeof(read_id_t));
            device_buffer<read_id_t> read_ids_d(read_ids_h.size());
            CGA_CU_CHECK_ERR(cudaMemcpy(read_ids_d.data(), read_ids_h.data(), read_ids_h.size() * sizeof(read_id_t),
                                        cudaMemcpyHostToDevice));

            const std::vector<SketchElement::DirectionOfRepresentation> &directions_of_reads_h = index.directions_of_reads();
            CGA_LOG_INFO("Allocating {} bytes for directions_of_reads_d",
                         directions_of_reads_h.size() * sizeof(SketchElement::DirectionOfRepresentation));
            device_buffer<SketchElement::DirectionOfRepresentation> directions_of_reads_d(directions_of_reads_h.size());
            CGA_CU_CHECK_ERR(cudaMemcpy(directions_of_reads_d.data(), directions_of_reads_h.data(),
                                        directions_of_reads_h.size() * sizeof(SketchElement::DirectionOfRepresentation),
                                        cudaMemcpyHostToDevice));

            CGA_LOG_INFO("Computing representation {} -> {}", representation_min_range, representation_max_range);
            // Each CUDA thread block is responsible for one read. For each sketch element in that read it checks all other reads for sketch elements with the same representation and records those pairs.
            // As read_ids are numbered from 0 to number_of_reads - 1 CUDA thread block is responsible for read with read_id = blockIdx.x.
            //
            // Overlapping is symmetric, i.e. if sketch element at position 8 in read 2 overlaps with (= has the same representation as) sketch element at position 4 in read 5 then sketch element
            // at position 4 in read 5 also overlaps with sketch element at position 8 in read 2. It is thus only necessary to check for overlapping in one direction. This is achieved by having each
            // CUDA thread block only check reads with read_ids greater than the read_id of that read.
            //
            // In order to be able to do this check CUDA thread block has to know which sketch elements belong to its read and which are candidates for a match (have read_id greater than CUDA thread block's
            // read_id and the same representation as one of sketch elements from CUDA thread block's read).
            // Note that positions_in_reads, read_ids and directions_of_reads (data arrays) have sketch elements grouped by representation and within one representation grouped by read_id
            // (both representations and read_ids are sorted in increasing order).
            //
            // Each section of read_id_to_sketch_elements belongs to one read_id. Each element in that section points to a section of data arrays that contains sketch elements belonging to that read_id
            // and some representation (it's not important which one).
            // Similarly to this each section of read_id_to_sketch_elements_to_check points to the section of data arrays with read_id greater than the gived read_id and the representation same as the
            // representation of the element in read_id_to_sketch_elements with the same index (it's still not importatn which representation is that).
            // This means that the kernel should match all sketch elements pointed by one element of read_id_to_sketch_elements with all sketch elements pointed to by the element of
            // read_id_to_sketch_elements_to_check with the same index.
            //
            // read_id_to_pointer_arrays_section maps a read_id to its section of read_id_to_sketch_elements and read_id_to_sketch_elements_to_check (pointer arrays).

            std::vector<ArrayBlock> read_id_to_sketch_elements_h; // TODO: we should be able to know this number -> reserve space?
            std::vector<ArrayBlock> read_id_to_sketch_elements_to_check_h;
            std::vector<ArrayBlock> read_id_to_pointer_arrays_section_h(index.number_of_reads(), {0, 0});

            // Anchor is one pair of sketch elements with the same representation in different reads
            // Anchors are symmetric (as explained above), so they are saved in only one direction
            // As only one direction is saved for each representation in each read_id there are going to be sketch_elements_with_that_representation_in_that_read * sketch_elements_with_that_representation_to_check_in_other_reads anchors.
            // This means we know upfront how many anchors are there going to be for each read_id and we can merge all anchors in one array and assign its sections to different read_ids
            std::vector<ArrayBlock> read_id_to_anchors_section_h(index.number_of_reads(), {0, 0});
            std::uint64_t total_anchors = 0;
            std::uint32_t largest_block_size = 0;

            for (std::size_t read_id = 0; read_id < index.number_of_reads(); ++read_id) {
                // First determine the starting index of section of pointer arrays that belong to read with read_id.
                // Reads are processed consecutively. Pointer arrays section for read 0 will start at index 0 and if we assume that all sketch elements in read 0 had a total of 10 unique representation its section will end at index 9. This means that the section for read 1 belongs at index 0 + 10 = 10.
                if (read_id != 0) {
                    read_id_to_pointer_arrays_section_h[read_id].first_element_ =
                            read_id_to_pointer_arrays_section_h[read_id - 1].first_element_ +
                            read_id_to_pointer_arrays_section_h[read_id - 1].block_size_;
                    read_id_to_anchors_section_h[read_id].first_element_ =
                            read_id_to_anchors_section_h[read_id - 1].first_element_ +
                            read_id_to_anchors_section_h[read_id - 1].block_size_;
                }

                const std::vector<Index::RepresentationToSketchElements> &array_blocks_for_this_read_id = index.read_id_and_representation_to_sketch_elements()[read_id];

                read_id_to_pointer_arrays_section_h[read_id].block_size_ = 0;

                // go through all representations in this read
                for (const auto &one_representation_in_this_read : array_blocks_for_this_read_id) {
                    //Check if we are in the correct range
                    if ((one_representation_in_this_read.representation_ < representation_max_range) && (
                            one_representation_in_this_read.representation_ >= representation_min_range)) {

                        const ArrayBlock &array_block_for_this_representation_and_read = one_representation_in_this_read.sketch_elements_for_representation_and_read_id_; // sketch elements with this representation and this read_id
                        const ArrayBlock &whole_data_arrays_section_for_representation = one_representation_in_this_read.sketch_elements_for_representation_and_all_read_ids_; // sketch elements with this representation in all read_ids
                        largest_block_size = std::max(largest_block_size,
                                                      array_block_for_this_representation_and_read.block_size_);
                        // Due to symmetry we only want to check reads with read_id greater than the current read_id.
                        // We are only interested in part of whole_data_arrays_section_for_representation that comes after array_block_for_this_representation_and_read because only sketch elements in that part have read_id greater than the current read_id
                        ArrayBlock section_to_check;
                        section_to_check.first_element_ = array_block_for_this_representation_and_read.first_element_ +
                                                          array_block_for_this_representation_and_read.block_size_; // element after the last element for this read_id
                        section_to_check.block_size_ = whole_data_arrays_section_for_representation.first_element_ +
                                                       whole_data_arrays_section_for_representation.block_size_ -
                                                       section_to_check.first_element_; // number of remaining elements

                        // TODO: if block_size_ == 0
                        if (section_to_check.block_size_) {
                            read_id_to_sketch_elements_h.emplace_back(array_block_for_this_representation_and_read);
                            read_id_to_sketch_elements_to_check_h.emplace_back(section_to_check);
                            // Determine the number of matches for this representation
                            read_id_to_anchors_section_h[read_id].block_size_ +=
                                    array_block_for_this_representation_and_read.block_size_ *
                                    section_to_check.block_size_;
                            ++read_id_to_pointer_arrays_section_h[read_id].block_size_;
                        }
                    }
                }
                total_anchors += read_id_to_anchors_section_h[read_id].block_size_;

                if (total_anchors > max_anchors) {
                    // If the maximum number of anchors has been exceeded all host buffers are re-initialised
                    // and the loop is restarted with a smaller representation range to compute.
                    read_id_to_sketch_elements_h.clear();
                    read_id_to_sketch_elements_to_check_h.clear();
                    total_anchors = 0;
                    largest_block_size = 0;
                    auto growth_coefficient = 4;
                    increment /= growth_coefficient; //TODO investigate best coefficient
                    representation_max_range = representation_min_range + increment;
                    read_id = 0;
                    read_id_to_anchors_section_h = std::vector<ArrayBlock>(index.number_of_reads(), {0, 0});
                    read_id_to_pointer_arrays_section_h = std::vector<ArrayBlock>(index.number_of_reads(), {0, 0});
                    CGA_LOG_INFO("Backing off - max range adjusted to {}", representation_max_range);
                }
            }

            // Now done with the read IDs

            CGA_LOG_INFO("Allocating {} bytes for read_id_to_sketch_elements_d",
                         read_id_to_sketch_elements_h.size() * sizeof(ArrayBlock));
            device_buffer<ArrayBlock> read_id_to_sketch_elements_d(read_id_to_sketch_elements_h.size());
            CGA_CU_CHECK_ERR(cudaMemcpy(read_id_to_sketch_elements_d.data(), read_id_to_sketch_elements_h.data(),
                                        read_id_to_sketch_elements_h.size() * sizeof(ArrayBlock),
                                        cudaMemcpyHostToDevice));

            read_id_to_sketch_elements_h.clear();
            read_id_to_sketch_elements_h.reserve(0);

            CGA_LOG_INFO("Allocating {} bytes for read_id_to_sketch_elements_to_check_d",
                         read_id_to_sketch_elements_to_check_h.size() * sizeof(ArrayBlock));
            device_buffer<ArrayBlock> read_id_to_sketch_elements_to_check_d(
                    read_id_to_sketch_elements_to_check_h.size());
            CGA_CU_CHECK_ERR(cudaMemcpy(read_id_to_sketch_elements_to_check_d.data(),
                                        read_id_to_sketch_elements_to_check_h.data(),
                                        read_id_to_sketch_elements_to_check_h.size() * sizeof(ArrayBlock),
                                        cudaMemcpyHostToDevice));
            read_id_to_sketch_elements_to_check_h.clear();
            read_id_to_sketch_elements_to_check_h.reserve(0);

            CGA_LOG_INFO("Allocating {} bytes for read_id_to_pointer_arrays_section_d",
                         read_id_to_pointer_arrays_section_h.size() * sizeof(ArrayBlock));
            device_buffer<ArrayBlock> read_id_to_pointer_arrays_section_d(read_id_to_pointer_arrays_section_h.size());
            CGA_CU_CHECK_ERR(
                    cudaMemcpy(read_id_to_pointer_arrays_section_d.data(), read_id_to_pointer_arrays_section_h.data(),
                               read_id_to_pointer_arrays_section_h.size() * sizeof(ArrayBlock),
                               cudaMemcpyHostToDevice));
            read_id_to_pointer_arrays_section_h.clear();
            read_id_to_pointer_arrays_section_h.reserve(0);

            CGA_LOG_INFO("Allocating {} bytes for anchors_d", total_anchors * sizeof(Anchor));
            device_buffer<Anchor> anchors_d(total_anchors);

            CGA_LOG_INFO("Allocating {} bytes for read_id_to_anchors_section_d",
                         read_id_to_anchors_section_h.size() * sizeof(ArrayBlock));
            device_buffer<ArrayBlock> read_id_to_anchors_section_d(read_id_to_anchors_section_h.size());
            CGA_CU_CHECK_ERR(cudaMemcpy(read_id_to_anchors_section_d.data(), read_id_to_anchors_section_h.data(),
                                        read_id_to_anchors_section_h.size() * sizeof(ArrayBlock),
                                        cudaMemcpyHostToDevice));
            read_id_to_anchors_section_h.clear();
            read_id_to_anchors_section_h.reserve(0);

            generate_anchors <<<index.number_of_reads(), 32, largest_block_size * sizeof(position_in_read_t)>>>
                                                               (positions_in_reads_d.data(),
                                                                       read_ids_d.data(),
                                                                       // directions_of_reads_d.data(), // currently we don't use direction
                                                                       read_id_to_sketch_elements_d.data(),
                                                                       read_id_to_sketch_elements_to_check_d.data(),
                                                                       read_id_to_pointer_arrays_section_d.data(),
                                                                       anchors_d.data(),
                                                                       read_id_to_anchors_section_d.data()
                                                               );

            cudaDeviceSynchronize();

            anchors_h_.resize(anchors_h_.size() + total_anchors);
            CGA_CU_CHECK_ERR(cudaMemcpy(anchors_h_.data(), anchors_d.data(), total_anchors * sizeof(Anchor),
                                        cudaMemcpyDeviceToHost));

            // clean up device memory
            CGA_LOG_INFO("Deallocating {} bytes from read_id_to_anchors_section_d",
                         read_id_to_anchors_section_d.size() *
                         sizeof(decltype(read_id_to_anchors_section_d)::value_type));
            read_id_to_anchors_section_d.free();
            CGA_LOG_INFO("Deallocating {} bytes from anchors_d",
                         anchors_d.size() * sizeof(decltype(anchors_d)::value_type));
            anchors_d.free();

            CGA_LOG_INFO("Deallocating {} bytes from read_id_to_sketch_elements_d",
                         read_id_to_sketch_elements_d.size() *
                         sizeof(decltype(read_id_to_sketch_elements_d)::value_type));
            read_id_to_sketch_elements_d.free();
            CGA_LOG_INFO("Deallocating {} bytes from read_id_to_sketch_elements_to_check_d",
                         read_id_to_sketch_elements_to_check_d.size() *
                         sizeof(decltype(read_id_to_sketch_elements_to_check_d)::value_type));
            read_id_to_sketch_elements_to_check_d.free();
            CGA_LOG_INFO("Deallocating {} bytes from read_id_to_pointer_arrays_section_d",
                         read_id_to_pointer_arrays_section_d.size() *
                         sizeof(decltype(read_id_to_pointer_arrays_section_d)::value_type));
            read_id_to_pointer_arrays_section_d.free();

            CGA_LOG_INFO("Deallocating {} bytes from positions_in_reads_d",
                         positions_in_reads_d.size() * sizeof(decltype(positions_in_reads_d)::value_type));
            positions_in_reads_d.free();
            CGA_LOG_INFO("Deallocating {} bytes from read_ids_d",
                         read_ids_d.size() * sizeof(decltype(read_ids_d)::value_type));
            read_ids_d.free();
            CGA_LOG_INFO("Deallocating {} bytes from directions_of_reads_d",
                         directions_of_reads_d.size() * sizeof(decltype(directions_of_reads_d)::value_type));
            directions_of_reads_d.free();

            representation_min_range += increment;
            increment *= 2; // TODO: investigate best coefficient
            representation_max_range += increment;

        }
    }

    const std::vector<Anchor> &Matcher::anchors() const {
        return anchors_h_;
    }

}
