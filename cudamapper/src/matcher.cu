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
#include <cudautils/cudautils.hpp>

namespace claragenomics {

    // TODO: move to cudautils?
    /// \brief creates a unique pointer to device memory which gets deallocated automatically
    ///
    /// \param num_of_elems number of elements to allocate
    ///
    /// \return unique pointer to allocated memory
    template<typename T>
    std::unique_ptr<T, void(*)(T*)> make_unique_cuda_malloc(std::size_t num_of_elems) {
        T* tmp_ptr_d = nullptr;
        CGA_CU_CHECK_ERR(cudaMalloc((void**)&tmp_ptr_d, num_of_elems*sizeof(T)));
        std::unique_ptr<T, void(*)(T*)> u_ptr_d(tmp_ptr_d, [](T* p) {CGA_CU_CHECK_ERR(cudaFree(p));}); // tmp_prt_d's ownership transfered to u_ptr_d
        return std::move(u_ptr_d);
    }

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
    /// q0p0t1p0, q0p0t1p1 .. q0p0t1p39, q0p0t2p0, q0p0t2p1 ... q0p0t2p49, q0p1t1p0, q0p1t1p1 ... q0p1t1p39, q0p1t2p0 .. q0p1t2p49, q0p2p1p0 ...
    ///
    /// \param positions_in_reads_d positions of sketch elements in their reads first sorted by representation and then by read id
    /// \param read_ids_d read ids of reads sketch elements belong to first sorted by representation and then by read id (elements with the same index from positions_in_reads_d belong to the same sketch element)
    /// \param read_id_to_sketch_elements_d every element points to a section of positions_in_reads_d and read_ids_d that belong to sketch elements with the same read_id and representation
    /// \param read_id_to_sketch_elements_to_check_d every element points to a section of positions_in_reads_d and read_ids_d that belong to sketch elements with the same representation and read_id larger than some value
    /// \param read_id_to_pointer_arrays_section_d every element belongs to one read_id and points to its sections of read_id_to_sketch_elements_d and read_id_to_sketch_elements_to_check_d
    /// \param anchors_d pairs of sketch elements with the same representation that belong to different reads
    /// \param read_id_to_anchors_section_d points to parts of anchors_d in which all anchors have the same read_id
    __global__ void generate_anchors(const position_in_read_t* const positions_in_reads_d,
                                     const read_id_t* const read_ids_d,
                                     // const SketchElement::DirectionOfRepresentation* const directions_of_reads_d, // currently we don't use direction
                                     ArrayBlock* read_id_to_sketch_elements_d,
                                     ArrayBlock* read_id_to_sketch_elements_to_check_d,
                                     ArrayBlock* read_id_to_pointer_arrays_section_d,
                                     Matcher::Anchor* const anchors_d,
                                     ArrayBlock* read_id_to_anchors_section_d) {

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
            for (auto i = threadIdx.x; i < target_sketch_elements_section.block_size_; ++i) {
                const read_id_t target_read_id = read_ids_d[target_sketch_elements_section.first_element_ + i];
                const position_in_read_t target_position_in_read = positions_in_reads_d[target_sketch_elements_section.first_element_ + i];
                for (int j = 0; j < query_sketch_elements_section.block_size_; ++j) {
                    // writing anchors in form (q1t1,q1t2,q1t3...q2t1,q2t2,q3t3....) for coalescing
                    // TODO: split anchors_d into four arrays for better coalescing?
                    anchors_d[read_id_to_anchors_section_d[query_read_id].first_element_ + anchors_written_so_far + j*target_sketch_elements_section.block_size_ + i].query_read_id_ = query_read_id;
                    anchors_d[read_id_to_anchors_section_d[query_read_id].first_element_ + anchors_written_so_far + j*target_sketch_elements_section.block_size_ + i].target_read_id_ = target_read_id;
                    anchors_d[read_id_to_anchors_section_d[query_read_id].first_element_ + anchors_written_so_far + j*target_sketch_elements_section.block_size_ + i].query_position_in_read_ = query_positions[j];
                    anchors_d[read_id_to_anchors_section_d[query_read_id].first_element_ + anchors_written_so_far + j*target_sketch_elements_section.block_size_ + i].target_position_in_read_ = target_position_in_read;
                }
            }
            __syncthreads();
            if (0 == threadIdx.x) anchors_written_so_far += target_sketch_elements_section.block_size_ * query_sketch_elements_section.block_size_;
            __syncthreads();
        }
    }

    Matcher::Matcher(const Index& index) {

        if (0 == index.number_of_reads()) {
            return;
        }

        const std::vector<position_in_read_t>& positions_in_reads_h = index.positions_in_reads();
        auto positions_in_reads_d = make_unique_cuda_malloc<position_in_read_t>(positions_in_reads_h.size());
        CGA_CU_CHECK_ERR(cudaMemcpy(positions_in_reads_d.get(), positions_in_reads_h.data(), positions_in_reads_h.size()*sizeof(position_in_read_t), cudaMemcpyHostToDevice));
        
        const std::vector<read_id_t>& read_ids_h = index.read_ids();
        auto read_ids_d = make_unique_cuda_malloc<read_id_t>(read_ids_h.size());
        CGA_CU_CHECK_ERR(cudaMemcpy(read_ids_d.get(), read_ids_h.data(), read_ids_h.size()*sizeof(read_id_t), cudaMemcpyHostToDevice));

        const std::vector<SketchElement::DirectionOfRepresentation>& directions_of_reads_h = index.directions_of_reads();
        auto directions_of_reads_d = make_unique_cuda_malloc<SketchElement::DirectionOfRepresentation>(directions_of_reads_h.size());
        CGA_CU_CHECK_ERR(cudaMemcpy(directions_of_reads_d.get(), directions_of_reads_h.data(), directions_of_reads_h.size()*sizeof(SketchElement::DirectionOfRepresentation), cudaMemcpyHostToDevice));

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
        std::vector<ArrayBlock> read_id_to_pointer_arrays_section_h(index.number_of_reads(), {0,0});

        // Anchor is one pair of sketch elements with the same representation in different reads
        // Anchors are symmetric (as explained above), so they are saved in only one direction
        // As only one direction is saved for each representation in each read_id there are going to be sketch_elements_with_that_representation_in_that_read * sketch_elements_with_that_representation_to_check_in_other_reads anchors.
        // This means we know upfront how many anchors are there going to be for each read_id and we can merge all anchors in one array and assign its sections to different read_ids
        std::vector<ArrayBlock> read_id_to_anchors_section_h(index.number_of_reads(), {0,0});
        std::uint64_t total_anchors = 0;

        std::uint32_t largest_block_size = 0;

        for (std::size_t read_id = 0; read_id < index.number_of_reads(); ++read_id) {
            // First determine the starting index of section of pointer arrays that belong to read with read_id.
            // Reads are processed consecutively. Pointer arrays section for read 0 will start at index 0 and if we assume that all sketch elements in read 0 had a total of 10 unique representation.
            // its section will end at index 9. This means that the section for read 1 belongs at index 0 + 10 = 10.
            if (read_id != 0) {
                read_id_to_pointer_arrays_section_h[read_id].first_element_ = read_id_to_pointer_arrays_section_h[read_id - 1].first_element_ + read_id_to_pointer_arrays_section_h[read_id - 1].block_size_;
                read_id_to_anchors_section_h[read_id].first_element_ = read_id_to_anchors_section_h[read_id - 1].first_element_ + read_id_to_anchors_section_h[read_id - 1].block_size_;
            }

            const std::map<representation_t, ArrayBlock>& representation_to_all_sketch_elements_with_this_read_id = index.read_id_and_representation_to_all_its_sketch_elements()[read_id];

            read_id_to_pointer_arrays_section_h[read_id].block_size_ = 0;

            // go through all representations in this read
            for (const auto& one_representation_in_this_read : representation_to_all_sketch_elements_with_this_read_id) { // this is a map so we know that representations are sorted in increasing order
                const representation_t& representation = one_representation_in_this_read.first;
                const ArrayBlock& data_arrays_section_for_this_representation_and_read = one_representation_in_this_read.second;
                largest_block_size = std::max(largest_block_size, data_arrays_section_for_this_representation_and_read.block_size_);
                const ArrayBlock& whole_data_arrays_section_for_representation = (*index.representation_to_all_its_sketch_elements().find(representation)).second; // sketch elements with this representation in all reads
                // Due to symmetry we only want to check reads with read_id greater than the current read_id.
                // We are only interested in part of whole_data_array_section_for_representation that comes after data_array_section_for_this_representation_and_read because only sketch
                // elements in that part have read_id greater than current read_id
                ArrayBlock section_to_check;
                section_to_check.first_element_ = data_arrays_section_for_this_representation_and_read.first_element_ + data_arrays_section_for_this_representation_and_read.block_size_; // element after the last element for this read_id
                section_to_check.block_size_ = whole_data_arrays_section_for_representation.first_element_ + whole_data_arrays_section_for_representation.block_size_ - section_to_check.first_element_; // number of remaining elements

                // TODO: if block_size_ == 0
                if (section_to_check.block_size_) {
                    read_id_to_sketch_elements_h.emplace_back(data_arrays_section_for_this_representation_and_read);
                    read_id_to_sketch_elements_to_check_h.emplace_back(section_to_check);
                    // Determine the number of matches for this representation
                    read_id_to_anchors_section_h[read_id].block_size_ += data_arrays_section_for_this_representation_and_read.block_size_ * section_to_check.block_size_;
                    ++read_id_to_pointer_arrays_section_h[read_id].block_size_;
                }
            }
            total_anchors += read_id_to_anchors_section_h[read_id].block_size_;
        }

        auto read_id_to_sketch_elements_d = make_unique_cuda_malloc<ArrayBlock>(read_id_to_sketch_elements_h.size());
        CGA_CU_CHECK_ERR(cudaMemcpy(read_id_to_sketch_elements_d.get(), read_id_to_sketch_elements_h.data(), read_id_to_sketch_elements_h.size()*sizeof(ArrayBlock), cudaMemcpyHostToDevice));
        read_id_to_sketch_elements_h.clear();
        read_id_to_sketch_elements_h.reserve(0);


        auto read_id_to_sketch_elements_to_check_d = make_unique_cuda_malloc<ArrayBlock>(read_id_to_sketch_elements_to_check_h.size());
        CGA_CU_CHECK_ERR(cudaMemcpy(read_id_to_sketch_elements_to_check_d.get(), read_id_to_sketch_elements_to_check_h.data(), read_id_to_sketch_elements_to_check_h.size()*sizeof(ArrayBlock), cudaMemcpyHostToDevice));
        read_id_to_sketch_elements_to_check_h.clear();
        read_id_to_sketch_elements_to_check_h.reserve(0);

        auto read_id_to_pointer_arrays_section_d = make_unique_cuda_malloc<ArrayBlock>(read_id_to_pointer_arrays_section_h.size());
        CGA_CU_CHECK_ERR(cudaMemcpy(read_id_to_pointer_arrays_section_d.get(), read_id_to_pointer_arrays_section_h.data(), read_id_to_pointer_arrays_section_h.size()*sizeof(ArrayBlock), cudaMemcpyHostToDevice));
        read_id_to_pointer_arrays_section_h.clear();
        read_id_to_pointer_arrays_section_h.reserve(0);

        auto anchors_d = make_unique_cuda_malloc<Anchor>(total_anchors);

        auto read_id_to_anchors_section_d = make_unique_cuda_malloc<ArrayBlock>(read_id_to_anchors_section_h.size());
        CGA_CU_CHECK_ERR(cudaMemcpy(read_id_to_anchors_section_d.get(), read_id_to_anchors_section_h.data(), read_id_to_anchors_section_h.size()*sizeof(ArrayBlock), cudaMemcpyHostToDevice));
        // TODO: do we need read_id_to_anchors_section_h after this point?

        generate_anchors<<<index.number_of_reads(),32,largest_block_size*sizeof(position_in_read_t)>>>(positions_in_reads_d.get(),
                                                                                                     read_ids_d.get(),
                                                                                                     // directions_of_reads_d.get(), // currently we don't use direction
                                                                                                     read_id_to_sketch_elements_d.get(),
                                                                                                     read_id_to_sketch_elements_to_check_d.get(),
                                                                                                     read_id_to_pointer_arrays_section_d.get(),
                                                                                                     anchors_d.get(),
                                                                                                     read_id_to_anchors_section_d.get());

        std::vector<Anchor> anchors_h_temp(total_anchors);
        CGA_CU_CHECK_ERR(cudaMemcpy(anchors_h_temp.data(), anchors_d.get(), total_anchors*sizeof(Anchor), cudaMemcpyDeviceToHost));
        std::swap(anchors_h_temp, anchors_h_);

        // clean up device memory
        read_id_to_anchors_section_d.reset(nullptr);
        anchors_d.reset(nullptr);

        read_id_to_sketch_elements_d.reset(nullptr);
        read_id_to_sketch_elements_to_check_d.reset(nullptr);
        read_id_to_pointer_arrays_section_d.reset(nullptr);

        positions_in_reads_d.reset(nullptr);
        read_ids_d.reset(nullptr);
        directions_of_reads_d.reset(nullptr);
    }

    const std::vector<Matcher::Anchor>& Matcher::anchors() const {
        return anchors_h_;
    }

}
