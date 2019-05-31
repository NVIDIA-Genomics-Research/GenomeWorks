/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <unordered_set>
#include <utility>
#include <vector>
#include "cudautils/cudautils.hpp"
#include "matcher.hpp"

// TODO: remove after dummy printf has been removed from the kernel
#include <stdio.h>

namespace genomeworks {
    // TODO: move to cudautils?
    /// \brief allocates pinned host memory
    ///
    /// \param num_of_elems number of elements to allocate
    ///
    /// \return unique pointer to allocated memory
    template<typename T>
    std::unique_ptr<T, void(*)(T*)> make_unique_cuda_malloc_host(std::size_t num_of_elems) {
        T* tmp_ptr_h = nullptr;
        GW_CU_CHECK_ERR(cudaMallocHost((void**)&tmp_ptr_h, num_of_elems*sizeof(T)));
        std::unique_ptr<T, void(*)(T*)> uptr_h(tmp_ptr_h, [](T* p) {GW_CU_CHECK_ERR(cudaFreeHost(p));});
        return std::move(uptr_h);
    }

    // TODO: move to cudautils?
    /// \brief allocates device memory
    ///
    /// \param num_of_elems number of elements to allocate
    ///
    /// \return unique pointer to allocated memory
    template<typename T>
    std::unique_ptr<T, void(*)(T*)> make_unique_cuda_malloc(std::size_t num_of_elems) {
        T* tmp_ptr_h = nullptr;
        GW_CU_CHECK_ERR(cudaMalloc((void**)&tmp_ptr_h, num_of_elems*sizeof(T)));
        std::unique_ptr<T, void(*)(T*)> uptr_h(tmp_ptr_h, [](T* p) {GW_CU_CHECK_ERR(cudaFree(p));});
        return std::move(uptr_h);
    }

    __global__ void access_data(const IndexGPU::MappingToDeviceArrays* sequence_mappings_d, const std::size_t* sequences_block_start_d, const std::size_t* sequences_block_past_the_end_d, const std::size_t* positions_d, const std::size_t* sequence_ids_d, const SketchElement::DirectionOfRepresentation* directions_d) {
        const int sequence_num = blockIdx.x;
        const std::size_t sequence_block_start = sequences_block_start_d[sequence_num];
        const std::size_t sequence_block_end = sequences_block_past_the_end_d[sequence_num];
        for (std::size_t position_group = sequence_block_start; position_group < sequence_block_end; ++position_group) {
            const IndexGPU::MappingToDeviceArrays& sequence_mapping = sequence_mappings_d[position_group];
            for (std::size_t position_position = threadIdx.x; position_position < sequence_mapping.block_size_; position_position += blockDim.x) {
                std::size_t elem_to_get = sequence_mapping.location_first_in_block_ + position_position;
                // dummy checks and printfs to make sure values are actually loaded into memory
                if(positions_d[elem_to_get] > 1000000000) {
                    printf("foo\n");
                }
                if (sequence_ids_d[elem_to_get] > 1000000000) {
                    printf("bar\n");
                }
                if(directions_d[elem_to_get] != SketchElement::DirectionOfRepresentation::FORWARD && directions_d[elem_to_get] != SketchElement::DirectionOfRepresentation::REVERSE) {
                    printf("buz\n");
                }
            }
            __syncthreads();
        }
    }

    Matcher::Matcher(const IndexGPU& index) {
        // IndexGPU::MappingToDeviceArrays points to a section of arrays returned by index.positions_d(), index.sequence_ids_d(), index.directions_d() that
        // correspond to a representation
        // Each section of sequences_mappings_h corresponds to one seqence. By iterating over such section one gets mappings to all occurrences (in all seqeunce)
        // of all representations within the given sequence

        const std::unordered_set<std::uint64_t>& sequence_ids_set = index.sequence_ids();

        // each block of this vector (as defined by sequences_block_start and sequence_block_end) contains mappings for one sequence
        std::vector<IndexGPU::MappingToDeviceArrays> sequences_mappings_h;

        // index of the first element of the section of sequences_mappings_h belonging to the given sequence
        auto sequences_block_start_h = make_unique_cuda_malloc_host<std::size_t>(sequence_ids_set.size());
        // index of past-the-end element of the section of sequences_mappings_h belonging to the given sequence
        auto sequences_block_past_the_end_h = make_unique_cuda_malloc_host<std::size_t>(sequence_ids_set.size());

        // maps sequence_id to all representations in that sequence
        const auto& sequence_id_to_representations = index.sequence_id_to_representations();
        // maps representation to its IndexGPU::MappingToDeviceArrays
        const auto& representation_to_device_arrays = index.representation_to_device_arrays();

        std::size_t sequence_num = 0;
        for (const std::uint64_t& sequence_id : sequence_ids_set) {
            sequences_block_start_h.get()[sequence_num] = sequences_mappings_h.size();
            const auto& representations_range = sequence_id_to_representations.equal_range(sequence_id);
            std::unordered_set<std::uint64_t> representations_seen;
            for (auto representation_it = representations_range.first; representation_it != representations_range.second; ++representation_it) {
                if (representations_seen.find((*representation_it).second) == std::end(representations_seen)) { // representation not seen already
                    const IndexGPU::MappingToDeviceArrays& mapping = (*representation_to_device_arrays.find((*representation_it).second)).second;
                    sequences_mappings_h.push_back(mapping);
                }
            }
            sequences_block_past_the_end_h.get()[sequence_num] = sequences_mappings_h.size();
            ++sequence_num;
        }


        auto sequences_mappings_d = make_unique_cuda_malloc<IndexGPU::MappingToDeviceArrays>(sequences_mappings_h.size());
        GW_CU_CHECK_ERR(cudaMemcpy(sequences_mappings_d.get(), &sequences_mappings_h[0], sequences_mappings_h.size()*sizeof(IndexGPU::MappingToDeviceArrays), cudaMemcpyHostToDevice));
        // clear sequences_mappings_h
        sequences_mappings_h.clear();
        sequences_mappings_h.reserve(0);

        auto sequences_block_start_d = make_unique_cuda_malloc<std::size_t>(sequence_ids_set.size());
        GW_CU_CHECK_ERR(cudaMemcpy(sequences_block_start_d.get(), sequences_block_start_h.get(), sequence_ids_set.size()*sizeof(std::size_t), cudaMemcpyHostToDevice));
        // clear sequences_block_start_h
        sequences_block_start_h.reset(nullptr);

        auto sequences_block_past_the_end_d = make_unique_cuda_malloc<std::size_t>(sequence_ids_set.size());
        GW_CU_CHECK_ERR(cudaMemcpy(sequences_block_past_the_end_d.get(), sequences_block_past_the_end_h.get(), sequence_ids_set.size()*sizeof(std::size_t), cudaMemcpyHostToDevice));
        // clear sequences_block_past_the_end_h
        sequences_block_past_the_end_h.reset(nullptr);

        access_data<<<sequence_ids_set.size(),128>>>(sequences_mappings_d.get(), sequences_block_start_d.get(), sequences_block_past_the_end_d.get(), index.positions_d().get(), index.sequence_ids_d().get(), index.directions_d().get());
        GW_CU_CHECK_ERR(cudaPeekAtLastError());
        GW_CU_CHECK_ERR(cudaDeviceSynchronize());
    }
}
