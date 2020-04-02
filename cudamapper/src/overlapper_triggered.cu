/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <cub/cub.cuh>

#include <fstream>
#include <cstdlib>
#include <omp.h>

#include <claragenomics/utils/cudautils.hpp>

#include "cudamapper_utils.hpp"
#include "overlapper_triggered.hpp"

#ifndef NDEBUG // only needed to check if input is sorted in assert
#include <algorithm>
#include <thrust/host_vector.h>
#endif

namespace claragenomics
{
namespace cudamapper
{

__host__ __device__ bool operator==(const Anchor& lhs,
                                    const Anchor& rhs)
{
    auto score_threshold = 1;

    // Very simple scoring function to quantify quality of overlaps.
    auto score = 1;

    if ((rhs.query_position_in_read_ - lhs.query_position_in_read_) < 150 and abs(int(rhs.target_position_in_read_) - int(lhs.target_position_in_read_)) < 150)
        score = 2;
    return ((lhs.query_read_id_ == rhs.query_read_id_) &&
            (lhs.target_read_id_ == rhs.target_read_id_) &&
            score > score_threshold);
}

struct cuOverlapKey
{
    const Anchor* anchor;
};

struct cuOverlapKey_transform
{
    const Anchor* d_anchors;
    const int32_t* d_chain_start;

    cuOverlapKey_transform(const Anchor* anchors, const int32_t* chain_start)
        : d_anchors(anchors)
        , d_chain_start(chain_start)
    {
    }

    __host__ __device__ __forceinline__ cuOverlapKey
    operator()(const int32_t& idx) const
    {
        auto anchor_idx = d_chain_start[idx];

        cuOverlapKey key;
        key.anchor = &d_anchors[anchor_idx];
        return key;
    }
};

__host__ __device__ bool operator==(const cuOverlapKey& key0,
                                    const cuOverlapKey& key1)
{
    const Anchor* a = key0.anchor;
    const Anchor* b = key1.anchor;

    int distance_difference = abs(abs(int(a->query_position_in_read_) - int(b->query_position_in_read_)) -
                                  abs(int(a->target_position_in_read_) - int(b->target_position_in_read_)));

    bool equal = (a->target_read_id_ == b->target_read_id_) &&
                 (a->query_read_id_ == b->query_read_id_) &&
                 distance_difference < 300;

    return equal;
}

struct cuOverlapArgs
{
    int32_t overlap_end;
    int32_t num_residues;
    int32_t overlap_start;
};

struct cuOverlapArgs_transform
{
    const int32_t* d_chain_start;
    const int32_t* d_chain_length;

    cuOverlapArgs_transform(const int32_t* chain_start, const int32_t* chain_length)
        : d_chain_start(chain_start)
        , d_chain_length(chain_length)
    {
    }

    __host__ __device__ __forceinline__ cuOverlapArgs
    operator()(const int32_t& idx) const
    {
        cuOverlapArgs overlap;
        auto overlap_start    = d_chain_start[idx];
        auto overlap_length   = d_chain_length[idx];
        overlap.overlap_end   = overlap_start + overlap_length;
        overlap.num_residues  = overlap_length;
        overlap.overlap_start = overlap_start;
        return overlap;
    }
};

struct FuseOverlapOp
{
    __host__ __device__ cuOverlapArgs operator()(const cuOverlapArgs& a,
                                                 const cuOverlapArgs& b) const
    {
        cuOverlapArgs fused_overlap;
        fused_overlap.num_residues = a.num_residues + b.num_residues;
        fused_overlap.overlap_end =
            a.overlap_end > b.overlap_end ? a.overlap_end : b.overlap_end;
        fused_overlap.overlap_start =
            a.overlap_start < b.overlap_start ? a.overlap_start : b.overlap_start;
        return fused_overlap;
    }
};

struct FilterOverlapOp
{
    size_t min_residues;
    size_t min_overlap_len;
    size_t min_bases_per_residue;
    float min_overlap_fraction;

    __host__ __device__ __forceinline__ FilterOverlapOp(size_t min_residues,
                                                        size_t min_overlap_len,
                                                        size_t min_bases_per_residue,
                                                        float min_overlap_fraction)
        : min_residues(min_residues)
        , min_overlap_len(min_overlap_len)
        , min_bases_per_residue(min_bases_per_residue)
        , min_overlap_fraction(min_overlap_fraction)
    {
    }

    __host__ __device__ __forceinline__ bool operator()(const Overlap& overlap) const
    {

        const auto target_overlap_length = overlap.target_end_position_in_read_ - overlap.target_start_position_in_read_;
        const auto query_overlap_length  = overlap.query_end_position_in_read_ - overlap.query_start_position_in_read_;
        const auto overlap_length        = max(target_overlap_length, query_overlap_length);

        return ((overlap.num_residues_ >= min_residues) &&
                ((overlap_length / overlap.num_residues_) < min_bases_per_residue) &&
                (query_overlap_length > min_overlap_len) &&
                (overlap.query_read_id_ != overlap.target_read_id_) &&
                ((static_cast<float>(target_overlap_length) / static_cast<float>(overlap_length)) > min_overlap_fraction) &&
                ((static_cast<float>(query_overlap_length) / static_cast<float>(overlap_length)) > min_overlap_fraction));
    }
};

struct CreateOverlap
{
    const Anchor* d_anchors;

    __host__ __device__ __forceinline__ CreateOverlap(const Anchor* anchors_ptr)
        : d_anchors(anchors_ptr)
    {
    }

    __host__ __device__ __forceinline__ Overlap
    operator()(cuOverlapArgs overlap)
    {
        Anchor overlap_start_anchor = d_anchors[overlap.overlap_start];
        Anchor overlap_end_anchor   = d_anchors[overlap.overlap_end - 1];

        Overlap new_overlap;

        new_overlap.query_read_id_  = overlap_end_anchor.query_read_id_;
        new_overlap.target_read_id_ = overlap_end_anchor.target_read_id_;
        new_overlap.num_residues_   = overlap.num_residues;
        new_overlap.target_end_position_in_read_ =
            overlap_end_anchor.target_position_in_read_;
        new_overlap.target_start_position_in_read_ =
            overlap_start_anchor.target_position_in_read_;
        new_overlap.query_end_position_in_read_ =
            overlap_end_anchor.query_position_in_read_;
        new_overlap.query_start_position_in_read_ =
            overlap_start_anchor.query_position_in_read_;
        new_overlap.overlap_complete = true;

        // If the target start position is greater than the target end position
        // We can safely assume that the query and target are template and
        // complement reads. TODO: Incorporate sketchelement direction value when
        // this is implemented
        if (new_overlap.target_start_position_in_read_ >
            new_overlap.target_end_position_in_read_)
        {
            new_overlap.relative_strand = RelativeStrand::Reverse;
            auto tmp                    = new_overlap.target_end_position_in_read_;
            new_overlap.target_end_position_in_read_ =
                new_overlap.target_start_position_in_read_;
            new_overlap.target_start_position_in_read_ = tmp;
        }
        else
        {
            new_overlap.relative_strand = RelativeStrand::Forward;
        }
        return new_overlap;
    };
};

OverlapperTriggered::OverlapperTriggered(DefaultDeviceAllocator allocator,
                                         const cudaStream_t cuda_stream)
    : _allocator(allocator)
    , _cuda_stream(cuda_stream)
{
}

void OverlapperTriggered::get_overlaps(std::vector<Overlap>& fused_overlaps,
                                       const device_buffer<Anchor>& d_anchors,
                                       int64_t min_residues,
                                       int64_t min_overlap_len,
                                       int64_t min_bases_per_residue,
                                       float min_overlap_fraction)
{
    CGA_NVTX_RANGE(profiler, "OverlapperTriggered::get_overlaps");
    const auto tail_length_for_chain = 3;
    auto n_anchors                   = d_anchors.size();

#ifndef NDEBUG
    // check if anchors are sorted properly

    // TODO: Copying data to host and doing the check there as using thrust::is_sorted
    //       leads to a compilaiton error. It is probably a bug in device_buffer implementation

    thrust::host_vector<Anchor> h_anchors(d_anchors.size());
    cudautils::device_copy_n(d_anchors.data(), d_anchors.size(), h_anchors.data()); // D2H

    auto comp_anchors = [](const Anchor& i, const Anchor& j) { return (i.query_read_id_ < j.query_read_id_) ||
                                                                      ((i.query_read_id_ == j.query_read_id_) &&
                                                                       (i.target_read_id_ < j.target_read_id_)) ||
                                                                      ((i.query_read_id_ == j.query_read_id_) &&
                                                                       (i.target_read_id_ == j.target_read_id_) &&
                                                                       (i.query_position_in_read_ < j.query_position_in_read_)) ||
                                                                      ((i.query_read_id_ == j.query_read_id_) &&
                                                                       (i.target_read_id_ == j.target_read_id_) &&
                                                                       (i.query_position_in_read_ == j.query_position_in_read_) &&
                                                                       (i.target_position_in_read_ < j.target_position_in_read_)); };

    assert(std::is_sorted(std::begin(h_anchors),
                          std::end(h_anchors),
                          comp_anchors));
#endif

    // temporary workspace buffer on device
    device_buffer<char> d_temp_buf(_allocator, _cuda_stream);

    // Do run length encode to compute the chains
    // note - identifies the start and end anchor of the chain without moving the anchors
    // >>>>>>>>>

    // d_start_anchor[i] contains the starting anchor of chain i
    device_buffer<Anchor> d_start_anchor(n_anchors, _allocator, _cuda_stream);

    // d_chain_length[i] contains the length of chain i
    device_buffer<int32_t> d_chain_length(n_anchors, _allocator, _cuda_stream);

    // total number of chains found
    device_buffer<int32_t> d_nchains(1, _allocator, _cuda_stream);

    //The equality of two anchors has been overriden, such that they are equal (members of the same chain) if their QID,TID are equal and they fall within a fixed distance of one another
    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;
    // calculate storage requirement for run length encoding
    cub::DeviceRunLengthEncode::Encode(
        d_temp_storage, temp_storage_bytes, d_anchors.data(), d_start_anchor.data(),
        d_chain_length.data(), d_nchains.data(), n_anchors, _cuda_stream);

    // allocate temporary storage
    d_temp_buf.resize(temp_storage_bytes, _cuda_stream);
    d_temp_storage = d_temp_buf.data();

    // run encoding
    cub::DeviceRunLengthEncode::Encode(
        d_temp_storage, temp_storage_bytes, d_anchors.data(), d_start_anchor.data(),
        d_chain_length.data(), d_nchains.data(), n_anchors, _cuda_stream);

    // <<<<<<<<<<

    // memcpy D2H
    auto n_chains = cudautils::get_value_from_device(d_nchains.data(), _cuda_stream); //We now know the number of chains we are working with.

    // use prefix sum to calculate the starting index position of all the chains
    // >>>>>>>>>>>>
    // for a chain i, d_chain_start[i] contains the index of starting anchor from d_anchors array
    device_buffer<int32_t> d_chain_start(n_chains, _allocator, _cuda_stream);

    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_chain_length.data(), d_chain_start.data(),
                                  n_chains, _cuda_stream);

    // allocate temporary storage
    d_temp_buf.resize(temp_storage_bytes, _cuda_stream);
    d_temp_storage = d_temp_buf.data();

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_chain_length.data(), d_chain_start.data(),
                                  n_chains, _cuda_stream);

    // <<<<<<<<<<<<

    // calculate overlaps where overlap is a chain with length > tail_length_for_chain
    // >>>>>>>>>>>>

    auto thrust_exec_policy = thrust::cuda::par(_allocator).on(_cuda_stream);

    // d_overlaps[j] contains index to d_chain_length/d_chain_start where
    // d_chain_length[d_overlaps[j]] and d_chain_start[d_overlaps[j]] corresponds
    // to length and index to starting anchor of the chain-d_overlaps[j] (also referred as overlap j)
    device_buffer<int32_t> d_overlaps(n_chains, _allocator, _cuda_stream);
    auto indices_end =
        thrust::copy_if(thrust_exec_policy, thrust::make_counting_iterator<int32_t>(0),
                        thrust::make_counting_iterator<int32_t>(n_chains),
                        d_chain_length.data(), d_overlaps.data(),
                        [=] __host__ __device__(const int32_t& len) -> bool {
                            return (len >= tail_length_for_chain);
                        });

    auto n_overlaps = indices_end - d_overlaps.data();

    // <<<<<<<<<<<<<

    // >>>>>>>>>>>>
    // fuse overlaps using reduce by key operations

    // key is a minimal data structure that is required to compare the overlaps
    cuOverlapKey_transform key_op(d_anchors.data(),
                                  d_chain_start.data());
    cub::TransformInputIterator<cuOverlapKey, cuOverlapKey_transform, int32_t*>
        d_keys_in(d_overlaps.data(),
                  key_op);

    // value is a minimal data structure that represents a overlap
    cuOverlapArgs_transform value_op(d_chain_start.data(),
                                     d_chain_length.data());

    cub::TransformInputIterator<cuOverlapArgs, cuOverlapArgs_transform, int32_t*>
        d_values_in(d_overlaps.data(),
                    value_op);

    device_buffer<cuOverlapKey> d_fusedoverlap_keys(n_overlaps, _allocator, _cuda_stream);
    device_buffer<cuOverlapArgs> d_fusedoverlaps_args(n_overlaps, _allocator, _cuda_stream);
    device_buffer<int32_t> d_nfused_overlaps(1, _allocator, _cuda_stream);

    FuseOverlapOp reduction_op;

    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceReduce::ReduceByKey(d_temp_storage,
                                   temp_storage_bytes,
                                   d_keys_in,
                                   d_fusedoverlap_keys.data(), d_values_in,
                                   d_fusedoverlaps_args.data(), d_nfused_overlaps.data(),
                                   reduction_op,
                                   n_overlaps,
                                   _cuda_stream);

    // allocate temporary storage
    d_temp_buf.resize(temp_storage_bytes, _cuda_stream);
    d_temp_storage = d_temp_buf.data();

    cub::DeviceReduce::ReduceByKey(d_temp_storage,
                                   temp_storage_bytes,
                                   d_keys_in,
                                   d_fusedoverlap_keys.data(), //Write out the unique keys here
                                   d_values_in,
                                   d_fusedoverlaps_args.data(), //Write out the values here
                                   d_nfused_overlaps.data(),
                                   reduction_op,
                                   n_overlaps,
                                   _cuda_stream);

    // memcpyD2H
    auto n_fused_overlap = cudautils::get_value_from_device(d_nfused_overlaps.data(), _cuda_stream);

    // construct overlap from the overlap args
    CreateOverlap fuse_op(d_anchors.data());
    device_buffer<Overlap> d_fused_overlaps(n_fused_overlap, _allocator, _cuda_stream); //Overlaps written here

    thrust::transform(thrust_exec_policy, d_fusedoverlaps_args.data(),
                      d_fusedoverlaps_args.data() + n_fused_overlap,
                      d_fused_overlaps.data(), fuse_op);

    device_buffer<Overlap> d_filtered_overlaps(n_fused_overlap, _allocator, _cuda_stream);

    FilterOverlapOp filterOp(min_residues, min_overlap_len, min_bases_per_residue, min_overlap_fraction);
    auto filtered_overlaps_end =
        thrust::copy_if(thrust_exec_policy,
                        d_fused_overlaps.data(), d_fused_overlaps.data() + n_fused_overlap,
                        d_filtered_overlaps.data(),
                        filterOp);

    auto n_filtered_overlaps = filtered_overlaps_end - d_filtered_overlaps.data();

    // memcpyD2H - move fused and filtered overlaps to host
    fused_overlaps.resize(n_filtered_overlaps);
    cudautils::device_copy_n(d_filtered_overlaps.data(), n_filtered_overlaps, fused_overlaps.data(), _cuda_stream);

    // This is not completely necessary, but if removed one has to make sure that the next step
    // uses the same stream or that sync is done in caller
    CGA_CU_CHECK_ERR(cudaStreamSynchronize(_cuda_stream));
}

} // namespace cudamapper
} // namespace claragenomics
