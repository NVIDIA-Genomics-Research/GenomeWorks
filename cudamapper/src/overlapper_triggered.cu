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
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "claragenomics/cudamapper/overlapper.hpp"
#include "cudamapper_utils.hpp"
#include "matcher.hpp"
#include "overlapper_triggered.hpp"
#include <claragenomics/utils/cudautils.hpp>
#include <fstream>
#include <omp.h>

namespace claragenomics {
namespace cudamapper {

__host__ __device__ bool operator==(const Anchor &prev_anchor,
                                    const Anchor &current_anchor) {
  uint16_t score_threshold = 1;
  // Very simple scoring function to quantify quality of overlaps.
  auto anchor_score = [] __host__ __device__(Anchor a, Anchor b) {
    if ((b.query_position_in_read_ - a.query_position_in_read_) < 350) {
      return 2;
    } else {
      return 1; // TODO change to a more sophisticated scoring method
    }
  };
  auto score = anchor_score(prev_anchor, current_anchor);
  return ((current_anchor.query_read_id_ == prev_anchor.query_read_id_) &&
	  (current_anchor.target_read_id_ == prev_anchor.target_read_id_) &&
	  score > score_threshold);
}
    
struct cuOverlapKey
{
    Anchor* anchor;
};

struct cuOverlapKey_transform
{
    Anchor* d_anchors;
    int32_t* d_chain_start;

    cuOverlapKey_transform(Anchor* anchors, int32_t* chain_start)
        : d_anchors(anchors)
        , d_chain_start(chain_start)
    {
    }

    __host__ __device__ __forceinline__ cuOverlapKey
    operator()(const int& idx) const
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
    Anchor* a = key0.anchor;
    Anchor* b = key1.anchor;
    return (a->target_read_id_ == b->target_read_id_) &&
           (a->query_read_id_ == b->query_read_id_);
}

struct cuOverlapArgs
{
    int32_t overlap_end;
    int32_t num_residues;
    int32_t overlap_start;
};

struct cuOverlapArgs_transform
{
    int32_t* d_chain_start;
    int32_t* d_chain_length;

    cuOverlapArgs_transform(int32_t* chain_start, int32_t* chain_length)
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
        // printf("%d %d %d\n", idx, overlap_start, overlap_length);
        return overlap;
    }
};

struct CustomReduceOp
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

struct CreateOverlap
{
    Anchor* d_anchors;

    __host__ __device__ __forceinline__ CreateOverlap(Anchor* anchors_ptr)
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
            // std::swap(new_overlap.target_end_position_in_read_,
            // new_overlap.target_start_position_in_read_);
            auto tmp = new_overlap.target_end_position_in_read_;
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

std::vector<Overlap>
fused_overlaps_ongpu(std::vector<Overlap>& fused_overlaps,
                     thrust::device_vector<Anchor>& d_anchors,
                     const Index& index)
{
    const auto& read_names   = index.read_id_to_read_name();
    const auto& read_lengths = index.read_id_to_read_length();
    auto n_anchors           = d_anchors.size();

    uint16_t tail_length_for_chain = 3;
    thrust::device_vector<int32_t> n_uniques(1);
    thrust::device_vector<int32_t> d_chain_length(n_anchors);

    thrust::device_vector<int32_t> d_chain_start(n_anchors);

    thrust::device_vector<Anchor> anchors_buf(d_anchors.size());

    Anchor* d_start_anchor = thrust::raw_pointer_cast(anchors_buf.data());

    auto d_num_runs_ptr = n_uniques.data();

    // run length encode to compute the overlaps start and end indices
    void* d_temp_storage      = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(
        d_temp_storage, temp_storage_bytes, d_anchors.data(), d_start_anchor,
        d_chain_length.data(), d_num_runs_ptr, n_anchors);

    // Allocate temporary storage
    CGA_CU_CHECK_ERR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Run encoding
    cub::DeviceRunLengthEncode::Encode(
        d_temp_storage, temp_storage_bytes, d_anchors.data(), d_start_anchor,
        d_chain_length.data(), d_num_runs_ptr, n_anchors);

    auto n_chains = n_uniques[0];

    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_chain_length.data(), d_chain_start.data(),
                                  n_chains);

    // Allocate temporary storage
    CGA_CU_CHECK_ERR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_chain_length.data(), d_chain_start.data(),
                                  n_chains);

    // storage for the nonzero indices
    // indices to d_chain_length/d_chain_start vector
    thrust::device_vector<int32_t> d_valid_chains_indices(n_chains);
    auto indices_end =
        thrust::copy_if(thrust::make_counting_iterator<int32_t>(0),
                        thrust::make_counting_iterator<int32_t>(n_chains),
                        d_chain_length.data(), d_valid_chains_indices.data(),
                        [=] __host__ __device__(const int32_t& len) -> bool {
                            return (len >= tail_length_for_chain);
                        });

    auto n_valid_chains = indices_end - d_valid_chains_indices.data();

    // std::ofstream glog;
    // glog.open ("glog.log", std::ios::app);
    // glog << " # valid chains/# chains - " << n_valid_chains << "/" << n_chains
    // << "\n"; glog.close();

    cuOverlapKey_transform key_op(thrust::raw_pointer_cast(d_anchors.data()),
                                  thrust::raw_pointer_cast(d_chain_start.data()));
    cub::TransformInputIterator<cuOverlapKey, cuOverlapKey_transform, int32_t*>
        d_keys_in(thrust::raw_pointer_cast(d_valid_chains_indices.data()),
                  key_op);

    cuOverlapArgs_transform value_op(
        thrust::raw_pointer_cast(d_chain_start.data()),
        thrust::raw_pointer_cast(d_chain_length.data()));

    cub::TransformInputIterator<cuOverlapArgs, cuOverlapArgs_transform, int32_t*>
        d_values_in(thrust::raw_pointer_cast(d_valid_chains_indices.data()),
                    value_op);

    thrust::device_vector<cuOverlapKey> d_unique_out(n_valid_chains);
    thrust::device_vector<cuOverlapArgs> d_aggregates_out(n_valid_chains);

    thrust::device_vector<int32_t> d_num_runs_out(1);

    CustomReduceOp reduction_op;

    // using namespace claragenomics::cudamapper::fused_overlap;
    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_keys_in,
                                   d_unique_out.data(), d_values_in,
                                   d_aggregates_out.data(), d_num_runs_out.data(),
                                   reduction_op, n_valid_chains);

    // Allocate temporary storage
    CGA_CU_CHECK_ERR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes, d_keys_in,
                                   d_unique_out.data(), d_values_in,
                                   d_aggregates_out.data(), d_num_runs_out.data(),
                                   reduction_op, n_valid_chains);

    cudaDeviceSynchronize();

    auto n_fused_overlap = d_num_runs_out[0];

    CreateOverlap fuse_op(thrust::raw_pointer_cast(d_anchors.data()));
    thrust::device_vector<Overlap> d_fused_overlaps(n_fused_overlap);
    thrust::transform(d_aggregates_out.data(),
                      d_aggregates_out.data() + n_fused_overlap,
                      d_fused_overlaps.data(), fuse_op);

    fused_overlaps.resize(n_fused_overlap);
    thrust::copy(d_fused_overlaps.begin(), d_fused_overlaps.end(),
                 fused_overlaps.begin());


#pragma omp parallel for
    for (auto i = 0; i < n_fused_overlap; ++i)
    {
        Overlap& new_overlap = fused_overlaps[i];

        std::string query_read_name  = read_names[new_overlap.query_read_id_];
        std::string target_read_name = read_names[new_overlap.target_read_id_];

        new_overlap.query_read_name_ = new char[query_read_name.length()];
        strcpy(new_overlap.query_read_name_, query_read_name.c_str());

        new_overlap.target_read_name_ = new char[target_read_name.length()];
        strcpy(new_overlap.target_read_name_, target_read_name.c_str());

        new_overlap.query_length_  = read_lengths[new_overlap.query_read_id_];
        new_overlap.target_length_ = read_lengths[new_overlap.target_read_id_];
    }

    CGA_CU_CHECK_ERR(cudaFree(d_temp_storage));

    return fused_overlaps;
}

bool operator==(const Overlap& o1, const Overlap& o2)
{
    bool same = (o1.query_read_id_ == o2.query_read_id_);
    same &= (o1.target_read_id_ == o2.target_read_id_);
    same &=
        (o1.query_start_position_in_read_ == o2.query_start_position_in_read_);
    same &=
        (o1.target_start_position_in_read_ == o2.target_start_position_in_read_);
    same &= (o1.query_end_position_in_read_ == o2.query_end_position_in_read_);
    same &= (o1.target_end_position_in_read_ == o2.target_end_position_in_read_);

    same &= (!strcmp(o1.query_read_name_, o2.query_read_name_));
    same &= (!strcmp(o1.target_read_name_, o2.target_read_name_));

    same &= (o1.relative_strand == o2.relative_strand);
    same &= (o1.num_residues_ == o2.num_residues_);
    same &= (o1.query_length_ == o2.query_length_);
    same &= (o1.target_length_ == o2.target_length_);
    return same;
}

void OverlapperTriggered::get_overlaps(std::vector<Overlap>& fused_overlaps,
                                       thrust::device_vector<Anchor>& d_anchors,
                                       const Index& index)
{

    CGA_NVTX_RANGE(profiler, "OverlapperTriggered::get_overlaps");
    const auto& read_names   = index.read_id_to_read_name();
    const auto& read_lengths = index.read_id_to_read_length();
    size_t total_anchors     = d_anchors.size();

    // comparison function object
    auto comp = [] __host__ __device__(Anchor i, Anchor j) -> bool {
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

    // sort on device
    thrust::sort(thrust::device, d_anchors.begin(), d_anchors.end(), comp);

    fused_overlaps_ongpu(fused_overlaps, d_anchors, index);
}
} // namespace cudamapper
} // namespace claragenomics
