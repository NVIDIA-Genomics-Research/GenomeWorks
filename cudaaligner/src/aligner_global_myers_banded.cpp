/*
* Copyright 2019-2021 NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "aligner_global_myers_banded.hpp"
#include "myers_gpu.cuh"
#include "batched_device_matrices.cuh"
#include "alignment_impl.hpp"

#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/utils/genomeutils.hpp>
#include <claraparabricks/genomeworks/utils/mathutils.hpp>
#include <claraparabricks/genomeworks/utils/pinned_host_vector.hpp>

#include <limits>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

namespace
{

constexpr int32_t n_alignments_initial_parameter = 1000; // for initial allocation - buffers will grow automatically if needed.

using myers::WordType;
constexpr int32_t word_size = sizeof(myers::WordType) * CHAR_BIT;

int64_t compute_matrix_size_for_alignment(const int32_t query_size, const int32_t target_size, const int32_t max_bandwidth)
{
    assert(max_bandwidth >= 0);
    assert(target_size >= 0);
    const int32_t p            = (max_bandwidth + 1) / 2;
    const int32_t bandwidth    = std::min(1 + 2 * p, query_size);
    const int64_t n_words_band = ceiling_divide(bandwidth, word_size);
    return n_words_band * (static_cast<int64_t>(target_size) + 1);
}

} // namespace

struct AlignerGlobalMyersBanded::InternalData
{
    InternalData(const DefaultDeviceAllocator& allocator, cudaStream_t stream)
        : seq_d(0, allocator, stream)
        , seq_starts_d(0, allocator, stream)
        , max_bandwidths_d(0, allocator, stream)
        , scheduling_index_d(0, allocator, stream)
        , scheduling_atomic_d(0, allocator, stream)
        , results_d(0, allocator, stream)
        , result_counts_d(0, allocator, stream)
        , result_starts_d(0, allocator, stream)
        , result_metadata_d(0, allocator, stream)
        , result_buffer_d(0, allocator, stream)
        , result_count_buffer_d(0, allocator, stream)
    {
    }

    void free_device_memory()
    {
        seq_d.free();
        seq_starts_d.free();
        max_bandwidths_d.free();
        scheduling_index_d.free();
        scheduling_atomic_d.free();
        results_d.free();
        result_counts_d.free();
        result_starts_d.free();
        result_metadata_d.free();
        result_buffer_d.free();
        result_count_buffer_d.free();
        pvs            = batched_device_matrices<WordType>();
        mvs            = batched_device_matrices<WordType>();
        scores         = batched_device_matrices<int32_t>();
        query_patterns = batched_device_matrices<WordType>();
    }

    pinned_host_vector<char> seq_h;
    pinned_host_vector<int64_t> seq_starts_h;
    pinned_host_vector<int32_t> max_bandwidths_h;
    pinned_host_vector<int32_t> scheduling_index_h;
    pinned_host_vector<int8_t> results_h;
    pinned_host_vector<int32_t> result_counts_h;
    pinned_host_vector<uint32_t> result_metadata_h;
    pinned_host_vector<int32_t> result_starts_h;
    device_buffer<char> seq_d;
    device_buffer<int64_t> seq_starts_d;
    device_buffer<int32_t> max_bandwidths_d;
    device_buffer<int32_t> scheduling_index_d;
    device_buffer<int32_t> scheduling_atomic_d;
    device_buffer<int8_t> results_d;
    device_buffer<int32_t> result_counts_d;
    device_buffer<int32_t> result_starts_d;
    device_buffer<uint32_t> result_metadata_d;
    batched_device_matrices<WordType> pvs;
    batched_device_matrices<WordType> mvs;
    batched_device_matrices<int32_t> scores;
    batched_device_matrices<WordType> query_patterns;
    device_buffer<int8_t> result_buffer_d;
    device_buffer<int32_t> result_count_buffer_d;
    std::vector<std::pair<int64_t, int64_t>> largest_matrix_sizes; // sorted from largest to smallest
    int64_t matrix_size_small            = 0;
    int32_t cur_query_pattern_size_large = 0;
    int32_t cur_query_pattern_size_small = 0;
    int32_t result_buffer_length         = 0;
    int32_t n_large_workloads            = 0;
    int32_t n_launch_blocks              = 0;
};

AlignerGlobalMyersBanded::AlignerGlobalMyersBanded(int64_t max_device_memory, int32_t max_bandwidth, DefaultDeviceAllocator allocator, cudaStream_t stream, int32_t device_id)
    : data_()
    , stream_(stream)
    , device_id_(device_id)
    , max_bandwidth_(throw_on_negative(max_bandwidth, "max_bandwidth cannot be negative."))
    , alignments_()
    , max_device_memory_(max_device_memory < 0 ? get_size_of_largest_free_memory_block(allocator) : max_device_memory)
{
    scoped_device_switch dev(device_id);
    data_                  = std::make_unique<AlignerGlobalMyersBanded::InternalData>(allocator, stream);
    const int32_t n_blocks = myers_banded_gpu_get_blocks_per_sm();
    int n_sms              = 0;
    GW_CU_CHECK_ERR(cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, device_id));
    data_->n_launch_blocks          = n_sms * n_blocks;
    const int64_t space_for_io_data = 2 * max_device_memory_ / (3 * (2 * sizeof(char) + sizeof(int32_t))); // estimate: 2/3s of the device mem go to the io buffers
    data_->seq_h.resize(space_for_io_data);
    // This aligner launches n_launch_blocks blocks, where each block has one workspace assigned to it and it aligns pairs of sequences until all alignments are done.
    // There are two different workspace: large workspaces and small workspaces. The size of these workspaces is determined dynamically.
    // The size of the large workspaces is defined by the n_large_workloads largest alignment problems in the batch.
    // The size of the small workspaces is defined by the largest of the remaining alignment problems.
    data_->n_large_workloads = n_sms; // The number of large workloads/workspaces - we arbitrarily choose one large workspace per SM (doesn't need to be a multiple).
    data_->largest_matrix_sizes.emplace_back(0, 0);
    AlignerGlobalMyersBanded::reset_max_bandwidth(max_bandwidth);
}

// Keep destructor definition in src file to keep InternalData type incomplete in the .hpp file.
AlignerGlobalMyersBanded::~AlignerGlobalMyersBanded() = default;

StatusType AlignerGlobalMyersBanded::add_alignment(const char* query, int32_t query_length, const char* target, int32_t target_length, bool reverse_complement_query, bool reverse_complement_target)
{
    return AlignerGlobalMyersBanded::add_alignment(max_bandwidth_, query, query_length, target, target_length, reverse_complement_query, reverse_complement_target);
}

StatusType AlignerGlobalMyersBanded::add_alignment(int32_t max_bandwidth, const char* query, int32_t query_length, const char* target, int32_t target_length, bool reverse_complement_query, bool reverse_complement_target)
{
    GW_NVTX_RANGE(profiler, "AlignerGlobalMyersBanded::add_alignment");
    if (max_bandwidth < 0 || query_length < 0 || target_length < 0 || query == nullptr || target == nullptr)
        return StatusType::generic_error;

    auto& seq_h                = data_->seq_h;
    auto& seq_starts_h         = data_->seq_starts_h;
    auto& max_bandwidths_h     = data_->max_bandwidths_h;
    auto& largest_matrix_sizes = data_->largest_matrix_sizes;
    const int32_t n_alignments = get_size<int32_t>(seq_starts_h) / 2;

    assert(seq_starts_h.size() % 2 == 1);

    if (max_bandwidth > query_length)
    {
        // we need to guarantee max_bandwidth % word_size != 1 for myers_banded_gpu().
        max_bandwidth = (query_length % word_size == 1 ? query_length + 1 : query_length);
    }

    const int64_t matrix_size        = compute_matrix_size_for_alignment(query_length, target_length, max_bandwidth);
    const int32_t n_words_query      = ceiling_divide(query_length, word_size);
    const int32_t query_pattern_size = n_words_query * 4;

    assert(!data_->largest_matrix_sizes.empty());
    int64_t new_matrix_size_large        = data_->largest_matrix_sizes.front().first;
    int64_t new_matrix_size_small        = data_->matrix_size_small;
    int32_t new_result_buffer_length     = std::max(data_->result_buffer_length, query_length + target_length);
    int32_t new_query_pattern_size_large = data_->cur_query_pattern_size_large;
    int32_t new_query_pattern_size_small = data_->cur_query_pattern_size_small;

    const auto smallest_large_workload = largest_matrix_sizes.back();
    if (matrix_size > smallest_large_workload.first)
    {
        new_matrix_size_large        = std::max(new_matrix_size_large, matrix_size);
        new_query_pattern_size_large = std::max(new_query_pattern_size_large, query_pattern_size);
        new_matrix_size_small        = smallest_large_workload.first;
        new_query_pattern_size_small = std::max<int64_t>(new_query_pattern_size_small, smallest_large_workload.second);
    }
    else
    {
        new_matrix_size_small        = std::max(new_matrix_size_small, matrix_size);
        new_query_pattern_size_small = std::max(new_query_pattern_size_small, query_pattern_size);
    }

    if (!fits_device_memory(new_matrix_size_large, new_matrix_size_small, new_query_pattern_size_large, new_query_pattern_size_small, query_length, target_length))
    {
        if (n_alignments == 0)
            throw std::runtime_error("Could not fit alignment into device or host memory.");
        return StatusType::exceeded_max_alignments;
    }
    if (seq_starts_h.back() + query_length + target_length > get_size(seq_h))
    {
        try
        {
            const int64_t new_size = std::max(seq_starts_h.back() + query_length + target_length, get_size(seq_h) + get_size(seq_h) / 4);
            seq_h.resize(new_size);
        }
        catch (std::bad_alloc const&)
        {
            return StatusType::exceeded_max_alignments;
        }
    }

    assert(get_size(seq_starts_h) % 2 == 1);
    const int64_t seq_start = seq_starts_h.back();
    genomeutils::copy_sequence(query, query_length, seq_h.data() + seq_start, reverse_complement_query);
    genomeutils::copy_sequence(target, target_length, seq_h.data() + seq_start + query_length, reverse_complement_target);
    try
    {
        seq_starts_h.push_back(seq_start + query_length);
        seq_starts_h.push_back(seq_start + query_length + target_length);
        max_bandwidths_h.push_back(max_bandwidth);
    }
    catch (const std::bad_alloc&)
    {
        if (n_alignments == 0)
            throw std::runtime_error("Could not fit alignment into device host memory.");
        seq_starts_h.resize(2 * n_alignments + 1);
        max_bandwidths_h.resize(n_alignments);
        return StatusType::exceeded_max_alignments;
    }

    // Update the workspace sizes
    if (matrix_size > data_->matrix_size_small)
    {
        auto it = std::lower_bound(begin(largest_matrix_sizes), end(largest_matrix_sizes), std::make_pair(matrix_size, int64_t(0)), [](std::pair<int64_t, int64_t> const& a, std::pair<int64_t, int64_t> const& b) { return a.first > b.first; });
        largest_matrix_sizes.emplace(it, matrix_size, query_pattern_size);
        if (get_size(largest_matrix_sizes) > data_->n_large_workloads)
        {
            largest_matrix_sizes.resize(data_->n_large_workloads);
        }
    }
    data_->matrix_size_small            = new_matrix_size_small;
    data_->result_buffer_length         = new_result_buffer_length;
    data_->cur_query_pattern_size_large = new_query_pattern_size_large;
    data_->cur_query_pattern_size_small = new_query_pattern_size_small;
    return StatusType::success;
}

StatusType AlignerGlobalMyersBanded::align_all()
{
    GW_NVTX_RANGE(profiler, "AlignerGlobalMyersBanded::align_all");
    using cudautils::device_copy_n_async;
    assert(data_->seq_starts_h.size() >= 1);
    assert(data_->seq_starts_h.size() % 2 == 1);
    const int32_t n_alignments = get_size<int32_t>(data_->seq_starts_h) / 2;
    if (n_alignments == 0)
        return StatusType::success;

    scoped_device_switch dev(device_id_);

    const auto& seq_h            = data_->seq_h;
    const auto& seq_starts_h     = data_->seq_starts_h;
    const auto& max_bandwidths_h = data_->max_bandwidths_h;
    auto& result_starts_h        = data_->result_starts_h;
    auto& scheduling_index_h     = data_->scheduling_index_h;

    auto& seq_d              = data_->seq_d;
    auto& seq_starts_d       = data_->seq_starts_d;
    auto& max_bandwidths_d   = data_->max_bandwidths_d;
    auto& scheduling_index_d = data_->scheduling_index_d;
    auto& results_d          = data_->results_d;
    auto& result_counts_d    = data_->result_counts_d;
    auto& result_starts_d    = data_->result_starts_d;
    auto& result_metadata_d  = data_->result_metadata_d;

    assert(get_size(seq_starts_h) == 2 * n_alignments + 1);

    const int64_t sequence_length_sum = seq_starts_h.back();
    seq_d.clear_and_resize(sequence_length_sum);
    seq_starts_d.clear_and_resize(2 * n_alignments + 1);
    results_d.clear_and_resize(sequence_length_sum);
    result_counts_d.clear_and_resize(sequence_length_sum);
    result_starts_d.clear_and_resize(n_alignments + 1);
    result_metadata_d.clear_and_resize(n_alignments);
    max_bandwidths_d.clear_and_resize(n_alignments);
    scheduling_index_d.clear_and_resize(n_alignments);
    data_->scheduling_atomic_d.clear_and_resize(1);
    device_copy_n_async(seq_h.data(), seq_starts_h.back(), seq_d.data(), stream_);
    device_copy_n_async(seq_starts_h.data(), 2 * n_alignments + 1, seq_starts_d.data(), stream_);
    device_copy_n_async(max_bandwidths_h.data(), n_alignments, max_bandwidths_d.data(), stream_);

    // Create an index of alignment tasks, which determines the processing
    // order. Sort this index by the sum of query and target lengths such
    // that large alignments get processed first.
    scheduling_index_h.clear();
    scheduling_index_h.resize(n_alignments);
    std::iota(begin(scheduling_index_h), end(scheduling_index_h), 0);
    std::sort(begin(scheduling_index_h), end(scheduling_index_h), [&seq_starts_h](int32_t i, int32_t j) { return seq_starts_h[2 * i + 2] - seq_starts_h[2 * i] > seq_starts_h[2 * j + 2] - seq_starts_h[2 * j]; });

    batched_device_matrices<WordType>& pvs            = data_->pvs;
    batched_device_matrices<WordType>& mvs            = data_->mvs;
    batched_device_matrices<int32_t>& scores          = data_->scores;
    batched_device_matrices<WordType>& query_patterns = data_->query_patterns;

    DefaultDeviceAllocator allocator = seq_d.get_allocator();

    const int32_t n_large_spaces        = get_size<int32_t>(data_->largest_matrix_sizes);
    const int32_t n_small_spaces        = data_->n_launch_blocks - n_large_spaces;
    const int64_t total_matrix_elements = n_large_spaces * data_->largest_matrix_sizes.front().first + n_small_spaces * data_->matrix_size_small;
    const int64_t total_query_patterns  = n_large_spaces * data_->cur_query_pattern_size_large + n_small_spaces * data_->cur_query_pattern_size_small;

    pvs = batched_device_matrices<WordType>(total_matrix_elements, allocator, stream_);
    pvs.reserve_n_matrices(data_->n_launch_blocks);
    mvs = batched_device_matrices<WordType>(total_matrix_elements, allocator, stream_);
    mvs.reserve_n_matrices(data_->n_launch_blocks);
    scores = batched_device_matrices<int32_t>(total_matrix_elements, allocator, stream_);
    scores.reserve_n_matrices(data_->n_launch_blocks);
    query_patterns = batched_device_matrices<WordType>(total_query_patterns, allocator, stream_);
    query_patterns.reserve_n_matrices(data_->n_launch_blocks);
    data_->result_buffer_d.clear_and_resize(data_->n_launch_blocks * data_->result_buffer_length);
    data_->result_count_buffer_d.clear_and_resize(data_->n_launch_blocks * data_->result_buffer_length);

    bool success = true;
    for (int32_t i = 0; i < n_large_spaces; ++i)
    {
        success = success && pvs.append_matrix(data_->largest_matrix_sizes.front().first);
        success = success && mvs.append_matrix(data_->largest_matrix_sizes.front().first);
        success = success && scores.append_matrix(data_->largest_matrix_sizes.front().first);
        success = success && query_patterns.append_matrix(data_->cur_query_pattern_size_large);
    }
    assert(n_small_spaces >= 0);
    for (int32_t i = 0; i < n_small_spaces; ++i)
    {
        success = success && pvs.append_matrix(data_->matrix_size_small);
        success = success && mvs.append_matrix(data_->matrix_size_small);
        success = success && scores.append_matrix(data_->matrix_size_small);
        success = success && query_patterns.append_matrix(data_->cur_query_pattern_size_small);
    }

    if (!success)
    {
        // Ideally this should not happen, since the memory requirements are checked when the alignment is added.
        throw std::runtime_error("Out of memory.");
    }

    pvs.construct_device_matrices_async(stream_);
    mvs.construct_device_matrices_async(stream_);
    scores.construct_device_matrices_async(stream_);
    query_patterns.construct_device_matrices_async(stream_);

    device_copy_n_async(scheduling_index_h.data(), n_alignments, scheduling_index_d.data(), stream_);

    myers_banded_gpu(results_d.data(), result_counts_d.data(), result_starts_d.data(), result_metadata_d.data(),
                     seq_d.data(), seq_starts_d.data(), max_bandwidths_d.data(), scheduling_index_d.data(), data_->scheduling_atomic_d.data(),
                     pvs, mvs, scores, query_patterns,
                     data_->result_buffer_d.data(), data_->result_count_buffer_d.data(), data_->result_buffer_length,
                     n_alignments, data_->n_launch_blocks, data_->n_large_workloads,
                     stream_);
    result_starts_h.clear();
    result_starts_h.resize(n_alignments + 1);
    device_copy_n_async(result_starts_d.data(), n_alignments + 1, result_starts_h.data(), stream_);
    return StatusType::success;
}

StatusType AlignerGlobalMyersBanded::sync_alignments()
{
    GW_NVTX_RANGE(profiler, "AlignerGlobalMyersBanded::sync");
    using cudautils::device_copy_n_async;
    assert(data_->seq_starts_h.size() >= 1);
    assert(data_->seq_starts_h.size() % 2 == 1);
    const int32_t n_alignments  = get_size<int32_t>(data_->seq_starts_h) / 2;
    auto& results_h             = data_->results_h;
    auto& result_counts_h       = data_->result_counts_h;
    const auto& result_starts_h = data_->result_starts_h;
    auto& result_metadata_h     = data_->result_metadata_h;
    result_metadata_h.clear();
    result_metadata_h.resize(n_alignments);
    device_copy_n_async(data_->result_metadata_d.data(), n_alignments, result_metadata_h.data(), stream_);
    alignments_.clear();
    alignments_.resize(n_alignments);
    GW_CU_CHECK_ERR(cudaStreamSynchronize(stream_)); // wait for result_starts_d copy from align_all()
    results_h.clear();
    results_h.resize(result_starts_h.back());
    device_copy_n_async(data_->results_d.data(), result_starts_h.back(), results_h.data(), stream_);
    result_counts_h.clear();
    result_counts_h.resize(result_starts_h.back());
    device_copy_n_async(data_->result_counts_d.data(), result_starts_h.back(), result_counts_h.data(), stream_);
    GW_CU_CHECK_ERR(cudaStreamSynchronize(stream_));

    GW_NVTX_RANGE(profiler_post, "AlignerGlobalMyersBanded::post-sync");
    for (int32_t i = 0; i < n_alignments; ++i)
    {
        const bool is_optimal         = result_metadata_h[i] >> 31;
        const int32_t index           = result_metadata_h[i] & DeviceAlignmentsPtrs::index_mask;
        const int64_t result_start    = std::max(0, result_starts_h[i]);
        const int64_t result_end      = std::max(0, result_starts_h[i + 1]);
        const int8_t* r_begin         = results_h.data() + result_start;
        const int8_t* r_end           = results_h.data() + result_end;
        const int32_t* r_counts_begin = result_counts_h.data() + result_start;
        const int32_t* r_counts_end   = result_counts_h.data() + result_end;

        const char* query           = data_->seq_h.data() + data_->seq_starts_h[2 * index];
        const int32_t query_length  = data_->seq_starts_h[2 * index + 1] - data_->seq_starts_h[2 * index];
        const char* target          = data_->seq_h.data() + data_->seq_starts_h[2 * index + 1];
        const int32_t target_length = data_->seq_starts_h[2 * index + 2] - data_->seq_starts_h[2 * index + 1];

        std::shared_ptr<AlignmentImpl> alignment = std::make_shared<AlignmentImpl>(query, query_length, target, target_length);
        alignment->set_alignment_type(AlignmentType::global_alignment);

        if (r_begin != r_end || (query_length == 0 && target_length == 0))
        {
            alignment->set_alignment({std::make_reverse_iterator(r_end), std::make_reverse_iterator(r_begin)}, {std::make_reverse_iterator(r_counts_end), std::make_reverse_iterator(r_counts_begin)}, is_optimal);
            alignment->set_status(StatusType::success);
        }
        alignments_[index] = std::move(alignment);
    }
    reset_data();
    return StatusType::success;
}

int32_t AlignerGlobalMyersBanded::num_alignments() const
{
    return get_size<int32_t>(data_->seq_starts_h) / 2;
}

void AlignerGlobalMyersBanded::reset()
{
    reset_data();
    alignments_.clear();
}

void AlignerGlobalMyersBanded::free_temporary_device_buffers()
{
    data_->result_count_buffer_d.free();
    data_->result_buffer_d.free();
    data_->query_patterns = batched_device_matrices<WordType>();
    data_->scores         = batched_device_matrices<int32_t>();
    data_->mvs            = batched_device_matrices<WordType>();
    data_->pvs            = batched_device_matrices<WordType>();
}

DeviceAlignmentsPtrs AlignerGlobalMyersBanded::get_alignments_device() const
{
    assert(data_->seq_starts_h.size() >= 1);
    DeviceAlignmentsPtrs r;
    r.cigar_operations = data_->results_d.data();
    r.cigar_runlengths = data_->result_counts_d.data();
    r.cigar_offsets    = data_->result_starts_d.data();
    r.metadata         = data_->result_metadata_d.data();
    r.total_length     = data_->result_starts_h.back();
    r.n_alignments     = get_size<int32_t>(data_->seq_starts_h) / 2;
    return r;
}

void AlignerGlobalMyersBanded::reset_max_bandwidth(const int32_t max_bandwidth)
{
    assert(max_device_memory_ >= 0);
    throw_on_negative(max_bandwidth, "max_bandwidth cannot be negative.");
    if (max_bandwidth % (sizeof(WordType) * CHAR_BIT) == 1)
    {
        throw std::invalid_argument("Invalid max_bandwidth. max_bandwidth % 32 == 1 is not allowed. Please change it by +/-1.");
    }
    reset();
    max_bandwidth_ = max_bandwidth;
}

void AlignerGlobalMyersBanded::reset_data()
{
    data_->seq_starts_h.clear();
    data_->max_bandwidths_h.clear();
    data_->result_metadata_h.clear();
    data_->result_starts_h.clear();

    data_->seq_starts_h.push_back(0);
    data_->free_device_memory();
    data_->largest_matrix_sizes.clear();
    data_->largest_matrix_sizes.emplace_back(0, 0);
    data_->matrix_size_small            = 0;
    data_->cur_query_pattern_size_small = 0;
    data_->cur_query_pattern_size_large = 0;
}

bool AlignerGlobalMyersBanded::fits_device_memory(int64_t matrix_size_large, int64_t matrix_size_small, int32_t query_pattern_size_large, int32_t query_pattern_size_small, int32_t query_length, int32_t target_length) const
{
    assert(matrix_size_large >= 0);
    assert(matrix_size_small >= 0);
    assert(query_pattern_size_large >= 0);
    assert(query_pattern_size_small >= 0);
    assert(query_length >= 0);
    assert(target_length >= 0);
    assert(data_->n_launch_blocks >= data_->n_large_workloads);

    constexpr int64_t mem_alignment = 256;
    // for the upper bound we simply add the full alignment to every allocation

    const int32_t new_n_alignments         = get_size<int32_t>(data_->seq_starts_h) / 2 + 1;
    const int64_t new_sequence_length_sum  = data_->seq_starts_h.back() + query_length + target_length;
    const int32_t new_result_buffer_length = std::max(query_length + target_length, data_->result_buffer_length);
    if ((static_cast<uint32_t>(new_n_alignments) & (~DeviceAlignmentsPtrs::index_mask)) != 0u)
    {
        // The alignments cannot be indexed with an 26bit index in DeviceAlignmentPtrs::metadata.
        return false;
    }
    if (new_sequence_length_sum > static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
    {
        // The theoretical maximum total length of the results does not fit into int32_t used for the result_starts_d.
        return false;
    }

    // If batched_device_matrices ever considers an alignment, these two quantities need to be adapted:
    const int64_t workspace_req_small = matrix_size_small * (2 * sizeof(WordType) + sizeof(int32_t)) + query_pattern_size_small * sizeof(WordType);
    const int64_t workspace_req_large = matrix_size_large * (2 * sizeof(WordType) + sizeof(int32_t)) + query_pattern_size_large * sizeof(WordType);

    int64_t memory_req = sizeof(decltype(*data_->scheduling_atomic_d.data())) + mem_alignment;
    memory_req += (workspace_req_large + 4 * sizeof(ptrdiff_t)) * data_->n_large_workloads;
    memory_req += (workspace_req_small + 4 * sizeof(ptrdiff_t)) * (data_->n_launch_blocks - data_->n_large_workloads);
    memory_req += 4 * mem_alignment + 4 * mem_alignment; // batched_device_matrices: data + offset arrays
    memory_req += 3 * sizeof(batched_device_matrices<WordType>::device_interface) + sizeof(batched_device_matrices<int32_t>::device_interface) + 4 * mem_alignment;
    memory_req += new_result_buffer_length * (sizeof(decltype(*data_->result_buffer_d.data())) + sizeof(decltype(*data_->result_count_buffer_d.data()))) * data_->n_launch_blocks + 2 * mem_alignment;

    memory_req += new_sequence_length_sum * (sizeof(decltype(*data_->seq_d.data())) + sizeof(decltype(*data_->results_d.data())) + sizeof(decltype(*data_->result_counts_d.data()))) + 3 * mem_alignment;
    memory_req += (2 * new_n_alignments + 1) * sizeof(decltype(*data_->seq_starts_d.data())) + mem_alignment;
    memory_req += new_n_alignments * (sizeof(decltype(*data_->max_bandwidths_d.data())) + sizeof(decltype(*data_->scheduling_index_d.data())) + sizeof(decltype(*data_->result_starts_d.data())) + sizeof(decltype(*data_->result_metadata_d.data()))) + 4 * mem_alignment;
    memory_req += sizeof(decltype(*data_->result_starts_d.data()));

    return memory_req < max_device_memory_;
}

DefaultDeviceAllocator AlignerGlobalMyersBanded::get_device_allocator() const
{
    return data_->seq_d.get_allocator();
}

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
