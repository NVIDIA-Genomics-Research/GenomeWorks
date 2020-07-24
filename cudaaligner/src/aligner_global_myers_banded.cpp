/*
* Copyright 2019-2020 NVIDIA CORPORATION.
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

struct memory_distribution
{
    int64_t sequence_memory;
    int64_t results_memory;
    int64_t query_patterns_memory;
    int64_t pmvs_matrix_memory;
    int64_t score_matrix_memory;
    int64_t remainder;
};

memory_distribution split_available_memory(const int64_t max_device_memory, const int32_t max_bandwidth)
{
    assert(max_device_memory >= 0);
    assert(max_bandwidth >= 0);

    // Memory requirements per alignment per base pair
    const float mem_req_sequence       = sizeof(char);
    const float mem_req_results        = 2 * sizeof(char);
    const float mem_req_query_patterns = 4.f / word_size;
    const float mem_req_pmvs_matrix    = (sizeof(WordType) * static_cast<float>(max_bandwidth)) / word_size;
    const float mem_req_score_matrix   = (sizeof(int32_t) * static_cast<float>(max_bandwidth)) / word_size;

    const float mem_req_total_per_bp = 2 * mem_req_sequence + mem_req_results + mem_req_query_patterns + 2 * mem_req_pmvs_matrix + mem_req_score_matrix;

    const float fmax_device_memory = static_cast<float>(max_device_memory) * 0.95; // reserve 5% for misc

    memory_distribution r;
    r.sequence_memory       = static_cast<int64_t>(fmax_device_memory / mem_req_total_per_bp * mem_req_sequence);
    r.results_memory        = static_cast<int64_t>(fmax_device_memory / mem_req_total_per_bp * mem_req_results);
    r.query_patterns_memory = static_cast<int64_t>(fmax_device_memory / mem_req_total_per_bp * mem_req_query_patterns);
    r.pmvs_matrix_memory    = static_cast<int64_t>(fmax_device_memory / mem_req_total_per_bp * mem_req_pmvs_matrix);
    r.score_matrix_memory   = static_cast<int64_t>(fmax_device_memory / mem_req_total_per_bp * mem_req_score_matrix);
    r.remainder             = max_device_memory - (2 * r.sequence_memory + r.results_memory + r.query_patterns_memory + 2 * r.pmvs_matrix_memory + r.score_matrix_memory);
    return r;
}

} // namespace

struct AlignerGlobalMyersBanded::InternalData
{
    InternalData(const memory_distribution mem, const int32_t n_alignments_initial, const DefaultDeviceAllocator& allocator, cudaStream_t stream)
        : seq_h(2 * mem.sequence_memory / sizeof(char))
        , seq_starts_h()
        , results_h(mem.results_memory / sizeof(char))
        , result_lengths_h()
        , result_starts_h()
        , seq_d(2 * mem.sequence_memory / sizeof(char), allocator, stream)
        , seq_starts_d(2 * n_alignments_initial + 1, allocator, stream)
        , results_d(mem.results_memory / sizeof(char), allocator, stream)
        , result_starts_d(n_alignments_initial + 1, allocator, stream)
        , result_lengths_d(n_alignments_initial, allocator, stream)
        , pvs(mem.pmvs_matrix_memory / sizeof(WordType), allocator, stream)
        , mvs(mem.pmvs_matrix_memory / sizeof(WordType), allocator, stream)
        , scores(mem.score_matrix_memory / sizeof(int32_t), allocator, stream)
        , query_patterns(mem.query_patterns_memory / sizeof(WordType), allocator, stream)
    {
        seq_starts_h.reserve(2 * n_alignments_initial + 1);
        result_starts_h.reserve(n_alignments_initial + 1);
        result_lengths_h.reserve(n_alignments_initial);
    }

    pinned_host_vector<char> seq_h;
    pinned_host_vector<int64_t> seq_starts_h;
    pinned_host_vector<int8_t> results_h;
    pinned_host_vector<int32_t> result_lengths_h;
    pinned_host_vector<int64_t> result_starts_h;
    device_buffer<char> seq_d;
    device_buffer<int64_t> seq_starts_d;
    device_buffer<int8_t> results_d;
    device_buffer<int64_t> result_starts_d;
    device_buffer<int32_t> result_lengths_d;
    batched_device_matrices<WordType> pvs;
    batched_device_matrices<WordType> mvs;
    batched_device_matrices<int32_t> scores;
    batched_device_matrices<WordType> query_patterns;
};

AlignerGlobalMyersBanded::AlignerGlobalMyersBanded(int64_t max_device_memory, int32_t max_bandwidth, DefaultDeviceAllocator allocator, cudaStream_t stream, int32_t device_id)
    : data_()
    , stream_(stream)
    , device_id_(device_id)
    , max_bandwidth_(max_bandwidth)
    , alignments_()
{
    if (max_bandwidth % (sizeof(WordType) * CHAR_BIT) == 1)
    {
        throw std::invalid_argument("Invalid max_bandwidth value. Please change it by +/-1.");
    }
    if (max_device_memory < 0)
    {
        max_device_memory = get_size_of_largest_free_memory_block(allocator);
    }
    scoped_device_switch dev(device_id);
    const memory_distribution mem = split_available_memory(max_device_memory, max_bandwidth);
    data_                         = std::make_unique<AlignerGlobalMyersBanded::InternalData>(mem, n_alignments_initial_parameter, allocator, stream_);
    data_->seq_starts_h.push_back(0);
    data_->result_starts_h.push_back(0);
}

AlignerGlobalMyersBanded::~AlignerGlobalMyersBanded()
{
    // Keep empty destructor to keep Workspace type incomplete in the .hpp file.
}

StatusType AlignerGlobalMyersBanded::add_alignment(const char* query, int32_t query_length, const char* target, int32_t target_length, bool reverse_complement_query, bool reverse_complement_target)
{
    throw_on_negative(query_length, "query_length should not be negative");
    throw_on_negative(target_length, "target_length should not be negative");
    if (query == nullptr || target == nullptr)
        return StatusType::generic_error;

    scoped_device_switch dev(device_id_);
    auto& seq_h           = data_->seq_h;
    auto& seq_starts_h    = data_->seq_starts_h;
    auto& result_starts_h = data_->result_starts_h;

    batched_device_matrices<WordType>& pvs            = data_->pvs;
    batched_device_matrices<WordType>& mvs            = data_->mvs;
    batched_device_matrices<int32_t>& scores          = data_->scores;
    batched_device_matrices<WordType>& query_patterns = data_->query_patterns;

    assert(!seq_starts_h.empty());
    assert(!result_starts_h.empty());
    assert(get_size(seq_h) == get_size(data_->seq_d));
    assert(get_size(data_->results_h) == get_size(data_->results_d));

    assert(pvs.remaining_free_matrix_elements() == scores.remaining_free_matrix_elements());
    assert(mvs.remaining_free_matrix_elements() == scores.remaining_free_matrix_elements());
    assert(pvs.number_of_matrices() == scores.number_of_matrices());
    assert(mvs.number_of_matrices() == scores.number_of_matrices());
    assert(query_patterns.number_of_matrices() == scores.number_of_matrices());

    const int64_t matrix_size        = compute_matrix_size_for_alignment(query_length, target_length, max_bandwidth_);
    const int32_t n_words_query      = ceiling_divide(query_length, word_size);
    const int32_t query_pattern_size = n_words_query * 4;

    if (matrix_size > scores.remaining_free_matrix_elements())
    {
        return StatusType::exceeded_max_alignments;
    }
    if (query_pattern_size > query_patterns.remaining_free_matrix_elements())
    {
        return StatusType::exceeded_max_alignments;
    }
    if (target_length + query_length > get_size<int64_t>(seq_h) - seq_starts_h.back() || target_length + query_length > get_size<int64_t>(data_->results_h) - result_starts_h.back())
    {
        return StatusType::exceeded_max_alignments;
    }

    // TODO handle reverse complements
    assert(reverse_complement_query == false);
    assert(reverse_complement_target == false);
    assert(get_size(seq_starts_h) % 2 == 1);
    const int64_t seq_start = seq_starts_h.back();
    genomeutils::copy_sequence(query, query_length, seq_h.data() + seq_start, reverse_complement_query);
    genomeutils::copy_sequence(target, target_length, seq_h.data() + seq_start + query_length, reverse_complement_target);
    seq_starts_h.push_back(seq_start + query_length);
    seq_starts_h.push_back(seq_start + query_length + target_length);
    result_starts_h.push_back(result_starts_h.back() + query_length + target_length);

    bool success = pvs.append_matrix(matrix_size);
    success      = success && mvs.append_matrix(matrix_size);
    success      = success && scores.append_matrix(matrix_size);
    success      = success && query_patterns.append_matrix(query_pattern_size);

    try
    {
        std::shared_ptr<AlignmentImpl> alignment = std::make_shared<AlignmentImpl>(query, query_length, target, target_length);
        alignment->set_alignment_type(AlignmentType::global_alignment);
        alignments_.push_back(alignment);
    }
    catch (...)
    {
        success = false;
    }

    if (!success)
    {
        // This should never trigger due to the size check before the append.
        this->reset();
        return StatusType::generic_error;
    }

    return StatusType::success;
}

StatusType AlignerGlobalMyersBanded::align_all()
{
    using cudautils::device_copy_n;
    const auto n_alignments = get_size(alignments_);
    if (n_alignments == 0)
        return StatusType::success;

    scoped_device_switch dev(device_id_);

    data_->pvs.construct_device_matrices_async(stream_);
    data_->mvs.construct_device_matrices_async(stream_);
    data_->scores.construct_device_matrices_async(stream_);
    data_->query_patterns.construct_device_matrices_async(stream_);

    const auto& seq_h           = data_->seq_h;
    const auto& seq_starts_h    = data_->seq_starts_h;
    auto& results_h             = data_->results_h;
    auto& result_lengths_h      = data_->result_lengths_h;
    const auto& result_starts_h = data_->result_starts_h;

    auto& seq_d            = data_->seq_d;
    auto& seq_starts_d     = data_->seq_starts_d;
    auto& results_d        = data_->results_d;
    auto& result_starts_d  = data_->result_starts_d;
    auto& result_lengths_d = data_->result_lengths_d;

    assert(get_size(seq_starts_h) == 2 * n_alignments + 1);
    assert(get_size(result_starts_h) == n_alignments + 1);
    if (get_size(seq_starts_d) < 2 * n_alignments + 1)
    {
        seq_starts_d.clear_and_resize(2 * n_alignments + 1);
    }
    if (get_size(result_starts_d) < n_alignments + 1)
    {
        result_starts_d.clear_and_resize(n_alignments + 1);
    }
    if (get_size(result_lengths_d) < n_alignments)
    {
        result_lengths_d.clear_and_resize(n_alignments);
    }
    device_copy_n(seq_h.data(), seq_starts_h.back(), seq_d.data(), stream_);
    device_copy_n(seq_starts_h.data(), 2 * n_alignments + 1, seq_starts_d.data(), stream_);
    device_copy_n(result_starts_h.data(), n_alignments + 1, result_starts_d.data(), stream_);

    myers_banded_gpu(results_d.data(), result_lengths_d.data(), result_starts_d.data(),
                     seq_d.data(), seq_starts_d.data(), n_alignments, max_bandwidth_,
                     data_->pvs, data_->mvs, data_->scores, data_->query_patterns,
                     stream_);

    result_lengths_h.clear();
    result_lengths_h.resize(n_alignments);

    device_copy_n(results_d.data(), result_starts_h.back(), results_h.data(), stream_);
    device_copy_n(result_lengths_d.data(), n_alignments, result_lengths_h.data(), stream_);

    return StatusType::success;
}

StatusType AlignerGlobalMyersBanded::sync_alignments()
{
    scoped_device_switch dev(device_id_);
    GW_CU_CHECK_ERR(cudaStreamSynchronize(stream_));

    const int32_t n_alignments = get_size<int32_t>(alignments_);
    std::vector<AlignmentState> al_state;
    for (int32_t i = 0; i < n_alignments; ++i)
    {
        al_state.clear();
        const int8_t* r_begin = data_->results_h.data() + data_->result_starts_h[i];
        const int8_t* r_end   = r_begin + std::abs(data_->result_lengths_h[i]);
        std::transform(r_begin, r_end, std::back_inserter(al_state), [](int8_t x) { return static_cast<AlignmentState>(x); });
        std::reverse(begin(al_state), end(al_state));
        if (!al_state.empty() || (alignments_[i]->get_query_sequence().empty() && alignments_[i]->get_target_sequence().empty()))
        {
            AlignmentImpl* alignment = dynamic_cast<AlignmentImpl*>(alignments_[i].get());
            const bool is_optimal    = (data_->result_lengths_h[i] >= 0);
            alignment->set_alignment(al_state, is_optimal);
            alignment->set_status(StatusType::success);
        }
    }
    reset_data();
    return StatusType::success;
}

void AlignerGlobalMyersBanded::reset()
{
    reset_data();
    alignments_.clear();
}

void AlignerGlobalMyersBanded::reset_data()
{
    data_->seq_starts_h.clear();
    data_->result_lengths_h.clear();
    data_->result_starts_h.clear();

    data_->seq_starts_h.push_back(0);
    data_->result_starts_h.push_back(0);

    data_->pvs.clear();
    data_->mvs.clear();
    data_->scores.clear();
    data_->query_patterns.clear();
}

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
