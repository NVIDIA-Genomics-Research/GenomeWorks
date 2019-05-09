#include <cstring>

#include <cuda_runtime_api.h>

#define GW_LOG_LEVEL gw_log_level_warn

#include "aligner_global.hpp"
#include "alignment_impl.hpp"
#include <cudautils/cudautils.hpp>
#include <logging/logging.hpp>


namespace genomeworks {

namespace cudaaligner {

AlignerGlobal::AlignerGlobal(uint32_t max_query_length, uint32_t max_target_length, uint32_t max_alignments, uint32_t device_id)
    : max_query_length_(max_query_length)
    , max_target_length_(max_target_length)
    , max_alignments_(max_alignments)
    , alignments_()
    , device_id_(device_id)
{
    // Allocate buffers
    GW_CU_CHECK_ERR(cudaSetDevice(device_id_));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &sequences_d_,
                2 * sizeof(uint8_t) * std::max(max_query_length_, max_target_length_) * max_alignments_));
    GW_CU_CHECK_ERR(cudaHostAlloc((void**) &sequences_h_,
                2 * sizeof(uint8_t) * std::max(max_query_length_, max_target_length_) * max_alignments_,
                cudaHostAllocDefault));

    GW_CU_CHECK_ERR(cudaMalloc((void**) &sequence_lengths_d_,
                2 * sizeof(uint32_t) * max_alignments_));
    GW_CU_CHECK_ERR(cudaHostAlloc((void**) &sequence_lengths_h_,
                2 * sizeof(uint32_t) * max_alignments_,
                cudaHostAllocDefault));

    GW_CU_CHECK_ERR(cudaMalloc((void**) &results_d_,
                sizeof(uint8_t) * (max_query_length_ + max_target_length_) * max_alignments_));
    GW_CU_CHECK_ERR(cudaHostAlloc((void**) &results_h_,
                sizeof(uint8_t) * (max_query_length_ + max_target_length_) * max_alignments_,
                cudaHostAllocDefault));

    GW_CU_CHECK_ERR(cudaMalloc((void**) &result_lengths_d_,
                sizeof(uint32_t) * max_alignments_));
    GW_CU_CHECK_ERR(cudaHostAlloc((void**) &result_lengths_h_,
                sizeof(uint32_t) * max_alignments_,
                cudaHostAllocDefault));
}

AlignerGlobal::~AlignerGlobal()
{
    GW_CU_CHECK_ERR(cudaSetDevice(device_id_));
    // Free up buffers
    GW_CU_CHECK_ERR(cudaFree(sequences_d_));
    GW_CU_CHECK_ERR(cudaFree(results_d_));
    GW_CU_CHECK_ERR(cudaFree(sequence_lengths_d_));
    GW_CU_CHECK_ERR(cudaFree(result_lengths_d_));

    GW_CU_CHECK_ERR(cudaFreeHost(sequences_h_));
    GW_CU_CHECK_ERR(cudaFreeHost(sequence_lengths_h_));
    GW_CU_CHECK_ERR(cudaFreeHost(results_h_));
    GW_CU_CHECK_ERR(cudaFreeHost(result_lengths_h_));
}

StatusType AlignerGlobal::add_alignment(const char* query, uint32_t query_length, const char* target, uint32_t target_length)
{
    uint32_t num_alignments = alignments_.size();
    if (num_alignments >= max_alignments_)
    {
        GW_LOG_INFO("{} {}", "Exceeded maximum number of alignments allowed : ", max_alignments_);
        return StatusType::exceeded_max_alignments; 
    }

    if (query_length >= max_query_length_)
    {
        GW_LOG_INFO("{} {}", "Exceeded maximum length of query allowed : ", max_query_length_);
        return StatusType::exceeded_max_length;
    }

    if (target_length >= max_target_length_)
    {
        GW_LOG_INFO("{} {}", "Exceeded maximum length of target allowed : ", max_target_length_);
        return StatusType::exceeded_max_length;
    }

    memcpy(&sequences_h_[(2 * num_alignments) * std::max(max_query_length_, max_target_length_)],
          query,
          sizeof(uint8_t) * query_length);
    memcpy(&sequences_h_[(2 * num_alignments + 1) * std::max(max_query_length_, max_target_length_)],
           target,
           sizeof(uint8_t) * target_length);

    sequence_lengths_h_[2 * num_alignments] = query_length;
    sequence_lengths_h_[2 * num_alignments + 1] = target_length;

    std::shared_ptr<AlignmentImpl> alignment = std::make_shared<AlignmentImpl>(query,
                                                                               query_length,
                                                                               target,
                                                                               target_length);
    alignment->set_alignment_type(AlignmentType::global);
    alignments_.push_back(alignment);

    return StatusType::success;
}

StatusType AlignerGlobal::align_all()
{
    uint32_t num_alignments = alignments_.size();
    GW_CU_CHECK_ERR(cudaSetDevice(device_id_));
    GW_CU_CHECK_ERR(cudaMemcpyAsync(sequences_d_,
                                    sequences_h_,
                                    2 * sizeof(uint8_t) * std::max(max_query_length_, max_target_length_) * num_alignments,
                                    cudaMemcpyHostToDevice,
                                    stream_));
    GW_CU_CHECK_ERR(cudaMemcpyAsync(sequence_lengths_d_,
                                    sequence_lengths_h_,
                                    2 * sizeof(uint32_t) * num_alignments,
                                    cudaMemcpyHostToDevice,
                                    stream_));

    // Run kernel

    GW_CU_CHECK_ERR(cudaMemcpyAsync(results_h_,
                                    results_d_,
                                    sizeof(uint8_t) * (max_query_length_ + max_target_length_) * num_alignments,
                                    cudaMemcpyDeviceToHost,
                                    stream_));
    GW_CU_CHECK_ERR(cudaMemcpyAsync(result_lengths_h_,
                                    result_lengths_d_,
                                    sizeof(uint32_t) * num_alignments,
                                    cudaMemcpyDeviceToHost,
                                    stream_));

    update_alignments_with_results();

    return StatusType::success;
}

void AlignerGlobal::update_alignments_with_results()
{
    uint32_t num_alignments = alignments_.size();
    std::vector<AlignmentState> al_state;
    for(uint32_t a = 0; a < num_alignments; a++)
    {
        al_state.clear();
        uint32_t alignment_length = result_lengths_h_[a];
        for(uint32_t pos = 0; pos < alignment_length; pos++)
        {
            uint8_t state = results_h_[a * (max_query_length_ + max_target_length_) + pos];            
            al_state.push_back(static_cast<AlignmentState>(state));
        }
        AlignmentImpl* alignment = static_cast<AlignmentImpl*>(alignments_.at(a).get());
        alignment->set_alignment(al_state);
        alignment->set_status(StatusType::success);
    }
}

void AlignerGlobal::reset()
{
    alignments_.clear();
}

}

}
