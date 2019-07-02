/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "aligner_global_myers.hpp"
#include "myers_gpu.cuh"
#include <utils/signed_integer_utils.hpp>
#include "batched_device_matrices.cuh"

namespace claragenomics
{

namespace cudaaligner
{

struct AlignerGlobalMyers::Workspace
{
    Workspace(int32_t max_alignments, int32_t max_elements_per_matrix, cudaStream_t stream, int32_t device_id)
        : pvs(max_alignments, max_elements_per_matrix, stream, device_id), mvs(max_alignments, max_elements_per_matrix, stream, device_id), scores(max_alignments, max_elements_per_matrix, stream, device_id)
    {
    }
    batched_device_matrices<myers::WordType> pvs;
    batched_device_matrices<myers::WordType> mvs;
    batched_device_matrices<int32_t> scores;
};

AlignerGlobalMyers::AlignerGlobalMyers(int32_t max_query_length, int32_t max_subject_length, int32_t max_alignments, cudaStream_t stream, int32_t device_id)
    : AlignerGlobal(max_query_length, max_subject_length, max_alignments, stream, device_id), workspace_(std::make_unique<Workspace>(max_alignments, ceiling_divide<int32_t>(max_query_length, sizeof(myers::WordType)) * (max_subject_length + 1), stream, device_id))
{
}

AlignerGlobalMyers::~AlignerGlobalMyers()
{
    // Keep empty destructor to keep Workspace type incomplete in the .hpp file.
}

void AlignerGlobalMyers::run_alignment(int8_t* results_d, int32_t* result_lengths_d, int32_t max_result_length, const char* sequences_d, int32_t* sequence_lengths_d, int32_t* sequence_lengths_h, int32_t max_sequence_length, int32_t num_alignments, cudaStream_t stream)
{
    static_cast<void>(sequence_lengths_h);
    myers_gpu(results_d, result_lengths_d, max_result_length, sequences_d, sequence_lengths_d, max_sequence_length, num_alignments, workspace_->pvs, workspace_->mvs, workspace_->scores, stream);
}

} // namespace cudaaligner
} // namespace claragenomics
