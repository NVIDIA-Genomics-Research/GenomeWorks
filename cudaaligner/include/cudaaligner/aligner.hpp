/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

#include "cudaaligner/cudaaligner.hpp"

namespace claragenomics
{

namespace cudaaligner
{

// Forward declaration of Alignment class.
class Alignment;

/// \addtogroup cudaaligner
/// \{

/// \class Aligner
/// CUDA Alignment object
class Aligner
{
public:
    /// \brief Virtual destructor for Aligner.
    virtual ~Aligner() = default;

    /// \brief Launch CUDA accelerated alignment
    ///
    /// Perform alignment on all Alignment objects previously
    /// inserted. This is an async call, and returns before alignment
    /// is fully finished. To sync the alignments, refer to the
    /// sync_alignments() call;
    /// To
    virtual StatusType align_all() = 0;

    /// \brief Waits for CUDA accelerated alignment to finish
    ///
    /// Blocking call that waits for all the alignments scheduled
    /// on the GPU to come to completion.
    virtual StatusType sync_alignments() = 0;

    /// \brief Add new alignment object
    ///
    /// \param query Query string
    /// \param query_length  Query string length
    /// \param subject Subject string
    /// \param subject_length Subject string length
    virtual StatusType add_alignment(const char* query, int32_t query_length, const char* subject, int32_t subject_length) = 0;

    /// \brief Return the computed alignments.
    ///
    /// \return Vector of Alignments.
    virtual const std::vector<std::shared_ptr<Alignment>>& get_alignments() const = 0;

    /// \brief Set CUDA stream for aligner.
    ///
    /// \param stream CUDA stream
    virtual void set_cuda_stream(cudaStream_t stream) = 0;

    /// \brief Reset aligner object.
    virtual void reset() = 0;
};

/// \brief Created Aligner object
///
/// \param max_query_length Maximum length of query string
/// \param max_subject_length Maximum length of subject string
/// \param max_alignments Maximum number of alignments to be performed
/// \param type Type of aligner to construct
/// \param device_id GPU device ID to run all CUDA operations on
///
/// \return Unique pointer to Aligner object
std::unique_ptr<Aligner> create_aligner(int32_t max_query_length, int32_t max_subject_length, int32_t max_alignments, AlignmentType type, int32_t device_id);

/// \}
} // namespace cudaaligner
} // namespace claragenomics
