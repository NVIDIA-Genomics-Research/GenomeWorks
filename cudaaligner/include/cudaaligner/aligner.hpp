#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

#include "cudaaligner/cudaaligner.hpp"

namespace genomeworks {

namespace cudaaligner {

// Forward declaration of Alignment class.
class Alignment;

/// \addtogroup cudaaligner
/// \{

/// \class Aligner
/// CUDA Alignment object
class Aligner {
    public:
        /// \brief Virtual destructor for Aligner.
        virtual ~Aligner() = default;

        /// \brief Perform CUDA accelerated alignment
        ///
        /// Perform alignment on all Alignment objects previously
        /// inserted. The alignment objects are directly updated
        /// with the results.
        virtual StatusType align_all() = 0;

        /// \brief Add new alignment object
        ///
        /// \param query Query string
        /// \param query_length  Query string length
        /// \param target Target string
        /// \param target_length Target string length
        virtual StatusType add_alignment(const char* query, uint32_t query_length, const char* target, uint32_t target_length) = 0;

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
/// \param max_target_length Maximum length of target string
/// \param max_alignments Maximum number of alignments to be performed
/// \param type Type of aligner to construct
///
/// \return Unique pointer to Aligner object
std::unique_ptr<Aligner> create_aligner(uint32_t max_query_length, uint32_t max_target_length, uint32_t max_alignments, AlignmentType type, uint32_t device_id);

/// \}

}

}
