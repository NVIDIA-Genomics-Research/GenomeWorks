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

#include <cstdint>
#include <unordered_map>
#include "cudamapper/index_generator.hpp"
#include "cudamapper/sequence.hpp"
#include "minimizer.hpp"

namespace genomeworks {
    /// IndexGeneratorCPU - generates and manages (k,w)-minimizer index for one or more sequences
    /// lifecycle managed by the host (not GPU)
    class IndexGeneratorCPU : public IndexGenerator {
    public:
        /// \brief generates an in-memory (k,w)-minimizer index
        ///
        /// \param query_filename filepath to reads in FASTA or FASTQ format
        /// \param minimizer_size k - the kmer length used as a minimizer
        /// \param window_size w - the length of the sliding window used to find minimizer
        IndexGeneratorCPU(const std::string& query_filename, std::uint64_t minimizer_size, std::uint64_t window_size);

        /// \brief return minimizer size
        /// \return minimizer size
        std::uint64_t minimizer_size() const;

        /// \brief returns window size
        /// \return window size
        std::uint64_t window_size() const;

        /// \brief return a hash table wich maps minimizers' representations and positions
        /// \return hash table
        const std::unordered_multimap<std::uint64_t, std::unique_ptr<SketchElement>>& representation_sketch_element_mapping() const override;

    private:
        /// \brief generates the index
        /// \param query_filename
        void generate_index(const std::string& query_filename);

        /// \brief finds minimizers and adds them to the index
        ///
        /// \param sequence
        /// \param sequence_id
        void add_sequence_to_index(const Sequence& sequence, std::uint64_t sequence_id);

        /// \brief finds "central" minimizers
        ///
        /// \param sequence
        /// \param sequence_id
        void find_central_minimizers(const Sequence& sequence, std::uint64_t sequence_id);

        /// \brief finds end minimizers
        ///
        /// \param sequence
        /// \param sequence_id
        void find_end_minimizers(const Sequence& sequence, std::uint64_t sequence_id);

        std::uint64_t minimizer_size_;
        std::uint64_t window_size_;
        std::unordered_multimap<std::uint64_t, std::unique_ptr<SketchElement>> index_;
    };
}
