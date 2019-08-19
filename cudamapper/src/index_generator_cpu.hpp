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
#include "cudamapper/index_generator.hpp"
#include "cudamapper/sequence.hpp"
#include "cudamapper/types.hpp"
#include "minimizer.hpp"

namespace claragenomics {
    /// IndexGeneratorCPU - generates data structures necessary for building the actualy index using (k,w)-kmer minimizers
    class IndexGeneratorCPU : public IndexGenerator {
    public:
        /// \brief generates data structures necessary for building the actualy index
        ///
        /// \param query_filename filepath to reads in FASTA or FASTQ format
        /// \param minimizer_size k - the kmer length used as a minimizer
        /// \param window_size w - the length of the sliding window used to find minimizer
        IndexGeneratorCPU(const std::string& query_filename, std::uint64_t minimizer_size, std::uint64_t window_size);

        /// \brief return minimizer size (k)
        /// \return minimizer size
        std::uint64_t minimizer_size() const;

        /// \brief returns window size (w)
        /// \return window size
        std::uint64_t window_size() const;

        /// \brief returns a mapping of minimizer representations to all minimizers with those representations
        /// \return mapping of minimzer representations to all minimizers with those representations
        const std::map<representation_t, std::vector<std::unique_ptr<SketchElement>>>& representations_to_sketch_elements() const override;

        /// \brief returns mapping of internal read id that goes from 0 to number_of_reads-1 to actual read name from the input
        /// \return mapping of internal read id that goes from 0 to number_of_reads-1 to actual read name from the input
        const std::vector<std::string>& read_id_to_read_name() const override;

        const std::vector<std::uint32_t>& read_id_to_read_length() const override;

        /// \brief returns number of reads
        /// \return number of reads
        std::uint64_t number_of_reads() const override;

    private:
        /// \brief generates the index
        /// \param query_filename
        void generate_index(const std::string& query_filename);

        /// \brief finds minimizers and adds them to the index
        ///
        /// \param read
        /// \param read_id
        void add_read_to_index(const Sequence& read, const read_id_t read_id);


        /// \brief finds all minimizers (central and end)
        ///
        /// \param read
        /// \param read_id
        void find_minimizers(const Sequence& read, const read_id_t read_id);

        /// \brief finds end minimizers
        ///
        /// \param read
        /// \param read_id
        /// \param first_window_minimizer_representation representations of the minimizer of the first window
        /// \param last_window_minimizer_representation representations of the minimizer of the last window
        void find_end_minimizers(const Sequence& read, const read_id_t sequence_id, const representation_t first_window_minimizer_representation, const representation_t last_window_minimizer_representation);

        std::uint64_t minimizer_size_;
        std::uint64_t window_size_;
        std::uint64_t number_of_reads_;
        std::map<representation_t, std::vector<std::unique_ptr<SketchElement>>> index_;
        std::vector<std::string> read_id_to_read_name_;
        std::vector<std::uint32_t> read_id_to_read_length_;
    };
}
