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

namespace genomeworks {
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
        const std::map<representation_t, std::vector<std::unique_ptr<SketchElement>>>& representation_to_sketch_elements() const override;

        /// \brief returns mapping of internal read id that goes from 0 to number_of_reads-1 to actual read name from the input
        /// returns mapping of internal read id that goes from 0 to number_of_reads-1 to actual read name from the input
        const std::vector<std::string>& read_id_to_read_name() const;

    private:
        /// \brief generates the index
        /// \param query_filename
        void generate_index(const std::string& query_filename);

        /// \brief finds minimizers and adds them to the index
        ///
        /// \param sequence
        /// \param sequence_id
        void add_read_to_index(const Sequence& read, read_id_t read_id);


        /// \brief finds "central" minimizers
        ///
        /// \param sequence
        /// \param sequence_id
        void find_central_minimizers(const Sequence& sequence, std::uint64_t sequence_id);

/*        /// \brief finds end minimizers
        ///
        /// \param sequence
        /// \param sequence_id
        void find_end_minimizers(const Sequence& sequence, std::uint64_t sequence_id);
*/
        std::uint64_t minimizer_size_;
        std::uint64_t window_size_;
        std::map<representation_t, std::vector<std::unique_ptr<SketchElement>>> index_;
        std::vector<std::string> read_id_to_read_name_;
    };
}
