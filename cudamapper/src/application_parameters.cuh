/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <memory>

#include <claragenomics/utils/allocator.hpp>

namespace claragenomics
{

namespace io
{
class FastaParser;
}

namespace cudamapper
{

/// @brief application parameteres, default or passed through command line
class ApplicationParameters
{
public:
    /// @brief constructor reads input from command line
    /// @param argc
    /// @param argv
    ApplicationParameters(int argc, char* argv[]);

    uint32_t kmer_size                           = 15;   // k
    uint32_t windows_size                        = 15;   // w
    std::int32_t num_devices                     = 1;    // d
    std::int32_t max_cached_memory               = 0;    // m
    std::int32_t index_size                      = 30;   // i
    std::int32_t target_index_size               = 30;   // t
    double filtering_parameter                   = 1.0;  // F
    std::int32_t alignment_engines               = 0;    // a
    std::int32_t min_residues                    = 10;   // r
    std::int32_t min_overlap_len                 = 500;  // l
    std::int32_t min_bases_per_residue           = 100;  // b
    float min_overlap_fraction                   = 0.95; // z
    std::int32_t query_indices_in_host_memory    = 10;   // q
    std::int32_t query_indices_in_device_memory  = 5;    // Q
    std::int32_t target_indices_in_host_memory   = 10;   // c
    std::int32_t target_indices_in_device_memory = 5;    // C
    bool all_to_all                              = false;
    std::string query_filepath;
    std::string target_filepath;
    std::shared_ptr<io::FastaParser> query_parser;
    std::shared_ptr<io::FastaParser> target_parser;

private:
    /// \brief gets query and target parsers
    /// \param query_parser nullptr on input, query parser on output
    /// \param target_parser nullptr on input, target parser on output
    void get_input_parsers(std::shared_ptr<io::FastaParser>& query_parser,
                           std::shared_ptr<io::FastaParser>& target_parser);

    /// @brief prints help message
    /// @param exit_code
    void help(int32_t exit_code = 0);
};

} // namespace cudamapper
} // namespace claragenomics
