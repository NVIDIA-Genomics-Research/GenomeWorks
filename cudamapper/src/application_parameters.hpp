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

#include <memory>

#include <claraparabricks/genomeworks/utils/allocator.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace io
{
class FastaParser;
} // namespace io

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

    uint32_t kmer_size                      = 15;    // k
    uint32_t windows_size                   = 10;    // w
    int32_t num_devices                     = 1;     // d
    int32_t max_cached_memory               = 0;     // m
    int32_t index_size                      = 30;    // i
    int32_t target_index_size               = 30;    // t
    double filtering_parameter              = 1e-5;  // F
    int32_t alignment_engines               = 0;     // a
    int32_t min_residues                    = 3;     // r, recommended range: 1 - 10. Higher: more accurate. Lower: more sensitive
    int32_t min_overlap_len                 = 250;   // l, recommended range: 100 - 1000
    int32_t min_bases_per_residue           = 1000;  // b
    float min_overlap_fraction              = 0.8;   // z
    bool perform_overlap_end_rescue         = false; // R
    bool drop_fused_overlaps                = false; // D
    bool drop_self_mappings                 = false; // X
    int32_t query_indices_in_host_memory    = 10;    // Q
    int32_t query_indices_in_device_memory  = 5;     // q
    int32_t target_indices_in_host_memory   = 10;    // C
    int32_t target_indices_in_device_memory = 5;     // c
    bool all_to_all                         = false;
    std::string query_filepath;
    std::string target_filepath;
    std::shared_ptr<io::FastaParser> query_parser;
    std::shared_ptr<io::FastaParser> target_parser;
    int64_t max_cached_memory_bytes;

private:
    /// \brief creates query and target parsers
    /// \param query_parser nullptr on input, query parser on output
    /// \param target_parser nullptr on input, target parser on output
    void create_input_parsers(std::shared_ptr<io::FastaParser>& query_parser,
                              std::shared_ptr<io::FastaParser>& target_parser);

    /// \brief Determines if filtering should be run and sets the filtering level parameter accordingly.
    /// \param query_parser A FAST(x) file parser for query sequences
    /// \param target_parser A FAST(x) file parser for target sequences
    /// \param custom_filtering_parameter A boolean signifying whether the user passed an argument for <-F>.
    void set_filtering_parameter(std::shared_ptr<io::FastaParser>& query_parser,
                                 std::shared_ptr<io::FastaParser>& target_parser,
                                 bool custom_filtering_parameter);

    /// \brief gets max number of bytes to cache by device allocator
    ///
    /// If max_cached_memory is set that value is used, finds almost complete amount of available memory otherwise
    /// Returns 0 if GW_ENABLE_CACHING_ALLOCATOR is not set
    ///
    /// \return max_cached_memory_bytes
    int64_t get_max_cached_memory_bytes();

    /// \brief prints cudamapper's version
    /// \param exit_on_completion
    void print_version(bool exit_on_completion = true);

    /// \brief prints help message
    /// \param exit_code
    [[noreturn]] void help(int32_t exit_code = 0);
};

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
