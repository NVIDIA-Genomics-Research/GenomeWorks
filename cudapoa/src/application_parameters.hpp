
#pragma once
#include <vector>
#include <string>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

/// @brief application parameteres, default or passed through command line
class ApplicationParameters
{
public:
    /// @brief constructor reads input from command line
    /// @param argc
    /// @param argv
    ApplicationParameters(int argc, char* argv[]);

    std::vector<std::string> input_paths;
    std::string graph_output_path;
    bool all_fasta            = true;
    bool msa                  = false; // consensus by default
    bool banded               = true;
    int32_t band_width        = 256; // Band width for banded mode
    int32_t max_groups        = -1;  // -1 => infinite
    int32_t mismatch_score    = -6;
    int32_t gap_score         = -8;
    int32_t match_score       = 8;
    double gpu_mem_allocation = 0.9;

private:
    /// \brief verifies input file formats
    /// \param input_paths input files to verify
    void verify_input_files(std::vector<std::string>& input_paths);

    /// \brief prints cudamapper's version
    /// \param exit_on_completion
    void print_version(bool exit_on_completion = true);

    /// \brief prints help message
    /// \param exit_code
    [[noreturn]] void help(int32_t exit_code = 0);
};

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
