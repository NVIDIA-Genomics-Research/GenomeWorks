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

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <cassert>

#include "../../src/allocate_block.hpp" // for estimate_max_poas()

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

/// \brief Parses window data file
///
/// \param[out] windows Reference to vector into which parsed window
///                     data is saved
/// \param[in] filename Name of file with window data
/// \param[in] total_windows Limit windows read to total windows, or
///                          loop over existing windows to fill remaining spots.
///                          -1 ignored the total_windows arg and uses all windows in the file.
inline void parse_window_data_file(std::vector<std::vector<std::string>>& windows, const std::string& filename, int32_t total_windows)
{
    std::ifstream infile(filename);
    if (!infile.good())
    {
        throw std::runtime_error("Cannot read file " + filename);
    }
    std::string line;
    int32_t num_sequences = 0;
    while (std::getline(infile, line))
    {
        if (num_sequences == 0)
        {
            std::istringstream iss(line);
            iss >> num_sequences;
            windows.emplace_back(std::vector<std::string>());
        }
        else
        {
            windows.back().push_back(line);
            num_sequences--;
        }
    }

    if (total_windows >= 0)
    {
        if (windows.size() > total_windows)
        {
            windows.erase(windows.begin() + total_windows, windows.end());
        }
        else if (windows.size() < total_windows)
        {
            int32_t windows_read = windows.size();
            while (windows.size() != total_windows)
            {
                windows.push_back(windows[windows.size() - windows_read]);
            }
        }

        assert(windows.size() == total_windows);
    }
}

/// \brief Parses golden value file with genome
///
/// \param[in] filename Name of file with reference genome
///
/// \return Genome string
inline std::string parse_golden_value_file(const std::string& filename)
{
    std::ifstream infile(filename);
    if (!infile.good())
    {
        throw std::runtime_error("Cannot read file " + filename);
    }

    std::string line;
    std::getline(infile, line);
    return line;
}

/// \brief Create a small set of batch-sizes to increase GPU parallelization.
///        Input windows are grouped based on their sizes. Similarly sized windows are grouped in the same bin.
///        Separating smaller windows from large ones allows to launch a larger number of windows concurrently.
///        This increase in parallelization translates to faster runtime
///        This multi-batch strategy is particularly useful when input windows sizes display a large variance.
///
/// \param list_of_batch_sizes [out] a set of batch-sizes, covering all input windows
/// \param list_of_windows_per_batch [out] corresponding windows per batch-size
/// \param windows [in] vector of input windows
/// \param banded_alignment [in] flag indicating whether banded-alignment or full-alignment is going to be used, default is banded-alignment
/// \param msa_flag [in] flag indicating whether MSA or consensus is going to be computed, default is consensus
/// \param bins_capacity [in] pointer to vector of bins used to create separate different-sized windows, if null as input, a set of default bins will be used
/// \param gpu_memory_usage_quota [in] portion of GPU available memory that will be used for compute each cudaPOA batch, default 0.9
/// \param mismatch_score [in] mismatch score, default -6
/// \param gap_score [in] gap score, default -8
/// \param match_score [in] match core, default 8
void get_multi_batch_sizes(std::vector<BatchSize>& list_of_batch_sizes,
                           std::vector<std::vector<int32_t>>& list_of_windows_per_batch,
                           const std::vector<std::vector<std::string>>& windows,
                           bool banded_alignment               = true,
                           bool msa_flag                       = false,
                           int32_t band_width                  = 256,
                           std::vector<int32_t>* bins_capacity = nullptr,
                           float gpu_memory_usage_quota        = 0.9,
                           int32_t mismatch_score              = -6,
                           int32_t gap_score                   = -8,
                           int32_t match_score                 = 8)
{
    // go through all the windows and evaluate maximum number of POAs of that size where can be processed in a single batch
    int32_t num_windows = get_size(windows);
    std::vector<int32_t> max_poas(num_windows);    // maximum number of POAs that canrun in parallel for windows of this size
    std::vector<int32_t> max_lengths(num_windows); // maximum sequence length within the window

    for (int32_t i = 0; i < num_windows; i++)
    {
        int32_t max_read_length = 0;
        for (auto& seq : windows[i])
        {
            max_read_length = std::max(max_read_length, get_size<int32_t>(seq) + 1);
        }
        max_poas[i]    = BatchBlock<int32_t, int32_t>::estimate_max_poas(BatchSize(max_read_length, windows[i].size(), band_width),
                                                                      banded_alignment, msa_flag,
                                                                      gpu_memory_usage_quota,
                                                                      mismatch_score, gap_score, match_score);
        max_lengths[i] = max_read_length;
    }

    int32_t num_bins = 20;
    if (bins_capacity != nullptr)
    {
        num_bins = get_size<int32_t>(*bins_capacity);
    }

    // create histogram based on number of max POAs
    std::vector<int32_t> bins_frequency(num_bins, 0);             // represents the count of windows that fall within the corresponding bin
    std::vector<int32_t> bins_max_length(num_bins, 0);            // represents the max sequence length of the largest window in the bin
    std::vector<int32_t> bins_num_reads(num_bins, 0);             // represents the number of reads of the largest window in the bin
    std::vector<int32_t> default_bins(num_bins, 1);               // represents maximum POAs of the bin
    std::vector<std::vector<int32_t>> bins_window_list(num_bins); // list of windows that are added to each bin

    // default bins, if not provided as input
    for (int32_t j = 1; j < num_bins; j++)
    {
        default_bins[j] = default_bins[j - 1] * 2;
    }
    if (bins_capacity == nullptr)
    {
        bins_capacity = &default_bins;
    }

    // go through all windows and keep track of the bin they fit in
    for (int32_t i = 0; i < num_windows; i++)
    {
        int32_t current_window = max_lengths[i] * windows[i].size();
        for (int32_t j = 0; j < num_bins; j++)
        {
            if (max_poas[i] <= bins_capacity->at(j) || j == num_bins - 1)
            {
                bins_frequency[j]++;
                bins_window_list[j].push_back(i);
                int32_t largest_window = bins_max_length[j] * bins_num_reads[j];
                if (largest_window < current_window)
                {
                    bins_max_length[j] = max_lengths[i];
                    bins_num_reads[j]  = windows[i].size();
                }
                break;
            }
        }
    }

    // a bin of capacity N means a batch made from this bin can launch up to N POAs. If the sum of bins frequency of higher capacities
    // is smaller than N, they can all fit in batch N and no need to create extra batches.
    // For example. consider the following:
    //
    // bins capacity    1        2       4       8       16      32      64      128     256     512
    // bins frequency   0        0       0       0       0       0       10      51      0       0
    // bins width       0        0       0       0       0       0       5120    3604    0       0
    //
    // Note that bins_capacity represent max POAs that can fit per bin. This means larger capacity bins correspond with smaller windows, meaning that windows
    // of larger capacity bins can be processed in lower capacity bins. This is the idea behind merging bins into each other where applicable.
    // In the example above, to process 10 windows that fall within bin capacity 64, we need to create one batch. This batch can process up to 64 windows of
    // max length 5120 or smaller. This means all the windows in bin capacity 128 can also be processed with the same batch and no need to launch an extra batch

    // loop to merge bins
    for (int32_t j = 0; j < num_bins; j++)
    {
        if (bins_frequency[j] > 0)
        {
            list_of_batch_sizes.emplace_back(bins_max_length[j], bins_num_reads[j]);
            list_of_windows_per_batch.push_back(bins_window_list[j]);
            // check if windows in the following bins can be merged into the current bin
            for (int32_t k = j + 1; k < num_bins; k++)
            {
                if (bins_frequency[k] > 0)
                {
                    if (bins_capacity->at(j) >= bins_frequency[k])
                    {
                        auto& list_of_windows_in_current_batch = list_of_windows_per_batch.back();
                        list_of_windows_in_current_batch.insert(list_of_windows_in_current_batch.end(), bins_window_list[k].begin(), bins_window_list[k].end());
                        bins_frequency[k] = 0;
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }
    }
}

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
