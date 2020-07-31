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

#include "allocate_block.hpp"

#include <claraparabricks/genomeworks/cudapoa/utils.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

void get_multi_batch_sizes(std::vector<BatchConfig>& list_of_batch_sizes,
                           std::vector<std::vector<int32_t>>& list_of_groups_per_batch,
                           const std::vector<Group>& poa_groups,
                           bool msa_flag /*= false*/,
                           int32_t band_width /*= 256*/,
                           BandMode band_mode /*= adaptive_band*/,
                           std::vector<int32_t>* bins_capacity /*= nullptr*/,
                           float gpu_memory_usage_quota /*= 0.9*/,
                           int32_t mismatch_score /*= -6*/,
                           int32_t gap_score /*= -8*/,
                           int32_t match_score /*= 8*/)
{
    // go through all the POA groups and evaluate maximum number of POAs of that size where can be processed in a single batch
    int32_t num_groups = get_size(poa_groups);
    std::vector<int32_t> max_poas(num_groups);    // maximum number of POAs that can run in parallel for poa_groups of this size
    std::vector<int32_t> max_lengths(num_groups); // maximum sequence length within the group

    for (int32_t i = 0; i < num_groups; i++)
    {
        int32_t max_read_length = 0;
        for (auto& entry : poa_groups[i])
        {
            max_read_length = std::max(max_read_length, entry.length);
        }
        max_poas[i]    = BatchBlock<int32_t, int32_t>::estimate_max_poas(BatchConfig(max_read_length, get_size<int32_t>(poa_groups[i]), band_width, band_mode),
                                                                      msa_flag, gpu_memory_usage_quota,
                                                                      mismatch_score, gap_score, match_score);
        max_lengths[i] = max_read_length;
    }

    int32_t num_bins = 20;
    if (bins_capacity != nullptr)
    {
        num_bins = get_size<int32_t>(*bins_capacity);
    }

    // create histogram based on number of max POAs
    std::vector<int32_t> bins_frequency(num_bins, 0);            // represents the count of poa_groups that fall within the corresponding bin
    std::vector<int32_t> bins_max_length(num_bins, 0);           // represents the max sequence length of the largest POA group in the bin
    std::vector<int32_t> bins_num_reads(num_bins, 0);            // represents the number of reads of the largest POA group in the bin
    std::vector<int32_t> default_bins(num_bins, 1);              // represents maximum POAs of the bin
    std::vector<std::vector<int32_t>> bins_group_list(num_bins); // list of poa_groups that are added to each bin

    // default bins, if not provided as input
    for (int32_t j = 1; j < num_bins; j++)
    {
        default_bins[j] = default_bins[j - 1] * 2;
    }
    if (bins_capacity == nullptr)
    {
        bins_capacity = &default_bins;
    }

    // go through all poa_groups and keep track of the bin they fit in
    for (int32_t i = 0; i < num_groups; i++)
    {
        int32_t current_group = max_lengths[i] * poa_groups[i].size();
        for (int32_t j = 0; j < num_bins; j++)
        {
            if (max_poas[i] <= bins_capacity->at(j) || j == num_bins - 1)
            {
                bins_frequency[j]++;
                bins_group_list[j].push_back(i);
                int32_t largest_group = bins_max_length[j] * bins_num_reads[j];
                if (largest_group < current_group)
                {
                    bins_max_length[j] = max_lengths[i];
                    bins_num_reads[j]  = poa_groups[i].size();
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
    // Note that bins_capacity represent max POAs that can fit per bin. This means larger capacity bins correspond with smaller poa_groups, meaning that poa_groups
    // of larger capacity bins can be processed in lower capacity bins. This is the idea behind merging bins into each other where applicable.
    // In the example above, to process 10 poa_groups that fall within bin capacity 64, we need to create one batch. This batch can process up to 64 poa_groups of
    // max length 5120 or smaller. This means all the poa_groups in bin capacity 128 can also be processed with the same batch and no need to launch an extra batch

    // loop to merge bins
    for (int32_t j = 0; j < num_bins; j++)
    {
        if (bins_frequency[j] > 0)
        {
            list_of_batch_sizes.emplace_back(bins_max_length[j], bins_num_reads[j], band_width, band_mode);
            list_of_groups_per_batch.push_back(bins_group_list[j]);
            // check if poa_groups in the following bins can be merged into the current bin
            for (int32_t k = j + 1; k < num_bins; k++)
            {
                if (bins_frequency[k] > 0)
                {
                    if (bins_capacity->at(j) >= bins_frequency[k])
                    {
                        auto& list_of_groups_in_current_batch = list_of_groups_per_batch.back();
                        list_of_groups_in_current_batch.insert(list_of_groups_in_current_batch.end(), bins_group_list[k].begin(), bins_group_list[k].end());
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
