/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <future>
#include <numeric>
#include "cudapoa/batch.hpp"
#include "../common/utils.hpp"

namespace genomeworks
{
namespace cudapoa
{

/// \class MultiBatch
/// Class encapsulating running multiple concurrent cudapoa Batch classes.
class MultiBatch
{
public:
    /// \brief Construct a multi batch processor
    ///
    /// \param num_batches Number of cudapoa batches
    /// \param max_poas_per_batch Batch size
    /// \param filename Filename with window data
    MultiBatch(int32_t num_batches, int32_t max_poas_per_batch, const std::string& filename, int32_t total_windows = -1)
        : num_batches_(num_batches)
        , max_poas_per_batch_(max_poas_per_batch)
    {
        parse_window_data_file(windows_, filename, total_windows);

        assert(windows.size() > 0);

        for (int32_t batch = 0; batch < num_batches_; batch++)
        {
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            batches_.emplace_back(create_batch(max_poas_per_batch, 200, 0));
            batches_.back()->set_cuda_stream(stream);
        }
    }

    ~MultiBatch()
    {
        batches_.clear();
    }

    /// \brief Process the batches in concurrent fashion
    void process_batches()
    {
        // Reserve space
        consensus_.resize(windows_.size());
        coverages_.resize(windows_.size());

        // Mutex for accessing the vector of windows.
        std::mutex mutex_windows;

        // Index of next window to be added to a batch.
        uint32_t next_window_index = 0;

        // Lambda function for adding windows to batches.
        auto fill_next_batch = [&mutex_windows, &next_window_index, this](Batch* batch) -> std::pair<uint32_t, uint32_t> {
            // Use mutex to read the vector containing windows in a threadsafe manner.
            std::lock_guard<std::mutex> guard(mutex_windows);

            // TODO: Reducing window wize by 10 for debugging.
            uint32_t initial_count = next_window_index;
            uint32_t count         = windows_.size();
            int32_t num_inserted   = 0;
            while (next_window_index < count)
            {
                if (num_inserted < max_poas_per_batch_)
                {
                    next_window_index++;
                    num_inserted++;
                }
                else
                {
                    break;
                }
            }

            return {initial_count, next_window_index};
        };

        // Lambda function for processing each batch.
        auto process_batch = [&fill_next_batch, this](Batch* batch) -> void {
            while (true)
            {
                batch->reset();

                std::pair<uint32_t, uint32_t> range = fill_next_batch(batch);
                for (int32_t i = range.first; i < range.second; i++)
                {
                    if (batch->add_poa() == StatusType::success)
                    {
                        const std::vector<std::string>& window = windows_[i];
                        for (auto& seq : window)
                        {
                            batch->add_seq_to_poa(seq.c_str(), NULL, seq.length());
                        }
                    }
                    else
                    {
                        throw std::runtime_error("Could not add new POA to batch");
                    }
                }
                if (batch->get_total_poas() > 0)
                {
                    std::vector<std::string> consensus_temp;
                    std::vector<std::vector<uint16_t>> coverages_temp;
                    std::vector<genomeworks::cudapoa::StatusType> output_status;

                    // Launch workload.
                    batch->generate_poa();
                    batch->get_consensus(consensus_temp, coverages_temp, output_status);

                    // Check if the number of batches processed is same as the range of
                    // of windows that were added.
                    if (consensus_temp.size() != (range.second - range.first))
                    {
                        throw std::runtime_error("Consensus processed doesn't match \
                                range of windows passed to batch\n");
                    }
                    if (coverages_temp.size() != (range.second - range.first))
                    {
                        throw std::runtime_error("Coverages processed doesn't match \
                                range of windows passed to batch\n");
                    }
                    for (int32_t i = range.first; i < range.second; i++)
                    {
                        consensus_[i] = consensus_temp[i - range.first];
                        coverages_[i] = coverages_temp[i - range.first];
                    }
                }
                else
                {
                    break;
                }
            }
        };

        // Process each of the batches in a separate thread.
        std::vector<std::future<void>> thread_futures;
        for (auto& batch : batches_)
        {
            thread_futures.emplace_back(std::async(std::launch::async,
                                                   process_batch,
                                                   batch.get()));
        }

        // Wait for threads to finish, and collect their results.
        for (const auto& future : thread_futures)
        {
            future.wait();
        }
    }

    /// \brief Concatenate consensus of finished POA windows to generate
    ///        final assembly. The coverage values is also used to trim the
    ///        assembly based on average covarge across the window.
    std::string assembly()
    {
        std::string genome = "";
        for (int32_t w = 0; w < windows_.size(); w++)
        {
            int32_t average_coverage = std::accumulate(coverages_[w].begin(),
                                                       coverages_[w].end(),
                                                       0) /
                                       coverages_[w].size();
            int32_t begin = 0, end = consensus_[w].length() - 1;
            for (; begin < static_cast<int32_t>(coverages_[w].size()); ++begin)
            {
                if (coverages_[w][begin] >= average_coverage)
                {
                    break;
                }
            }
            for (; end >= 0; --end)
            {
                if (coverages_[w][end] >= average_coverage)
                {
                    break;
                }
            }

            if (begin < end)
            {
                genome += consensus_[w].substr(begin, end - begin + 1);
            }
        }
        return genome;
    }

private:
    int32_t num_batches_;
    int32_t max_poas_per_batch_;
    std::vector<std::unique_ptr<Batch>> batches_;
    std::vector<std::vector<std::string>> windows_;
    std::vector<std::string> consensus_;
    std::vector<std::vector<uint16_t>> coverages_;
};
} // namespace cudapoa
} // namespace genomeworks
