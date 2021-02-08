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

#include <claraparabricks/genomeworks/cudapoa/utils.hpp>
#include <claraparabricks/genomeworks/cudapoa/batch.hpp>
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{
class SingleBatch
{

    /// \class SingleBatch
    /// Class encapsulating running single cudapoa Batch classes with varying batch size.
public:
    /// \brief Construct a single batch processor
    ///
    /// \param max_poas_per_batch Batch size
    /// \param filename Filename with window data
    SingleBatch(int32_t max_poas_per_batch, const std::string& filename, int32_t total_windows)
        : stream_(make_cuda_stream())
        , max_poas_per_batch_(max_poas_per_batch)
    {
        parse_cudapoa_file(windows_, filename, total_windows);

        assert(get_size(windows_) > 0);

        size_t total = 0, free = 0;
        cudaSetDevice(0);
        cudaMemGetInfo(&free, &total);
        size_t mem_per_batch = 0.9 * free;

        BatchConfig batch_size(1024, 200);

        batch_ = create_batch(0, stream_.get(), mem_per_batch, OutputType::consensus, batch_size, -8, -6, 8);
    }

    ~SingleBatch()
    {
        batch_.reset();
    }

    /// \brief Add windows to the batch class
    void add_windows()
    {
        batch_->reset();

        int32_t total_windows = get_size(windows_);
        for (int32_t i = 0; i < max_poas_per_batch_; i++)
        {
            Group poa_group;
            const auto& window = windows_[i % total_windows];
            for (int32_t s = 0; s < get_size(window); s++)
            {
                Entry e{};
                e.seq     = window[s].c_str();
                e.weights = NULL;
                e.length  = window[s].length();
                poa_group.push_back(e);
            }
            std::vector<StatusType> error_status;
            batch_->add_poa_group(error_status, poa_group);
        }
    }

    /// \brief Process POA and generate consensus
    void process_consensus()
    {
        batch_->generate_poa();
        std::vector<std::string> consensus;
        std::vector<std::vector<uint16_t>> coverage;
        std::vector<genomeworks::cudapoa::StatusType> output_status;
        batch_->get_consensus(consensus, coverage, output_status);
    }

private:
    std::unique_ptr<Batch> batch_;
    CudaStream stream_;
    std::vector<std::vector<std::string>> windows_;
    int32_t max_poas_per_batch_;
};
} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
