/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "../benchmarks/common/utils.hpp"

#include <file_location.hpp>
#include <claragenomics/cudapoa/cudapoa.hpp>
#include <claragenomics/cudapoa/batch.hpp>
#include <claragenomics/utils/signed_integer_utils.hpp>
#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/utils/genomeutils.hpp>

#include "spoa/spoa.hpp"

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <unistd.h>
#include <random>
#include <iomanip>

using namespace claragenomics;
using namespace claragenomics::cudapoa;

std::unique_ptr<Batch> initialize_batch(bool msa, const BatchSize& batch_size)
{
    // Get device information.
    int32_t device_count = 0;
    CGA_CU_CHECK_ERR(cudaGetDeviceCount(&device_count));
    assert(device_count > 0);

    size_t total = 0, free = 0;
    cudaSetDevice(0); // Using first GPU for sample.
    cudaMemGetInfo(&free, &total);

    // Initialize internal logging framework.
    Init();

    // Initialize CUDAPOA batch object for batched processing of POAs on the GPU.
    const int32_t device_id      = 0;
    cudaStream_t stream          = 0;
    size_t mem_per_batch         = 0.9 * free; // Using 90% of GPU available memory for CUDAPOA batch.
    const int32_t mismatch_score = -6, gap_score = -8, match_score = 8;
    bool banded_alignment = false;

    std::unique_ptr<Batch> batch = create_batch(device_id,
                                                stream,
                                                mem_per_batch,
                                                msa ? OutputType::msa : OutputType::consensus,
                                                batch_size,
                                                gap_score,
                                                mismatch_score,
                                                match_score,
                                                banded_alignment);

    return std::move(batch);
}

void process_batch(Batch* batch, bool msa, bool print)
{
    batch->generate_poa();

    StatusType status = StatusType::success;
    if (msa)
    {
        // Grab MSA results for all POA groups in batch.
        std::vector<std::vector<std::string>> msa; // MSA per group
        std::vector<StatusType> output_status;     // Status of MSA generation per group

        status = batch->get_msa(msa, output_status);
        if (status != StatusType::success)
        {
            std::cerr << "Could not generate MSA for batch : " << status << std::endl;
        }

        for (int32_t g = 0; g < get_size(msa); g++)
        {
            if (output_status[g] != StatusType::success)
            {
                std::cerr << "Error generating  MSA for POA group " << g << ". Error type " << output_status[g] << std::endl;
            }
            else
            {
                if (print)
                {
                    for (const auto& alignment : msa[g])
                    {
                        std::cout << alignment << std::endl;
                    }
                }
            }
        }
    }
    else
    {
        // Grab consensus results for all POA groups in batch.
        std::vector<std::string> consensus;          // Consensus string for each POA group
        std::vector<std::vector<uint16_t>> coverage; // Per base coverage for each consensus
        std::vector<StatusType> output_status;       // Status of consensus generation per group

        status = batch->get_consensus(consensus, coverage, output_status);
        if (status != StatusType::success)
        {
            std::cerr << "Could not generate consensus for batch : " << status << std::endl;
        }

        for (int32_t g = 0; g < get_size(consensus); g++)
        {
            if (output_status[g] != StatusType::success)
            {
                std::cerr << "Error generating consensus for POA group " << g << ". Error type " << output_status[g] << std::endl;
            }
            else
            {
                if (print)
                {
                    std::cout << consensus[g] << std::endl;
                }
            }
        }
    }
}

void spoa_compute(const std::vector<std::vector<std::string>>& groups, const int32_t start_id, const int32_t end_id, bool msa, bool print)
{
    spoa::AlignmentType atype = spoa::AlignmentType::kNW;
    int match_score           = 8;
    int mismatch_score        = -6;
    int gap_score             = -8;

    auto alignment_engine = spoa::createAlignmentEngine(atype, match_score, mismatch_score, gap_score);
    auto graph            = spoa::createGraph();

    if (msa)
    {
        // Grab MSA results for all groups within the range
        std::vector<std::vector<std::string>> msa(end_id - start_id); // MSA per group

        for (int32_t g = start_id; g < end_id; g++)
        {
            for (const auto& it : groups[g])
            {
                auto alignment = alignment_engine->align(it, graph);
                graph->add_alignment(alignment, it);
            }
            graph->generate_multiple_sequence_alignment(msa[g - start_id]);
        }

        if (print)
        {
            std::cout << std::endl;
            for (int32_t i = 0; i < get_size(msa); i++)
            {
                {
                    for (const auto& alignment : msa[i])
                    {
                        std::cout << alignment << std::endl;
                    }
                }
            }
        }
    }
    else
    {
        // Grab consensus results for all POA groups within the range
        std::vector<std::string> consensus(end_id - start_id);          // Consensus string for each POA group
        std::vector<std::vector<uint32_t>> coverage(end_id - start_id); // Per base coverage for each consensus

        for (int32_t g = start_id; g < end_id; g++)
        {
            for (const auto& it : groups[g])
            {
                auto alignment = alignment_engine->align(it, graph);
                graph->add_alignment(alignment, it);
            }
            consensus[g - start_id] = graph->generate_consensus(coverage[g - start_id]);
        }

        if (print)
        {
            std::cout << std::endl;
            for (int32_t i = 0; i < get_size(consensus); i++)
            {
                std::cout << consensus[i] << std::endl;
            }
        }
    }
}

void generate_short_reads(std::vector<std::vector<std::string>>& windows, BatchSize& batch_size,
                          const int32_t number_of_windows = 1000, const int32_t sequence_size = 1024, const int32_t group_size = 100)
{
    const std::string input_data = std::string(CUDAPOA_BENCHMARK_DATA_DIR) + "/sample-windows.txt";
    parse_window_data_file(windows, input_data, number_of_windows); // Generate windows.
    assert(get_size(windows) > 0);
    batch_size = BatchSize(sequence_size, group_size);
}

void generate_bonito_long_reads(std::vector<std::vector<std::string>>& windows, BatchSize& batch_size,
                                const int32_t number_of_windows = 5, const int32_t sequence_size = 20000, const int32_t group_size = 6)
{
    const std::string input_data = std::string(CUDAPOA_BENCHMARK_DATA_DIR) + "/sample-bonito.txt";
    parse_window_data_file(windows, input_data, number_of_windows); // Generate windows.
    assert(get_size(windows) > 0);

    int32_t max_read_length = 0;
    for (auto& window : windows)
    {
        for (auto& seq : window)
        {
            max_read_length = std::max(max_read_length, get_size<int>(seq) + 1);
        }
    }

    assert(sequence_size >= max_read_length);

    batch_size = BatchSize(max_read_length, group_size);
}

void generate_simulated_long_reads(std::vector<std::vector<std::string>>& windows, BatchSize& batch_size,
                                   const int32_t number_of_windows = 2, const int32_t sequence_size = 10000, const int32_t group_size = 5)
{
    constexpr uint32_t random_seed = 5827349;
    std::minstd_rand rng(random_seed);

    int32_t max_sequence_length = sequence_size + 1;

    std::vector<std::pair<int, int>> variation_ranges;
    variation_ranges.push_back(std::pair<int, int>(30, 50));
//    variation_ranges.push_back(std::pair<int, int>(300, 500));
//    variation_ranges.push_back(std::pair<int, int>(1000, 1300));
//    variation_ranges.push_back(std::pair<int, int>(2000, 2200));
//    variation_ranges.push_back(std::pair<int, int>(3000, 3500));
//    variation_ranges.push_back(std::pair<int, int>(4000, 4200));
//    variation_ranges.push_back(std::pair<int, int>(5000, 5400));
//    variation_ranges.push_back(std::pair<int, int>(6000, 6200));
//    variation_ranges.push_back(std::pair<int, int>(8000, 8300));

    std::vector<std::string> long_reads(group_size);

    for (int w = 0; w < number_of_windows; w++)
    {
        long_reads[0] = claragenomics::genomeutils::generate_random_genome(sequence_size, rng);
        for (int i = 1; i < group_size; i++)
        {
            long_reads[i]       = claragenomics::genomeutils::generate_random_sequence(long_reads[0], rng, sequence_size, sequence_size, sequence_size, &variation_ranges);
            max_sequence_length = max_sequence_length > get_size(long_reads[i]) ? max_sequence_length : get_size(long_reads[i]) + 1;
        }
        // add long reads as one window
        windows.push_back(long_reads);
    }

    // Define upper limits for sequence size, graph size ....
    batch_size = BatchSize(max_sequence_length, sequence_size);
}

int main(int argc, char** argv)
{
    // Process options
    int c            = 0;
    bool msa         = false;
    bool long_read   = false;
    bool help        = false;
    bool print       = false;
    bool print_graph = false;
    bool benchmark   = false;

    // following parameters are used in benchmarking only
    int32_t number_of_windows = 0;
    int32_t sequence_size     = 0;
    int32_t group_size        = 0;

    while ((c = getopt(argc, argv, "mlhpgbW:S:N:")) != -1)
    {
        switch (c)
        {
        case 'm':
            msa = true;
            break;
        case 'l':
            long_read = true;
            break;
        case 'p':
            print = true;
            break;
        case 'g':
            print_graph = true;
            break;
        case 'b':
            benchmark = true;
            break;
        case 'W':
            number_of_windows = atoi(optarg);
            break;
        case 'S':
            sequence_size = atoi(optarg);
            break;
        case 'N':
            group_size = atoi(optarg);
            break;
        case 'h':
            help = true;
            break;
        }
    }

    if (help)
    {
        std::cout << "CUDAPOA API sample program. Runs consensus or MSA generation on pre-canned data." << std::endl;
        std::cout << "Usage:" << std::endl;
        std::cout << "./sample_cudapoa [-m] [-h]" << std::endl;
        std::cout << "-m : Generate MSA (if not provided, generates consensus by default)" << std::endl;
        std::cout << "-l : Perform long-read sample (if not provided, will run short-read sample by default)" << std::endl;
        std::cout << "-p : Print the MSA or consensus output to stdout" << std::endl;
        std::cout << "-g : Print POA graph in dot format, this option is only for long-read sample" << std::endl;
        std::cout << "-b : Benchmark against SPOA" << std::endl;
        std::cout << "-W : Number of total windows used in benchmarking" << std::endl;
        std::cout << "-S : Maximum sequence length in benchmarking" << std::endl;
        std::cout << "-N : Number of sequences per POA group" << std::endl;
        std::cout << "-h : Print help message" << std::endl;
        std::exit(0);
    }

    // if not defined as input args, set default values for benchmarking parameters
    number_of_windows = number_of_windows == 0 ? (long_read ? 10 : 1000) : number_of_windows;
    sequence_size     = sequence_size == 0 ? (long_read ? 10000 : 1024) : sequence_size;
    group_size        = group_size == 0 ? (long_read ? 6 : 100) : group_size;

    // Load input data. Each POA group is represented as a vector of strings. The sample
    // data for short reads has many such POA groups to process, hence the data is loaded into a vector
    // of vector of strings. Long read sample creates one POA group.
    std::vector<std::vector<std::string>> windows;

    // Define upper limits for sequence size, graph size ....
    BatchSize batch_size;

    if (benchmark)
    {
        if (long_read)
        {
            generate_simulated_long_reads(windows, batch_size, number_of_windows, sequence_size, group_size);
            //generate_bonito_long_reads(windows, batch_size, sequence_size, group_size);
        }
        else
        {
            generate_short_reads(windows, batch_size, sequence_size, group_size);
        }
    }
    else
    {
        if (long_read)
        {
            generate_bonito_long_reads(windows, batch_size);
        }
        else
        {
            generate_short_reads(windows, batch_size);
        }
    }

    // Initialize batch.
    std::unique_ptr<Batch> batch = initialize_batch(msa, batch_size);

    // Loop over all the POA groups, add them to the batch and process them.
    int32_t window_count = 0;
    // to avoid potential infinite loop
    int32_t error_count = 0;
    // for benchmarking
    float cudapoa_time = 0.f;
    float spoa_time    = 0.f;
    ChronoTimer timer;

    for (int32_t i = 0; i < get_size(windows);)
    {
        const std::vector<std::string>& window = windows[i];

        Group poa_group;
        // Create a new entry for each sequence and add to the group.
        for (const auto& seq : window)
        {
            Entry poa_entry{};
            poa_entry.seq     = seq.c_str();
            poa_entry.length  = seq.length();
            poa_entry.weights = nullptr;
            poa_group.push_back(poa_entry);
        }

        std::vector<StatusType> seq_status;
        StatusType status = batch->add_poa_group(seq_status, poa_group);

        // NOTE: If number of windows smaller than batch capacity, then run POA generation
        // once last window is added to batch.
        if (status == StatusType::exceeded_maximum_poas || status == StatusType::exceeded_batch_size || (i == get_size(windows) - 1))
        {
            // No more POA groups can be added to batch. Now process batch.
            if (benchmark)
            {
                timer.start_timer();
                process_batch(batch.get(), msa, print);
                cudapoa_time += timer.stop_timer();

                timer.start_timer();
                spoa_compute(windows, window_count, window_count + batch->get_total_poas(), msa, print);
                spoa_time += timer.stop_timer();
            }
            else
            {
                process_batch(batch.get(), msa, print);
            }

            if (print_graph && long_read)
            {
                std::vector<DirectedGraph> graph;
                std::vector<StatusType> graph_status;
                batch->get_graphs(graph, graph_status);
                for (auto& g : graph)
                {
                    std::cout << g.serialize_to_dot() << std::endl;
                }
            }

            // After MSA/consensus is generated for batch, reset batch to make room for next set of POA groups.
            batch->reset();

            if (status == StatusType::success)
            {
                std::cout << "Processed windows " << window_count << " - " << i << std::endl;
            }
            else
            {
                std::cout << "Processed windows " << window_count << " - " << i - 1 << std::endl;
            }

            window_count = i;
        }

        if (status == StatusType::success)
        {
            // Check if all sequences in POA group wre added successfully.
            for (const auto& s : seq_status)
            {
                if (s == StatusType::exceeded_maximum_sequence_size)
                {
                    std::cerr << "Dropping sequence because sequence exceeded maximum size" << std::endl;
                }
            }
            i++;
        }

        if (status != StatusType::exceeded_maximum_poas && status != StatusType::exceeded_batch_size && status != StatusType::success)
        {
            std::cerr << "Could not add POA group to batch. Error code " << status << std::endl;
            error_count++;
            if (error_count > get_size(windows))
                break;
        }
    }

    if (benchmark)
    {
        std::cerr << "benchmark summary:\n";
        std::cerr << "=========================================================================================================\n";
        std::cerr << "Number of windows(W) " << std::left << std::setw(13) << std::setfill(' ') << number_of_windows;
        std::cerr << "Sequence lengthv(S) " << std::left << std::setw(10) << std::setfill(' ') << sequence_size;
        std::cerr << "Number of sequences per window(N) " << std::left << std::setw(30) << std::setfill(' ') << group_size << std::endl;
        std::cerr << "Compute time:                     cudaPOA " << cudapoa_time << "(sec),           SPOA " << spoa_time << "(sec)\n";
        int32_t number_of_bases = number_of_windows * sequence_size * group_size;
        std::cerr << "Expected performance:             cudaPOA " << (float)number_of_bases / cudapoa_time << "(bases/sec),    SPOA ";
        std::cerr << (float)number_of_bases / spoa_time << "(bases/sec)" << std::endl;
        int32_t actual_number_of_bases = 0;
        for (auto& w : windows)
        {
            for (auto& seq : w)
            {
                actual_number_of_bases += get_size(seq);
            }
        }
        std::cerr << "Effective performance:            cudaPOA " << (float)actual_number_of_bases / cudapoa_time << "(bases/sec),    SPOA ";
        std::cerr << (float)actual_number_of_bases / spoa_time << "(bases/sec)" << std::endl;
        std::cerr << "Expected number of bases (S x N x W) = " << number_of_bases << std::endl;
        std::cerr << "Actual total number of bases         = " << actual_number_of_bases << std::endl;
        std::cerr << "=========================================================================================================\n";
    }

    return 0;
}
