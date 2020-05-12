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
#include "../src/cudapoa_kernels.cuh" // for estimate_max_poas()

#include <file_location.hpp>
#include <claragenomics/cudapoa/cudapoa.hpp>
#include <claragenomics/cudapoa/batch.hpp>
#include <claragenomics/utils/signed_integer_utils.hpp>
#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/utils/genomeutils.hpp>

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <unistd.h>
#include <random>

using namespace claragenomics;
using namespace claragenomics::cudapoa;

std::unique_ptr<Batch> initialize_batch(bool msa, bool banded_alignment, const BatchSize& batch_size)
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

void generate_window_data(const std::string& input_file, const int number_of_windows, const int max_sequences_per_poa,
                          std::vector<std::vector<std::string>>& windows, BatchSize& batch_size)
{
    parse_window_data_file(windows, input_file, number_of_windows); // Generate windows.
    assert(get_size(windows) > 0);

    int32_t max_read_length = 0;
    for (auto& window : windows)
    {
        for (auto& seq : window)
        {
            max_read_length = std::max(max_read_length, get_size<int>(seq));
        }
    }

    batch_size = BatchSize(max_read_length, max_sequences_per_poa);
}

size_t estimate_max_poas(const BatchSize& batch_size, const bool banded_alignment, const bool msa_flag)
{
    int32_t matrix_sequence_dimension = banded_alignment ? CUDAPOA_BANDED_MAX_MATRIX_SEQUENCE_DIMENSION : batch_size.max_matrix_sequence_dimension;
    int32_t matrix_graph_dimension    = banded_alignment ? batch_size.max_matrix_graph_dimension_banded : batch_size.max_matrix_graph_dimension;
    int32_t max_nodes_per_window      = banded_alignment ? batch_size.max_nodes_per_window_banded : batch_size.max_nodes_per_window;

    // Initialize CUDAPOA batch object for batched processing of POAs on the GPU.
    size_t total = 0, free = 0;
    cudaMemGetInfo(&free, &total);
    size_t mem_per_batch         = 0.9 * free; // Using 90% of GPU available memory for CUDAPOA batch.
    const int32_t mismatch_score = -6, gap_score = -8, match_score = 8;

    int64_t sizeof_ScoreT = use32bitScore(batch_size, gap_score, mismatch_score, match_score) ? 4 : 2;
    int64_t sizeof_SizeT  = use32bitSize(batch_size, banded_alignment) ? 4 : 2;

    // Calculate memory requirements for POA arrays
    int64_t device_size_per_poa = 0;
    int64_t input_size_per_poa  = batch_size.max_sequences_per_poa * batch_size.max_sequence_size;
    int64_t output_size_per_poa = batch_size.max_concensus_size;
    device_size_per_poa += output_size_per_poa * sizeof(uint8_t);                                                                                // output_details_d_->consensus
    device_size_per_poa += (!msa_flag) ? output_size_per_poa * sizeof(uint16_t) : 0;                                                             // output_details_d_->coverage
    device_size_per_poa += (msa_flag) ? output_size_per_poa * batch_size.max_sequences_per_poa * sizeof(uint8_t) : 0;                            // output_details_d_->multiple_sequence_alignments
    device_size_per_poa += input_size_per_poa * sizeof(uint8_t);                                                                                 // input_details_d_->sequences
    device_size_per_poa += input_size_per_poa * sizeof(int8_t);                                                                                  // input_details_d_->base_weights
    device_size_per_poa += batch_size.max_sequences_per_poa * sizeof_SizeT;                                                                      // input_details_d_->sequence_lengths
    device_size_per_poa += sizeof(WindowDetails);                                                                                                // input_details_d_->window_details
    device_size_per_poa += (msa_flag) ? batch_size.max_sequences_per_poa * sizeof_SizeT : 0;                                                     // input_details_d_->sequence_begin_nodes_ids
    device_size_per_poa += sizeof(uint8_t) * max_nodes_per_window;                                                                               // graph_details_d_->nodes
    device_size_per_poa += sizeof_SizeT * max_nodes_per_window * CUDAPOA_MAX_NODE_ALIGNMENTS;                                                    // graph_details_d_->node_alignments
    device_size_per_poa += sizeof(uint16_t) * max_nodes_per_window;                                                                              // graph_details_d_->node_alignment_count
    device_size_per_poa += sizeof_SizeT * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES;                                                         // graph_details_d_->incoming_edges
    device_size_per_poa += sizeof(uint16_t) * max_nodes_per_window;                                                                              // graph_details_d_->incoming_edge_count
    device_size_per_poa += sizeof_SizeT * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES;                                                         // graph_details_d_->outgoing_edges
    device_size_per_poa += sizeof(uint16_t) * max_nodes_per_window;                                                                              // graph_details_d_->outgoing_edge_count
    device_size_per_poa += sizeof(uint16_t) * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES;                                                     // graph_details_d_->incoming_edge_weights
    device_size_per_poa += sizeof(uint16_t) * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES;                                                     // graph_details_d_->outgoing_edge_weights
    device_size_per_poa += sizeof_SizeT * max_nodes_per_window;                                                                                  // graph_details_d_->sorted_poa
    device_size_per_poa += sizeof_SizeT * max_nodes_per_window;                                                                                  // graph_details_d_->sorted_poa_node_map
    device_size_per_poa += sizeof(uint16_t) * max_nodes_per_window;                                                                              // graph_details_d_->sorted_poa_local_edge_count
    device_size_per_poa += (!msa_flag) ? sizeof(int32_t) * max_nodes_per_window : 0;                                                             // graph_details_d_->consensus_scores
    device_size_per_poa += (!msa_flag) ? sizeof_SizeT * max_nodes_per_window : 0;                                                                // graph_details_d_->consensus_predecessors
    device_size_per_poa += sizeof(int8_t) * max_nodes_per_window;                                                                                // graph_details_d_->node_marks
    device_size_per_poa += sizeof(bool) * max_nodes_per_window;                                                                                  // graph_details_d_->check_aligned_nodes
    device_size_per_poa += sizeof_SizeT * max_nodes_per_window;                                                                                  // graph_details_d_->nodes_to_visit
    device_size_per_poa += sizeof(uint16_t) * max_nodes_per_window;                                                                              // graph_details_d_->node_coverage_counts
    device_size_per_poa += (msa_flag) ? sizeof(uint16_t) * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES * batch_size.max_sequences_per_poa : 0; // graph_details_d_->outgoing_edges_coverage
    device_size_per_poa += (msa_flag) ? sizeof(uint16_t) * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES : 0;                                    // graph_details_d_->outgoing_edges_coverage_count
    device_size_per_poa += (msa_flag) ? sizeof_SizeT * max_nodes_per_window : 0;                                                                 // graph_details_d_->node_id_to_msa_pos
    device_size_per_poa += sizeof_SizeT * matrix_graph_dimension;                                                                                // alignment_details_d_->alignment_graph
    device_size_per_poa += sizeof_SizeT * matrix_graph_dimension;                                                                                // alignment_details_d_->alignment_read

    // Compute required memory for score matrix
    int64_t device_size_per_score_matrix = (int64_t)matrix_sequence_dimension * (int64_t)matrix_graph_dimension * sizeof_ScoreT;

    // Calculate max POAs possible based on available memory.
    size_t max_poas = mem_per_batch / (device_size_per_poa + device_size_per_score_matrix);

    return max_poas;
}

void generate_batch_sizes(const std::vector<std::vector<std::string>>& windows, const bool banded_alignment, const bool msa_flag,
                          std::vector<BatchSize>& list_of_batch_sizes, std::vector<std::vector<size_t>>& list_of_windows_per_batch)
{
    // got through all the windows and evaluate maximum number of POAs of that size where can be processed in a single batch
    size_t num_windows = windows.size();
    std::vector<size_t> max_poas(num_windows);    // maximum number of POAs that canrun in parallel for windows of this size
    std::vector<size_t> max_lengths(num_windows); // maximum sequence length within the window

    for (size_t i = 0; i < windows.size(); i++)
    {
        size_t max_read_length = 0;
        for (auto& seq : windows[i])
        {
            max_read_length = std::max(max_read_length, get_size<size_t>(seq) + 1);
        }
        max_poas[i]    = estimate_max_poas(BatchSize(max_read_length, windows[i].size()), banded_alignment, msa_flag);
        max_lengths[i] = max_read_length;
    }

    // create histogram based on number of max POAs
    size_t num_bins = 20;
    std::vector<size_t> bins_frequency(num_bins, 0);             // count the windows that fall within corresponding range
    std::vector<size_t> bins_max_length(num_bins, 0);            // represents the length of the window with maximum sequence length in the bin
    std::vector<size_t> bins_num_reads(num_bins, 0);             // represents the number of reads in the window with maximum sequence length in the bin
    std::vector<size_t> bins_ranges(num_bins, 1);                // represents maximum POAs
    std::vector<std::vector<size_t>> bins_window_list(num_bins); // list of windows that are added to each bin

    for (size_t j = 1; j < num_bins; j++)
    {
        bins_ranges[j] = bins_ranges[j - 1] * 2;
    }

    // go through all windows and keep track of the bin they fit
    for (size_t i = 0; i < num_windows; i++)
    {
        for (size_t j = 0; j < num_bins; j++)
        {
            if (max_poas[i] <= bins_ranges[j] || j == num_bins - 1)
            {
                bins_frequency[j]++;
                bins_window_list[j].push_back(i);
                if (bins_max_length[j] < max_lengths[i])
                {
                    bins_max_length[j] = max_lengths[i];
                    bins_num_reads[j]  = windows[i].size();
                }
                break;
            }
        }
    }

    // a bin in range N means a batch made based on this bean, can launch up to N POAs. If the sum of bins frequency of higher ranges
    // is smaller than N, they can all fit in batch N and no need to create extra batches.
    // For example. consider the following:
    //
    // bins_ranges      1        2       4       8       16      32      64      128     256     512
    // bins frequency   0        0       0       0       0       0       10      51      0       0
    // bins width       0        0       0       0       0       0       5120    3604    0       0
    //
    // note that bin_ranges represent max POAs. This means larger bin ranges follow with smaller corresponding max lengths
    // In the example above, to process 10 windows that fall within bin range 64, we need to create one batch. This batch can process up to 64 windows of
    // max length 5120 or smaller. This means all the windows in bin range 128 can also be processed with the same batch and no need to launch an extra batch

    size_t remaining_windows = num_windows;
    for (size_t j = 0; j < num_bins; j++)
    {
        if (bins_frequency[j] > 0)
        {
            list_of_batch_sizes.emplace_back(bins_max_length[j], bins_num_reads[j]);
            list_of_windows_per_batch.push_back(bins_window_list[j]);
            remaining_windows -= bins_frequency[j];
            if (bins_ranges[j] > remaining_windows)
            {
                auto& remaining_list = list_of_windows_per_batch.back();
                for (auto it = bins_window_list.begin() + j + 1; it != bins_window_list.end(); it++)
                {
                    remaining_list.insert(remaining_list.end(), it->begin(), it->end());
                }
                break;
            }
        }
    }
}

int main(int argc, char** argv)
{
    // Process options
    int c            = 0;
    bool msa         = false;
    bool long_read   = false;
    bool banded      = true;
    bool help        = false;
    bool print       = false;
    bool print_graph = false;

    while ((c = getopt(argc, argv, "mlfpgh")) != -1)
    {
        switch (c)
        {
        case 'm':
            msa = true;
            break;
        case 'l':
            long_read = true;
            break;
        case 'f':
            banded = false;
            break;
        case 'p':
            print = true;
            break;
        case 'g':
            print_graph = true;
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
        std::cout << "-f : Perform full alignment (if not provided, banded alignment is used by default)" << std::endl;
        std::cout << "-p : Print the MSA or consensus output to stdout" << std::endl;
        std::cout << "-g : Print POA graph in dot format, this option is only for long-read sample" << std::endl;
        std::cout << "-h : Print help message" << std::endl;
        std::exit(0);
    }

    // Load input data. Each POA group is represented as a vector of strings. The sample
    // data for short reads has many such POA groups to process, hence the data is loaded into a vector
    // of vector of strings. Long read sample creates one POA group.
    std::vector<std::vector<std::string>> windows;

    // Define upper limits for sequence size, graph size ....
    BatchSize batch_size;

    if (long_read)
    {
        const std::string input_file = std::string(CUDAPOA_BENCHMARK_DATA_DIR) + "/sample-bonito.txt";
        generate_window_data(input_file, 55, 6, windows, batch_size);
    }
    else
    {
        const std::string input_file = std::string(CUDAPOA_BENCHMARK_DATA_DIR) + "/sample-windows.txt";
        generate_window_data(input_file, 1000, 100, windows, batch_size);
    }

    // analyze the windows and create a minimal set of batches to process them all
    std::vector<BatchSize> list_of_batch_sizes;
    std::vector<std::vector<size_t>> list_of_windows_per_batch;
    generate_batch_sizes(windows, banded, msa, list_of_batch_sizes, list_of_windows_per_batch);

    int32_t window_count_offset = 0;

    for (size_t b = 0; b < list_of_batch_sizes.size(); b++)
    {
        auto& batch_size       = list_of_batch_sizes[b];
        auto& batch_window_ids = list_of_windows_per_batch[b];

        // Initialize batch.
        std::unique_ptr<Batch> batch = initialize_batch(msa, banded, batch_size);

        // Loop over all the POA groups for the current batch, add them to the batch and process them.
        int32_t window_count = 0;

        for (int32_t i = 0; i < get_size(batch_window_ids);)
        {
            const std::vector<std::string>& window = windows[batch_window_ids[i]];

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

            // NOTE: If number of batch windows smaller than batch capacity, then run POA generation
            // once last window is added to batch.
            if (status == StatusType::exceeded_maximum_poas || (i == get_size(batch_window_ids) - 1))
            {
                // at least one POA should have been added before processing the batch
                if (batch->get_total_poas() > 0)
                {
                    // No more POA groups can be added to batch. Now process batch.
                    process_batch(batch.get(), msa, print);

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

                    // In case that number of batch windows is more than the capacity available on GPU, the for loop breaks into smaller number of windows.
                    // if adding window i in batch->add_poa_group is not successful, it wont be processed in this iteration, therefore we print i-1
                    // to account for the fact that window i was excluded at this round.
                    if (status == StatusType::success)
                    {
                        std::cout << "Processed windows " << window_count + window_count_offset << " - " << i + window_count_offset << " (batch " << b << ")" << std::endl;
                    }
                    else
                    {
                        std::cout << "Processed windows " << window_count + window_count_offset << " - " << i - 1 + window_count_offset << " (batch " << b << ")" << std::endl;
                    }
                }
                else
                {
                    // the POA was too large to be added to the GPU, skip and move on
                    std::cout << "Could not add POA group " << batch_window_ids[i] << "to batch " << b << ". Error code " << status << std::endl;
                    i++;
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

            if (status != StatusType::exceeded_maximum_poas && status != StatusType::success)
            {
                std::cout << "Could not add POA group " << batch_window_ids[i] << "to batch " << b << ". Error code " << status << std::endl;
                i++;
            }
        }

        window_count_offset += get_size(batch_window_ids);
    }

    return 0;
}
