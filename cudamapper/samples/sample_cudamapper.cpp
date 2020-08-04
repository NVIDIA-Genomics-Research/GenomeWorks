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

#include <claraparabricks/genomeworks/cudamapper/cudamapper.hpp>
#include <claraparabricks/genomeworks/io/fasta_parser.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>
#include "../src/index_descriptor.hpp"
#include <cudamapper_file_location.hpp>

#include <claraparabricks/genomeworks/cudamapper/index.hpp>
#include <claraparabricks/genomeworks/cudamapper/matcher.hpp>
#include <claraparabricks/genomeworks/cudamapper/overlapper.hpp>
#include <claraparabricks/genomeworks/cudamapper/utils.hpp>

#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include <mutex>

// define constants. See cudamapper/src/application_parameters.hpp for more.
// constants used in multiple places
static constexpr int32_t INDEX_SIZE    = 30;
static constexpr uint32_t KMER_SIZE    = 15;
static constexpr uint32_t WINDOWS_SIZE = 5;

// constants used in the overlapper
static constexpr int32_t MIN_RESIDUES          = 3;
static constexpr int32_t MIN_OVERLAP_LEN       = 250;
static constexpr int32_t MIN_BASES_PER_RESIDUE = 1000;
static constexpr float MIN_OVERLAP_FRACTION    = 0.8;

// constant used in the indices
static constexpr float FILTERING_PARAMETER = 1e-5;

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

namespace
{
std::vector<IndexDescriptor> initialize_batch(const std::shared_ptr<const io::FastaParser> parser,
                                              DefaultDeviceAllocator allocator)
{
    //Init();

    // group reads into indices. For small inputs, there will only be 1 index
    std::vector<IndexDescriptor> index_descriptors = group_reads_into_indices(*parser,
                                                                              INDEX_SIZE * 1'000'000);

    return index_descriptors;
}

void process_batch(std::vector<IndexDescriptor>& query_index_descriptors,
                   std::vector<IndexDescriptor>& target_index_descriptors,
                   const std::shared_ptr<const io::FastaParser> query_parser,
                   const std::shared_ptr<const io::FastaParser> target_parser,
                   DefaultDeviceAllocator allocator,
                   bool print)
{
    // extra variables used in print_paf. Note "cigars" are typically found during alignment.
    const std::vector<std::string> cigars(0);
    std::mutex print_mutex;

    // process the pairs of query and target indices
    for (const IndexDescriptor& query_index_descriptor : query_index_descriptors)
    {
        std::unique_ptr<Index> query_index = Index::create_index(allocator,
                                                                 *query_parser,
                                                                 query_index_descriptor.first_read(),
                                                                 query_index_descriptor.first_read() + query_index_descriptor.number_of_reads(),
                                                                 KMER_SIZE,
                                                                 WINDOWS_SIZE,
                                                                 true,                 // hash representations
                                                                 FILTERING_PARAMETER); // filter parameter

        for (const IndexDescriptor& target_index_descriptor : target_index_descriptors)
        {
            // skip pairs in which target batch has smaller id than query batch as it will be covered by symmetry
            if (target_index_descriptor.first_read() >= query_index_descriptor.first_read())
            {
                std::unique_ptr<Index> target_index = Index::create_index(allocator,
                                                                          *target_parser,
                                                                          target_index_descriptor.first_read(),
                                                                          target_index_descriptor.first_read() + target_index_descriptor.number_of_reads(),
                                                                          KMER_SIZE,
                                                                          WINDOWS_SIZE,
                                                                          true,                 // hash representations
                                                                          FILTERING_PARAMETER); // filter parameter

                // find anchors & find overlaps
                auto matcher = Matcher::create_matcher(allocator,
                                                       *query_index,
                                                       *target_index);

                // Output vector of overlaps
                std::vector<Overlap> overlaps;
                auto overlapper = Overlapper::create_overlapper(allocator);

                overlapper->get_overlaps(overlaps,
                                         matcher->anchors(),
                                         true,
                                         MIN_RESIDUES,
                                         MIN_OVERLAP_LEN,
                                         MIN_BASES_PER_RESIDUE,
                                         MIN_OVERLAP_FRACTION);

                // post process the overlaps
                Overlapper::post_process_overlaps(overlaps, false);

                // print overlaps
                if (print)
                {
                    print_paf(overlaps, cigars, *query_parser, *target_parser, KMER_SIZE, print_mutex);
                }
            }
        }
    }

    return;
}

int main(int argc, char** argv)
{
    // parse command line options
    int c      = 0;
    bool help  = false;
    bool print = false;

    while ((c = getopt(argc, argv, "hp")) != -1)
    {
        switch (c)
        {
        case 'p':
            print = true;
            break;
        case 'h':
            help = true;
            break;
        }
    }

    // print help string
    if (help)
    {
        std::cout << "CUDA Mapper API sample program. Runs minimizer-based approximate mapping" << std::endl;
        std::cout << "Usage:" << std::endl;
        std::cout << "./sample_cudamapper [-p] [-h]" << std::endl;
        std::cout << "-p : Print the overlaps to stdout" << std::endl;
        std::cout << "-h : Print help message" << std::endl;
        std::exit(0);
    }

    // Load FASTA/FASTQ file. Assume all-to-all
    const std::string query_file  = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/covid-reads.fasta.gz";
    const std::string target_file = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/covid-reads.fasta.gz";

    const std::size_t max_gpu_memory = cudautils::find_largest_contiguous_device_memory_section();
    DefaultDeviceAllocator allocator = create_default_device_allocator(max_gpu_memory);

    // create FASTA parser
    std::shared_ptr<io::FastaParser> query_parser;
    std::shared_ptr<io::FastaParser> target_parser;
    query_parser  = io::create_kseq_fasta_parser(query_file, KMER_SIZE + WINDOWS_SIZE - 1); // defaults taken from application parser
    target_parser = query_parser;                                                           // assume all to all

    // group the indices
    std::vector<IndexDescriptor> query_index_descriptors  = initialize_batch(query_parser, allocator);
    std::vector<IndexDescriptor> target_index_descriptors = initialize_batch(target_parser, allocator);

    process_batch(query_index_descriptors, target_index_descriptors, query_parser, target_parser, allocator, print);

    return 0;
}

} // namespace

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks

/// \brief main function
/// main function cannot be in a namespace so using this function to call actual main function
int main(int argc, char* argv[])
{
    return claraparabricks::genomeworks::cudamapper::main(argc, argv);
}
