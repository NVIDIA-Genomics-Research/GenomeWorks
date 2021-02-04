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

#include "cudaaligner_test_cases.hpp"
#include "cudaaligner_file_location.hpp"
#include <claraparabricks/genomeworks/types.hpp>

#include <random>
#include <memory>
#include <claraparabricks/genomeworks/io/fasta_parser.hpp>
#include <claraparabricks/genomeworks/utils/genomeutils.hpp>

namespace
{

constexpr int32_t n_random_testcases  = 10;
constexpr int32_t max_sequence_length = 5000;
constexpr uint32_t random_seed        = 5827349;

claraparabricks::genomeworks::TestCaseData generate_random_test_case(std::minstd_rand& rng)
{
    using claraparabricks::genomeworks::get_size;
    claraparabricks::genomeworks::TestCaseData t;
    std::uniform_int_distribution<int> random_length(0, max_sequence_length);
    t.target = claraparabricks::genomeworks::genomeutils::generate_random_genome(random_length(rng), rng);
    t.query  = claraparabricks::genomeworks::genomeutils::generate_random_sequence(t.target, rng, get_size(t.target), get_size(t.target), get_size(t.target));
    return t;
}

std::vector<claraparabricks::genomeworks::TestCaseData> load_test_case(const std::string& target, const std::string& query)
{
    std::vector<claraparabricks::genomeworks::TestCaseData> tests;

    std::unique_ptr<claraparabricks::genomeworks::io::FastaParser> target_parser = claraparabricks::genomeworks::io::create_kseq_fasta_parser(target, 0, false);
    std::unique_ptr<claraparabricks::genomeworks::io::FastaParser> query_parser  = claraparabricks::genomeworks::io::create_kseq_fasta_parser(query, 0, false);

    assert(target_parser->get_num_seqences() == query_parser->get_num_seqences());

    for (claraparabricks::genomeworks::read_id_t read = 0; read < target_parser->get_num_seqences(); read++)
    {
        tests.push_back({.target = target_parser->get_sequence_by_id(read).seq, .query = query_parser->get_sequence_by_id(read).seq});
    }

    return tests;
}
} // namespace

namespace claraparabricks
{

namespace genomeworks
{
std::vector<TestCaseData> create_cudaaligner_test_cases()
{
    std::vector<TestCaseData> tests = load_test_case(
        std::string(CUDAALIGNER_BENCHMARK_DATA_DIR) + "/target_cudaaligner_cases.fasta",
        std::string(CUDAALIGNER_BENCHMARK_DATA_DIR) + "/query_cudaaligner_cases.fasta");

    std::minstd_rand rng(random_seed);
    for (int32_t i = 0; i < n_random_testcases; ++i)
        tests.push_back(generate_random_test_case(rng));
    return tests;
}
} // namespace genomeworks

} // namespace claraparabricks
