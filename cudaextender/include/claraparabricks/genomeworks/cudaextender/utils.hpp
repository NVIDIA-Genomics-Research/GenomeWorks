/*
* Copyright 2020 NVIDIA CORPORATION.
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

#pragma once
#include <claraparabricks/genomeworks/io/fasta_parser.hpp>
#include <claraparabricks/genomeworks/cudaextender/extender.hpp>

#include <string>
#include <vector>
#include <fstream>
#include <cassert>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaextender
{

/// Sequence encoding scheme:
constexpr int8_t A_NT = 0;
constexpr int8_t C_NT = 1;
constexpr int8_t G_NT = 2;
constexpr int8_t T_NT = 3;
constexpr int8_t L_NT = 4;
constexpr int8_t N_NT = 5;
constexpr int8_t X_NT = 6;
constexpr int8_t E_NT = 7;
constexpr int8_t NUC  = 8;
constexpr int8_t NUC2 = NUC * NUC;

/// \brief Parses seed pairs from a csv file in the following format:
///        target_position_in_read_1, query_position_in_read_1
///        target_position_in_read_2, query_position_in_read_2
///        target_position_in_read_n, query_position_in_read_n
///
/// \param[out] seed_pairs Reference to vector into which parsed seed pairs are saved
/// \param[in]  filepath   Reference to the string containing the path of the seed pairs csv
inline void parse_seed_pairs(const std::string& filepath, std::vector<SeedPair>& seed_pairs)
{
    std::ifstream seed_pairs_file(filepath);
    if (!seed_pairs_file.is_open())
        throw std::runtime_error("Cannot open file");
    if (seed_pairs_file.good())
    {
        std::string line;
        while (std::getline(seed_pairs_file, line, ','))
        {
            SeedPair seed_pair;
            seed_pair.target_position_in_read = std::atoi(line.c_str());
            std::getline(seed_pairs_file, line); // Get the next value
            seed_pair.query_position_in_read = std::atoi(line.c_str());
            seed_pairs.push_back(seed_pair);
        }
    }
}

/// \brief Parses scored segment pairs from a csv file in the following format:
///        target_position_in_read_1, query_position_in_read_1, length_1, score_1
///        target_position_in_read_2, query_position_in_read_2, length_2, score_2
///        target_position_in_read_n, query_position_in_read_n, length_n, score_n
///
/// \param[out] scored_segment_pairs Reference to vector into which parsed scored segment pairs are saved
/// \param[in]  filepath   Reference to the string containing the path of the scored segment pairs csv
inline void parse_scored_segment_pairs(const std::string& filepath, std::vector<ScoredSegmentPair>& scored_segment_pairs)
{
    std::ifstream scored_segment_pairs_file(filepath);
    if (!scored_segment_pairs_file.is_open())
        throw std::runtime_error("Cannot open file");
    if (scored_segment_pairs_file.good())
    {
        std::string line;
        while (std::getline(scored_segment_pairs_file, line, ','))
        {
            ScoredSegmentPair scored_segment_pair;
            scored_segment_pair.seed_pair.target_position_in_read = std::atoi(line.c_str());
            std::getline(scored_segment_pairs_file, line, ',');
            scored_segment_pair.seed_pair.query_position_in_read = std::atoi(line.c_str());
            std::getline(scored_segment_pairs_file, line, ',');
            scored_segment_pair.length = std::atoi(line.c_str());
            std::getline(scored_segment_pairs_file, line);
            scored_segment_pair.score = std::atoi(line.c_str());
            scored_segment_pairs.push_back(scored_segment_pair);
        }
    }
}

/// \brief Encodes character sequence as integer sequence
///
/// \param[out] dst_seq    Pointer to pre-allocated storage for encoded sequence
/// \param[in]  src_seq    Pointer to input sequence
/// \param[in]  length     Length of the sequence
inline void encode_sequence(int8_t* dst_seq, const char* src_seq, const int32_t length)
{
    for (int32_t i = 0; i < length; i++)
    {
        const char ch = src_seq[i];
        switch (ch)
        {
        case 'A':
            dst_seq[i] = A_NT;
            break;
        case 'C':
            dst_seq[i] = C_NT;
            break;
        case 'G':
            dst_seq[i] = G_NT;
            break;
        case 'T':
            dst_seq[i] = T_NT;
            break;
        case '&':
            dst_seq[i] = E_NT;
            break;
        case 'n':
        case 'N':
            dst_seq[i] = N_NT;
            break;
        case 'a':
        case 'c':
        case 'g':
        case 't':
            dst_seq[i] = L_NT;
            break;
        default:
            dst_seq[i] = X_NT;
            break;
        }
    }
}

} // namespace cudaextender

} // namespace genomeworks

} // namespace claraparabricks
