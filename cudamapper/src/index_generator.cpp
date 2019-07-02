/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "bioparser/bioparser.hpp"
#include "index_generator_cpu.hpp"

namespace claragenomics {
    std::unique_ptr<IndexGenerator> IndexGenerator::create_index_generator(const std::string &query_filename, std::uint64_t kmer_length, std::uint64_t window_size) {
        return std::make_unique<IndexGeneratorCPU>(query_filename, kmer_length, window_size);
    }
}
