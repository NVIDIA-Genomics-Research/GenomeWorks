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

#include <random>
#include <stdexcept>
#include <claragenomics/utils/signed_integer_utils.hpp>

namespace claragenomics
{

namespace genomeutils
{

inline std::string generate_random_genome(const int32_t length, std::minstd_rand& rng)
{
    const char alphabet[4] = {'A', 'C', 'G', 'T'};
    std::uniform_int_distribution<int32_t> random_index(0, 3);
    std::string genome = "";
    for (int32_t i = 0; i < length; i++)
    {
        genome += alphabet[random_index(rng)];
    }
    return genome;
}

inline std::string generate_random_sequence(const std::string& backbone, std::minstd_rand& rng, int max_mutations, int max_insertions, int max_deletions)
{
    throw_on_negative(max_mutations, "max_mutations cannot be negative.");
    throw_on_negative(max_insertions, "max_insertions cannot be negative.");
    throw_on_negative(max_deletions, "max_deletions cannot be negative.");

    if (get_size<int>(backbone) < max_deletions)
        throw std::invalid_argument("max_deletions should be smaller than backbone's length.");

    const char alphabet[4] = {'A', 'C', 'G', 'T'};
    std::uniform_int_distribution<int> random_base(0, 3);

    std::string sequence = backbone;

    std::uniform_real_distribution<double> random_prob(0, 1);

    for (int j = 0; j < max_deletions; j++)
    {
        if (random_prob(rng) > 0.5)
        {
            int length = sequence.length();
            std::uniform_int_distribution<int> random_del_pos(0, length - 1);
            int del_pos = random_del_pos(rng);
            sequence.erase(del_pos, 1);
        }
    }

    for (int j = 0; j < max_insertions; j++)
    {
        if (random_prob(rng) > 0.5)
        {
            int length = sequence.length();
            std::uniform_int_distribution<int> random_ins_pos(0, length);
            int ins_pos  = random_ins_pos(rng);
            int ins_base = random_base(rng);

            sequence.insert(ins_pos, 1, alphabet[ins_base]);
        }
    }

    int length = sequence.length();
    std::uniform_int_distribution<int> random_mut_pos(0, length - 1);
    for (int j = 0; j < max_mutations; j++)
    {
        if (random_prob(rng) > 0.5)
        {
            int mut_pos       = random_mut_pos(rng);
            int swap_base     = random_base(rng);
            sequence[mut_pos] = alphabet[swap_base];
        }
    }
    return sequence;
}

inline std::vector<std::string> generate_random_sequences(std::string const& backbone,
                                                          int n,
                                                          std::minstd_rand& rng,
                                                          int max_mutations = 1,
                                                          int max_insertion = 1,
                                                          int max_deletions = 1)
{
    throw_on_negative(n, "n cannot be negative!");
    std::vector<std::string> sequences;
    sequences.reserve(n);
    sequences.push_back(backbone);
    for (int i = 1; i < n; i++)
        sequences.push_back(generate_random_sequence(backbone, rng, max_mutations, max_insertion, max_deletions));

    return sequences;
}

} // namespace genomeutils

} // namespace claragenomics
