/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <string>

#include "cudamapper/index.hpp"
#include <logging/logging.hpp>

int main(int argc, char *argv[])
{
    genomeworks::logging::Init();

    GW_LOG_INFO("Creating index generator");
    // TODO: pass kmer and window size as parameters
    std::unique_ptr<genomeworks::IndexGenerator> index_generator = genomeworks::IndexGenerator::create_index_generator(std::string(argv[1]), 4, 4);
    GW_LOG_INFO("Created index generator");
    GW_LOG_INFO("Creating index");
    std::unique_ptr<genomeworks::Index> index = genomeworks::Index::create_index(*index_generator);
    GW_LOG_INFO("Created index");
    return  0;
}
