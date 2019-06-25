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
#include "matcher.hpp"
#include <logging/logging.hpp>

#include <iostream>
#include <chrono>

int main(int argc, char *argv[])
{
    genomeworks::logging::Init();
    auto start_time = std::chrono::high_resolution_clock::now();
    GW_LOG_INFO("Creating index generator");
    // TODO: pass kmer and window size as parameters
    std::unique_ptr<genomeworks::IndexGenerator> index_generator = genomeworks::IndexGenerator::create_index_generator(std::string(argv[1]), 15, 15);
    GW_LOG_INFO("Created index generator");
    std::cout << "Index generator execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    GW_LOG_INFO("Creating index");
    std::unique_ptr<genomeworks::Index> index = genomeworks::Index::create_index(*index_generator);
    GW_LOG_INFO("Created index");
    std::cout << "Index execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    GW_LOG_INFO("Started matcher");
    genomeworks::Matcher matcher(static_cast<genomeworks::IndexGPU&>(*(index.get())));
    GW_LOG_INFO("Finished matcher");
    std::cout << "Matcher execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

    return  0;
}
