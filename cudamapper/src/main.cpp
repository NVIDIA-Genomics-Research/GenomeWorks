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
#include <iostream>
#include <chrono>

#include <logging/logging.hpp>

#include "cudamapper/index.hpp"
#include "matcher.hpp"
#include "overlapper.hpp"

int main(int argc, char *argv[])
{
    claragenomics::logging::Init();

    auto start_time = std::chrono::high_resolution_clock::now();
    CGA_LOG_INFO("Creating index generator");
    // TODO: pass kmer and window size as parameters
    std::unique_ptr<claragenomics::IndexGenerator> index_generator = claragenomics::IndexGenerator::create_index_generator(std::string(argv[1]), 15, 15);
    CGA_LOG_INFO("Created index generator");
    std::cout << "Index generator execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    CGA_LOG_INFO("Creating index");
    std::unique_ptr<claragenomics::Index> index = claragenomics::Index::create_index(*index_generator);
    CGA_LOG_INFO("Created index");
    std::cout << "Index execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    CGA_LOG_INFO("Started matcher");
    claragenomics::Matcher matcher(static_cast<claragenomics::IndexCPU&>(*(index.get())));
    CGA_LOG_INFO("Finished matcher");
    std::cout << "Matcher execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    CGA_LOG_INFO("Started overlap detector");
    auto overlaps = claragenomics::get_overlaps(matcher.anchors(), static_cast<claragenomics::IndexCPU&>(*(index.get())));
    CGA_LOG_INFO("Finished overlap detector");
    std::cout << "Overlap detection execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

    claragenomics::print_paf(overlaps);
    return  0;
}
