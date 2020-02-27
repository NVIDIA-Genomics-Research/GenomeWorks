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

#include <cstdint>
#include <string>
// the following headers are for BenchmarkData
#include <sys/resource.h>
#include <chrono>
#include <cuda_runtime_api.h>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <assert.h>

namespace claragenomics
{

namespace cudamapper
{

/// ArrayBlock - points to a part of an array
///
/// Contains the index of the first element in the block and the number of elements
struct ArrayBlock
{
    /// index of the first element of the block
    size_t first_element_;
    /// number of elements of the block
    std::uint32_t block_size_;
};

/// position_in_read_t
using position_in_read_t = std::uint32_t;
/// representation_t
using representation_t = std::uint64_t; // this depends on kmer size, in some cases could also be 32-bit
/// read_id_t
using read_id_t = std::uint32_t;

/// Relative strand - represents whether query and target
/// are on the same DNA strand (i.e Forward) or not (i.e Reverse).
enum class RelativeStrand : unsigned char
{
    Forward = '+',
    Reverse = '-',
};

/// Anchor - represents one anchor
///
/// Anchor is a pair of two sketch elements with the same sketch element representation from different reads
struct Anchor
{
    /// read ID of query
    read_id_t query_read_id_;
    /// read ID of target
    read_id_t target_read_id_;
    /// position of first sketch element in query_read_id_
    position_in_read_t query_position_in_read_;
    /// position of second sketch element in target_read_id_
    position_in_read_t target_position_in_read_;
};

/// Overlap - represents one overlap between two substrings
///
/// Overlap is a region of two strings which is considered to be the same underlying biological sequence.
/// The overlapping region need not be identical across both substrings.
typedef struct Overlap
{
    /// internal read ID for query
    read_id_t query_read_id_;
    /// internal read ID for target
    read_id_t target_read_id_;
    /// start position in the query
    position_in_read_t query_start_position_in_read_;
    /// start position in the target
    position_in_read_t target_start_position_in_read_;
    /// end position in the query
    position_in_read_t query_end_position_in_read_;
    /// end position in the target
    position_in_read_t target_end_position_in_read_;
    /// query read name (e.g from FASTA)
    char* query_read_name_ = nullptr;
    /// target read name (e.g from FASTA)
    char* target_read_name_ = nullptr;
    /// Relative strand: Forward ("+") or Reverse("-")
    RelativeStrand relative_strand;
    /// Number of residues (e.g anchors) between the two reads
    std::uint32_t num_residues_ = 0;
    /// Length of query sequence
    std::uint32_t query_length_ = 0;
    /// Length of target sequence
    std::uint32_t target_length_ = 0;
    /// Whether the overlap is considered valid by the generating overlapper
    bool overlap_complete = false;
    /// CIGAR string for alignment of mapped section.
    char* cigar_ = nullptr;

    //TODO add a destructor and copy constructor to remove need for this function
    /// \brief Free memory associated with Overlap.
    /// Since query_read_name_, target_read_name_ and cigar_ are char * types,
    /// they are not freed when Overlap is deleted.
    void clear()
    {
        delete[] target_read_name_;
        target_read_name_ = nullptr;

        delete[] query_read_name_;
        query_read_name_ = nullptr;

        delete[] cigar_;
        cigar_ = nullptr;
    }

} Overlap;

/// Data structure for storing benchmark data
///
/// contains vectors keep record of time and memory per benchmark iterations
/// a benchmark iteration refers to processing one batch of query indices
struct BenchmarkData
{
    /// time (secs) spent to complete indexer for each benchmark iteration
    std::vector<float> indexer_time;
    /// time (secs) spent to complete matcher for each benchmark iteration
    std::vector<float> matcher_time;
    /// time (secs) spent to complete overlapper for each benchmark iteration
    std::vector<float> overlapper_time;

    /// keep track of max device memory (GB) per benchmark iteration
    std::vector<float> device_mem;
    /// keep track of max host memory per (GB) benchmark iteration
    std::vector<float> host_mem;

    /// size of query index batch per benchmark iteration
    std::vector<std::int32_t> index_size;

private:
    /// resource usage stamps to keep track of cpu time and memory usage
    rusage start_;
    rusage stop_;
    /// time stamps to measure runtime
    std::chrono::time_point<std::chrono::system_clock> start_wall_clock_;
    std::chrono::time_point<std::chrono::system_clock> stop_wall_clock_;
    /// a debug flag to ensure start_timer is called before any stop_timer usage
    bool timer_initilized_ = false;
    /// accumulative compute time per benchmark iteration for indexer
    float index_time_ = 0.f;
    /// accumulative compute time per benchmark iteration for matcher
    float match_time_ = 0.f;
    /// accumulative compute time per benchmark iteration for overlapper
    float overlap_time_ = 0.f;
    /// max used memory on GPU (GB)
    float device_mem_ = 0.f;
    /// max used RAM in (GB)
    float host_mem_ = 0.f;

    // cudamapper arguments used in benchmark
    int k_arg_    = 0;
    int w_arg_    = 0;
    int d_arg_    = 0;
    int c_arg_    = 0;
    int C_arg_    = 0;
    int m_arg_    = 0;
    int i_arg_    = 0;
    int t_arg_    = 0;
    double F_arg_ = 0.;

    /// if set to false, timing measurements will be user-cpu-time, otherwise wall-clock-time
    bool wall_clock_time_ = true;

public:
    /// \brief sets arguments used in cudamapper benchmark
    /// \param k length of kmer to use for minimizers
    /// \param w length of window to use for minimizers
    /// \param d number of GPUs to use
    /// \param c number of indices cached on device
    /// \param C number of indices cached on host
    /// \param m maximum aggregate cached memory per device in GB
    /// \param i length of batch size used for query in MB
    /// \param t length of batch size used for target in MB
    /// \param F filter parameter
    void set_benchmark_args(int k, int w, int d, int c, int C, int m, int i, int t, double F)
    {
        k_arg_ = k;
        w_arg_ = w;
        d_arg_ = d;
        c_arg_ = c;
        C_arg_ = C;
        m_arg_ = m;
        i_arg_ = i;
        t_arg_ = t;
        F_arg_ = F;
    }

    /// \brief This will record beginning of a session to be measured in the code
    /// \param enabled indicates if benchmark mode is enabled, if not function exits without doing anything
    void start_timer(const bool enabled)
    {
        if (!enabled)
        {
            return;
        }

        if (wall_clock_time_)
        {
            start_wall_clock_ = std::chrono::system_clock::now();
        }
        else
        {
            getrusage(RUSAGE_SELF, &start_);
        }
        timer_initilized_ = true;
    }

    /// \brief records end of a session to be measured, should be called in pair with and following start_timer().
    /// \brief This function will gather both timing and host memory usage data.
    /// \brief The measured time (in sec) will update the record for the corresponding block among indexer, matcher or overlapper
    /// \param enabled indicates if benchmark mode is enabled, if not function exits without doing anything
    /// \param x can be either 'i', 'm' or 'o', denoting indexer, matcher and overlapper respectively
    void stop_timer_and_gather_data(char x, const bool enabled)
    {
        if (!enabled)
        {
            return;
        }
        // start_timer was not used before calling stop_timer
        assert(timer_initilized_ == true);
        getrusage(RUSAGE_SELF, &stop_);
        timer_initilized_ = false;

        float spent_time;
        if (wall_clock_time_)
        {
            stop_wall_clock_ = std::chrono::system_clock::now();
            auto diff        = stop_wall_clock_ - start_wall_clock_;
            auto msec        = std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
            spent_time       = (float)msec / 1000;
        }
        else
        {
            float sec  = stop_.ru_utime.tv_sec - start_.ru_utime.tv_sec;
            float usec = stop_.ru_utime.tv_usec - start_.ru_utime.tv_usec;
            spent_time = sec + (usec / 1000000);
        }

        switch (x)
        {
        case 'i':
            index_time_ += spent_time;
            break;
        case 'm':
            match_time_ += spent_time;
            break;
        case 'o':
            overlap_time_ += spent_time;
            break;
        default:
            assert(false);
        }

        // update memory info
        host_mem_ = std::max(host_mem_, (float)(stop_.ru_maxrss) / 1000000.f);
        size_t free_mem_d, total_mem_d;
        cudaMemGetInfo(&free_mem_d, &total_mem_d);
        device_mem_ = std::max(device_mem_, (float)(total_mem_d - free_mem_d) / 1000000000.f);
    }

    /// \brief stores benchmark iteration data in corresponding vectors and reset local variables
    /// \param enabled indicates if benchmark mode is enabled, if not function exits without doing anything
    /// \param idx_sz is the size of query index batch
    void update_iteration_data(const bool enabled, const std::int32_t idx_sz)
    {
        if (!enabled)
        {
            return;
        }
        indexer_time.push_back(index_time_);
        matcher_time.push_back(match_time_);
        overlapper_time.push_back(overlap_time_);
        device_mem.push_back(device_mem_);
        host_mem.push_back(host_mem_);
        index_size.push_back(idx_sz);
        // reset per iteration data
        index_time_   = 0.f;
        match_time_   = 0.f;
        overlap_time_ = 0.f;
        device_mem_   = 0.f;
        host_mem_     = 0.f;
    }

    /// \brief BenchmarkData by default uses wall-clock timing, this can change to cpu-timing by calling this fiunction
    /// \param use_cpu_timer if true, will use cpu-timer, otherwise will measure runtime
    void set_using_cpu_timer(bool use_cpu_timer)
    {
        wall_clock_time_ = !use_cpu_timer;
    }

    /// \brief displays a summary of benchmark at the end of runtime
    void display()
    {
        size_t num_itr = indexer_time.size();
        std::cerr << "\nbenchmark summary:\n";
        std::cerr << "==============================================================================\n";
        for (size_t i = 0; i < num_itr; i++)
        {
            float total_time = indexer_time[i] + matcher_time[i] + overlapper_time[i];
            int n_i          = (int)std::ceil(indexer_time[i] * 50.f / total_time);
            int n_m          = (int)std::ceil(matcher_time[i] * 50.f / total_time);
            int n_o          = (int)std::ceil(overlapper_time[i] * 50.f / total_time);
            std::cerr << "iteration " << i << std::endl;
            std::cerr << "indexer (sec)     " << std::left << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(2) << indexer_time[i] << " ";
            for (int j = 0; j < n_i; j++)
            {
                std::cerr << ".";
            }
            std::cerr << "\nmatcher (sec)     " << std::left << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(2) << matcher_time[i] << " ";
            for (int j = 0; j < n_m; j++)
            {
                std::cerr << ".";
            }
            std::cerr << "\noverlapper (sec)  " << std::left << std::setw(8) << std::setfill(' ') << std::fixed << std::setprecision(2) << overlapper_time[i] << " ";
            for (int j = 0; j < n_o; j++)
            {
                std::cerr << ".";
            }
            std::cerr << "\nhost mem. (GB)    " << std::fixed << std::setprecision(2) << host_mem[i];
            std::cerr << "\ndevice mem. (GB)  " << std::fixed << std::setprecision(2) << device_mem[i];
            float perf = index_size[i] > 0 ? total_time * 1000 / index_size[i] : -1;
            std::cerr << "\nperformance (s/k) " << std::fixed << std::setprecision(1) << perf;
            std::cerr << "\n______________________________________________________________________________\n";
        }
        std::cerr << "number of benchmark iterations : " << num_itr << std::endl;
        std::cerr << "input args : -k " << k_arg_ << " -w " << w_arg_ << " -d " << d_arg_ << " -c " << c_arg_ << " -C " << C_arg_;
        std::cerr << " -m " << m_arg_ << " -i " << i_arg_ << " -t " << t_arg_ << " -F " << F_arg_ << std::endl;
        std::cerr << "maximum used device memory (GB): " << std::fixed << std::setprecision(2) << *std::max_element(device_mem.begin(), device_mem.end()) << std::endl;
        std::cerr << "maximum used host memory (GB)  : " << std::fixed << std::setprecision(2) << *std::max_element(host_mem.begin(), host_mem.end()) << std::endl;
        std::cerr << "==============================================================================\n";
    }
};

} // namespace cudamapper

} // namespace claragenomics
