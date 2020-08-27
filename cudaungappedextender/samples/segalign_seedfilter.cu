#include "parameters.h"
#include "seed_filter.h"
#include <iostream>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>
#include <thrust/unique.h>

#define MAX_SEED_HITS_PER_GB 8388608
#define MAX_UNGAPPED_PER_GB 4194304

using namespace claraparabricks::genomeworks;
using namespace claraparabricks::genomeworks::cudaungappedextender;

// Control Variables
std::mutex mu;
std::condition_variable cv;
std::vector<int> available_gpus;
std::vector<UngappedExtender> g_cuda_ungapped_extenders; // indexed by gpu_id
std::vector<cudaStream_t> g_cuda_streams;                // indexed by gpu_id

int NUM_DEVICES;

// Seed Variables
uint32_t MAX_SEEDS;
uint32_t MAX_SEED_HITS;

char** d_ref_seq;
uint32_t ref_len;

char** d_query_seq;
char** d_query_rc_seq;
uint32_t query_length[BUFFER_DEPTH];

uint32_t seed_size;
uint32_t** d_index_table;
uint32_t** d_pos_table;

uint64_t** d_seed_offsets;

uint32_t** d_hit_num_array;
std::vector<thrust::device_vector<uint32_t>> d_hit_num_vec;

Anchor** d_hit;
std::vector<thrust::device_vector<Anchor>> d_hit_vec;

ScoredSegment** d_hsp;
std::vector<thrust::device_vector<ScoredSegment>> d_hsp_vec;

// UngappedExtend Variables (ideally not visible to the user in the API)
uint32_t MAX_UNGAPPED; // maximum extensions per iteration in the
// UngappedExtension function

int** d_sub_mat; // substitution score matrix
int xdrop;       // xdrop parameter for the UngappedExtension function
int hspthresh;   // score threshold for qualifying as an HSP
bool noentropy;  // whether or not to adjust scores of segments as a factor of
// the Shannon entropy

// convert input sequence from alphabet to integers
__global__ void compress_string(char* dst_seq, char* src_seq, uint32_t len){
    ...}

// convert input sequence to its reverse complement and convert from alphabet to
// integers
__global__ void compress_rev_comp_string(char* dst_seq, char* src_seq,
                                         uint32_t len){
    ...}

///////////////////// End Ungapped Extension related functions executed on the
/// GPU ///////////////

__global__ void find_num_hits(int num_seeds,
                              const uint32_t* __restrict__ d_index_table,
                              uint64_t* seed_offsets, uint32_t* seed_hit_num){
    ...}

__global__ void find_hits(const uint32_t* __restrict__ d_index_table,
                          const uint32_t* __restrict__ d_pos_table,
                          uint64_t* d_seed_offsets, uint32_t seed_size,
                          uint32_t* seed_hit_num, int num_hits, Anchor* d_hit,
                          uint32_t start_seed_index, uint32_t start_hit_index)
{
    ...
}

int InitializeProcessor(int num_gpu, bool transition, uint32_t WGA_CHUNK,
                        uint32_t input_seed_size, int* sub_mat, int input_xdrop,
                        int input_hspthresh, bool input_noentropy)
{

    int nDevices;

    cudaError_t err = cudaGetDeviceCount(&nDevices);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: No GPU device found!\n");
        exit(1);
    }

    if (num_gpu == -1)
    {
        NUM_DEVICES = nDevices;
    }
    else
    {
        if (num_gpu <= nDevices)
        {
            NUM_DEVICES = num_gpu;
        }
        else
        {
            fprintf(stderr, "Requested GPUs greater than available GPUs\n");
            exit(10);
        }
    }

    fprintf(stderr, "Using %d GPU(s)\n", NUM_DEVICES);

    seed_size = input_seed_size;

    if (transition)
        MAX_SEEDS = 13 * WGA_CHUNK;
    else
        MAX_SEEDS = WGA_CHUNK;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    float global_mem_gb =
        static_cast<float>(deviceProp.totalGlobalMem / 1073741824.0f);
    MAX_SEED_HITS = global_mem_gb * MAX_SEED_HITS_PER_GB;

    Anchor zeroHit;
    zeroHit.query_position_in_read_  = 0;
    zeroHit.target_position_in_read_ = 0;

    ScoredSegment zeroHsp;
    zeroHsp.anchor.query_position_in_read_  = 0;
    zeroHsp.anchor.target_position_in_read_ = 0;
    zeroHsp.len                             = 0;
    zeroHsp.score                           = 0;

    d_ref_seq      = (char**)malloc(NUM_DEVICES * sizeof(char*));
    d_query_seq    = (char**)malloc(BUFFER_DEPTH * NUM_DEVICES * sizeof(char*));
    d_query_rc_seq = (char**)malloc(BUFFER_DEPTH * NUM_DEVICES * sizeof(char*));

    d_index_table = (uint32_t**)malloc(NUM_DEVICES * sizeof(uint32_t*));
    d_pos_table   = (uint32_t**)malloc(NUM_DEVICES * sizeof(uint32_t*));

    d_seed_offsets = (uint64_t**)malloc(NUM_DEVICES * sizeof(uint64_t*));

    d_hit_num_array = (uint32_t**)malloc(NUM_DEVICES * sizeof(int32_t*));
    d_hit_num_vec.reserve(NUM_DEVICES);

    d_hit = (seedHit**)malloc(NUM_DEVICES * sizeof(Anchor*));
    d_hit_vec.reserve(NUM_DEVICES);

    d_hsp = (segment**)malloc(NUM_DEVICES * sizeof(ScoredSegment*));
    d_hsp_vec.reserve(NUM_DEVICES);

    for (int g = 0; g < NUM_DEVICES; g++)
    {

        check_cuda_setDevice(g, "InitializeProcessor");

        check_cuda_malloc((void**)&d_seed_offsets[g], MAX_SEEDS * sizeof(uint64_t),
                          "seed_offsets");

        d_hit_num_vec.emplace_back(MAX_SEEDS, 0);
        d_hit_num_array[g] = thrust::raw_pointer_cast(d_hit_num_vec.at(g).data());

        d_hit_vec.emplace_back(MAX_SEED_HITS, zeroHit);
        d_hit[g] = thrust::raw_pointer_cast(d_hit_vec.at(g).data());

        d_hsp_vec.emplace_back(MAX_SEED_HITS, zeroHsp);
        d_hsp[g] = thrust::raw_pointer_cast(d_hsp_vec.at(g).data());
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        g_cuda_streams.push_back(stream);
        g_cuda_ungapped_extenders.emplace_back(g, sub_mat, input_xdrop,
                                               input_noentropy, stream);
        available_gpus.push_back(g);
    }
    return NUM_DEVICES;
}

void InclusivePrefixScan(uint32_t* data, uint32_t len)
{
    ...
}

void SendSeedPosTable(uint32_t* index_table, uint32_t index_table_size,
                      uint32_t* pos_table, uint32_t num_index,
                      uint32_t max_pos_index)
{
    ...
}

void SendRefWriteRequest(size_t start_addr, uint32_t len)
{
    ...
}

void SendQueryWriteRequest(size_t start_addr, uint32_t len, uint32_t buffer){
    ...}

std::vector<segment> SeedAndFilter(std::vector<uint64_t> seed_offset_vector,
                                   bool rev, uint32_t buffer,
                                   int input_hspthresh)
{

    uint32_t num_hits      = 0;
    uint32_t total_anchors = 0;

    uint32_t num_seeds = seed_offset_vector.size();

    uint64_t* tmp_offset = (uint64_t*)malloc(num_seeds * sizeof(uint64_t));
    for (uint32_t i = 0; i < num_seeds; i++)
    {
        tmp_offset[i] = seed_offset_vector[i];
    }

    int g;
    std::unique_lock<std::mutex> locker(mu);
    if (available_gpus.empty())
    {
        cv.wait(locker, []() { return !available_gpus.empty(); });
    }
    g = available_gpus.back();
    available_gpus.pop_back();
    locker.unlock();

    check_cuda_setDevice(g, "SeedAndFilter");

    check_cuda_memcpy((void*)d_seed_offsets[g], (void*)tmp_offset,
                      num_seeds * sizeof(uint64_t), cudaMemcpyHostToDevice,
                      "seed_offsets");

    find_num_hits<<<MAX_BLOCKS, MAX_THREADS>>>(
        num_seeds, d_index_table[g], d_seed_offsets[g], d_hit_num_array[g]);

    thrust::inclusive_scan(d_hit_num_vec[g].begin(),
                           d_hit_num_vec[g].begin() + num_seeds,
                           d_hit_num_vec[g].begin());

    check_cuda_memcpy((void*)&num_hits,
                      (void*)(d_hit_num_array[g] + num_seeds - 1),
                      sizeof(uint32_t), cudaMemcpyDeviceToHost, "num_hits");

    int num_iter            = num_hits / MAX_UNGAPPED + 1;
    uint32_t iter_hit_limit = MAX_UNGAPPED;
    thrust::device_vector<uint32_t> limit_pos(num_iter);

    for (int i = 0; i < num_iter - 1; i++)
    {
        thrust::device_vector<uint32_t>::iterator result_end = thrust::lower_bound(
            d_hit_num_vec[g].begin(), d_hit_num_vec[g].begin() + num_seeds,
            iter_hit_limit);
        uint32_t pos   = thrust::distance(d_hit_num_vec[g].begin(), result_end) - 1;
        iter_hit_limit = d_hit_num_vec[g][pos] + MAX_UNGAPPED;
        limit_pos[i]   = pos;
    }

    limit_pos[num_iter - 1] = num_seeds - 1;

    segment** h_hsp       = (segment**)malloc(num_iter * sizeof(segment*));
    uint32_t* num_anchors = (uint32_t*)calloc(num_iter, sizeof(uint32_t));

    uint32_t start_seed_index = 0;
    uint32_t start_hit_val    = 0;
    uint32_t iter_num_seeds, iter_num_hits;
    int32_t* d_num_anchors;
    GW_CU_CHECK_ERR(cudaMalloc((void**)&d_num_anchors, sizeof(int32_t)));

    if (num_hits > 0)
    {

        for (int i = 0; i < num_iter; i++)
        {
            iter_num_seeds = limit_pos[i] + 1 - start_seed_index;
            iter_num_hits  = d_hit_num_vec[g][limit_pos[i]] - start_hit_val;

            find_hits<<<iter_num_seeds, BLOCK_SIZE>>>(
                d_index_table[g], d_pos_table[g], d_seed_offsets[g], seed_size,
                d_hit_num_array[g], iter_num_hits, d_hit[g], start_seed_index,
                start_hit_val);

            if (rev)
            {
                if (!g_cuda_ungapped_extenders[g].ungapped_extend(
                        d_query_rc_seq[buffer * NUM_DEVICES + g], query_length[buffer],
                        d_ref_seq[g], ref_len, input_hspthresh, d_hit[g], iter_num_hits,
                        d_hsp[g], d_num_anchors))
                    ;
                {
                    // err...
                }
            }
            else
            {
                if (!g_cuda_ungapped_extenders[g].ungapped_extend(
                        d_query_seq[buffer * NUM_DEVICES + g], query_length[buffer],
                        d_ref_seq[g], ref_len, input_hspthresh, d_hit[g], iter_num_hits,
                        d_hsp[g], d_num_anchors))
                    ;
                {
                    // err...
                }
            }
            GW_CU_CHECK_ERR(cudaMemcpyAsync(&num_anchors[i], d_num_anchors,
                                            sizeof(int32_t), cudaMemcpyDeviceToHost,
                                            g_cuda_streams[g]));
            cudaStreamSynchronize(g_cuda_streams[g]);

            total_anchors += num_anchors[i];

            if (num_anchors[i] > 0)
            {
                h_hsp[i] = (segment*)calloc(num_anchors[i], sizeof(segment));

                check_cuda_memcpy((void*)h_hsp[i], (void*)d_hsp[g],
                                  num_anchors[i] * sizeof(segment),
                                  cudaMemcpyDeviceToHost, "hsp_output");
            }

            start_seed_index = limit_pos[i] + 1;
            start_hit_val    = d_hit_num_vec[g][limit_pos[i]];
        }
    }

    limit_pos.clear();

    {
        std::unique_lock<std::mutex> locker(mu);
        available_gpus.push_back(g);
        locker.unlock();
        cv.notify_one();
    }

    std::vector<segment> gpu_filter_output;

    segment first_el;
    first_el.len   = total_anchors;
    first_el.score = num_hits;
    gpu_filter_output.push_back(first_el);

    if (total_anchors > 0)
    {
        for (int it = 0; it < num_iter; it++)
        {

            for (int i = 0; i < num_anchors[it]; i++)
            {
                gpu_filter_output.push_back(h_hsp[it][i]);
            }
        }
        free(h_hsp);
    }

    free(tmp_offset);
    return gpu_filter_output;
}

void clearRef()
{
    ...
}

void clearQuery(uint32_t buffer)
{
    ...
}

void ShutdownProcessor()
{

    d_hit_num_vec.clear();
    d_hit_vec.clear();
    d_hsp_vec.clear();

    g_cuda_ungapped_extenders.clear();
    for (auto& cudaStream : g_cuda_streams)
    {
        cudaStreamDestroy(cudaStream);
    }
    g_cuda_streams.clear();
    cudaDeviceReset();
}

///// Start Ungapped Extension related functions /////

void CompressSeq(char* input_seq, char* output_seq, uint32_t len)
{
    ...
}

void CompressRevCompSeq(char* input_seq, char* output_seq, uint32_t len){...}

CompressSeq_ptr g_CompressSeq               = CompressSeq;
CompressRevCompSeq_ptr g_CompressRevCompSeq = CompressRevCompSeq;

///// End Ungapped Extension related functions /////

InitializeProcessor_ptr g_InitializeProcessor     = InitializeProcessor;
InclusivePrefixScan_ptr g_InclusivePrefixScan     = InclusivePrefixScan;
SendSeedPosTable_ptr g_SendSeedPosTable           = SendSeedPosTable;
SendRefWriteRequest_ptr g_SendRefWriteRequest     = SendRefWriteRequest;
SendQueryWriteRequest_ptr g_SendQueryWriteRequest = SendQueryWriteRequest;
SeedAndFilter_ptr g_SeedAndFilter                 = SeedAndFilter;
clearRef_ptr g_clearRef                           = clearRef;
clearQuery_ptr g_clearQuery                       = clearQuery;
ShutdownProcessor_ptr g_ShutdownProcessor         = ShutdownProcessor;
