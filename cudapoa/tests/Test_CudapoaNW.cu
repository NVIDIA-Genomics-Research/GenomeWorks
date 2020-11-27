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

#include "../src/cudapoa_nw.cuh" //runNW, CUDAPOA_*
#include "sorted_graph.hpp"      //SortedGraph

#include <claraparabricks/genomeworks/utils/cudautils.hpp>            //GW_CU_CHECK_ERR
#include <claraparabricks/genomeworks/utils/stringutils.hpp>          //array_to_string
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp> //get_size
#include <numeric>

#include "gtest/gtest.h"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

class BasicNW
{

public:
    const static int16_t gap_score_      = -8;
    const static int16_t mismatch_score_ = -6;
    const static int16_t match_score_    = 8;

public:
    BasicNW(std::vector<uint8_t> nodes, std::vector<int16_t> sorted_graph, SizeTVec2D outgoing_edges,
            std::vector<uint8_t> read)
        : graph_(nodes, sorted_graph, outgoing_edges)
        , read_(read)
    {
        // do nothing
    }

    BasicNW() = delete;

    void get_graph_buffers(int16_t* incoming_edges, uint16_t* incoming_edge_count,
                           int16_t* outgoing_edges, uint16_t* outgoing_edge_count,
                           uint8_t* nodes, int16_t* node_count,
                           int16_t* graph, int16_t* node_id_to_pos) const
    {
        graph_.get_edges(incoming_edges, incoming_edge_count, outgoing_edges, outgoing_edge_count);
        graph_.get_nodes(nodes, node_count);
        graph_.get_sorted_graph(graph);
        graph_.get_node_id_to_pos(node_id_to_pos);
    }

    void get_read_buffers(uint8_t* read, uint16_t* read_count) const
    {
        for (int i = 0; i < get_size(read_); i++)
        {
            read[i] = read_[i];
        }
        *read_count = get_size(read_);
    }

protected:
    SortedGraph graph_;
    std::vector<uint8_t> read_;
};

typedef std::pair<std::string, std::string> NWAnswer;
typedef std::pair<NWAnswer, BasicNW> NWTestPair;
// create a vector of test cases
std::vector<NWTestPair> getNWTestCases()
{

    std::vector<NWTestPair> test_cases;

    /*
     * read:            A   A   T   A
     * graph:           A — A — A — A
     * alignment graph: 0   1   2   3
     * alignment read:  0   1   2   3
     *                        T
     *                       / \
     * final graph      A — A   A
     *                       \ /
     *                        A
     */

    NWAnswer ans_1("3,2,1,0", "3,2,1,0"); //alginment_graph & alignment_read are reversed
    BasicNW nw_1({'A', 'A', 'A', 'A'},    //nodes
                 {0, 1, 2, 3},            //sorted_graph
                 {{1}, {2}, {3}, {}},     //outgoing_edges
                 {'A', 'A', 'T', 'A'});   //read
    test_cases.emplace_back(std::move(ans_1), std::move(nw_1));

    /*
     * read:            A   T   C   G   A
     * graph:           A — T — C — G
     * alignment graph: 0   1   2   3  -1
     * alignment read:  0   1   2   3   4
     *                        
     * final graph      A — T — C — G — A
     * 
     */
    NWAnswer ans_2("-1,3,2,1,0", "4,3,2,1,0"); //alginment_graph & alignment_read are reversed
    BasicNW nw_2({'A', 'T', 'C', 'G'},         //nodes
                 {0, 1, 2, 3},                 //sorted_graph
                 {{1}, {2}, {3}, {}},          //outgoing_edges
                 {'A', 'T', 'C', 'G', 'A'});   //read

    test_cases.emplace_back(std::move(ans_2), std::move(nw_2));

    /*
     * read:            A   T   C   G
     *                      A
     *                    /   \
     * graph:           A — C — C — G
     * alignment graph: 0   1   2   3
     * alignment read:  0   1   2   3
     *                      T
     *                    /   \
     * final graph      A — C — C — G
     *                    \   /
     *                      A
     */
    NWAnswer ans_3("3,2,1,0", "3,2,1,0");     //alginment_graph & alignment_read are reversed
    BasicNW nw_3({'A', 'A', 'C', 'G', 'C'},   //nodes
                 {0, 4, 1, 2, 3},             //sorted_graph
                 {{1, 4}, {2}, {3}, {}, {2}}, //outgoing_edges
                 {'A', 'T', 'C', 'G'});       //read

    test_cases.emplace_back(std::move(ans_3), std::move(nw_3));

    /*
     * read:            A   A  
     * graph:           A — T — T — G — A
     * alignment graph: 0   1   2   3   4
     * alignment read:  0  -1  -1  -1   1
     *                        
     * final graph      A — T — T — G — A
     *                   \_____________/
     * 
     */
    NWAnswer ans_4("4,3,2,1,0", "1,-1,-1,-1,0"); //alginment_graph & alignment_read are reversed
    BasicNW nw_4({'A', 'T', 'T', 'G', 'A'},      //nodes
                 {0, 1, 2, 3, 4},                //sorted_graph
                 {{1}, {2}, {3}, {4}, {}},       //outgoing_edges
                 {'A', 'A'});                    //read
    test_cases.emplace_back(std::move(ans_4), std::move(nw_4));

    /*
     * read:            A   C   T   T   A
     *                      T — G
     *                    /       \ 
     * graph:           A — C — A — T — A
     * alignment graph: 0   5   6   3   4
     * alignment read:  0   1   2   3   4
     *                      T — G   
     *                    /       \
     * final graph      A — C — A — T — A
     *                        \   /
     *                          T
     * 
     */
    NWAnswer ans_5("4,3,6,5,0", "4,3,2,1,0");           //alignment_graph & alignment_read are reversed
    BasicNW nw_5({'A', 'T', 'G', 'T', 'A', 'C', 'A'},   //nodes
                 {0, 5, 1, 6, 2, 3, 4},                 //sorted_graph
                 {{1, 5}, {2}, {3}, {4}, {}, {6}, {3}}, //outgoing_edges
                 {'A', 'C', 'T', 'T', 'A'});            //read

    test_cases.emplace_back(std::move(ans_5), std::move(nw_5));

    //add more test cases below

    return test_cases;
}

// host function for calling the kernel to test full-band NW device function.
NWAnswer testNW(const BasicNW& obj)
{
    //declare device buffer
    uint8_t* nodes;
    int16_t* graph;
    int16_t* node_id_to_pos;
    int16_t graph_count; //local
    uint16_t* incoming_edge_count;
    int16_t* incoming_edges;
    uint16_t* outgoing_edge_count;
    int16_t* outgoing_edges;
    uint8_t* read;
    uint16_t read_count; //local
    int16_t* scores;
    int16_t* alignment_graph;
    int16_t* alignment_read;
    int32_t gap_score;
    int32_t mismatch_score;
    int32_t match_score;
    int16_t* aligned_nodes; //local; to store num of nodes aligned (length of alignment_graph and alignment_read)
    BatchConfig batch_size; //default max_sequence_size = 1024, max_sequences_per_poa = 100

    //allocate unified memory so they can be accessed by both host and device.
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&nodes, batch_size.max_nodes_per_graph * sizeof(uint8_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&graph, batch_size.max_nodes_per_graph * sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&node_id_to_pos, batch_size.max_nodes_per_graph * sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&incoming_edges, batch_size.max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES * sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&incoming_edge_count, batch_size.max_nodes_per_graph * sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&outgoing_edges, batch_size.max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES * sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&outgoing_edge_count, batch_size.max_nodes_per_graph * sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&scores, batch_size.max_nodes_per_graph * batch_size.matrix_sequence_dimension * sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&alignment_graph, batch_size.max_nodes_per_graph * sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&read, batch_size.max_sequence_size * sizeof(uint8_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&alignment_read, batch_size.max_nodes_per_graph * sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&aligned_nodes, sizeof(int16_t)));

    //initialize all 'count' buffers
    memset((void**)incoming_edge_count, 0, batch_size.max_nodes_per_graph * sizeof(uint16_t));
    memset((void**)outgoing_edge_count, 0, batch_size.max_nodes_per_graph * sizeof(uint16_t));
    memset((void**)node_id_to_pos, 0, batch_size.max_nodes_per_graph * sizeof(int16_t));
    memset((void**)scores, 0, batch_size.max_nodes_per_graph * batch_size.matrix_sequence_dimension * sizeof(int16_t));

    //calculate edge counts on host
    obj.get_graph_buffers(incoming_edges, incoming_edge_count,
                          outgoing_edges, outgoing_edge_count,
                          nodes, &graph_count,
                          graph, node_id_to_pos);
    obj.get_read_buffers(read, &read_count);
    gap_score      = BasicNW::gap_score_;
    mismatch_score = BasicNW::mismatch_score_;
    match_score    = BasicNW::match_score_;

    //call the host wrapper of nw kernel
    runNW(nodes,
          graph,
          node_id_to_pos,
          graph_count,
          incoming_edge_count,
          incoming_edges,
          outgoing_edge_count,
          read,
          read_count,
          scores,
          batch_size.matrix_sequence_dimension,
          alignment_graph,
          alignment_read,
          gap_score,
          mismatch_score,
          match_score,
          aligned_nodes);

    GW_CU_CHECK_ERR(cudaDeviceSynchronize());

    //input and output buffers are the same ones in unified memory, so the results are updated in place
    //results are stored in alignment_graph and alignment_read; return string representation of those
    auto res = std::make_pair(genomeworks::stringutils::array_to_string<int16_t>(alignment_graph, *aligned_nodes, ","),
                              genomeworks::stringutils::array_to_string<int16_t>(alignment_read, *aligned_nodes, ","));

    GW_CU_CHECK_ERR(cudaFree(nodes));
    GW_CU_CHECK_ERR(cudaFree(graph));
    GW_CU_CHECK_ERR(cudaFree(node_id_to_pos));
    GW_CU_CHECK_ERR(cudaFree(incoming_edges));
    GW_CU_CHECK_ERR(cudaFree(incoming_edge_count));
    GW_CU_CHECK_ERR(cudaFree(outgoing_edges));
    GW_CU_CHECK_ERR(cudaFree(outgoing_edge_count));
    GW_CU_CHECK_ERR(cudaFree(scores));
    GW_CU_CHECK_ERR(cudaFree(alignment_graph));
    GW_CU_CHECK_ERR(cudaFree(read));
    GW_CU_CHECK_ERR(cudaFree(alignment_read));
    GW_CU_CHECK_ERR(cudaFree(aligned_nodes));

    return res;
}

using ::testing::TestWithParam;
using ::testing::ValuesIn;

class NWTest : public TestWithParam<NWTestPair>
{
public:
    void SetUp() {}

    NWAnswer runNWTest(const BasicNW& nw)
    {
        return testNW(nw);
    }
};

TEST_P(NWTest, TestNWCorrectness)
{
    const auto test_case = GetParam();
    EXPECT_EQ(test_case.first, runNWTest(test_case.second));
}

INSTANTIATE_TEST_SUITE_P(TestNW, NWTest, ValuesIn(getNWTestCases()));

//---------------------------------------------------------------------------------------

// host function for calling the kernels to test static/adaptive-band NW with/without traceback buffer
NWAnswer testNWbanded(const BasicNW& obj, bool adaptive, bool traceback = false)
{
    //declare device buffer
    uint8_t* nodes;
    int16_t* graph;
    int16_t* node_id_to_pos;
    int16_t graph_count; //local
    uint16_t* incoming_edge_count;
    int16_t* incoming_edges;
    uint16_t* outgoing_edge_count;
    int16_t* outgoing_edges;
    uint8_t* read;
    uint16_t read_count; //local
    int16_t* scores;
    int16_t* traces;
    int16_t* alignment_graph;
    int16_t* alignment_read;
    int32_t gap_score;
    int32_t mismatch_score;
    int32_t match_score;
    int16_t* aligned_nodes; //local; to store num of nodes aligned (length of alignment_graph and alignment_read)
    BandMode band_mode = traceback ? (adaptive ? BandMode::adaptive_band_traceback : BandMode::static_band_traceback)
                                   : (adaptive ? BandMode::adaptive_band : BandMode::static_band);
    BatchConfig batch_size(1024 /*max_sequence_size*/, 2 /*max_sequences_per_poa*/,
                           128 /*= band_width*/, band_mode);

    //allocate unified memory so they can be accessed by both host and device.
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&nodes, batch_size.max_nodes_per_graph * sizeof(uint8_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&graph, batch_size.max_nodes_per_graph * sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&node_id_to_pos, batch_size.max_nodes_per_graph * sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&incoming_edges, batch_size.max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES * sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&incoming_edge_count, batch_size.max_nodes_per_graph * sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&outgoing_edges, batch_size.max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES * sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&outgoing_edge_count, batch_size.max_nodes_per_graph * sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&alignment_graph, batch_size.max_nodes_per_graph * sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&read, batch_size.max_sequence_size * sizeof(uint8_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&alignment_read, batch_size.max_nodes_per_graph * sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void**)&aligned_nodes, sizeof(int16_t)));
    if (traceback)
    {
        GW_CU_CHECK_ERR(cudaMallocManaged((void**)&scores, batch_size.max_banded_pred_distance * batch_size.matrix_sequence_dimension * sizeof(int16_t)));
        GW_CU_CHECK_ERR(cudaMallocManaged((void**)&traces, batch_size.max_nodes_per_graph * batch_size.matrix_sequence_dimension * sizeof(int16_t)));
    }
    else
    {
        GW_CU_CHECK_ERR(cudaMallocManaged((void**)&scores, batch_size.max_nodes_per_graph * batch_size.matrix_sequence_dimension * sizeof(int16_t)));
    }

    //initialize all 'count' buffers
    memset((void**)incoming_edge_count, 0, batch_size.max_nodes_per_graph * sizeof(uint16_t));
    memset((void**)outgoing_edge_count, 0, batch_size.max_nodes_per_graph * sizeof(uint16_t));
    memset((void**)node_id_to_pos, 0, batch_size.max_nodes_per_graph * sizeof(int16_t));
    memset((void**)scores, 0, batch_size.max_nodes_per_graph * batch_size.matrix_sequence_dimension * sizeof(int16_t));

    //calculate edge counts on host
    obj.get_graph_buffers(incoming_edges, incoming_edge_count,
                          outgoing_edges, outgoing_edge_count,
                          nodes, &graph_count,
                          graph, node_id_to_pos);
    obj.get_read_buffers(read, &read_count);
    gap_score      = BasicNW::gap_score_;
    mismatch_score = BasicNW::mismatch_score_;
    match_score    = BasicNW::match_score_;

    //call the host wrapper of nw kernels
    if (traceback)
    {
        runNWbandedTB(nodes,
                               graph,
                               node_id_to_pos,
                               graph_count,
                               incoming_edge_count,
                               incoming_edges,
                               outgoing_edge_count,
                               read,
                               read_count,
                               scores,
                               traces,
                               batch_size.matrix_sequence_dimension,
                               batch_size.max_nodes_per_graph,
                               alignment_graph,
                               alignment_read,
                               batch_size.alignment_band_width,
                               batch_size.max_banded_pred_distance,
                               gap_score,
                               mismatch_score,
                               match_score,
                               aligned_nodes,
                               adaptive);
    }
    else
    {
        runNWbanded(nodes,
                             graph,
                             node_id_to_pos,
                             graph_count,
                             incoming_edge_count,
                             incoming_edges,
                             outgoing_edge_count,
                             read,
                             read_count,
                             scores,
                             batch_size.matrix_sequence_dimension,
                             batch_size.max_nodes_per_graph,
                             alignment_graph,
                             alignment_read,
                             batch_size.alignment_band_width,
                             gap_score,
                             mismatch_score,
                             match_score,
                             aligned_nodes,
                             adaptive);
    }

    GW_CU_CHECK_ERR(cudaDeviceSynchronize());

    //input and output buffers are the same ones in unified memory, so the results are updated in place
    //results are stored in alignment_graph and alignment_read; return string representation of those
    auto res = std::make_pair(genomeworks::stringutils::array_to_string<int16_t>(alignment_graph, *aligned_nodes, ","),
                              genomeworks::stringutils::array_to_string<int16_t>(alignment_read, *aligned_nodes, ","));

    GW_CU_CHECK_ERR(cudaFree(nodes));
    GW_CU_CHECK_ERR(cudaFree(graph));
    GW_CU_CHECK_ERR(cudaFree(node_id_to_pos));
    GW_CU_CHECK_ERR(cudaFree(incoming_edges));
    GW_CU_CHECK_ERR(cudaFree(incoming_edge_count));
    GW_CU_CHECK_ERR(cudaFree(outgoing_edges));
    GW_CU_CHECK_ERR(cudaFree(outgoing_edge_count));
    GW_CU_CHECK_ERR(cudaFree(scores));
    GW_CU_CHECK_ERR(cudaFree(alignment_graph));
    GW_CU_CHECK_ERR(cudaFree(read));
    GW_CU_CHECK_ERR(cudaFree(alignment_read));
    GW_CU_CHECK_ERR(cudaFree(aligned_nodes));
    if (traceback)
    {
        GW_CU_CHECK_ERR(cudaFree(traces));
    }

    return res;
}

class NWbandedTest : public ::testing::Test
{
public:
    BasicNW* nw;

public:
    void SetUp()
    {
        // initialize nw graph and read with the following data
        std::string nodes_str = "TTTAACCTAATAAATCAGTGAAGATTTAAAATATGATAATTATTGATTTTGGTGAGAGTGCAAAGAAATTTGTTACCCTCATAAGCTGAGCAGACAGATAAGATAGAAAAACAGAAGATAGAATATTAAAACCATGATAGGTACAGACTGAAAAATTCTTGGATAAATATTAAAATTTAGGCTTTAGTAGTAGATTGATGACTGTGAGGAAAAAGGATGTCCAATTGTTGAGTGACATGTAGAATGCCTTAAAATAATTTTACACGTCACTGAAAGCTATATTTATATTCAGGAAGGATATATCCCAGTCATGATTTTCTTAATAAGTTGCCCCATTTTCCAAGTTTAGCTAATTAACATTTATGTCTTCTATAATCAGGAATAGTCATTAACTGACACAGAAACAATTGGAAGCATATGTAGCCAAAAACATAAAAATTATTGCATCCAAATAATGATAAAGTAAAATATTAAAAAATATAGTCTTCTAAAT";
        std::string read_str  = "TTTCACCTAGAAAATCAGTGAAGATTTAACAAAAAAAAAAAAAAAAAAAAAAAAATATTGATAATTATTGATTTTGGTGAGAGTGCAAAGCAATTGGCTACCCTCATAAGCTGAGCAGAAGATAAGATAGACAACAGAAGATAGAATAGTTAAACCATGATAGGTACAGACTGCAAAAAAATTCGATAAATATTAAAATTTAGGGCTTTAGTATATATTGATGACTGAGAAAAATCGTGATGTGCAATTGTGCGTGACATGTAGAATTGCCTTAAATAAAATTTAATCTGTCACTGAAGCTATATTTATATTCAGGAAGGATATATCCCAGTCATTGCTTTTCTTAATAAGTGCCCATGTTCCAAGTTTAGCCTAATTAAAAACTTTATGTCTTCTATATCAGAATAGTCATTAATGCACAGAAACAATTTGCGAAGGCATTATGTAGCAAAAACATAAAAAATTATTGCAGCCAAATAATGAATAAAAGTAACACAATCATTTAAAAAAATTATTATGTACTTCTAAAC";
        // extract data to build BasicNW
        std::vector<uint8_t> nodes(nodes_str.begin(), nodes_str.end());
        std::vector<int16_t> sorted_graph(nodes.size());
        std::iota(sorted_graph.begin(), sorted_graph.end(), 0);
        SizeTVec2D outgoing_edges(nodes.size());
        for (size_t i = 0; i < outgoing_edges.size() - 1; i++)
        {
            outgoing_edges[i].push_back(i + 1);
        }
        std::vector<uint8_t> read(read_str.begin(), read_str.end());
        // setup nw
        nw = new BasicNW(nodes, sorted_graph, outgoing_edges, read);
    }

    void TearDown() { delete nw; }
};

TEST_F(NWbandedTest, NWSaticBandvsFull)
{
    auto full_alignment_results = testNW(*nw);
    auto static_banded_results  = testNWbanded(*nw, false);
    // verify alignment_graph
    EXPECT_EQ(full_alignment_results.first, static_banded_results.first);
    // verify alignment_read
    EXPECT_EQ(full_alignment_results.second, static_banded_results.second);
}

TEST_F(NWbandedTest, NWAdaptiveBandvsFull)
{
    auto full_alignment_results  = testNW(*nw);
    auto adaptive_banded_results = testNWbanded(*nw, true);
    // verify alignment_graph
    EXPECT_EQ(full_alignment_results.first, adaptive_banded_results.first);
    // verify alignment_read
    EXPECT_EQ(full_alignment_results.second, adaptive_banded_results.second);
}

TEST_F(NWbandedTest, NWSaticBandTracebackvsFull)
{
    auto full_alignment_results   = testNW(*nw);
    auto static_banded_tb_results = testNWbanded(*nw, false, true);
    // verify alignment_graph
    EXPECT_EQ(full_alignment_results.first, static_banded_tb_results.first);
    // verify alignment_read
    EXPECT_EQ(full_alignment_results.second, static_banded_tb_results.second);
}

TEST_F(NWbandedTest, NWAdaptiveBandTracebackvsFull)
{
    auto full_alignment_results     = testNW(*nw);
    auto adaptive_banded_tb_results = testNWbanded(*nw, true, true);
    // verify alignment_graph
    EXPECT_EQ(full_alignment_results.first, adaptive_banded_tb_results.first);
    // verify alignment_read
    EXPECT_EQ(full_alignment_results.second, adaptive_banded_tb_results.second);
}

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
