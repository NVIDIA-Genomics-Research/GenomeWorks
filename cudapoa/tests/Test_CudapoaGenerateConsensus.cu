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

#include "../src/cudapoa_generate_consensus.cuh" //generateConsensusHost, CUDAPOA_MAX_NODE_EDGES, CUDAPOA_MAX_NODE_ALIGNMENTS
#include "sorted_graph.hpp"                      //SortedGraph

#include <claraparabricks/genomeworks/utils/cudautils.hpp>            //GW_CU_CHECK_ERR
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp> //get_size

#include "gtest/gtest.h"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

class BasicGenerateConsensus
{

public:
    BasicGenerateConsensus(std::vector<uint8_t> nodes, std::vector<int16_t> sorted_graph, Int16Vec2D node_alignments,
                           Int16Vec2D outgoing_edges, std::vector<uint16_t> node_coverage_counts, Uint16Vec2D outgoing_edge_w)
        : graph_(nodes, sorted_graph, node_alignments, node_coverage_counts, outgoing_edges)
        , outgoing_edge_w_(outgoing_edge_w)
        , outgoing_edges_(outgoing_edges)
    {
    }

    void get_graph_buffers(uint8_t* nodes, int16_t* node_count,
                           int16_t* sorted_poa, int16_t* node_id_to_pos,
                           int16_t* incoming_edges, uint16_t* incoming_edge_count,
                           int16_t* outgoing_edges, uint16_t* outgoing_edge_count,
                           uint16_t* incoming_edge_w, uint16_t* node_coverage_counts,
                           int16_t* node_alignments, uint16_t* node_alignment_count) const
    {
        graph_.get_nodes(nodes, node_count);
        graph_.get_sorted_graph(sorted_poa);
        graph_.get_node_id_to_pos(node_id_to_pos);
        graph_.get_node_coverage_counts(node_coverage_counts);
        graph_.get_edges(incoming_edges, incoming_edge_count, outgoing_edges, outgoing_edge_count);
        graph_.get_node_alignments(node_alignments, node_alignment_count);
        get_incoming_edge_w(incoming_edge_w);
    }

    void get_incoming_edge_w(uint16_t* incoming_edge_w) const
    {
        auto outgoing_edges = graph_.get_outgoing_edges();
        for (int i = 0; i < get_size(outgoing_edges); i++)
        {
            for (int j = 0; j < get_size(outgoing_edges[i]); j++)
            {
                int16_t to_node                                       = outgoing_edges[i][j];
                incoming_edge_w[to_node * CUDAPOA_MAX_NODE_EDGES + i] = outgoing_edge_w_[i][j];
            }
        }
    }

protected:
    SortedGraph graph_;
    Int16Vec2D outgoing_edges_;
    Uint16Vec2D outgoing_edge_w_;
};

typedef std::pair<std::string, BasicGenerateConsensus> GenerateConsensusTestPair;
// create a vector of test cases
std::vector<GenerateConsensusTestPair> getGenerateConsensusTestCases()
{

    std::vector<GenerateConsensusTestPair> test_cases;

    /*
     *                  T
     *                 / \
     * graph      A — A   A
     *                 \ /
     *                  A
     */
    std::string ans_1 = "ATAA";
    BasicGenerateConsensus gc_1({'A', 'A', 'A', 'A', 'T'},    //nodes
                                {0, 1, 2, 4, 3},              //sorted_graph
                                {{}, {}, {4}, {}, {2}},       //node_alignments
                                {{1}, {2, 4}, {3}, {}, {3}},  //outgoing_edges
                                {2, 2, 1, 2, 1},              //node_coverage_counts
                                {{5}, {4, 3}, {2}, {}, {1}}); //outgoing_edge_w
    test_cases.emplace_back(std::move(ans_1), std::move(gc_1));

    /*
     * graph   A — T — C — G — A
     */
    std::string ans_2 = "AGCTA";
    BasicGenerateConsensus gc_2({'A', 'T', 'C', 'G', 'A'}, //nodes
                                {0, 1, 2, 3, 4},           //sorted_graph
                                {{}, {}, {}, {}, {}},      //node_alignments
                                {{1}, {2}, {3}, {4}, {}},  //outgoing_edges
                                {1, 1, 1, 1, 1},           //node_coverage_counts
                                {{4}, {3}, {2}, {1}, {}});
    test_cases.emplace_back(std::move(ans_2), std::move(gc_2));

    /*
     *                T
     *              /   \
     * graph      A — C — C — G
     *              \   /
     *                A
     */
    std::string ans_3 = "GCCA";
    BasicGenerateConsensus gc_3({'A', 'A', 'C', 'G', 'C', 'T'},       //nodes
                                {0, 1, 4, 5, 2, 3},                   //sorted_graph
                                {{}, {4, 5}, {}, {}, {1, 5}, {1, 4}}, //node_alignments
                                {{1, 4, 5}, {2}, {3}, {}, {2}, {2}},  //outgoing_edges
                                {3, 1, 3, 3, 1, 1},                   //node_coverage_counts
                                {{7, 6, 5}, {4}, {3}, {}, {2}, {1}});
    test_cases.emplace_back(std::move(ans_3), std::move(gc_3));

    /*
     * graph      A — T — T — G — A
     *             \_____________/
     */
    std::string ans_4 = "AGTTA";
    BasicGenerateConsensus gc_4({'A', 'T', 'T', 'G', 'A'},   //nodes
                                {0, 1, 2, 3, 4},             //sorted_graph
                                {{}, {}, {}, {}, {}},        //node_alignments
                                {{1, 4}, {2}, {3}, {4}, {}}, //outgoing_edges
                                {2, 1, 1, 1, 2},             //node_coverage_counts
                                {{5, 4}, {3}, {2}, {1}, {}});
    test_cases.emplace_back(std::move(ans_4), std::move(gc_4));

    /*
     *                T — G   
     *              /       \
     * graph      A — C — A — T — A
     *                  \   /
     *                    T
     */
    std::string ans_5 = "ATTCA";
    BasicGenerateConsensus gc_5({'A', 'T', 'G', 'T', 'A', 'C', 'A', 'T'},       //nodes
                                {0, 1, 5, 2, 6, 7, 3, 4},                       //sorted_graph
                                {{}, {5}, {6, 7}, {}, {}, {1}, {2, 7}, {2, 6}}, //node_alignments
                                {{1, 5}, {2}, {3}, {4}, {}, {6, 7}, {3}, {3}},  //outgoing_edges
                                {3, 1, 1, 3, 3, 2, 1, 1},                       //node_coverage_counts
                                {{9, 8}, {7}, {6}, {5}, {}, {4, 3}, {2}, {1}});

    test_cases.emplace_back(std::move(ans_5), std::move(gc_5));

    //add more test cases below

    return test_cases;
}

// host function for calling the kernel to test topsort device function.
std::string testGenerateConsensus(const BasicGenerateConsensus& obj)
{
    //declare device buffer
    uint8_t* nodes                 = nullptr;
    int16_t* node_count            = nullptr;
    int16_t* graph                 = nullptr;
    int16_t* node_id_to_pos        = nullptr;
    int16_t* incoming_edges        = nullptr;
    uint16_t* incoming_edge_count  = nullptr;
    int16_t* outgoing_edges        = nullptr;
    uint16_t* outgoing_edge_count  = nullptr;
    uint16_t* incoming_edge_w      = nullptr;
    uint16_t* node_coverage_counts = nullptr;
    int16_t* node_alignments       = nullptr;
    uint16_t* node_alignment_count = nullptr;

    //buffers that don't need initialization
    int16_t* predecessors = nullptr;
    int32_t* scores       = nullptr;
    uint8_t* consensus    = nullptr;
    uint16_t* coverage    = nullptr;

    //default data size limits
    BatchConfig batch_size;

    //allocate unified memory so they can be accessed by both host and device.
    GW_CU_CHECK_ERR(cudaMallocManaged(&nodes, batch_size.max_nodes_per_graph * sizeof(uint8_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged(&node_count, sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged(&graph, batch_size.max_nodes_per_graph * sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged(&node_id_to_pos, batch_size.max_nodes_per_graph * sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged(&incoming_edges, batch_size.max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES * sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged(&incoming_edge_count, batch_size.max_nodes_per_graph * sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged(&outgoing_edges, batch_size.max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES * sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged(&outgoing_edge_count, batch_size.max_nodes_per_graph * sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged(&incoming_edge_w, batch_size.max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES * sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged(&node_coverage_counts, batch_size.max_nodes_per_graph * sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged(&node_alignments, batch_size.max_nodes_per_graph * CUDAPOA_MAX_NODE_ALIGNMENTS * sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged(&node_alignment_count, batch_size.max_nodes_per_graph * sizeof(uint16_t)));

    GW_CU_CHECK_ERR(cudaMallocManaged(&predecessors, batch_size.max_nodes_per_graph * sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged(&scores, batch_size.max_nodes_per_graph * sizeof(int32_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged(&consensus, batch_size.max_consensus_size * sizeof(uint8_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged(&coverage, batch_size.max_consensus_size * sizeof(uint16_t)));

    //initialize all 'count' buffers
    memset(incoming_edge_count, 0, batch_size.max_nodes_per_graph * sizeof(uint16_t));
    memset(outgoing_edge_count, 0, batch_size.max_nodes_per_graph * sizeof(uint16_t));
    memset(node_coverage_counts, 0, batch_size.max_nodes_per_graph * sizeof(uint16_t));
    memset(node_alignment_count, 0, batch_size.max_nodes_per_graph * sizeof(uint16_t));

    //calculate edge counts on host
    obj.get_graph_buffers(nodes, node_count,
                          graph, node_id_to_pos,
                          incoming_edges, incoming_edge_count,
                          outgoing_edges, outgoing_edge_count,
                          incoming_edge_w, node_coverage_counts,
                          node_alignments, node_alignment_count);

    // call the host wrapper of topsort kernel
    generateConsensusTestHost(nodes,
                              *node_count,
                              graph,
                              node_id_to_pos,
                              incoming_edges,
                              incoming_edge_count,
                              outgoing_edges,
                              outgoing_edge_count,
                              incoming_edge_w,
                              predecessors,
                              scores,
                              consensus,
                              coverage,
                              node_coverage_counts,
                              node_alignments,
                              node_alignment_count,
                              batch_size.max_consensus_size);

    GW_CU_CHECK_ERR(cudaDeviceSynchronize());

    //input and output buffers are the same ones in unified memory, so the results are updated in place
    //create and return a new BasicGraph object that encodes the resulting graph structure after adding the alignment
    std::string res((char*)consensus);

    GW_CU_CHECK_ERR(cudaFree(nodes));
    GW_CU_CHECK_ERR(cudaFree(node_count));
    GW_CU_CHECK_ERR(cudaFree(graph));
    GW_CU_CHECK_ERR(cudaFree(node_id_to_pos));
    GW_CU_CHECK_ERR(cudaFree(incoming_edges));
    GW_CU_CHECK_ERR(cudaFree(incoming_edge_count));
    GW_CU_CHECK_ERR(cudaFree(outgoing_edges));
    GW_CU_CHECK_ERR(cudaFree(outgoing_edge_count));
    GW_CU_CHECK_ERR(cudaFree(incoming_edge_w));
    GW_CU_CHECK_ERR(cudaFree(node_coverage_counts));
    GW_CU_CHECK_ERR(cudaFree(node_alignments));
    GW_CU_CHECK_ERR(cudaFree(node_alignment_count));
    GW_CU_CHECK_ERR(cudaFree(predecessors));
    GW_CU_CHECK_ERR(cudaFree(scores));
    GW_CU_CHECK_ERR(cudaFree(consensus));
    GW_CU_CHECK_ERR(cudaFree(coverage));

    return res;
}

using ::testing::TestWithParam;
using ::testing::ValuesIn;

class GenerateConsensusTest : public TestWithParam<GenerateConsensusTestPair>
{
public:
    void SetUp() {}

    std::string runGenerateConsensus(const BasicGenerateConsensus& obj)
    {
        return testGenerateConsensus(obj);
    }
};

TEST_P(GenerateConsensusTest, TestGenerateConsensuesCorrectness)
{
    const auto test_case = GetParam();
    EXPECT_EQ(test_case.first, runGenerateConsensus(test_case.second));
}

INSTANTIATE_TEST_SUITE_P(TestGenerateConsensus, GenerateConsensusTest, ValuesIn(getGenerateConsensusTestCases()));

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
