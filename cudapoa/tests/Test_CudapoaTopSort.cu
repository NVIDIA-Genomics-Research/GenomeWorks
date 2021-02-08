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

#include "../src/cudapoa_topsort.cuh" //runTopSort

#include <claraparabricks/genomeworks/cudapoa/batch.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>            //GW_CU_CHECK_ERR
#include <claraparabricks/genomeworks/utils/stringutils.hpp>          //array_to_string
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp> //get_size

#include "gtest/gtest.h"
#include "basic_graph.hpp"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

// alias for a test case (answer, graph)
typedef std::pair<std::string, Int16Vec2D> TopSortTestPair;

using ::testing::TestWithParam;
using ::testing::ValuesIn;

// create a vector of test cases
std::vector<TopSortTestPair> getTopSortTestCases()
{

    std::vector<TopSortTestPair> test_cases;

    Int16Vec2D outgoing_edges_1 = {{}, {}, {3}, {1}, {0, 1}, {0, 2}};
    std::string answer_1        = "4-5-0-2-3-1";
    test_cases.emplace_back(answer_1, outgoing_edges_1);

    Int16Vec2D outgoing_edges_2 = {{1, 3}, {2, 3}, {3, 4, 5}, {4, 5}, {5}, {}};
    std::string answer_2        = "0-1-2-3-4-5";
    test_cases.emplace_back(answer_2, outgoing_edges_2);

    Int16Vec2D outgoing_edges_3 = {{}, {}, {3}, {1}, {0, 1, 7}, {0, 2}, {4}, {5}};
    std::string answer_3        = "6-4-7-5-0-2-3-1";
    test_cases.emplace_back(answer_3, outgoing_edges_3);

    //add more test cases below

    return test_cases;
}

// host function for calling the kernel to test topsort device function.
std::string testTopSortDeviceUtil(int16_t node_count, Int16Vec2D outgoing_edges_vec)
{
    //declare device buffer
    int16_t* sorted_poa                 = nullptr;
    int16_t* sorted_poa_node_map        = nullptr;
    uint16_t* incoming_edge_count       = nullptr;
    int16_t* outgoing_edges             = nullptr;
    uint16_t* outgoing_edge_count       = nullptr;
    uint16_t* local_incoming_edge_count = nullptr;

    size_t graph_size = node_count * sizeof(uint16_t);

    //allocate unified memory so they can be accessed by both host and device.
    GW_CU_CHECK_ERR(cudaMallocManaged(&sorted_poa, node_count * sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged(&sorted_poa_node_map, node_count * sizeof(int16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged(&incoming_edge_count, graph_size));
    GW_CU_CHECK_ERR(cudaMallocManaged(&outgoing_edges, node_count * sizeof(int16_t) * CUDAPOA_MAX_NODE_EDGES));
    GW_CU_CHECK_ERR(cudaMallocManaged(&outgoing_edge_count, graph_size));
    GW_CU_CHECK_ERR(cudaMallocManaged(&local_incoming_edge_count, graph_size));

    //initialize incoming_edge_count & local_incoming_edge_count
    memset(incoming_edge_count, 0, graph_size);
    memset(local_incoming_edge_count, 0, graph_size);

    //calculate edge counts on host

    int16_t out_node;
    for (int i = 0; i < node_count; i++)
    {
        outgoing_edge_count[i] = get_size(outgoing_edges_vec[i]);
        for (int j = 0; j < get_size(outgoing_edges_vec[i]); j++)
        {
            out_node = outgoing_edges_vec[i][j];
            incoming_edge_count[out_node]++;
            local_incoming_edge_count[out_node]++;
            outgoing_edges[i * CUDAPOA_MAX_NODE_EDGES + j] = out_node;
        }
    }

    // call the host wrapper of topsort kernel
    runTopSort(sorted_poa,
               sorted_poa_node_map,
               node_count,
               incoming_edge_count,
               outgoing_edges,
               outgoing_edge_count,
               local_incoming_edge_count);

    GW_CU_CHECK_ERR(cudaDeviceSynchronize());

    std::string res = genomeworks::stringutils::array_to_string(sorted_poa, node_count);

    GW_CU_CHECK_ERR(cudaFree(sorted_poa));
    GW_CU_CHECK_ERR(cudaFree(sorted_poa_node_map));
    GW_CU_CHECK_ERR(cudaFree(incoming_edge_count));
    GW_CU_CHECK_ERR(cudaFree(outgoing_edges));
    GW_CU_CHECK_ERR(cudaFree(outgoing_edge_count));
    GW_CU_CHECK_ERR(cudaFree(local_incoming_edge_count));

    return res;
}

class TopSortDeviceUtilTest : public TestWithParam<TopSortTestPair>
{
public:
    void SetUp() {}

    std::string runTopSortDevice(Int16Vec2D outgoing_edges)
    {
        return testTopSortDeviceUtil(get_size(outgoing_edges), outgoing_edges);
    }
};

TEST_P(TopSortDeviceUtilTest, TestTopSotCorrectness)
{
    const auto test_case = GetParam();
    EXPECT_EQ(test_case.first, runTopSortDevice(test_case.second));
}

INSTANTIATE_TEST_SUITE_P(TestTopSort, TopSortDeviceUtilTest, ValuesIn(getTopSortTestCases()));

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
