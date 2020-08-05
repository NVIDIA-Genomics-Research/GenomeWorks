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

#include <claraparabricks/genomeworks/utils/graph.hpp>

#include "gtest/gtest.h"

namespace claraparabricks
{

namespace genomeworks
{

TEST(GraphTest, DirectedGraph)
{
    DirectedGraph graph;

    // Sample graph
    //      3
    //      ^
    //      |
    // 1 -> 2 -> 5
    //      |    ^
    //      u    |
    //      4 ---|

    graph.add_edge(1, 2);
    graph.add_edge(2, 5);
    graph.add_edge(2, 3);
    graph.add_edge(2, 4);
    graph.add_edge(4, 5);

    const auto& adjacent_nodes_to_2 = graph.get_adjacent_nodes(2);
    EXPECT_NE(std::find(adjacent_nodes_to_2.begin(), adjacent_nodes_to_2.end(), 3), adjacent_nodes_to_2.end());
    EXPECT_NE(std::find(adjacent_nodes_to_2.begin(), adjacent_nodes_to_2.end(), 4), adjacent_nodes_to_2.end());
    EXPECT_NE(std::find(adjacent_nodes_to_2.begin(), adjacent_nodes_to_2.end(), 5), adjacent_nodes_to_2.end());
    EXPECT_EQ(std::find(adjacent_nodes_to_2.begin(), adjacent_nodes_to_2.end(), 1), adjacent_nodes_to_2.end());

    const auto& adjacent_nodes_to_3 = graph.get_adjacent_nodes(3);
    EXPECT_EQ(std::find(adjacent_nodes_to_3.begin(), adjacent_nodes_to_3.end(), 2), adjacent_nodes_to_3.end());
}

TEST(GraphTest, UndirectedGraph)
{
    UndirectedGraph graph;

    // Sample graph
    //      3
    //      |
    //      |
    // 1 -- 2 -- 5
    //      |    |
    //      |    |
    //      4 ---|

    graph.add_edge(1, 2);
    graph.add_edge(2, 5);
    graph.add_edge(2, 3);
    graph.add_edge(2, 4);
    graph.add_edge(4, 5);

    const auto& adjacent_nodes_to_2 = graph.get_adjacent_nodes(2);
    EXPECT_NE(std::find(adjacent_nodes_to_2.begin(), adjacent_nodes_to_2.end(), 3), adjacent_nodes_to_2.end());
    EXPECT_NE(std::find(adjacent_nodes_to_2.begin(), adjacent_nodes_to_2.end(), 4), adjacent_nodes_to_2.end());
    EXPECT_NE(std::find(adjacent_nodes_to_2.begin(), adjacent_nodes_to_2.end(), 5), adjacent_nodes_to_2.end());
    EXPECT_NE(std::find(adjacent_nodes_to_2.begin(), adjacent_nodes_to_2.end(), 1), adjacent_nodes_to_2.end());

    const auto& adjacent_nodes_to_3 = graph.get_adjacent_nodes(3);
    EXPECT_EQ(std::find(adjacent_nodes_to_3.begin(), adjacent_nodes_to_3.end(), 1), adjacent_nodes_to_3.end());
    EXPECT_NE(std::find(adjacent_nodes_to_3.begin(), adjacent_nodes_to_3.end(), 2), adjacent_nodes_to_3.end());
}

} // namespace genomeworks

} // namespace claraparabricks
