#include "gtest/gtest.h"
#include "../src/cudapoa_kernels.cuh" //addAlignment, CUDAPOA_MAX_NODE_EDGES, CUDAPOA_MAX_NODE_ALIGNMENTS
#include <cudautils/cudautils.hpp>    //GW_CU_CHECK_ERR
#include <utils/stringutils.hpp>      //array_to_string

namespace genomeworks
{

namespace cudapoa
{

// alias for the 2d vector graph representation
typedef std::vector<std::vector<uint16_t>> Uint16Vec2D;


class BasicGraph {
public:
    BasicGraph(std::vector<uint8_t> nodes, Uint16Vec2D outgoing_edges, Uint16Vec2D node_alignments, std::vector<uint16_t> node_coverage_counts)
      : nodes_(nodes), outgoing_edges_(outgoing_edges), node_alignments_(node_alignments), node_coverage_counts_(node_coverage_counts)
    {
        graph_complete_ = true;
        node_count_ = nodes_.size();
    }

    BasicGraph(uint16_t *outgoing_edges, uint16_t *outgoing_edge_count, uint16_t node_count) 
    {
        graph_complete_ = false;
        outgoing_edges_ = edges_to_graph(outgoing_edges, outgoing_edge_count, node_count);
    }

    BasicGraph(Uint16Vec2D outgoing_edges)
    {
        graph_complete_ = false;
        outgoing_edges_ = outgoing_edges;
    }

    BasicGraph() = delete;

    //fill in the edge-related pointers based on stored graph
    void get_edges(uint16_t *incoming_edges, uint16_t *incoming_edge_count,
                       uint16_t *outgoing_edges, uint16_t *outgoing_edge_count) const
    {
        uint16_t out_node;
        for (int i = 0; i < node_count_; i++)
        {
            outgoing_edge_count[i] = outgoing_edges_[i].size();
            for (int j = 0; j < (int)outgoing_edges_[i].size(); j++)
            {
                out_node = outgoing_edges_[i][j];
                uint16_t in_edge_count = incoming_edge_count[out_node];
                incoming_edge_count[out_node] = in_edge_count + 1;
                incoming_edges[out_node * CUDAPOA_MAX_NODE_EDGES + in_edge_count] =  i;
                outgoing_edges[i * CUDAPOA_MAX_NODE_EDGES + j] = out_node;
            }
        }
    }
    //fill in the nodes and node_count pointer
    void get_nodes(uint8_t *nodes, uint16_t *node_count) const {
        for (int i = 0; i < nodes_.size(); i++) {
            nodes[i] = nodes_[i];
        }
        *node_count = node_count_;
    }
    //fill in the node_alignments and node_alignment_count pointers
    void get_node_alignments(uint16_t *node_alignments, uint16_t *node_alignment_count) const {
        uint16_t aligned_node;
        for (int i = 0; i < node_alignments_.size(); i++) {
            for (int j = 0; j < node_alignments_[i].size(); j++) {
                aligned_node = node_alignments_[i][j];
                node_alignments[i * CUDAPOA_MAX_NODE_ALIGNMENTS + j] = aligned_node;
                node_alignment_count[i]++;
            }
        }
    }
    //fill in node_coverage_counts pointer
    void get_node_coverage_counts(uint16_t *node_coverage_counts) const {
        for (int i = 0; i < node_coverage_counts_.size(); i++) {
            node_coverage_counts[i] = node_coverage_counts_[i];
        }
    }

    // convert results from outgoing_edges to Uint16Vec2D graph
    Uint16Vec2D edges_to_graph(uint16_t *outgoing_edges, uint16_t *outgoing_edge_count, uint16_t node_count)
    {
        Uint16Vec2D graph(node_count);
        for (uint16_t i = 0; i < node_count; i++)
        {
            for (uint16_t j = 0; j < outgoing_edge_count[i]; j++)
            {
                graph[i].push_back(outgoing_edges[i * CUDAPOA_MAX_NODE_EDGES + j]);
            }
        }
        return graph;
    }

    bool is_complete() const {
        return graph_complete_;
    }

    bool operator==(const BasicGraph& rhs) const {
        return this->outgoing_edges_ == rhs.outgoing_edges_;
    }

  protected:
    bool graph_complete_;
    std::vector<uint8_t> nodes_;
    Uint16Vec2D outgoing_edges_; //this uniquely represents the graph structure; equality of BasicGraph is based on this member.
    Uint16Vec2D node_alignments_;
    std::vector<uint16_t> node_coverage_counts_;
    uint16_t node_count_;

};

class BasicAlignment {
public:
    BasicAlignment(std::vector<uint8_t> nodes, Uint16Vec2D outgoing_edges,
                 Uint16Vec2D node_alignments, std::vector<uint16_t> node_coverage_counts,
                 std::vector<uint8_t> read, std::vector<int16_t> alignment_graph, std::vector<int16_t> alignment_read)
      : graph(nodes, outgoing_edges, node_alignments, node_coverage_counts), read_(read), alignment_graph_(alignment_graph), alignment_read_(alignment_read)
    {
      //do nothing for now
    }

    void get_alignments(int16_t *alignment_graph, int16_t *alignment_read, uint16_t *alignment_length) const {
        for (int i = 0; i < alignment_graph_.size(); i++) {
            alignment_graph[i] = alignment_graph_[i];
            alignment_read[i] = alignment_read_[i];
        }
        *alignment_length = alignment_graph_.size();
    }

    void get_read(uint8_t *read) const {
        for (int i = 0; i < read_.size(); i++) {
            read[i] = read_[i];
        }
    }

    void get_graph_buffers(uint16_t *incoming_edges, uint16_t *incoming_edge_count,
                           uint16_t *outgoing_edges, uint16_t *outgoing_edge_count,
                           uint8_t *nodes, uint16_t *node_count,
                           uint16_t *node_alignments, uint16_t *node_alignment_count,
                           uint16_t *node_coverage_counts) const {
        if (!graph.is_complete()) {
            throw "graph is incomplete; unable to fill the buffers.";
        }
        graph.get_edges(incoming_edges, incoming_edge_count, outgoing_edges, outgoing_edge_count);
        graph.get_nodes(nodes, node_count);
        graph.get_node_alignments(node_alignments, node_alignment_count);
        graph.get_node_coverage_counts(node_coverage_counts);
    }

    void get_alignment_buffers(int16_t *alignment_graph, int16_t *alignment_read, uint16_t *alignment_length,
                                    uint8_t *read) const {
        get_alignments(alignment_graph, alignment_read, alignment_length);
        get_read(read);
    }

protected: 
    BasicGraph graph;
    std::vector<uint8_t> read_;
    std::vector<int16_t> alignment_graph_;
    std::vector<int16_t> alignment_read_;
};

typedef std::pair<BasicGraph, BasicAlignment> AddAlginmentTestPair;
    // create a vector of test cases
std::vector<AddAlginmentTestPair> getAddAlignmentTestCases()
{

    std::vector<AddAlginmentTestPair> test_cases;

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
    BasicGraph ans_1({{}, {0}, {1}, {2, 4}, {1}});
    BasicAlignment ali_1({'A', 'A', 'A', 'A'},                  //nodes
                         {{}, {0}, {1}, {2}},                   //outgoing_edges
                         {{}, {}, {}, {}},                      //node_alignments
                         {1, 1, 1, 1},                          //node_coverage_counts
                         {'A', 'A', 'T', 'A'},                  //read
                         {0, 1, 2, 3},                          //alignment_graph
                         {0, 1, 2, 3});                         //alignment_read
    test_cases.emplace_back(std::move(ans_1), std::move(ali_1));

    /*
     * read:            A   T   C   G   A
     * graph:           A — T — C — G
     * alignment graph: 0   1   2   3  -1
     * alignment read:  0   1   2   3   4
     *                        
     * final graph      A — T — C — G — A
     * 
     */
    BasicGraph ans_2({{}, {0}, {1}, {2}, {3}});
    BasicAlignment ali_2({'A', 'T', 'C', 'G'},                  //nodes
                         {{}, {0}, {1}, {2}},                   //outgoing_edges
                         {{}, {}, {}, {}},                      //node_alignments
                         {1, 1, 1, 1},                          //node_coverage_counts
                         {'A', 'T', 'C', 'G', 'A'},             //read
                         {0, 1, 2, 3, -1},                      //alignment_graph
                         {0, 1, 2, 3, 4});                      //alignment_read
    test_cases.emplace_back(std::move(ans_2), std::move(ali_2));

    /*
     * read:            A   T   C   G
     *                      A
     *                    /   \
     * graph:           A — C — C — G
     * alignment graph: 0   4   2   3
     * alignment read:  0   1   2   3
     *                      T
     *                    /   \
     * final graph      A — C — C — G
     *                    \   /
     *                      A
     */
    BasicGraph ans_3({{}, {0}, {1, 4, 5}, {2}, {0}, {0}});
    BasicAlignment ali_3({'A', 'A', 'C', 'G', 'C'},             //nodes
                         {{}, {0}, {1, 4}, {2}, {0}},           //outgoing_edges
                         {{}, {}, {}, {}},                      //node_alignments
                         {2, 1, 2, 2, 1},                       //node_coverage_counts
                         {'A', 'T', 'C', 'G'},                  //read
                         {0, 4, 2, 3},                          //alignment_graph
                         {0, 1, 2, 3});                         //alignment_read
    test_cases.emplace_back(std::move(ans_3), std::move(ali_3));

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
    BasicGraph ans_4({{}, {0}, {1}, {2}, {3, 0}});
    BasicAlignment ali_4({'A', 'T', 'T', 'G', 'A'},             //nodes
                         {{}, {0}, {1}, {2}, {3}},              //outgoing_edges
                         {{}, {}, {}, {}},                      //node_alignments
                         {1, 1, 1, 1, 1},                       //node_coverage_counts
                         {'A', 'A'},                            //read
                         {0, 1, 2, 3, 4},                       //alignment_graph
                         {0, -1, -1, -1, 1});                   //alignment_read
    test_cases.emplace_back(std::move(ans_4), std::move(ali_4));

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
    BasicGraph ans_5({{}, {0}, {1}, {2, 6, 7}, {3}, {0}, {5}, {5}});
    BasicAlignment ali_5({'A', 'T', 'G', 'T', 'A', 'C', 'A'},   //nodes
                         {{}, {0}, {1}, {2, 6}, {3}, {0}, {5}}, //outgoing_edges
                         {{}, {}, {}, {}},                      //node_alignments
                         {2, 1, 1, 2, 2, 1, 1},                 //node_coverage_counts
                         {'A', 'C', 'T', 'T', 'A'},             //read
                         {0, 5, 6, 3, 4},                       //alignment_graph
                         {0, 1, 2, 3, 4});                      //alignment_read
    test_cases.emplace_back(std::move(ans_5), std::move(ali_5));

    //add more test cases below

    return test_cases;
}


// host function for calling the kernel to test topsort device function.
BasicGraph testAddAlignment(const BasicAlignment &obj)
{
    //declare device buffer
    uint8_t *nodes;
    uint16_t *node_count;
    uint16_t *node_alignments;
    uint16_t *node_alignment_count;
    uint16_t *incoming_edges;
    uint16_t *incoming_edge_count;
    uint16_t *outgoing_edges;
    uint16_t *outgoing_edge_count;
    uint16_t *incoming_edge_w;
    uint16_t *outgoing_edge_w;
    uint16_t *alignment_length;
    uint16_t *graph;
    int16_t  *alignment_graph;
    uint8_t  *read;
    int16_t  *alignment_read;
    uint16_t *node_coverage_counts;

    //allocate unified memory so they can be accessed by both host and device.
    GW_CU_CHECK_ERR(cudaMallocManaged((void **)&nodes, CUDAPOA_MAX_NODES_PER_WINDOW * sizeof(uint8_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void **)&node_count, sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void **)&node_alignments, CUDAPOA_MAX_NODES_PER_WINDOW * CUDAPOA_MAX_NODE_ALIGNMENTS * sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void **)&node_alignment_count, CUDAPOA_MAX_NODES_PER_WINDOW * sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void **)&incoming_edges, CUDAPOA_MAX_NODES_PER_WINDOW * CUDAPOA_MAX_NODE_EDGES * sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void **)&incoming_edge_count, CUDAPOA_MAX_NODES_PER_WINDOW * sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void **)&outgoing_edges, CUDAPOA_MAX_NODES_PER_WINDOW * CUDAPOA_MAX_NODE_EDGES * sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void **)&outgoing_edge_count, CUDAPOA_MAX_NODES_PER_WINDOW * sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void **)&incoming_edge_w, CUDAPOA_MAX_NODES_PER_WINDOW * CUDAPOA_MAX_NODE_EDGES * sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void **)&outgoing_edge_w, CUDAPOA_MAX_NODES_PER_WINDOW * CUDAPOA_MAX_NODE_EDGES * sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void **)&alignment_length, sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void **)&graph, CUDAPOA_MAX_NODES_PER_WINDOW * sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void **)&alignment_graph, CUDAPOA_MAX_SEQUENCE_SIZE * sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void **)&read, CUDAPOA_MAX_SEQUENCE_SIZE * sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void **)&alignment_read, CUDAPOA_MAX_SEQUENCE_SIZE * sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMallocManaged((void **)&node_coverage_counts, CUDAPOA_MAX_NODES_PER_WINDOW * sizeof(uint16_t)));

    //initialize all 'count' buffers
    memset((void **)node_alignment_count, 0, CUDAPOA_MAX_NODES_PER_WINDOW * sizeof(uint16_t));
    memset((void **)incoming_edge_count, 0, CUDAPOA_MAX_NODES_PER_WINDOW * sizeof(uint16_t));
    memset((void **)outgoing_edge_count, 0, CUDAPOA_MAX_NODES_PER_WINDOW * sizeof(uint16_t));
    memset((void **)node_coverage_counts, 0, CUDAPOA_MAX_NODES_PER_WINDOW * sizeof(uint16_t));

    //calculate edge counts on host
    //3 buffers are disregarded because they don't affect correctness -- incoming_edge_w, outgoing_edge_w, graph 
    obj.get_graph_buffers(incoming_edges, incoming_edge_count, outgoing_edges, outgoing_edge_count,
                          nodes, node_count,
                          node_alignments, node_alignment_count,
                          node_coverage_counts);
    obj.get_alignment_buffers(alignment_graph, alignment_read, alignment_length, read);

    // call the host wrapper of topsort kernel
    addAlignment(nodes,
                 node_count,
                 node_alignments, node_alignment_count,
                 incoming_edges, incoming_edge_count,
                 outgoing_edges, outgoing_edge_count,
                 incoming_edge_w, outgoing_edge_w,
                 alignment_length,
                 graph,
                 alignment_graph,
                 read,
                 alignment_read,
                 node_coverage_counts);

    GW_CU_CHECK_ERR(cudaDeviceSynchronize());

    //input and output buffers are the same ones in unified memory, so the results are updated in place
    //create and return a new BasicGraph object that encodes the resulting graph structure after adding the alignment
    BasicGraph res(outgoing_edges, outgoing_edge_count, *node_count);

    GW_CU_CHECK_ERR(cudaFree(nodes));
    GW_CU_CHECK_ERR(cudaFree(node_count));
    GW_CU_CHECK_ERR(cudaFree(node_alignments));
    GW_CU_CHECK_ERR(cudaFree(node_alignment_count));
    GW_CU_CHECK_ERR(cudaFree(incoming_edges));
    GW_CU_CHECK_ERR(cudaFree(incoming_edge_count));
    GW_CU_CHECK_ERR(cudaFree(outgoing_edges));
    GW_CU_CHECK_ERR(cudaFree(outgoing_edge_count));
    GW_CU_CHECK_ERR(cudaFree(incoming_edge_w));
    GW_CU_CHECK_ERR(cudaFree(outgoing_edge_w));
    GW_CU_CHECK_ERR(cudaFree(alignment_length));
    GW_CU_CHECK_ERR(cudaFree(graph));
    GW_CU_CHECK_ERR(cudaFree(alignment_graph));
    GW_CU_CHECK_ERR(cudaFree(read));
    GW_CU_CHECK_ERR(cudaFree(alignment_read));
    GW_CU_CHECK_ERR(cudaFree(node_coverage_counts));

    return res;
}



using ::testing::TestWithParam;
using ::testing::ValuesIn;


class AddAlignmentTest : public TestWithParam<AddAlginmentTestPair>
{
  public:
    void SetUp() {}

    BasicGraph runAddAlignment(const BasicAlignment& ali)
    {
        return testAddAlignment(ali);
    }
};

TEST_P(AddAlignmentTest, TestAddAlignmentCorrectness)
{
    const auto test_case = GetParam();
    EXPECT_EQ(test_case.first, runAddAlignment(test_case.second));
}

INSTANTIATE_TEST_SUITE_P(TestAddAlginment, AddAlignmentTest, ValuesIn(getAddAlignmentTestCases()));

} // namespace cudapoa

} // namespace genomeworks
