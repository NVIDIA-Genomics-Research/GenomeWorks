#include "gtest/gtest.h"
#include "../src/aligner_global.hpp"
#include "cudaaligner/alignment.hpp"

namespace genomeworks {

namespace cudaaligner {

// Test adding alignments to Aligner objects
TEST(TestCudaAligner, TestAlignmentAddition)
{
    std::unique_ptr<AlignerGlobal> aligner = std::make_unique<AlignerGlobal>(5, 5, 5, 0);
    ASSERT_EQ(StatusType::success, aligner->add_alignment("ATCG", 4, "TACG", 4));
    ASSERT_EQ(StatusType::success, aligner->add_alignment("ATCG", 4, "TACG", 4));
    ASSERT_EQ(StatusType::success, aligner->add_alignment("ATCG", 4, "TACG", 4));

    ASSERT_EQ(3, aligner->num_alignments());

    ASSERT_EQ(StatusType::exceeded_max_length, aligner->add_alignment("ATCGAT", 6, "TACG", 4));
    ASSERT_EQ(StatusType::exceeded_max_length, aligner->add_alignment("ATCG", 4, "TACGAG", 6));

    ASSERT_EQ(3, aligner->num_alignments());

    ASSERT_EQ(StatusType::success, aligner->add_alignment("ATCG", 4, "TACG", 4));
    ASSERT_EQ(StatusType::success, aligner->add_alignment("ATCG", 4, "TACG", 4));

    ASSERT_EQ(5, aligner->num_alignments());

    ASSERT_EQ(StatusType::exceeded_max_alignments, aligner->add_alignment("ATCG", 4, "TACG", 4));

    ASSERT_EQ(5, aligner->num_alignments());
}

#pragma message("TODO: Add test for checking proper alignment")
TEST(TestCudaAligner, TestAlignmentKernel)
{
    std::unique_ptr<AlignerGlobal> aligner = std::make_unique<AlignerGlobal>(5, 5, 1, 0);
    ASSERT_EQ(StatusType::success, aligner->add_alignment("AAAA", 4, "TTAT", 4));

    aligner->align_all();

    const std::vector<std::shared_ptr<Alignment>>& alignments = aligner->get_alignments();
    ASSERT_EQ(StatusType::success, alignments.back()->get_status());
    ASSERT_EQ(AlignmentType::global, alignments.back()->get_alignment_type());
}

}

}
