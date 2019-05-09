#include "gtest/gtest.h"
#include "../src/alignment_impl.hpp"

namespace genomeworks {

namespace cudaaligner {

static std::string QUERY= "AAAA";
static std::string TARGET = "TTATG";
static const std::vector<AlignmentState> ALIGNMENT {
    AlignmentState::mismatch,
    AlignmentState::mismatch,
    AlignmentState::match,
    AlignmentState::mismatch,
    AlignmentState::insert_into_target
};

class TestAlignmentImpl : public ::testing::Test {
    public:
        void SetUp()
        {
            alignment_.reset(new AlignmentImpl(QUERY.c_str(), QUERY.size(), TARGET.c_str(), TARGET.size()));
        }

    protected:
        std::unique_ptr<AlignmentImpl> alignment_;
};

TEST_F(TestAlignmentImpl, StringGetters)
{
    ASSERT_STREQ(QUERY.c_str(), alignment_->get_query_sequence().c_str()) << "Query doesn't match original string";
    ASSERT_EQ(QUERY.size(), alignment_->get_query_sequence().size()) << "Query length match original string";

    ASSERT_STREQ(TARGET.c_str(), alignment_->get_target_sequence().c_str()) << "Target doesn't match original string";
    ASSERT_EQ(TARGET.size(), alignment_->get_target_sequence().size()) << "Target length match original string";
}

TEST_F(TestAlignmentImpl, Status)
{
    ASSERT_EQ(StatusType::uninitialized, alignment_->get_status()) << "Initial status incorrect";

    alignment_->set_status(StatusType::success);
    ASSERT_EQ(StatusType::success, alignment_->get_status()) << "Status not set properly";
}

TEST_F(TestAlignmentImpl, Type)
{
    ASSERT_EQ(AlignmentType::unset, alignment_->get_alignment_type()) << "Initial type incorrect";

    alignment_->set_alignment_type(AlignmentType::global);
    ASSERT_EQ(AlignmentType::global, alignment_->get_alignment_type()) << "Type not set properly";
}

TEST_F(TestAlignmentImpl, AlignmentState)
{
    alignment_->set_alignment(ALIGNMENT);

    const std::vector<AlignmentState>& al_read = alignment_->get_alignment();
    ASSERT_EQ(ALIGNMENT.size(), al_read.size());
    for(std::size_t i = 0; i < ALIGNMENT.size(); i++)
    {
        ASSERT_EQ(ALIGNMENT.at(i), al_read.at(i));
    }
}

TEST_F(TestAlignmentImpl, AlignmentFormatting)
{
    alignment_->set_alignment(ALIGNMENT);

    FormattedAlignment formatted_alignment = alignment_->format_alignment();
    std::string query = formatted_alignment.first;
    std::string target = formatted_alignment.second;
    ASSERT_STREQ("AAAA-", query.c_str());
    ASSERT_STREQ("TTATG", target.c_str());
}

}

}
