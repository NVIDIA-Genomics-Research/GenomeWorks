#include "gtest/gtest.h"
#include "cudapoa/batch.hpp"

namespace genomeworks
{

namespace cudapoa
{

class TestCudapoaBatch : public ::testing::Test
{
  public:
    void SetUp()
    {
        // Do noting for now, but place for
        // constructing test objects.
    }
};

TEST_F(TestCudapoaBatch, DummyTest)
{
    ASSERT_EQ(StatusType::SUCCESS, StatusType::SUCCESS);
}

} // namespace cudapoa

} // namespace genomeworks
