

#include <claraparabricks/genomeworks/utils/mathutils.hpp>

#include "gtest/gtest.h"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

TEST(TestCudaAlignerMisc, CeilingDivide)
{
    EXPECT_EQ(ceiling_divide(0, 5), 0);
    EXPECT_EQ(ceiling_divide(5, 5), 1);
    EXPECT_EQ(ceiling_divide(10, 5), 2);
    EXPECT_EQ(ceiling_divide(20, 5), 4);

    EXPECT_EQ(ceiling_divide(6, 5), 2);
    EXPECT_EQ(ceiling_divide(4, 5), 1);
}

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
