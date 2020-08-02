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
