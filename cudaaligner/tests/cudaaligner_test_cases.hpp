

#pragma once

#include <vector>
#include <string>

namespace claraparabricks
{

namespace genomeworks
{

struct TestCaseData
{
    std::string target;
    std::string query;
};

std::vector<TestCaseData> create_cudaaligner_test_cases();

} // namespace genomeworks

} // namespace claraparabricks
