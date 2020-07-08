

#pragma once

#include "matrix_cpu.hpp"

#include <vector>
#include <string>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

matrix<int> needleman_wunsch_build_score_matrix_naive(std::string const& text, std::string const& query);

std::vector<int8_t> needleman_wunsch_cpu(std::string const& text, std::string const& query);

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
