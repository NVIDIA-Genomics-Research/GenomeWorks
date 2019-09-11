/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma  once

#include "gmock/gmock.h"

#include "../src/index_gpu.cuh"
#include "../src/minimizer.hpp"

namespace claragenomics {

    class MockIndex : public IndexGPU<Minimizer> {
    public:
        MOCK_CONST_METHOD0(read_id_to_read_name, std::vector<std::string>&());
        MOCK_CONST_METHOD0(read_id_to_read_length, std::vector<std::uint32_t>&());
    };

}
