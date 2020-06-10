#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

macro (GitVersion)
    execute_process(COMMAND
        git describe --tag --dirty
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE CLARA_PARABRICKS_GENOMEWORKS_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE)

endmacro ()