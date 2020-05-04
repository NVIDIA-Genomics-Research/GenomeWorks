#!/bin/bash
#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

######################################
# ClaraGenomicsAnalysis CPU build script for CI #
######################################
set -e

START_TIME=$(date +%s)

export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
PARALLEL_LEVEL=4

# Set home to the job's workspace
export HOME=$WORKSPACE

cd "${WORKSPACE}"

################################################################################
# Init
################################################################################

source ci/common/logger.sh

logger "Calling prep-init-env..."
source ci/common/prep-init-env.sh "${WORKSPACE}" "${CONDA_ENV_NAME}"

################################################################################
# SDK build/test
################################################################################

logger "Build SDK..."
CMAKE_COMMON_VARIABLES=(-DCMAKE_BUILD_TYPE=Release)
source ci/common/build-test-sdk.sh "${WORKSPACE}" "${CMAKE_COMMON_VARIABLES[@]}" "${PARALLEL_LEVEL}" 0

rm -rf "${WORKSPACE}"/build

################################################################################
# Pyclaragenomics tests
################################################################################
logger "Build Pyclaragenomics..."
cd "${WORKSPACE}"
source ci/common/test-pyclaragenomics.sh "${WORKSPACE}"/pyclaragenomics

logger "Upload Wheel to PyPI..."
cd "${WORKSPACE}"
source ci/release/pypi_uploader.sh

logger "Done..."
