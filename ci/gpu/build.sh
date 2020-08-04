#!/bin/bash

#
# Copyright 2019-2020 NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#



######################################
# GenomeWorks CPU/GPU conda build script for CI #
######################################
set -e

START_TIME=$(date +%s)

export PATH=/conda/bin:/usr/local/cuda/bin:$PATH

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

logger "Build SDK in Release mode..."
CMAKE_COMMON_VARIABLES=(-DCMAKE_BUILD_TYPE=Release -Dgw_profiling=ON)
source ci/common/build-test-sdk.sh "${WORKSPACE}" "${CMAKE_COMMON_VARIABLES[@]}"

cd "${WORKSPACE}"
rm -rf "${WORKSPACE}"/build

logger "Build SDK in Debug mode..."
CMAKE_COMMON_VARIABLE=(-DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS_DEBUG="-g -O2")
source ci/common/build-test-sdk.sh "${WORKSPACE}" "${CMAKE_COMMON_VARIABLES[@]}"

rm -rf "${WORKSPACE}"/build

################################################################################
# Pygenomeworks tests
################################################################################
logger "Build Pygenomeworks..."
cd "${WORKSPACE}"
source ci/common/test-pygenomeworks.sh "${WORKSPACE}"/pygenomeworks

logger "Upload Wheel to PyPI..."
cd "${WORKSPACE}"
source ci/release/pypi_uploader.sh

logger "Done..."
