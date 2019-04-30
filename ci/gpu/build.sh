#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION.
######################################
# GenomeWorks CPU/GPU conda build script for CI #
######################################
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

################################################################################
# Init
################################################################################

export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
PARALLEL_LEVEL=4

# Set home to the job's workspace
export HOME=$WORKSPACE

cd ${WORKSPACE}

source ci/common/prep-init-env.sh ${WORKSPACE}

CMAKE_COMMON_VARIABLES="-DCMAKE_BUILD_TYPE=Release"

# If we are building for GPU, we do 2 builds:
# 1) the SDK on its own
# 2) use racon-gpu as a build flow
# If we are building for "CPU", we just build locally as the SDK

################################################################################
# SDK build/test
################################################################################

logger "Build SDK..."
source ci/common/build-test-sdk.sh ${WORKSPACE} ${CMAKE_COMMON_VARIABLES} ${PARALLEL_LEVEL} ${TEST_ON_GPU}

cd ${WORKSPACE}
rm -rf ${WORKSPACE}/build

################################################################################
# racon-gpu build/test
################################################################################

APP_REPO="ssh://git@gitlab-master.nvidia.com:12051/genomics/racon-gpu.git"
APP_NAME=racon-gpu
APP_DIR=$WORKSPACE/${APP_NAME}

logger "Pull racon-gpu..."

# pull from scratch each time
cd ${WORKSPACE}
rm -rf racon-gpu
mkdir ${APP_NAME}

source ci/common/prep-pull-repo-mr-branch.sh ${APP_REPO} ${APP_DIR}

logger "Build racon-gpu for CUDA..."

cd ${WORKSPACE}
source ci/common/build-test-racon-gpu.sh ${APP_DIR} ${WORKSPACE} ${CMAKE_COMMON_VARIABLES} ${PARALLEL_LEVEL} ${TEST_ON_GPU}



