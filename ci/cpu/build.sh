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
# GenomeWorks CPU build script for CI #
######################################
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

################################################################################
# Init
################################################################################

export TEST_PYGENOMEWORKS=1
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
PARALLEL_LEVEL=4

# Set home to the job's workspace
export HOME=$WORKSPACE

cd ${WORKSPACE}

source ci/common/prep-init-env.sh ${WORKSPACE}

CMAKE_COMMON_VARIABLES="-DCMAKE_BUILD_TYPE=Release"

################################################################################
# SDK build/test
################################################################################

logger "Build SDK..."
source ci/common/build-test-sdk.sh ${WORKSPACE} ${CMAKE_COMMON_VARIABLES} ${PARALLEL_LEVEL} 0

################################################################################
# racon-gpu build/test (CPU version)
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

logger "Build racon-gpu for CPU version..."

cd ${WORKSPACE}
source ci/common/build-test-racon-gpu.sh ${APP_DIR} ${WORKSPACE} ${CMAKE_COMMON_VARIABLES} ${PARALLEL_LEVEL} 0 0
export PATH=${APP_DIR}/build/bin:$PATH

################################################################################
# Pygenomeworks tests
################################################################################

cd ${WORKSPACE}
if [ "${TEST_PYGENOMEWORKS}" == '1' ]; then
    source ci/common/test-pygenomeworks.sh $WORKSPACE/pygenomeworks
fi
