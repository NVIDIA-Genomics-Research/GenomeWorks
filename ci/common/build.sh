#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
######################################
# GenomeWorks CPU/GPU conda build script for CI #
######################################
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

export APP_REPO="ssh://git@gitlab-master.nvidia.com:12051/genomics/racon-gpu.git"
export APP_NAME=racon-gpu
export APP_DIR=$WORKSPACE/${APP_NAME}

export TEST_PYGENOMEWORKS=1

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4

# Set home to the job's workspace
export HOME=$WORKSPACE

# Switch to project root; also root of repo checkout
cd ${WORKSPACE}

################################################################################
# SETUP - Check environment
################################################################################

logger "Get env..."
env

logger "Activate conda env..."
source activate gdf

logger "Check versions..."
gcc --version
g++ --version

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

################################################################################
# BUILD - Conda package builds 
################################################################################

CUDA_REL=${CUDA:0:3}
if [ "${CUDA:0:2}" == '10' ]; then
  # CUDA 10 release
  CUDA_REL=${CUDA:0:4}
fi

# Cleanup local git
git clean -xdf

CMAKE_COMMON_VARIABLES="-DCMAKE_BUILD_TYPE=Release"

# If we are building for GPU, we do 2 builds:
# 1) the SDK on its own
# 2) use racon-gpu as a build flow
# If we are buildubg for "CPU", we just build locally as the SDK
if [ "${BUILD_FOR_GPU}" == '1' ]; then
  logger "Pull racon-gpu..."

  # pull from scratch each time
  cd ${WORKSPACE}
  rm -rf racon-gpu
  mkdir ${APP_NAME}

  # is this is a merge request and there a branch with a name matching the MR branch in the
  # other repo, pull that
  export BRANCH_FOUND=""
  if [ "${BUILD_CAUSE_SCMTRIGGER}" == "true" ]; then
    logger "This is an SCM-caused build"
    if [ "${gitlabActionType}" == 'MERGE' ]; then
    logger "This is a merge-request-caused build"
      if [ "${gitlabSourceBranch}" != "" ]; then
        logger "The specified branch is: ${gitlabSourceBranch}"
        export BRANCH_FOUND=`git ls-remote -h ${APP_REPO} | grep "refs/heads/${gitlabSourceBranch}$"`
        logger "Branch found test ${BRANCH_FOUND}"
      fi
    fi
  fi

  if [ "${BRANCH_FOUND}" == "" ]; then
    logger "No specified branch - is there a target branch?: ${gitlabTargetBranch}"
    if [ "${gitlabTargetBranch}" != "" ]; then
      logger "A target branch is specified: ${gitlabTargetBranch}"
      export BRANCH_FOUND=`git ls-remote -h ${APP_REPO} | grep "refs/heads/${gitlabTargetBranch}$"`
      logger "Branch found test ${BRANCH_FOUND}"
      if [ "${BRANCH_FOUND}" != "" ]; then
        export MR_BRANCH=${gitlabTargetBranch}
      else
        export MR_BRANCH=master
      fi
    else
      export MR_BRANCH=master
    fi
  else
      export MR_BRANCH=${gitlabSourceBranch}
  fi

  git clone --branch ${MR_BRANCH} --single-branch --depth 1 ${APP_REPO}

  # Switch to project root; also root of repo checkout
  cd ${APP_DIR}

  git pull
  git submodule update --init --recursive
fi

if [ "${TEST_ON_GPU}" == '1' ]; then
    logger "GPU config..."
    nvidia-smi
fi

logger "Build SDK..."
CMAKE_BUILD_GPU=""
export LOCAL_BUILD_ROOT=${WORKSPACE}

cd ${LOCAL_BUILD_ROOT}
export LOCAL_BUILD_DIR=${LOCAL_BUILD_ROOT}/build

# Use CMake-based build procedure
mkdir --parents ${LOCAL_BUILD_DIR}
cd ${LOCAL_BUILD_DIR}

# configure
cmake $CMAKE_COMMON_VARIABLES ${CMAKE_BUILD_GPU} -Dgw_enable_tests=ON -DCMAKE_INSTALL_PREFIX=${LOCAL_BUILD_DIR}/install ..
# build
make -j${PARALLEL_LEVEL} VERBOSE=1 install

if [ "${TEST_ON_GPU}" == '1' ]; then
    logger "Running GenomeWorks unit tests..."
    run-parts -v ${LOCAL_BUILD_DIR}/install/tests
fi

cd ${LOCAL_BUILD_ROOT}
rm -rf build

# Build related application repo for end to end test.
if [ "${BUILD_FOR_GPU}" == '1' ]; then
  logger "Build racon-gpu for CUDA..."
  CMAKE_BUILD_GPU="-Dracon_enable_cuda=ON -DGENOMEWORKS_SRC_PATH=${WORKSPACE}"

  export LOCAL_BUILD_ROOT=${APP_DIR}

  cd ${LOCAL_BUILD_ROOT}
  export LOCAL_BUILD_DIR=${LOCAL_BUILD_ROOT}/build

  # Use CMake-based build procedure
  mkdir --parents ${LOCAL_BUILD_DIR}
  cd ${LOCAL_BUILD_DIR}

  # configure
  cmake $CMAKE_COMMON_VARIABLES ${CMAKE_BUILD_GPU} ..
  # build
  make -j${PARALLEL_LEVEL} VERBOSE=1 all

  if [ "${TEST_ON_GPU}" == '1' ]; then
    logger "Pulling GPU test data..."
    cd ${WORKSPACE}
    if [ ! -d "ont-racon-data" ]; then
      if [ ! -f "${ont-racon-data.tar.gz}" ]; then
        wget -q -L https://s3.us-east-2.amazonaws.com/racon-data/ont-racon-data.tar.gz
      fi
      tar xvzf ont-racon-data.tar.gz
    fi

    logger "Running Racon end to end test..."

    logger "Test results..."
    cd ${LOCAL_BUILD_DIR}/bin
    ./cuda_test.sh
  fi
fi

################################################################################
# Pygenomeworks tests
################################################################################

if [ "${TEST_PYGENOMEWORKS}" == '1' ]; then
    cd $WORKSPACE/pygenomeworks
    python3 -m pip install -r requirements.txt
    python3 setup.py install
    python3 -m pytest
fi
