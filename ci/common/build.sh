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

export APP_DIR=$WORKSPACE/racon-gpu

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

# If we are building for GPU, we use racon-gpu as the build flow
# If we are buildubg for "CPU", we just build locally as the SDK
if [ "${BUILD_FOR_GPU}" == '1' ]; then
  logger "Pull racon-gpu..."
  if [ ! -d "racon-gpu" ]; then
    git clone ssh://git@gitlab-master.nvidia.com:12051/genomics/racon-gpu.git
  fi

  # Switch to project root; also root of repo checkout
  cd ${APP_DIR}

  git pull
  git submodule update --init --recursive

  logger "Build racon-gpu for CUDA..."
  CMAKE_BUILD_GPU="-Dracon_enable_cuda=ON -DGENOMEWORKS_SRC_PATH=${WORKSPACE}"

  export LOCAL_BUILD_ROOT=${APP_DIR}
else
  # Forced disable GPU testing
  export TEST_ON_GPU=0

  logger "Build SDK..."
  CMAKE_BUILD_GPU=""
  export LOCAL_BUILD_ROOT=${WORKSPACE}
fi

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

  logger "Running test..."
  logger "GPU config..."
  nvidia-smi

  logger "Test results..."
  cd ${LOCAL_BUILD_DIR}/bin
  ./cuda_test.sh
fi
