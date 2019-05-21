#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

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

# Conda add custom packages for GenomeWorks CI
conda install -c conda-forge doxygen
conda install -c anaconda llvm

logger "Conda install minimap2"
# Conda install minimap2
conda install -c bioconda minimap2

logger "Conda install miniasm"
# Conda install miniasm
conda install -c bioconda miniasm

logger "Conda install GIT LFS"
conda install -c conda-forge git-lfs
git lfs install
git-lfs pull

################################################################################
# BUILD - Conda package builds 
################################################################################

CUDA_REL=${CUDA:0:3}
if [ "${CUDA:0:2}" == '10' ]; then
  # CUDA 10 release
  CUDA_REL=${CUDA:0:4}
fi

# Cleanup local git
cd $1
git clean -xdf

