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

logger "Check versions..."
gcc --version
g++ --version

logger "Activate anaconda enviroment..."
source /conda/etc/profile.d/conda.sh
conda activate "${2}"
conda info --envs

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

logger "Check Python version..."
python --version


# Conda add custom packages for ClaraGenomicsAnalysis CI
# Split setup into several steps to prevent the 15 minutes no
# output to stdout timeout limit in CI jobs when solving environment
logger "Conda install ClaraGenomicsAnalysis custom packages - clang-format"
conda install --override-channels -c sarcasm clang-format

logger "Conda install ClaraGenomicsAnalysis custom packages - doxygen ninja cmake"
conda install --override-channels -c conda-forge doxygen ninja cmake

logger "Conda install ClaraGenomicsAnalysis custom packages - minimap2 miniasm racon htslib"
conda install --override-channels -c bioconda minimap2 miniasm racon htslib

logger "Update LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# Show currentl installed paths
set -x
ls /usr/local/include
ls /usr/local/lib
set +x

################################################################################
# BUILD - Conda package builds 
################################################################################

CUDA_REL=${CUDA:0:3}
if [ "${CUDA:0:2}" == '10' ]; then
  # CUDA 10 release
  CUDA_REL=${CUDA:0:4}
fi

# Cleanup local git
cd "$1"
git clean -xdf
