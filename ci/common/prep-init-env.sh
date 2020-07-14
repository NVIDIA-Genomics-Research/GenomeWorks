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



################################################################################
# SETUP - Check environment
################################################################################

logger "Get env..."
env

logger "Check versions..."
gcc --version
g++ --version

logger "Activate anaconda enviroment..."
CONDA_NEW_ACTIVATION_CMD_VERSION="4.4"
CONDA_VERSION=$(conda --version | awk '{print $2}')
if [ "$CONDA_NEW_ACTIVATION_CMD_VERSION" == "$(echo -e "$CONDA_VERSION\n$CONDA_NEW_ACTIVATION_CMD_VERSION" | sort -V | head -1)" ]; then
  logger "Version is higer than ${CONDA_NEW_ACTIVATION_CMD_VERSION}, using conda activate"
  source /conda/etc/profile.d/conda.sh
  conda activate "${2}"
else
  logger "Version is lower than ${CONDA_NEW_ACTIVATION_CMD_VERSION}, using source activate"
  source activate "${2}"
fi
conda info --envs

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

logger "Check Python version..."
python --version


# Conda add custom packages for GenomeWorks CI
# Split setup into several steps to prevent the 15 minutes no
# output to stdout timeout limit in CI jobs when solving environment
logger "Conda install GenomeWorks custom packages - clang-format"
conda install --override-channels -c sarcasm clang-format

logger "Conda install GenomeWorks custom packages - doxygen ninja cmake"
conda install --override-channels -c conda-forge doxygen ninja cmake">=3.10.2"

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
