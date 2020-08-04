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

run_tests() {
  cd test/
  if [ "${TEST_ON_GPU}" == '1' ]; then
      python -m pytest -m gpu -s
  else
      python -m pytest -m cpu -s
  fi
}

PYGENOMEWORKS_DIR=$1
cd "$PYGENOMEWORKS_DIR"

logger "Install pygenomeworks external dependencies..."
python -m pip install -r requirements.txt

logger "Install pygenomeworks..."
python setup_pygenomeworks.py --build_output_folder gw_build

logger "Run Tests..."
run_tests
cd "$PYGENOMEWORKS_DIR"

logger "Uninstall pygenomeworks..."
pip uninstall -y pygenomeworks

logger "Create pygenomeworks Wheel package..."
CUDA_VERSION_FOR_PACKAGE_NAME=$(echo "$CUDA_VERSION" | cut -d"." -f1-2 | sed -e "s/\./_/g")
if [ "${COMMIT_HASH}" == "master" ]; then
  PYGW_VERSION=$(cat ../VERSION)
else
  PYGW_VERSION=$(cat ../VERSION | tr -d "\n")\.dev$(date +%y%m%d) # for nightly build
fi
python setup_pygenomeworks.py \
        --build_output_folder gw_build_wheel \
        --create_wheel_only \
        --overwrite_package_name genomeworks_cuda_"$CUDA_VERSION_FOR_PACKAGE_NAME" \
        --overwrite_package_version "$PYGW_VERSION"

logger "Install pygenomeworks Wheel package..."
yes | pip install "$PYGENOMEWORKS_DIR"/genomeworks_wheel/genomeworks*.whl

logger "Run Tests..."
run_tests
