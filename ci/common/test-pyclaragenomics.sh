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
# ClaraGenomicsAnalysis CPU/GPU conda build script for CI #
######################################
set -e

WORKSPACE=$1
cd $WORKSPACE

#Install external dependencies.
python3 -m pip install -r requirements.txt

# Build and install internal modules.
CGA_INSTALL_DIR=`pwd`/cga_install_dir

export LIBRARY_PATH=$LIBRARY_PATH:$CGA_INSTALL_DIR/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CGA_INSTALL_DIR/lib

python3 setup.py build_cga --cga-install-dir=$CGA_INSTALL_DIR --clean-build
python3 setup.py develop

# Run tests.
if [ "$GPU_TEST" == '1' ]; then
    python3 -m pytest -m gpu -s
else
    python3 -m pytest -m cpu -s
fi
