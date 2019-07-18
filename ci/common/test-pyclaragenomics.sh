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

cd $1
python3 -m pip install -r requirements.txt
python3 setup.py build_ext --inplace
python3 setup.py install
python3 -m pytest -s
