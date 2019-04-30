#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
######################################
# GenomeWorks CPU/GPU conda build script for CI #
######################################
set -e

cd $1
python3 -m pip install -r requirements.txt
python3 setup.py install
python3 -m pytest