#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
######################################
# cuDF GPU conda build script for CI #
######################################
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

. ci/common/build.sh


