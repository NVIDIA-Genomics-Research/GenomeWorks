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

# Ignore errors and set path
set +e
PATH=/conda/bin:$PATH

# Activate common conda env
source activate gdf

# Run flake8 and get results/return code
#FLAKE=`flake8 python`
#RETVAL=$?
RETVAL=0

# Output results if failure otherwise show pass
#if [ "$FLAKE" != "" ]; then
#  echo -e "\n\n>>>> FAILED: flake8 style check; begin output\n\n"
#  echo -e "$FLAKE"
#  echo -e "\n\n>>>> FAILED: flake8 style check; end output\n\n"
#else
#  echo -e "\n\n>>>> PASSED: flake8 style check\n\n"
#fi

exit $RETVAL
