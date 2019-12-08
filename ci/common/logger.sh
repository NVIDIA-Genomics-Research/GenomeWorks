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

# Logger function for build status output
function logger() {
  ELAPSED_TIME=$(( $(date +%s)-START_TIME ));
  PRINT_TIME=$(printf '%dh:%dm:%ds\n' $((ELAPSED_TIME/3600)) $((ELAPSED_TIME%3600/60)) $((ELAPSED_TIME%60)));
  echo -e "\n${PRINT_TIME} >>>> $* <<<<\n"
}
