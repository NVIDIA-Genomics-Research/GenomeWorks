#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
#########################
# cuDF CHANGELOG Tester #
#########################

# Checkout master for comparison
git checkout --quiet master

# Switch back to tip of PR branch
git checkout --quiet current-pr-branch

# Ignore errors during searching
set +e

# Get list of modified files between matster and PR branch
CHANGELOG=`git diff --name-only master...current-pr-branch | grep CHANGELOG.md`
RETVAL=0

exit $RETVAL
