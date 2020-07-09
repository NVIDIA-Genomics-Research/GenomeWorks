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
