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



## Usage
# bash update-version.sh
# Promotes patch version, version format : v<Year>.<Month>.<patch>

set -e

# Grab argument for release type
RELEASE_TYPE=$1

# Calender versioning only supports "minor"= a PR crated from clara-parabricks/dev branch
# or "patch"= a PR merged from any other branch
if [ "$RELEASE_TYPE" != "patch" ] && [ "$RELEASE_TYPE" != "minor" ]; then
  echo "Incorrect release type; use 'minor', or 'patch' as an argument"
  exit 1
fi

# Get current version and calculate next versions
CURRENT_TAG=$(git tag | grep -xE "v$(date '+%Y\.%m')\.[0-9]+" | sort --version-sort | tail -n 1 | tr -d 'v')

# In case no tag exists for this month
if [ -z "${CURRENT_TAG}" ]; then
  if [ "$RELEASE_TYPE" == "minor" ]; then
    # In case no tag exists for this month & this is a
    # "minor" release (merged from branch clara-parabricks/dev) - create new tag for this month
    CURRENT_TAG="$(date '+%Y.%m').-1"
  elif [ "$RELEASE_TYPE" == "patch" ]; then
    # In case no tag exists for this month & this is a "patch" release
    # (merged from some branch that is not clara-parabricks/dev) -
    # take latest Tag that was pushed and increase Patch location in 1
    CURRENT_TAG=$(git tag | grep -xE "v[0-9]+\.[0-9]+\.[0-9]+" | sort --version-sort | tail -n 1 | tr -d 'v')
  fi
fi

CURRENT_MAJOR=`echo $CURRENT_TAG | awk '{split($0, a, "."); print a[1]}'`
CURRENT_MINOR=`echo $CURRENT_TAG | awk '{split($0, a, "."); print a[2]}'`
CURRENT_PATCH=`echo $CURRENT_TAG | awk '{split($0, a, "."); print a[3]}'`
NEXT_PATCH=$((CURRENT_PATCH + 1))
NEXT_FULL_TAG=""
NEXT_SHORT_TAG=""

NEXT_FULL_TAG="${CURRENT_MAJOR}.${CURRENT_MINOR}.${NEXT_PATCH}"
NEXT_SHORT_TAG="${CURRENT_MAJOR}.${CURRENT_MINOR}"


if [[ "$CURRENT_PATCH" == "-1" ]]; then
  echo "Preparing release [$NEXT_FULL_TAG]"
else
  echo "Preparing release [$CURRENT_TAG -> $NEXT_FULL_TAG]"
fi

echo "${NEXT_FULL_TAG}" > VERSION

# Set gpuCI auto-releaser shell variable to an empty string
REL_TYPE=""
