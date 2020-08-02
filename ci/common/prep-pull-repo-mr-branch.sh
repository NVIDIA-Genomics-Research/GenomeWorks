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



PULL_REPO=$1
PULL_DEST=$2

# is this is a merge request and there a branch with a name matching the MR branch in the
# other repo, pull that
export BRANCH_FOUND=""
if [ "${BUILD_CAUSE_SCMTRIGGER}" == "true" ]; then
logger "This is an SCM-caused build"
if [ "${gitlabActionType}" == 'MERGE' ]; then
logger "This is a merge-request-caused build"
    if [ "${gitlabSourceBranch}" != "" ]; then
    logger "The specified branch is: ${gitlabSourceBranch}"
    export BRANCH_FOUND=`git ls-remote -h ${PULL_REPO} | grep "refs/heads/${gitlabSourceBranch}$"`
    logger "Branch found test ${BRANCH_FOUND}"
    fi
fi
fi

if [ "${BRANCH_FOUND}" == "" ]; then
logger "No specified branch - is there a target branch?: ${gitlabTargetBranch}"
if [ "${gitlabTargetBranch}" != "" ]; then
    logger "A target branch is specified: ${gitlabTargetBranch}"
    export BRANCH_FOUND=`git ls-remote -h ${PULL_REPO} | grep "refs/heads/${gitlabTargetBranch}$"`
    logger "Branch found test ${BRANCH_FOUND}"
    if [ "${BRANCH_FOUND}" != "" ]; then
    export MR_BRANCH=${gitlabTargetBranch}
    else
    export MR_BRANCH=master
    fi
else
    export MR_BRANCH=master
fi
else
    export MR_BRANCH=${gitlabSourceBranch}
fi

git clone --branch ${MR_BRANCH} --single-branch --depth 1 ${PULL_REPO} ${PULL_DEST}

# Switch to project root; also root of repo checkout
pushd ${PULL_DEST}

git pull
git submodule update --init --recursive

popd

