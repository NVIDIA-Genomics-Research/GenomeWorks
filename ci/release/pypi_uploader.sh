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



set -e

# Skip upload if CI is executed locally
if [[ ${RUNNING_CI_LOCALLY} = true  ]]; then
    echo "Skipping PyPi upload - running ci locally"
    return 0
fi

# Skip upload if current branch is not master or starts with "dev-"
if [ "${COMMIT_HASH}" != "master" ] && [[ ! "${COMMIT_HASH}" =~ ^dev-.+ ]]; then
    echo "Skipping PyPI upload - not master or development branch"
    return 0
fi

for f in "${WORKSPACE}"/pygenomeworks/genomeworks_wheel/*.whl; do
    if [ ! -e "${f}" ]; then
        echo "genomeworks Whl file does not exist"
        exit 1
    else
        conda install -c conda-forge twine
        python3 -m pip install 'readme-renderer>=21.0' # to support py3.5 images
        # Change .whl package name to support PyPI upload
        MODIFIED_WHL_NAME=$(dirname ${f})/$(basename "${f}" | sed -r "s/(.*-.+-.+)-.+-.+.whl/\1-none-any.whl/")
        mv "${f}" "${MODIFIED_WHL_NAME}"
        echo "File name ${f} was changed into ${MODIFIED_WHL_NAME}"
        # Perform Upload
        python3 -m twine upload --skip-existing "${WORKSPACE}"/pygenomeworks/genomeworks_wheel/*
    fi
done
