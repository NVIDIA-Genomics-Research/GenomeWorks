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



LOCAL_BUILD_ROOT=$1
CMAKE_OPTS=$2

cd ${LOCAL_BUILD_ROOT}
export LOCAL_BUILD_DIR=${LOCAL_BUILD_ROOT}/build

# Use CMake-based build procedure
mkdir --parents ${LOCAL_BUILD_DIR}
cd ${LOCAL_BUILD_DIR}

logger "Configure CMake..."
cmake .. "${CMAKE_COMMON_VARIABLES[@]}" \
    -Dgw_enable_tests=ON \
    -Dgw_enable_benchmarks=ON \
    -Dgw_build_shared=ON \
    -Dgw_cuda_gen_all_arch=ON \
    -DCMAKE_INSTALL_PREFIX="${LOCAL_BUILD_DIR}/install" \
    -GNinja

logger "Run build..."
NINJA_STATUS="[%r->%f/%t]" ninja all install package

logger "Install package..."
DISTRO=$(awk -F= '/^NAME/{print $2}' /etc/os-release)
DISTRO=${DISTRO//\"/}

PACKAGE_DIR=${LOCAL_BUILD_DIR}/gw-package
mkdir -p $PACKAGE_DIR
if [ "$DISTRO" == "Ubuntu" ]; then
    dpkg-deb -X ${LOCAL_BUILD_DIR}/*.deb $PACKAGE_DIR
elif [ "$DISTRO" == "CentOS Linux" ]; then
    rpm2cpio ${LOCAL_BUILD_DIR}/*.rpm | cpio -idmv
    mv usr/ $PACKAGE_DIR/
else
    echo "Unknown OS found - ${DISTRO}."
    exit 1
fi

logger "Creating symlink to installed package..."
UNPACK_ROOT=$(readlink -f "$PACKAGE_DIR/usr/local")
GW_SYMLINK_PATH="$UNPACK_ROOT/GenomeWorks"
ln -s $UNPACK_ROOT/GenomeWorks-* $GW_SYMLINK_PATH
GW_LIB_DIR=${GW_SYMLINK_PATH}/lib

# Run tests
if [ "$TEST_ON_GPU" == '1' ]; then
  logger "GPU config..."
  nvidia-smi

  logger "Running GenomeWorks unit tests..."
  # Avoid using 'find' which reutrns 0 even if -exec command fails
  for binary_test in "${LOCAL_BUILD_DIR}"/install/tests/*; do
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GW_LIB_DIR "${binary_test}";
  done

  logger "Running GenomeWorks benchmarks..."
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GW_LIB_DIR ${LOCAL_BUILD_DIR}/install/benchmarks/cudapoa/benchmark_cudapoa --benchmark_filter="BM_SingleBatchTest"
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GW_LIB_DIR ${LOCAL_BUILD_DIR}/install/benchmarks/cudaaligner/benchmark_cudaaligner --benchmark_filter="BM_SingleAlignment"
fi

