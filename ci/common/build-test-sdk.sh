CMAKE_BUILD_GPU=""
LOCAL_BUILD_ROOT=$1
CMAKE_OPTS=$2
BUILD_THREADS=$3
GPU_TEST=$4

cd ${LOCAL_BUILD_ROOT}
export LOCAL_BUILD_DIR=${LOCAL_BUILD_ROOT}/build

# Use CMake-based build procedure
mkdir --parents ${LOCAL_BUILD_DIR}
cd ${LOCAL_BUILD_DIR}

# configure
cmake $CMAKE_COMMON_VARIABLES ${CMAKE_BUILD_GPU} -Dgw_enable_tests=ON -DCMAKE_INSTALL_PREFIX=${LOCAL_BUILD_DIR}/install ..
# build
make -j${BUILD_THREADS} VERBOSE=1 all docs install

if [ "$GPU_TEST" == '1' ]; then
  logger "GPU config..."
  nvidia-smi

  logger "Running GenomeWorks unit tests..."
  run-parts -v ${LOCAL_BUILD_DIR}/install/tests
fi

