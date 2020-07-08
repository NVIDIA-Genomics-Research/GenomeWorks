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



set(GW_ENABLE_PACKAGING TRUE)

# Find Linux Distribution
EXECUTE_PROCESS(
    COMMAND "awk" "-F=" "/^NAME/{print $2}" "/etc/os-release"
    OUTPUT_VARIABLE LINUX_OS_NAME
    )

if (${LINUX_OS_NAME} MATCHES "Ubuntu")
    MESSAGE(STATUS "Package generator - DEB")
    SET(CPACK_GENERATOR "DEB")
elseif(${LINUX_OS_NAME} MATCHES "CentOS")
    MESSAGE(STATUS "Package generator - RPM")
    SET(CPACK_GENERATOR "RPM")
else()
    MESSAGE(STATUS "Unrecognized Linux distribution - ${LINUX_OS_NAME}. Disabling packaging.")
    set(GW_ENABLE_PACKAGING FALSE)
endif()

if (GW_ENABLE_PACKAGING)
    SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "NVIDIA Corporation")
    SET(CPACK_PACKAGE_VERSION "${GW_VERSION}")
    SET(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local/${GW_PROJECT_NAME}-${GW_VERSION}")

    include(CPack)
endif()
