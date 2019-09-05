#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# Get flavor of distribution
find_program(LSB_RELEASE_EXEC lsb_release)
execute_process(COMMAND ${LSB_RELEASE_EXEC} -is
    OUTPUT_VARIABLE LINUX_DISTRIBUTION_NAME
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (${LINUX_DISTRIBUTION_NAME} MATCHES "Ubuntu")
    SET(CPACK_GENERATOR "DEB")
    message(STATUS "Package generator - DEB")
elseif (${LINUX_DISTRIBUTION_NAME} MATCHES "CentOS")
    SET(CPACK_GENERATOR "RPM")
    message(STATUS "Package generator - RPM")
else()
    message(FATAL_ERROR "Unrecognized Linux distribution. Packaging not available")
endif()

SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "NVIDIA Corporation")
SET(CPACK_PACKAGE_VERSION "${CGA_VERSION}")

include(CPack)
