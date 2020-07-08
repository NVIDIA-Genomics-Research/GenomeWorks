

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
