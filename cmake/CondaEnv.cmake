

# Add Conda environment to CMake include and library paths.
if (DEFINED ENV{CONDA_PREFIX})
    message(STATUS "Found Conda environment in $ENV{CONDA_PREFIX}")
    include_directories($ENV{CONDA_PREFIX}/include)
    link_directories($ENV{CONDA_PREFIX}/lib)
endif()
