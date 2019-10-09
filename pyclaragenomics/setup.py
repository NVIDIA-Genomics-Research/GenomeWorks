#!/usr/bin/env python3

#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os.path
import os
import subprocess

from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize


class CMakeWrapper():
    """Class to encapsulate building a CMake project."""

    def __init__(self, cmake_root_dir, cmake_build_path="cmake_build", cmake_extra_args=""):
        """
        Class constructor.

        Args:
            cmake_root_dir : Root directory of CMake project
            cmake_install_dir : Install location for CMake project
            cmake_extra_args : Extra string arguments to be passed to CMake during setup
        """
        self.build_path = os.path.abspath(cmake_build_path)
        self.cmake_root_dir = os.path.abspath(cmake_root_dir)
        self.cmake_install_dir = os.path.join(self.build_path, "install")
        self.cmake_extra_args = cmake_extra_args
        self.cuda_toolkit_root_dir = os.environ.get("CUDA_TOOLKIT_ROOT_DIR")

    def run_cmake_cmd(self):
        cmake_args = ['-DCMAKE_INSTALL_PREFIX=' + self.cmake_install_dir,
                      '-DCMAKE_BUILD_TYPE=' + 'Release',
                      '-DCMAKE_INSTALL_RPATH=' + os.path.join(self.cmake_install_dir, "lib")]
        cmake_args += [self.cmake_extra_args]

        if self.cuda_toolkit_root_dir:
            cmake_args += ["-DCUDA_TOOLKIT_ROOT_DIR=%s" % self.cuda_toolkit_root_dir]

        if not os.path.exists(self.build_path):
            os.makedirs(self.build_path)

        subprocess.check_call(['cmake', self.cmake_root_dir] + cmake_args, cwd=self.build_path)

    def run_build_cmd(self):
        build_args = ['--', '-j16', 'install']
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_path)

    def build(self):
        self.run_cmake_cmd()
        self.run_build_cmd()

    def get_installed_path(self, component=""):
        installed_path = os.path.abspath(os.path.join(self.cmake_install_dir, component))
        if (not os.path.exists(installed_path)):
            raise RuntimeError("No valid path for requested component exists")
        return installed_path


# Initialize builds.
pycga_directory = os.path.dirname(os.path.realpath(__file__))
cmake_root_dir = os.path.dirname(pycga_directory)
cmake_proj = CMakeWrapper(cmake_root_dir,
                          cmake_build_path=os.path.join(pycga_directory, "cga_build"),
                          cmake_extra_args="-Dcga_build_shared=ON")
cmake_proj.build()

extensions = [
    Extension(
        "*",
        sources=[os.path.join(pycga_directory, "claragenomics/**/*.pyx")],
        include_dirs=[
            "/usr/local/cuda/include",
            os.path.join(cmake_root_dir, "cudapoa/include"),
            os.path.join(cmake_root_dir, "cudaaligner/include"),
        ],
        library_dirs=["/usr/local/cuda/lib64", cmake_proj.get_installed_path("lib")],
        runtime_library_dirs=["/usr/local/cuda/lib64", cmake_proj.get_installed_path("lib")],
        libraries=["cudapoa", "cudaaligner", "cudart"],
        language="c++",
        extra_compile_args=["-std=c++14"],
    )
]

# Run from the pycga directory
os.chdir(pycga_directory)

setup(name='pyclaragenomics',
      version='0.2.0',
      description='NVIDIA genomics python libraries an utiliites',
      author='NVIDIA Corporation',
      packages=find_packages(where=pycga_directory),
      ext_modules=cythonize(extensions, compiler_directives={'embedsignature': True}),
      scripts=[os.path.join(pycga_directory, 'bin', 'genome_simulator'),
               os.path.join(pycga_directory, 'bin', 'assembly_evaluator')],
      )
