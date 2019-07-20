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
import shutil
import subprocess

from distutils.sysconfig import get_python_lib
from distutils.cmd import Command
from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize

class BuildCgaCommand(Command):
    """
    Custom command to build ClaraGenomicsAnalysis CMake project
    required for python bindings in pyclaragenomics.
    """
    description = "Build ClaraGenomicsAnalysis C++ project"
    user_options = [
            ('cga-install-dir=', None, 'Path to build directory for CGA'),
            ('clean-build', None, 'Build CGA from scratch'),
            ]

    def initialize_options(self):
        self.cga_install_dir = ''
        self.clean_build = False

    def finalize_options(self):
        if (not self.cga_install_dir):
            raise RuntimeError("Please pass in an install path for the "
                    "CGA build files using --cga_install_dir=")

    def run(self):
        build_path = os.path.abspath('cga_build')
        cmake_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cmake_args = ['-DCMAKE_INSTALL_PREFIX=' + self.cga_install_dir,
                      '-Dcga_build_shared=ON']
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + 'Release']

        build_args = ['--', '-j16', 'docs', 'install']

        if os.path.exists(build_path) and self.clean_build:
            shutil.rmtree(build_path)

        if not os.path.exists(build_path):
            os.makedirs(build_path)

        if not os.path.exists(self.cga_install_dir):
            os.makedirs(self.cga_install_dir)

        subprocess.check_call(['cmake', cmake_root_dir] + cmake_args, cwd=build_path)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_path)

extensions = [
    Extension(
        "*",
        sources=["claragenomics/**/*.pyx"],
        include_dirs=[
            "/usr/local/cuda/include",
            "../cudapoa/include",
        ],
        library_dirs=["/usr/local/cuda/lib64"],
        runtime_library_dirs=["/usr/local/cuda/lib64"],
        libraries=["cudapoa", "cudart"],
        language="c++",
        extra_compile_args=["-std=c++14"],
    )
]

setup(name='pyclaragenomics',
      version='0.1',
      description='NVIDIA genomics python libraries an utiliites',
      author='NVIDIA Corporation',
      setup_requires=["cython"],
      packages=find_packages(),
      cmdclass={
          'build_cga': BuildCgaCommand,
          },
      ext_modules=cythonize(extensions),
      scripts=[os.path.join('bin', 'genome_simulator'),
               os.path.join('bin', 'assembly_evaluator')],
      install_requires=["cython"]
      )
