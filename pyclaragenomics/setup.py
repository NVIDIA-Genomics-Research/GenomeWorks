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

from distutils.sysconfig import get_python_lib
from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize

def build_cga(cmake_root_dir="..", cmake_build_folder="build", cmake_install_prefix="install"):
    build_path = os.path.abspath(cmake_build_folder)
    root_dir = os.path.abspath(cmake_root_dir)
    cmake_args = ['-DCMAKE_INSTALL_PREFIX=' + os.environ['CONDA_PREFIX'],
                  '-Dcga_build_shared=ON']

    cmake_args += ['-DCMAKE_BUILD_TYPE=' + 'Release']
    build_args = ['--', '-j16', 'docs', 'install']

    if not os.path.exists(build_path):
        os.makedirs(build_path)
    subprocess.check_call(['cmake', root_dir] + cmake_args, cwd=build_path)
    subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_path)

build_cga(cmake_build_folder="py_build")

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
      ext_modules=cythonize(extensions),
      scripts=[os.path.join('bin', 'genome_simulator'),
               os.path.join('bin', 'assembly_evaluator')],
      install_requires=["cython"]
      )
