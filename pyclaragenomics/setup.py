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

import os
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize


def get_verified_path(path):
    installed_path = os.path.abspath(path)
    if not os.path.exists(installed_path):
        raise RuntimeError("No valid path for requested component exists")
    return installed_path


# Must be set before calling pip
try:
    pycga_dir = os.environ['PYCGA_DIR']
    cga_install_dir = os.environ['CGA_INSTALL_DIR']
except KeyError as e:
    raise EnvironmentError(
        'PYCGA_DIR CGA_INSTALL_DIR environment variables must be set').with_traceback(e.__traceback__)


extensions = [
    Extension(
        "*",
        sources=[os.path.join(pycga_dir, "claragenomics/**/*.pyx")],
        include_dirs=[
            "/usr/local/cuda/include",
            get_verified_path(os.path.join(cga_install_dir, "include")),
        ],
        library_dirs=["/usr/local/cuda/lib64", get_verified_path(os.path.join(cga_install_dir,  "lib"))],
        runtime_library_dirs=["/usr/local/cuda/lib64", get_verified_path(os.path.join(cga_install_dir, "lib"))],
        libraries=["cudapoa", "cudaaligner", "cudart"],
        language="c++",
        extra_compile_args=["-std=c++14"],
    )
]

setup(name='pyclaragenomics',
      version='0.3.0',
      description='NVIDIA genomics python libraries and utiliites',
      author='NVIDIA Corporation',
      packages=find_packages(where=pycga_dir),
      ext_modules=cythonize(extensions, compiler_directives={'embedsignature': True}),
      scripts=[get_verified_path(os.path.join(pycga_dir, 'bin', 'genome_simulator')),
               get_verified_path(os.path.join(pycga_dir, 'bin', 'assembly_evaluator'))],
      )
