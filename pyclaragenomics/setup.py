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
import glob
import os
from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize


def get_verified_path(path):
    installed_path = os.path.abspath(path)
    if not os.path.exists(installed_path):
        raise RuntimeError("No valid path for requested component exists")
    return installed_path


def get_installation_requirments(file_path):
    with open(file_path, 'r') as file:
        requirements_file_content = \
            [line.strip() for line in file if line.strip() and not line.lstrip().startswith('#')]
    return requirements_file_content


# Must be set before calling pip
try:
    pycga_dir = os.environ['PYCGA_DIR']
    cga_install_dir = os.environ['CGA_INSTALL_DIR']
    cga_runtime_lib_dir = os.environ['CGA_RUNTIME_LIB_DIR']
except KeyError as e:
    raise EnvironmentError(
        'PYCGA_DIR CGA_INSTALL_DIR CGA_RUNTIME_LIB_DIR \
        environment variables must be set').with_traceback(e.__traceback__)

# Classifiers for PyPI
pycga_classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Operating System :: POSIX :: Linux',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'License :: OSI Approved :: Apache Software License',
    'Natural Language :: English',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
],

extensions = [
    Extension(
        "*",
        sources=[os.path.join(pycga_dir, "claragenomics/**/*.pyx")],
        include_dirs=[
            "/usr/local/cuda/include",
            get_verified_path(os.path.join(cga_install_dir, "include")),
        ],
        library_dirs=["/usr/local/cuda/lib64", get_verified_path(os.path.join(cga_install_dir, "lib"))],
        runtime_library_dirs=["/usr/local/cuda/lib64", cga_runtime_lib_dir],
        libraries=["cudapoa", "cudaaligner", "cudart"],
        language="c++",
        extra_compile_args=["-std=c++14"],
    )
]

setup(name='pyclaragenomics',
      version='0.4.0',
      description='NVIDIA genomics python libraries and utiliites',
      author='NVIDIA Corporation',
      url="https://github.com/clara-genomics/ClaraGenomicsAnalysis",
      package_data={
          'claragenomics': glob.glob(os.path.join(pycga_dir, 'claragenomics/shared_libs/*.so'))
      },
      install_requires=get_installation_requirments(os.path.join(pycga_dir, 'requirements.txt')),
      packages=find_packages(where=pycga_dir),
      python_requires='>=3.6',
      license='Apache License 2.0',
      long_description='Python libraries and utilities for manipulating genomics data',
      classifiers=pycga_classifiers,
      ext_modules=cythonize(extensions, compiler_directives={'embedsignature': True}),
      scripts=[get_verified_path(os.path.join(pycga_dir, 'bin', 'genome_simulator')),
               get_verified_path(os.path.join(pycga_dir, 'bin', 'assembly_evaluator'))],
      )
