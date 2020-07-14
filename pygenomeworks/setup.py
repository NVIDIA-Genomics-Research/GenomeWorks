#!/usr/bin/env python3

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


"""Python setuptools setup."""

import glob
import os
import shutil
from setuptools import setup, Extension, find_packages

from Cython.Build import cythonize


def get_verified_absolute_path(path):
    """Verify and return absolute path of argument.

    Args:
        path : Relative/absolute path

    Returns:
        Absolute path
    """
    installed_path = os.path.abspath(path)
    if not os.path.exists(installed_path):
        raise RuntimeError("The requested path does not exist:{}".format(installed_path))
    return installed_path


def get_installation_requirments(file_path):
    """Parse pip requirements file.

    Args:
        file_path : path to pip requirements file

    Returns:
        list of requirement strings
    """
    with open(file_path, 'r') as file:
        requirements_file_content = \
            [line.strip() for line in file if line.strip() and not line.lstrip().startswith('#')]
    return requirements_file_content


def copy_all_files_in_directory(src, dest, file_ext="*.so"):
    """Copy files with given extension from source to destination directories.

    Args:
        src : source directory
        dest : destination directory
        file_ext : a regular expression string capturing relevant files
    """
    files_to_copy = glob.glob(os.path.join(src, file_ext))
    if not files_to_copy:
        raise RuntimeError("No {} files under {}".format(src, file_ext))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        for file in files_to_copy:
            shutil.copy(file, dest)
            print("{} was copied into {}".format(file, dest))
    except (shutil.Error, PermissionError) as err:
        print('Could not copy {}. Error: {}'.format(file, err))
        raise err


# Must be set before calling pip
for envvar in ['GW_INSTALL_DIR', 'GW_VERSION', 'GW_ROOT_DIR']:
    if envvar not in os.environ.keys():
        raise EnvironmentError(
            '{} environment variables must be set'.format(envvar))

gw_root_dir = os.environ['GW_ROOT_DIR']
gw_install_dir = os.environ['GW_INSTALL_DIR']
gw_version = os.environ['GW_VERSION']
pygw_name = os.getenv('PYGW_RENAME', 'genomeworks')
cuda_root = os.getenv('CUDA_TOOLKIT_ROOT_DIR', '/usr/local/cuda')
cuda_include_path = os.path.join(cuda_root, 'include')
cuda_library_path = os.path.join(cuda_root, 'lib64')

# Get current dir (pygenomeworks folder is copied into a temp directory created by pip)
current_dir = os.path.dirname(os.path.realpath(__file__))


# Copies shared libraries into genomeworks package
copy_all_files_in_directory(
    get_verified_absolute_path(os.path.join(gw_install_dir, "lib")),
    os.path.join(current_dir, "genomeworks", "shared_libs/"),
)

# Copies license from genomeworks root dir for packaging
copy_all_files_in_directory(
    get_verified_absolute_path(gw_root_dir),
    get_verified_absolute_path(current_dir),
    file_ext="LICENSE"
)

# Classifiers for PyPI
pygw_classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9"
]

extensions = [
    Extension(
        "genomeworks.cuda.cuda",
        sources=[os.path.join("genomeworks/cuda/cuda.pyx")],
        include_dirs=[
            cuda_include_path,
        ],
        library_dirs=[cuda_library_path],
        runtime_library_dirs=[cuda_library_path],
        libraries=["cudart"],
        language="c++",
        extra_compile_args=["-std=c++14"],
    ),
    Extension(
        "genomeworks.cudapoa.cudapoa",
        sources=[os.path.join("genomeworks/cudapoa/cudapoa.pyx")],
        include_dirs=[
            cuda_include_path,
            get_verified_absolute_path(os.path.join(gw_install_dir, "include")),
            get_verified_absolute_path(os.path.join(gw_root_dir, "3rdparty", "spdlog", "include")),
        ],
        library_dirs=[cuda_library_path, get_verified_absolute_path(os.path.join(gw_install_dir, "lib"))],
        runtime_library_dirs=[cuda_library_path, os.path.join('$ORIGIN', os.pardir, 'shared_libs')],
        libraries=["cudapoa", "cudart", "gwbase"],
        language="c++",
        extra_compile_args=["-std=c++14"],
    ),
    Extension(
        "genomeworks.cudaaligner.cudaaligner",
        sources=[os.path.join("genomeworks/cudaaligner/cudaaligner.pyx")],
        include_dirs=[
            cuda_include_path,
            get_verified_absolute_path(os.path.join(gw_install_dir, "include")),
            get_verified_absolute_path(os.path.join(gw_root_dir, "3rdparty", "cub")),
            get_verified_absolute_path(os.path.join(gw_root_dir, "3rdparty", "spdlog", "include")),
        ],
        library_dirs=[cuda_library_path, get_verified_absolute_path(os.path.join(gw_install_dir, "lib"))],
        runtime_library_dirs=[cuda_library_path, os.path.join('$ORIGIN', os.pardir, 'shared_libs')],
        libraries=["cudaaligner", "cudart", "gwbase"],
        language="c++",
        extra_compile_args=["-std=c++14"],
    )
]


setup(name=pygw_name,
      version=gw_version,
      description='NVIDIA genomics python libraries and utiliites',
      author='NVIDIA Corporation',
      url="https://github.com/clara-parabricks/GenomeWorks",
      include_package_data=True,
      data_files=[
          ('gw_shared_objects', glob.glob('genomeworks/shared_libs/*.so'))
      ],
      install_requires=get_installation_requirments(
          get_verified_absolute_path(os.path.join(current_dir, 'requirements.txt'))
      ),
      packages=find_packages(where=current_dir, include=['genomeworks*']),
      python_requires='>=3.5',
      license='Apache License 2.0',
      long_description='Python libraries and utilities for manipulating genomics data',
      long_description_content_type='text/plain',
      classifiers=pygw_classifiers,
      platforms=['any'],
      ext_modules=cythonize(extensions, compiler_directives={'embedsignature': True}),
      scripts=[os.path.join('bin', 'genome_simulator')],
      )
