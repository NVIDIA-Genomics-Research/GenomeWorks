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

"""pyclaragenomics setup script.

A script to build and install pyclaragenomics from source. More information
about usage can be found by running
    python setup_pyclaragenomics.py -h

"""

import argparse
import os
import subprocess


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='build & install pyclaragenomics SDK.')
    parser.add_argument('--build_output_folder',
                        required=False,
                        default="cga_build",
                        help="Choose an output folder for building")
    parser.add_argument('--create_wheel_only',
                        required=False,
                        action='store_true',
                        help="Creates a python wheel package from pyclaragenomics (no installation)")
    parser.add_argument('--develop',
                        required=False,
                        action='store_true',
                        help="Install using pip editble mode")
    parser.add_argument("--overwrite_package_name",
                        required=False,
                        default=None,
                        help="Overwrite package name")
    parser.add_argument("--overwrite_package_version",
                        required=False,
                        default=None,
                        help="Overwrite package version")
    return parser.parse_args()


class CMakeWrapper:
    """Class to encapsulate building a CMake project."""

    def __init__(self,
                 cmake_root_dir,
                 cmake_build_path="cmake_build",
                 cga_install_dir="cmake_build/install",
                 cmake_extra_args=""):
        """Class constructor.

        Args:
            cmake_root_dir : Root directory of CMake project
            cmake_build_path : cmake build output folder
            cga_install_dir: pyclaragenomics installation directory
            cmake_extra_args : Extra string arguments to be passed to CMake during setup
        """
        self.cmake_root_dir = os.path.abspath(cmake_root_dir)
        self.build_path = os.path.abspath(cmake_build_path)
        self.cga_install_dir = os.path.abspath(cga_install_dir)
        self.cmake_extra_args = cmake_extra_args
        self.cuda_toolkit_root_dir = os.environ.get("CUDA_TOOLKIT_ROOT_DIR")

    def _run_cmake_cmd(self):
        """Build and call CMake command."""
        cmake_args = ['-DCMAKE_INSTALL_PREFIX=' + self.cga_install_dir,
                      '-DCMAKE_BUILD_TYPE=' + 'Release',
                      '-DCMAKE_INSTALL_RPATH=' + os.path.join(self.cga_install_dir, "lib"),
                      '-Dcga_generate_docs=OFF']
        cmake_args += [self.cmake_extra_args] if self.cmake_extra_args else []

        if self.cuda_toolkit_root_dir:
            cmake_args += ["-DCUDA_TOOLKIT_ROOT_DIR=%s" % self.cuda_toolkit_root_dir]

        if not os.path.exists(self.build_path):
            os.makedirs(self.build_path)

        subprocess.check_call(['cmake', self.cmake_root_dir] + cmake_args, cwd=self.build_path)

    def _run_build_cmd(self):
        """Build and run make command."""
        build_args = ['--', '-j16', 'install']
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_path)

    def build(self):
        """Run full CMake build pipeline."""
        self._run_cmake_cmd()
        self._run_build_cmd()


def get_package_version(overwritten_package_version, cga_dir):
    """Returns the correct version for pyclaragenomics python package.

    In case the user didn't overwrite the package name returns cga version found in VERSION file otherwise,
    returns the overwritten package name
    """
    if overwritten_package_version is not None:
        return overwritten_package_version
    # Get cga version from VERSION file
    with open(os.path.join(cga_dir, 'VERSION'), 'r') as f:
        return f.read().replace('\n', '')


def setup_python_binding(is_develop_mode, wheel_output_folder, cga_dir, pycga_dir, cga_install_dir,
                         pyclaragenomics_rename, pyclaragenomics_version):
    """Setup python bindings and pyclaragenomics modules for pyclaragenomics.

    Args:
        is_develop_mode : Develop or install mode for installation
        wheel_output_folder : Output directory for pyclaragenomics wheel file
        cga_dir : Root pyclaragenomics directory
        pycga_dir : Root pyclaragenomics directory
        cga_install_dir : Directory with pyclaragenomics SDK installation
        pyclaragenomics_rename : rename pyclaragenomics package
        pyclaragenomics_version : pyclaragenomics package version
    """
    if wheel_output_folder:
        setup_command = [
            'python3', '-m',
            'pip', 'wheel', '.',
            '--global-option', 'sdist',
            '--wheel-dir', wheel_output_folder, '--no-deps'
        ]
        completion_message = \
            "A wheel file was create for pyclaragenomics under {}".format(wheel_output_folder)
    else:
        setup_command = ['python3', '-m', 'pip', 'install'] + (['-e'] if is_develop_mode else []) + ["."]
        completion_message = \
            "pyclaragenomics was successfully setup in {} mode!".format(
                "development" if args.develop else "installation")

    subprocess.check_call(setup_command, env={
        **{
            **os.environ,
            'CGA_ROOT_DIR': cga_dir,
            'CGA_INSTALL_DIR': cga_install_dir,
            'CGA_VERSION': pyclaragenomics_version
        },
        **({} if pyclaragenomics_rename is None else {'PYCGA_RENAME': pyclaragenomics_rename})
    }, cwd=pycga_dir)
    print(completion_message)


if __name__ == "__main__":

    args = parse_arguments()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    cga_root_dir = os.path.dirname(current_dir)
    cga_installation_directory = os.path.join(args.build_output_folder, "install")
    # Build & install pyclaragenomics SDK
    cmake_proj = CMakeWrapper(cmake_root_dir=cga_root_dir,
                              cmake_build_path=args.build_output_folder,
                              cga_install_dir=cga_installation_directory,
                              cmake_extra_args="-Dcga_build_shared=ON")
    cmake_proj.build()
    # Setup pyclaragenomics
    setup_python_binding(
        is_develop_mode=args.develop,
        wheel_output_folder='pyclaragenomics_wheel/' if args.create_wheel_only else None,
        cga_dir=cga_root_dir,
        pycga_dir=current_dir,
        cga_install_dir=os.path.realpath(cga_installation_directory),
        pyclaragenomics_rename=args.overwrite_package_name,
        pyclaragenomics_version=get_package_version(args.overwrite_package_version, cga_root_dir)
    )
