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


"""genomeworks setup script.

A script to build and install genomeworks from source. More information
about usage can be found by running
    python setup_genomeworks.py -h

"""

import argparse
import os
import subprocess


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='build & install GenomeWorks SDK.')
    parser.add_argument('--build_output_folder',
                        required=False,
                        default="gw_build",
                        help="Choose an output folder for building")
    parser.add_argument('--create_wheel_only',
                        required=False,
                        action='store_true',
                        help="Creates a python wheel package from genomeworks (no installation)")
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
                 gw_install_dir="cmake_build/install",
                 cmake_extra_args=""):
        """Class constructor.

        Args:
            cmake_root_dir : Root directory of CMake project
            cmake_build_path : cmake build output folder
            gw_install_dir: GenomeWorks installation directory
            cmake_extra_args : Extra string arguments to be passed to CMake during setup
        """
        self.cmake_root_dir = os.path.abspath(cmake_root_dir)
        self.build_path = os.path.abspath(cmake_build_path)
        self.gw_install_dir = os.path.abspath(gw_install_dir)
        self.cmake_extra_args = cmake_extra_args
        self.cuda_toolkit_root_dir = os.environ.get("CUDA_TOOLKIT_ROOT_DIR")

    def _run_cmake_cmd(self):
        """Build and call CMake command."""
        cmake_args = ['-DCMAKE_INSTALL_PREFIX=' + self.gw_install_dir,
                      '-DCMAKE_BUILD_TYPE=' + 'Release',
                      '-DCMAKE_INSTALL_RPATH=' + os.path.join(self.gw_install_dir, "lib"),
                      '-Dgw_generate_docs=OFF',
                      '-Dgw_cuda_gen_all_arch=OFF']
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


def get_package_version(overwritten_package_version, gw_dir):
    """Returns the correct version for genomeworks python package.

    In case the user didn't overwrite the package name returns GW version found in VERSION file otherwise,
    returns the overwritten package name
    """
    if overwritten_package_version is not None:
        return overwritten_package_version
    # Get GW version from VERSION file
    with open(os.path.join(gw_dir, 'VERSION'), 'r') as f:
        return f.read().replace('\n', '')


def setup_python_binding(is_develop_mode, wheel_output_folder, gw_dir, pygw_dir, gw_install_dir,
                         genomeworks_rename, genomeworks_version):
    """Setup python bindings and genomeworks modules for genomeworks.

    Args:
        is_develop_mode : Develop or install mode for installation
        wheel_output_folder : Output directory for genomeworks wheel file
        gw_dir : Root GenomeWorks directory
        pygw_dir : Root genomeworks directory
        gw_install_dir : Directory with GenomeWorks SDK installation
        genomeworks_rename : rename genomeworks package
        genomeworks_version : genomeworks package version
    """
    if wheel_output_folder:
        setup_command = [
            'python3', '-m',
            'pip', 'wheel', '.',
            '--global-option', 'sdist',
            '--wheel-dir', wheel_output_folder, '--no-deps'
        ]
        completion_message = \
            "A wheel file was create for genomeworks under {}".format(wheel_output_folder)
    else:
        setup_command = ['python3', '-m', 'pip', 'install'] + (['-e'] if is_develop_mode else []) + ["."]
        completion_message = \
            "genomeworks was successfully setup in {} mode!".format(
                "development" if args.develop else "installation")

    subprocess.check_call(setup_command, env={
        **{
            **os.environ,
            'GW_ROOT_DIR': gw_dir,
            'GW_INSTALL_DIR': gw_install_dir,
            'GW_VERSION': genomeworks_version
        },
        **({} if genomeworks_rename is None else {'PYGW_RENAME': genomeworks_rename})
    }, cwd=pygw_dir)
    print(completion_message)


if __name__ == "__main__":

    args = parse_arguments()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    gw_root_dir = os.path.dirname(current_dir)
    gw_installation_directory = os.path.join(args.build_output_folder, "install")
    # Build & install GenomeWorks SDK
    cmake_proj = CMakeWrapper(cmake_root_dir=gw_root_dir,
                              cmake_build_path=args.build_output_folder,
                              gw_install_dir=gw_installation_directory,
                              cmake_extra_args="-Dgw_build_shared=ON")
    cmake_proj.build()
    # Setup genomeworks
    setup_python_binding(
        is_develop_mode=args.develop,
        wheel_output_folder='genomeworks_wheel/' if args.create_wheel_only else None,
        gw_dir=gw_root_dir,
        pygw_dir=current_dir,
        gw_install_dir=os.path.realpath(gw_installation_directory),
        genomeworks_rename=args.overwrite_package_name,
        genomeworks_version=get_package_version(args.overwrite_package_version, gw_root_dir)
    )
