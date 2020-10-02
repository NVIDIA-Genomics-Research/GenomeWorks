#!/usr/bin/env python

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

"""
Script to check for NVIDIA Copyright headers in source files and
add optionally copyright header to them.
"""

import argparse
import io
import os
import re
import stat
from subprocess import check_output
import sys

cpp_exts = {".hpp", ".cpp", ".cu", ".cuh", ".cc", ".h", ".c"}

cpp_copyright = r"""/*
* Copyright 2019-2020 NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

"""

other_copyright = r"""#
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

"""


def get_tracked_files(repo_root):
    """
    Get list of files tracked by git.

    Args:
    repo_root - Root folder for git
    """
    os.chdir(repo_root)
    branch = check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode('utf-8')
    branch = branch.replace('\n', '')
    filelist = check_output(["git", "ls-tree", "-r", branch, "--name-only"])
    tracked_files = list()
    for tracked_file in filelist.splitlines():
        tracked_files.append(os.path.join(repo_root, tracked_file.decode('utf-8')))
    return tracked_files


def filter_files(files, regex):
    """
    Filter a list of file names by removing entries that match
    the regular expression passed in.

    Args:
    files - List of file names
    regex - Regular expression for filtering files
    """
    filtered_files = []
    for fname in files:
        if (not os.path.isfile(fname)):
            continue
        elif (not re.search(regex, fname)):
            filtered_files.append(fname)
    return filtered_files


def add_copyright(f):
    """
    Add copyright header to the file passed in.

    Args:
    f - Path to file
    """
    extension = os.path.splitext(f)[1]
    original_permissions = os.stat(f).st_mode
    temp_f = f + ".copyright"

    with open(temp_f, "w") as myfile:
        copyright_text = cpp_copyright if extension in cpp_exts else other_copyright
        with open(f, "r") as orig:
            line = orig.readline()
            if (line and '#!' in line):
                # If file is a script and starts
                # with #!, then keep that line
                # as the first line and insert header
                # after that.
                myfile.write(line)
                myfile.write("\n")
                myfile.write(copyright_text)
                line = orig.readline()
            else:
                myfile.write(copyright_text)
            while line:
                myfile.write(line)
                line = orig.readline()
    os.rename(temp_f, f)
    # Resotre original file permissions
    os.chmod(f, stat.S_IMODE(original_permissions))


def copyright_present(f):
    """
    Check if file already has copyright header.

    Args:
    f - Path to file
    """
    with io.open(f, "r", encoding="utf-8") as fh:
        return re.search('Copyright (20[0-9][0-9]-)?20[0-9][0-9] NVIDIA CORPORATION', fh.read())


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Add NVIDIA copyright headers to source files")
    parser.add_argument('--add-header',
                        help='Add copyright header to the files which do not have it',
                        action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Get list of files to check.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.abspath(os.path.dirname(os.path.dirname(script_dir)))
    files = get_tracked_files(root_dir)
    files = filter_files(files, r"LICENSE|README|VERSION|data\/|docs\/")

    # Git list of files missing headers.
    missing_header_files = [f for f in files if not copyright_present(f)]

    # Check/Add headers if need be.
    if (args.add_header):
        [add_copyright(f) for f in missing_header_files]
    else:
        if (len(missing_header_files) > 0):
            print("List of files missing copyright headers - ")
            print("\n".join(missing_header_files))
            sys.exit(1)
