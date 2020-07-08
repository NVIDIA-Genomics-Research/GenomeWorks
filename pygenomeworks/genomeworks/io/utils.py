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


"""Utility functions for I/O of custom files."""


def read_poa_group_file(file_path, num_windows=0):
    """Parses data file containing POA groups.

    Args:
        file_path   : Path to POA group file.
                      File format is as follows -
                      <num sequences>
                      seq 1...
                      seq 2...
                      <num sequences>
                      seq 1...
                      seq 2...
                      seq 3...
        num_windows : Number of windows to extract from
                      file. If requested is more than available
                      in file, windows are repoeated in a circular
                      loop like fashion.
                      0 (default) implies using only those windows
                      available in file.

    """
    with open(file_path, "r") as window_file:
        num_seqs_in_group = 0
        group_list = []
        current_seq_list = []
        first_seq = True

        for line in window_file:
            line = line.strip()

            # First line is num sequences in group
            if (num_seqs_in_group == 0):
                if first_seq:
                    first_seq = False
                else:
                    group_list.append(current_seq_list)
                    current_seq_list = []
                num_seqs_in_group = int(line)
            else:
                current_seq_list.append(line)
                num_seqs_in_group = num_seqs_in_group - 1

        if len(group_list) == 0:
            group_list.append(current_seq_list)

        if (num_windows > 0):
            if (num_windows < len(group_list)):
                group_list = group_list[:num_windows]
            else:
                original_num_windows = len(group_list)
                num_windows_to_add = num_windows - original_num_windows
                for i in range(num_windows_to_add):
                    group_list.append(group_list[i % original_num_windows])

        return group_list
