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


import pytest

from genomeworks.simulators import readsim

test_reads = [
    ((("read_0",
       "AACGTCA",
       100,
       900),
      ("read_1",
       "AACGTCA",
       100,
       900)), 1),
    ((("read_0",
       "AACGTCA",
       100,
       900),
      ("read_1",
       "AACGTCA",
       1000,
       9000)), 0),
    ((("read_1",
       "AACGTCA",
       100,
       900),
      ("read_0",
       "AACGTCA",
       100,
       900)), 1),
    ((("read_1",
       "AACGTCA",
       100,
       900),
      ("read_0",
       "AACGTCA",
       100,
       900)), 1),
    ((("read_1",
       "AACGTCA",
       100,
       900),
      ("read_2",
       "AACGTCA",
       100,
       900),
      ("read_3",
       "AACGTCA",
       100,
       900)), 3),
]


@pytest.mark.cpu
@pytest.mark.parametrize("reads, expected_overlaps", test_reads)
def test_generates_overlaps(reads, expected_overlaps):
    """ Test that the number of overlaps detected is correct"""
    overlaps = readsim.generate_overlaps(reads, gzip_compressed=False)
    assert(len(overlaps) == expected_overlaps)
