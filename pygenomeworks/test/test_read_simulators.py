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
from genomeworks.simulators import genomesim
from genomeworks import simulators


num_reads_data = [
    (1, 100000, 1, 100),
    (20, 2000, 20, 100),
    pytest.param(1, 10, 1, 100, marks=pytest.mark.xfail(reason="Reads longer than reference")),
]


@pytest.mark.cpu
@pytest.mark.parametrize("num_reads, reference_length, num_reads_expected, read_median_length", num_reads_data)
def test_noisy_generator_number(num_reads, reference_length, num_reads_expected, read_median_length):
    """ Test generated length for Markovian genome simulator is correct"""

    genome_simulator = genomesim.MarkovGenomeSimulator()
    reference_string = genome_simulator.build_reference(reference_length,
                                                        transitions=simulators.HIGH_GC_HOMOPOLYMERIC_TRANSITIONS)
    read_generator = readsim.NoisyReadSimulator()
    num_reads_generated = 0
    for _ in range(num_reads):
        read_generator.generate_read(reference_string, read_median_length)
        num_reads_generated += 1

    assert(num_reads_generated == num_reads_expected)
