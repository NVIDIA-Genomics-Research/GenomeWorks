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

from genomeworks.simulators import genomesim
from genomeworks import simulators


genome_lengths_data = [
    (4, 4),
    (100, 100),
    (2000, 2000),
    (1e4, int(1e4)),
    (1e6, int(1e6)),
]


@pytest.mark.cpu
@pytest.mark.parametrize("reference_length, expected", genome_lengths_data)
def test_markov_length(reference_length, expected):
    """ Test generated length for Markovian genome simulator is correct"""

    genome_simulator = genomesim.PoissonGenomeSimulator()
    reference_string = genome_simulator.build_reference(reference_length)
    assert(len(reference_string) == expected)


@pytest.mark.cpu
@pytest.mark.parametrize("reference_length, expected", genome_lengths_data)
def test_poisson_length(reference_length, expected):
    """ Test generated length for Poisson genome simulator is correct"""

    genome_simulator = genomesim.MarkovGenomeSimulator()
    reference_string = genome_simulator.build_reference(reference_length,
                                                        transitions=simulators.HIGH_GC_HOMOPOLYMERIC_TRANSITIONS)
    assert(len(reference_string) == expected)
