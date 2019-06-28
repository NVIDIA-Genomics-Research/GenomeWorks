import pytest

from cga.simulators import genomesim
from cga import simulators


genome_lengths_data = [
    (4, 4),
    (100, 100),
    (2000, 2000),
    (1e4, int(1e4)),
    (1e6, int(1e6)),
]


@pytest.mark.parametrize("reference_length, expected", genome_lengths_data)
def test_markov_length(reference_length, expected):
    """ Test generated length for Markovian genome simulator is correct"""

    genome_simulator = genomesim.PoissonGenomeSimulator()
    reference_string = genome_simulator.build_reference(reference_length)
    assert(len(reference_string) == expected)


@pytest.mark.parametrize("reference_length, expected", genome_lengths_data)
def test_poisson_length(reference_length, expected):
    """ Test generated length for Poisson genome simulator is correct"""

    genome_simulator = genomesim.MarkovGenomeSimulator()
    reference_string = genome_simulator.build_reference(reference_length,
                                                        transitions=simulators.HIGH_GC_HOMOPOLYMERIC_TRANSITIONS)
    assert(len(reference_string) == expected)
