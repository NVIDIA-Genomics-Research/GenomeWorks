"""Classes to simulate genomes"""
import abc
import random

import numpy as np
from tqdm  import tqdm

from genomeworks.simulators import NUCLEOTIDES


class GenomeSimulator(abc.ABC):
    """Base class for genome simulators"""
    @abc.abstractmethod
    def build_reference(self):
        pass


class PoissonGenomeSimulator(GenomeSimulator):
    """Simulates genome with poisson process"""

    def __init__(self):
        pass

    def build_reference(self, reference_length):
        """Simulates genome with poisson process

        Args:
          reference_length: The desired genome length

        Returns:
          String corresponding to reference genome. Each character is a
          nucleotide radnomly selected from a uniform, flat distribution.
        """
        reference_length = int(reference_length)
        return ''.join(random.choice(tuple(NUCLEOTIDES)) for x in range(reference_length))


class MarkovGenomeSimulator(GenomeSimulator):
    """Simulates genome with a Markovian process"""

    def __init__(self):
        pass

    def build_reference(self, reference_length, transitions):
        """Simulates genome with a Markovian process

        Args:
          reference_length: The desired genome length
          transitions: dict of dict with all transition probabilities
            e.g {'A': {'A':0.1,'C':0.3',...}, 'C':{'A':0.3,...}...}

        Returns:
          String corresponding to reference genome. Each character is a
          nucleotide radnomly selected from a uniform, flat distribution.
        """
        reference_length = int(reference_length)
        prev_base = random.choice(list(NUCLEOTIDES))
        ref_bases = [prev_base]
        print("Simulating genome:")
        for _ in tqdm(range(reference_length - 1)):
            next_base_choices = list(zip(*transitions[prev_base].items()))
            next_base_candidates = next_base_choices[0]
            next_base_pd = np.array(next_base_choices[1])
            next_base_pd = next_base_pd / next_base_pd.sum()
            prev_base = np.random.choice(next_base_candidates, 1, p=next_base_pd)[0]
            ref_bases.append(prev_base)
        return "".join(ref_bases)
