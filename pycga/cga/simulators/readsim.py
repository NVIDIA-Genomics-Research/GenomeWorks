"""Classes to simulate reads from a known reference, mimicking sequencing errors"""
import abc
import random

from claragenomics.simulators import NUCLEOTIDES


class ReadSimulator:
    @abc.abstractmethod
    def generate_read(self, reference, median_length, error_rate):
        pass


class NoisyReadSimulator(ReadSimulator):
    """Simulate sequencing errors in reads"""
    def __init__(self):
        pass

    def _add_snv_errors(self, read, error_rate):
        """Randomly introduce SNV errors

        Args:
          read (str): The nucleotide string
          error_rate (int): the ratio of bases which will be converted to SNVs

        Returns: The read (string) with SNVs introduced
        """
        noisy_bases = []
        for r in read:
            rand = random.uniform(0, 1)
            if rand > error_rate:
                noisy_bases.append(r)
            else:
                candidate_bases = NUCLEOTIDES ^ set((r,))
                new_base = random.choice(tuple(candidate_bases))
                noisy_bases.append(new_base)
        return "".join(noisy_bases)

    def _add_deletion_errors(self, read, error_rate):
        """Randomly introduce SNV errors

        Args:
          read (str): The nucleotide string
          error_rate (int): the ratio of bases which will be deleted

        Returns: The read (string) with deletions introduced
        """
        noisy_bases = []
        for r in read:
            rand = random.uniform(0, 1)
            if rand > error_rate:
                noisy_bases.append(r)
        return "".join(noisy_bases)

    def _add_insertion_errors(self, read, error_rate):
        """Randomly introduce SNV errors

        Args:
          read (str): The nucleotide string
          error_rate (int): the ratio of bases which will be reference insertions in the read

        Returns: The read (string) with insertions introduced
        """
        noisy_bases = []
        for r in read:
            rand = random.uniform(0, 1)
            if rand > error_rate:
                noisy_bases.append(r)
            else:
                new_base = random.choice(tuple(NUCLEOTIDES))
                noisy_bases.append(r)
                noisy_bases.append(new_base)
        return "".join(noisy_bases)

    def _add_homopolymer_clipping(self, read, homopolymer_survival_length, clip_rate):
        """Randomly reduce homopolymer length

        Args:
          read (str): The nucleotide string
          homopolymer_survival_length: Homopolymers with this length will not be clipped
          clip_rate: bases above this length in a homopolymer will be removed with this probability

        Returns: The read (string) with clipped homopolymers
        """
        homopolymer_len = 1
        prev_base = read[0]
        noisy_bases = [prev_base]
        for r in read[1:]:
            if r == prev_base:
                homopolymer_len += 1
                if homopolymer_len > homopolymer_survival_length:
                    if random.uniform(0, 1) > clip_rate:
                        noisy_bases.append(r)
                else:
                    noisy_bases.append(r)
            else:
                prev_base = r
                homopolymer_len = 1
                noisy_bases.append(r)

        return "".join(noisy_bases)

    def generate_read(self, reference,
                      median_length,
                      snv_error_rate=2.5e-2,
                      insertion_error_rate=1.25e-2,
                      deletion_error_rate=1.25e-2,
                      homopolymer_survival_length=4,
                      homopolymer_clip_rate=0.5):
        """Generate reads

        Args:
          reference (str): The reference nucleotides from which the read is generated
          median_length (int): Median length of generated read
          snv_error_rate (int): the ratio of bases which will be converted to SNVs
          insertion_error_rate (int): the ratio of bases which will be reference insertions in the read
          deletion_error_rate (int): the ratio of bases from the reference which will be deleted
          homopolymer_survival_length: Homopolymers with this length will not be clipped
          homopolyumer_clip_rate: bases above this length in a homopolymer will be removed with this probability

        Returns: A read randomly generated from the reference, with noise applied
        """

        reference_length = len(reference)
        pos = random.randint(0, reference_length - 1)

        def clamp(x):
            return max(0, min(x, reference_length-1))

        start = clamp(pos - median_length // 2)
        end = clamp(pos + median_length // 2) + median_length % 2
        substring = reference[start: end]

        substring = self._add_snv_errors(substring, snv_error_rate)

        substring = self._add_insertion_errors(substring, insertion_error_rate)

        substring = self._add_deletion_errors(substring, deletion_error_rate)

        read = self._add_homopolymer_clipping(substring,
                                              homopolymer_survival_length=homopolymer_survival_length,
                                              clip_rate=homopolymer_clip_rate)

        return read
