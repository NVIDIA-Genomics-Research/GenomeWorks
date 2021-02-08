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

from genomeworks.cudaaligner import CudaAlignerBatch
import genomeworks.cuda as cuda
from genomeworks.simulators.genomesim import PoissonGenomeSimulator
from genomeworks.simulators.readsim import NoisyReadSimulator


@pytest.mark.gpu
@pytest.mark.parametrize("query, target, cigar", [
    ("AAAAAAA", "TTTTTTT", "7M"),
    ("AAATC", "TACGTTTT", "3M1I2M2I"),
    ("TACGTA", "ACATAC", "1D5M1I"),
    ("TGCA", "ATACGCT", "1I1M2I3M"),
    pytest.param("ACGT", "TCGA", "5M", marks=pytest.mark.xfail),
])
def test_cudaaligner_simple_batch(query, target, cigar):
    """Test valid calculation of alignments by checking cigar strings.
    """
    device = cuda.cuda_get_device()
    stream = cuda.CudaStream()
    batch = CudaAlignerBatch(len(query), len(target), 1, alignment_type="global", stream=stream, device_id=device)
    batch.add_alignment(query, target)
    batch.align_all()
    alignments = batch.get_alignments()

    assert(len(alignments) == 1)
    assert(alignments[0].cigar == cigar)


@pytest.mark.gpu
@pytest.mark.parametrize("ref_length, num_alignments", [
    (5000, 30),
    (10000, 10),
    (500, 100)
])
def test_cudaaligner_long_alignments(ref_length, num_alignments):
    """Test varying batches of long and short alignments and check for successful
    completion of alignment.
    """
    device = cuda.cuda_get_device()
    genome_sim = PoissonGenomeSimulator()
    read_sim = NoisyReadSimulator()

    batch = CudaAlignerBatch(ref_length, ref_length, num_alignments, device_id=device)

    for _ in range(num_alignments):
        reference = genome_sim.build_reference(ref_length)
        query, start, end = read_sim.generate_read(reference, ref_length, insertion_error_rate=0.0)
        target, start, end = read_sim.generate_read(reference, ref_length, insertion_error_rate=0.0)

        batch.add_alignment(query, target)

    batch.align_all()
    batch.get_alignments()

    # Test reset
    batch.reset()
    assert(len(batch.get_alignments()) == 0)


@pytest.mark.gpu
@pytest.mark.parametrize("max_seq_len, max_alignments, seq_len, num_alignments, should_succeed", [
    (1000, 100, 10000, 10, False),
    (1000, 100, 100, 10, True),
    (1000, 100, 1000, 100, True),
    (100, 10, 100, 1000, False),
])
def test_cudaaligner_various_arguments(max_seq_len, max_alignments, seq_len, num_alignments, should_succeed):
    """
    Pass legal and illegal arguments, and test for correct exception throwing behavior.
    """
    device = cuda.cuda_get_device()
    genome_sim = PoissonGenomeSimulator()
    read_sim = NoisyReadSimulator()

    batch = CudaAlignerBatch(max_seq_len, max_seq_len, max_alignments, device_id=device)

    success = True
    for _ in range(num_alignments):
        reference = genome_sim.build_reference(seq_len)
        query, start, end = read_sim.generate_read(reference, seq_len, insertion_error_rate=0.0)
        target, start, end = read_sim.generate_read(reference, seq_len, insertion_error_rate=0.0)

        status = batch.add_alignment(query, target)
        if status != 0:
            success &= False

    batch.align_all()

    assert(success is should_succeed)
