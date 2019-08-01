#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from difflib import SequenceMatcher
import pytest
import random

from claragenomics.bindings.cudapoa import CudaPoaBatch
from claragenomics.bindings.cuda import CudaStream


@pytest.mark.gpu
def test_cudapoa_simple_batch():
    batch = CudaPoaBatch(10, 10, 1e9)
    poa_1 = [b"ACTGACTG", b"ACTTACTG", b"ACGGACTG", b"ATCGACTG"]
    poa_2 = [b"ACTGAC", b"ACTTAC", b"ACGGAC", b"ATCGAC"]
    batch.add_poa_group(poa_1)
    batch.add_poa_group(poa_2)
    batch.generate_poa()
    (consensus, coverage, status) = batch.get_consensus()

    assert(len(consensus) == 2)
    assert(batch.total_poas == 2)


@pytest.mark.gpu
def test_cudapoa_reset_batch():
    batch = CudaPoaBatch(10, 10, 1e9)
    poa_1 = [b"ACTGACTG", b"ACTTACTG", b"ACGGACTG", b"ATCGACTG"]
    batch.add_poa_group(poa_1)
    batch.generate_poa()
    (consensus, coverage, status) = batch.get_consensus()

    assert(batch.total_poas == 1)

    batch.reset()

    assert(batch.total_poas == 0)


@pytest.mark.gpu
def test_cudapoa_complex_batch():
    random.seed(2)
    read_len = 500
    ref = ''.join([random.choice(['A', 'C', 'G', 'T']) for _ in range(read_len)])
    num_reads = 100
    mutation_rate = 0.02
    reads = []
    for _ in range(num_reads):
        new_read = ''.join([r if random.random() > mutation_rate else random.choice(['A', 'C', 'G', 'T']) for r in ref])
        reads.append(new_read.encode())

    stream = CudaStream()
    batch = CudaPoaBatch(10, 1000, 2*1e9, stream=stream)
    (add_status, seq_status) = batch.add_poa_group(reads)
    batch.generate_poa()

    (consensus, coverage, status) = batch.get_consensus()

    consensus = consensus[0].decode('utf-8')
    assert(len(consensus) == len(ref))
    matcher = SequenceMatcher(None, ref, consensus)
    match_ratio = matcher.ratio()
    assert(match_ratio == 1.0)
