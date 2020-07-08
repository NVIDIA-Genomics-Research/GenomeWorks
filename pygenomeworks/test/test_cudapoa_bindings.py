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


from difflib import SequenceMatcher
import pytest
import random

from genomeworks.cudapoa import CudaPoaBatch
import genomeworks.cuda as cuda


@pytest.mark.gpu
def test_cudapoa_simple_batch():
    device = cuda.cuda_get_device()
    free, total = cuda.cuda_get_mem_info(device)
    batch = CudaPoaBatch(10, 1024, 0.9 * free, deivce_id=device,
                         output_mask='consensus')
    poa_1 = ["ACTGACTG", "ACTTACTG", "ACGGACTG", "ATCGACTG"]
    poa_2 = ["ACTGAC", "ACTTAC", "ACGGAC", "ATCGAC"]
    batch.add_poa_group(poa_1)
    batch.add_poa_group(poa_2)
    batch.generate_poa()
    consensus, coverage, status = batch.get_consensus()

    assert(len(consensus) == 2)
    assert(batch.total_poas == 2)


def test_cudapoa_banded_aligned_batch():
    device = cuda.cuda_get_device()
    free, total = cuda.cuda_get_mem_info(device)
    batch = CudaPoaBatch(10, 1024, 0.9 * free,
                         deivce_id=device,
                         output_mask='consensus',
                         cuda_banded_alignment=True)
    poa_1 = ["ACTGACTG", "ACTTACTG", "ACGGACTG", "ATCGACTG"]
    poa_2 = ["ACTGAC", "ACTTAC", "ACGGAC", "ATCGAC"]
    batch.add_poa_group(poa_1)
    batch.add_poa_group(poa_2)
    batch.generate_poa()
    consensus, coverage, status = batch.get_consensus()

    assert(len(consensus) == 2)
    assert(batch.total_poas == 2)


@pytest.mark.gpu
def test_cudapoa_incorrect_output_type():
    device = cuda.cuda_get_device()
    free, total = cuda.cuda_get_mem_info(device)
    try:
        CudaPoaBatch(10, 1024, 0.9 * free, deivce_id=device,
                     output_type='error_input')
        assert(False)
    except RuntimeError:
        pass


@pytest.mark.gpu
def test_cudapoa_valid_output_type():
    device = cuda.cuda_get_device()
    free, total = cuda.cuda_get_mem_info(device)
    try:
        CudaPoaBatch(10, 1024, 0.9 * free, deivce_id=device,
                     output_type='consensus')
    except RuntimeError:
        assert(False)


@pytest.mark.gpu
def test_cudapoa_reset_batch():
    device = cuda.cuda_get_device()
    free, total = cuda.cuda_get_mem_info(device)
    batch = CudaPoaBatch(10, 1024, 0.9 * free, device_id=device)
    poa_1 = ["ACTGACTG", "ACTTACTG", "ACGGACTG", "ATCGACTG"]
    batch.add_poa_group(poa_1)
    batch.generate_poa()
    consensus, coverage, status = batch.get_consensus()

    assert(batch.total_poas == 1)

    batch.reset()

    assert(batch.total_poas == 0)


@pytest.mark.gpu
def test_cudapoa_graph():
    device = cuda.cuda_get_device()
    free, total = cuda.cuda_get_mem_info(device)
    batch = CudaPoaBatch(10, 1024, 0.9 * free, device_id=device)
    poa_1 = ["ACTGACTG", "ACTTACTG", "ACTCACTG"]
    batch.add_poa_group(poa_1)
    batch.generate_poa()
    consensus, coverage, status = batch.get_consensus()

    assert(batch.total_poas == 1)

    # Expected graph
    #           - -> G -> -
    #           |         |
    # A -> C -> T -> T -> A -> C -> T -> G
    #           |         |
    #           - -> C -> -

    graphs, status = batch.get_graphs()
    assert(len(graphs) == 1)

    digraph = graphs[0]
    assert(digraph.number_of_nodes() == 10)
    assert(digraph.number_of_edges() == 11)


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
        reads.append(new_read)

    device = cuda.cuda_get_device()
    free, total = cuda.cuda_get_mem_info(device)
    stream = cuda.CudaStream()
    batch = CudaPoaBatch(1000, 1024, 0.9 * free, stream=stream, device_id=device)
    (add_status, seq_status) = batch.add_poa_group(reads)
    batch.generate_poa()

    consensus, coverage, status = batch.get_consensus()

    consensus = consensus[0]
    assert(len(consensus) == len(ref))

    match_ratio = SequenceMatcher(None, ref, consensus).ratio()
    assert(match_ratio == 1.0)
