import random
from difflib import SequenceMatcher

import claragenomics.bindings.cudapoa as cudapoa
from claragenomics.bindings.cuda import *

stream = CudaStream()

a = cudapoa.PyCudapoa()
a.set_cuda_stream(stream)

a.add_poa()
a.add_seq_to_poa("ACTGACTG")
a.add_seq_to_poa("ACTTACTG")
a.add_seq_to_poa("ACGGACTG")
a.add_seq_to_poa("ATCGACTG")
a.generate_poa()
con = a.get_consensus()
print(con)

a.add_poa()
a.add_seq_to_poa("ACTGAC")
a.add_seq_to_poa("ACTTAC")
a.add_seq_to_poa("ACGGAC")
a.add_seq_to_poa("ATCGAC")
a.generate_poa()
con = a.get_consensus()
print(con)


read_len = 500
ref = ''.join([random.choice(['A','C', 'G', 'T']) for _ in range(read_len)])
num_reads=100
mutation_rate=0.02
reads = []
for _ in range(num_reads):
    new_read = ''.join([r if random.random() > mutation_rate else random.choice(['A','C', 'G', 'T']) for r in ref])
    reads.append(new_read)


b = cudapoa.PyCudapoa()
b.set_cuda_stream(stream)

b.add_poa()
for r in reads:
    b.add_seq_to_poa(r)

b.generate_poa()
consensus = b.get_consensus()[0].decode('utf-8')


print("Similarity beteen original string and consensus is ", SequenceMatcher(None, ref, consensus).ratio())
