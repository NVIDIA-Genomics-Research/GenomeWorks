import random
from difflib import SequenceMatcher

from claragenomics.bindings.cudapoa import CudaPoaBatch
from claragenomics.bindings.cuda import CudaStream

# Test CudaPoaBatch without any streams
a = CudaPoaBatch(10, 10)

windows = list()
windows.append(["ACTGACTG", "ACTTACTG", "ACGGACTG", "ATCGACTG"])
windows.append(["ACTGAC", "ACTTAC", "ACGGAC", "ATCGAC"])
a.add_poas(windows)
a.generate_poa()
con = a.get_consensus()
print(con)


# Test CudaPoaBatch with stream
read_len = 500
ref = ''.join([random.choice(['A','C', 'G', 'T']) for _ in range(read_len)])
num_reads=100
mutation_rate=0.02
reads = []
for _ in range(num_reads):
    new_read = ''.join([r if random.random() > mutation_rate else random.choice(['A','C', 'G', 'T']) for r in ref])
    reads.append(new_read)

stream = CudaStream()
b = CudaPoaBatch(10, 1000, stream)
b.add_poas(reads)
b.generate_poa()
consensus = b.get_consensus()[0].decode('utf-8')

print("Similarity beteen original string and consensus is ", SequenceMatcher(None, ref, consensus).ratio())
