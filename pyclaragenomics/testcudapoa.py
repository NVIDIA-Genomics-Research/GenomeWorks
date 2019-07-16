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
