import claragenomics.poa as cudapoa

a = cudapoa.PyCudapoa()
a.add_poa()
a.add_seq_to_poa("ACTGACTG")
a.add_seq_to_poa("ACTTACTG")
a.add_seq_to_poa("ACGGACTG")
a.add_seq_to_poa("ATCGACTG")
a.generate_poa()
con = a.get_consensus()
print(con)
