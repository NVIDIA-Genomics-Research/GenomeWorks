"""
Functions for FASTA file I/O
"""


def write_fasta(seqs, filepath):
    """Writes a fasta file for sequences.

    Args:
      seqs: list of 2-tuples containnig sequnces and their names, e.g [('seq1', 'ACGTC...'), ('seq2', 'TTGGC...'), ...]]
      filepath: path to file for writing out FASTA.

    Returns:
      None.
    """
    fasta_string = ""
    for s in seqs:
        fasta_string += ">{}\n".format(s[0])

        lines = [s[1][n*80:(n+1)*80] for n in range((len(s[1])//80)+1)]

        fasta_string += "\n".join(lines)
        if fasta_string[-1] != "\n":
            fasta_string += "\n"

    with open(filepath, 'w') as f:
        f.write(fasta_string)
