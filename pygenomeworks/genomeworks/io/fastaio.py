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


"""Functions for FASTA file I/O."""
import gzip


def write_fasta(seqs, filepath, gzip_compressed=False):
    """Writes a fasta file for sequences.

    Args:
      seqs: list of 2-tuples containnig sequnces and their names, e.g [('seq1', 'ACGTC...'), ('seq2', 'TTGGC...'), ...]]
      filepath: path to file for writing out FASTA.
      gzip_compressed bool: If True then the read component of the sequence has been compressed with gzip

    Returns:
      None.
    """
    with open(filepath, 'w') as f:
        for s in seqs:
            fasta_string = ">{}\n".format(s[0])

            if gzip_compressed:
                read = str(gzip.decompress(s[1]), "utf-8")
            else:
                read = s[1]

            lines = [read[n * 80:(n + 1) * 80] for n in range((len(read) // 80) + 1)]

            fasta_string += "\n".join(lines)
            if fasta_string[-1] != "\n":
                fasta_string += "\n"

            f.write(fasta_string)
