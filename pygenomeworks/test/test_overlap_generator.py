

import pytest

from genomeworks.simulators import readsim

test_reads = [
    ((("read_0",
      "AACGTCA",
       100,
       900),
     ("read_1",
      "AACGTCA",
      100,
      900)), 1),
    ((("read_0",
      "AACGTCA",
       100,
       900),
     ("read_1",
      "AACGTCA",
      1000,
      9000)), 0),
    ((("read_1",
      "AACGTCA",
       100,
       900),
     ("read_0",
      "AACGTCA",
      100,
      900)), 1),
    ((("read_1",
      "AACGTCA",
       100,
       900),
     ("read_0",
      "AACGTCA",
      100,
      900)), 1),
    ((("read_1",
      "AACGTCA",
       100,
       900),
      ("read_2",
      "AACGTCA",
       100,
       900),
     ("read_3",
      "AACGTCA",
      100,
      900)), 3),
]


@pytest.mark.cpu
@pytest.mark.parametrize("reads, expected_overlaps", test_reads)
def test_generates_overlaps(reads, expected_overlaps):
    """ Test that the number of overlaps detected is correct"""
    overlaps = readsim.generate_overlaps(reads, gzip_compressed=False)
    assert(len(overlaps) == expected_overlaps)
