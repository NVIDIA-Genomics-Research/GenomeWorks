#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import pytest

from claragenomics.bindings.cudaaligner import CudaAlignerBatch
import claragenomics.bindings.cuda as cuda


@pytest.mark.gpu
def test_cudapoa_simple_batch():
    stream = cuda.CudaStream()
    batch = CudaAlignerBatch(100, 100, 2, "global", stream, 0)
    batch.add_alignment(["AAAAAAA", "TTTTTTT"])
    batch.align_all()
    batch.sync_alignments()
    alignments = batch.get_alignments()

    assert(a.cigar == '7M')
