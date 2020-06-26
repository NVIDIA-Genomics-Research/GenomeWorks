#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# This file declares public cython utility objects for CUDA.

cdef class CudaStream:
    # Using size_t to store stream since underlying
    # representation of cudaStream_t is a (void *)
    # and python doesn't know how to deal with converting
    # (void *) to python objects.
    cdef size_t stream
