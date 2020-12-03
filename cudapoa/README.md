# CUDAPOA

The `cudapoa` package provides a GPU-accelerated implementation of the [Partial Order Alignment](https://simpsonlab.github.io/2015/05/01/understanding-poa/)
algorithm. It is heavily influenced by [SPOA](https://github.com/rvaser/spoa) and in many cases can be considered a GPU-accelerated replacement. Features include:

## Tool - cudapoa

A command line tool for generating consensus and MSA from a list of `fasta`/`fastq` files. The tool
is built on top of `libcudapoa.so` and showcases optimization strategies for writing high performance
applications with `libcudapoa.so`.

## Library - libcudapoa.so

* Generation of consensus sequences
* Generation of multi-sequence alignments (MSAs)
* Custom adaptive band implementation of POA
* Support for long and short read sequences

## Sample - sample_cudapoa

A prototypical binary to showcase the use of `libcudapoa.so` APIs.
