# cudaaligner

The `cudaaligner` package provides GPU-accelerated global alignment. Features include:

## Library
Built as `libcudaaligner.[so|a]`.

* Short and long read support
* Banded implementation with configurable band width for flexible performance and accuracy trade-off

APIs documented in [include](include/claraparabricks/genomeworks/cudaaligner) folder.

## Sample
[sample_cudaaligner](samples/sample_cudaaligner.cpp) - A prototypical binary to showcase the use of `libcudaaligner.so` APIs.

