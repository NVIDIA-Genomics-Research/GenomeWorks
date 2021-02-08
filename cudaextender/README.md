# cudaextender

This package implements CUDA-accelerated seed-extension algorithms that use seed positions in 
encoded input strands to extend and compute the alignment between the strands. 
Currently this module implements the ungapped X-drop algorithm, adapted from 
[SegAlign's](https://github.com/gsneha26/SegAlign) Ungapped Extender  authored by 
Sneha Goenka (gsneha@stanford.edu) and Yatish Turakhia (yturakhi@ucsc.edu).
Citing SegAlign: S. Goenka, Y. Turakhia, B. Paten and M. Horowitz,  "SegAlign: A Scalable 
GPU-Based Whole Genome Aligner," in 2020 SC20: International Conference for High Performance 
Computing, Networking, Storage and Analysis (SC), Atlanta, GA, US, 2020 pp. 540-552.
url: https://doi.ieeecomputersociety.org/10.1109/SC41405.2020.00043

## Library
Built as `libcudaextender.[so|a]`

* Ungapped X-Drop extension

`cudaextender` provides host and device pointer APIs to enable ease of integration with other
producer/consumer modules. The user is expected to handle all memory transactions and device
sychronizations for the device pointer API. The host pointer API abstracts those operations away.
Both APIs are documented here: [extender.hpp](include/claraparabricks/genomeworks/cudaextender/extender.hpp)

### Encoded Input
`cudaextender` expects the input strands to be encoded as integer sequences. 
This encoding scheme is documented here: [utils.hpp](include/claraparabricks/genomeworks/cudaextender/utils.hpp)
file. The provided `encode_sequence()` helper function will encode the input strands on CPU with
the expected scheme. 

## Sample
[sample_cudaextender](samples/sample_cudaextender.cpp) - Protoype to show the usage of host and device pointer APIs on FASTA sequences.
