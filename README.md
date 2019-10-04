# Clara Genomics Analysis

## Overview

Clara Genomics Analysis is a GPU-accelerated library for biological sequence analysis. This section provides a brief overview of the different components of ClaraGenomicsAnalysis.
For more detailed API documentation please refer to the [documentation](#enable-doc-generation).

### cudapoa

The `cudapoa` package provides a GPU-accelerated implementation of the [Partial Order Alignment](https://simpsonlab.github.io/2015/05/01/understanding-poa/)
algorithm. It is heavily influenced by [SPOA](https://github.com/rvaser/spoa) and in many cases can be considered a GPU-accelerated replacement. Features include:

1. Generation of consensus sequences
2. Generation of multi-sequence alignments (MSAs)

### cudaaligner

The `cudaaligner` package provides GPU-accelerated global alignment.

## Clone Clara Genomics Analysis
```bash
git clone --recursive git@github.com:clara-genomics/ClaraGenomicsAnalysis.git
```

## System Requirements
Minimum requirements -

1. Ubuntu 16.04 or Ubuntu 18.04
2. CUDA 9.0+
3. gcc/g++ 5.4.0+
4. Python 3.6.7+
5. htslib 1.9+ (https://github.com/samtools/htslib)

## Clara Genomics Analysis Setup

### Build
To build Clara Genomics Analysis -

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=install
make -j install
```

### Install
To install the SDK -

```bash
make install
```

### Package generation
Package generation puts the libraries, headers and binaries built by the `make` command above
into a `.deb`/`.rpm` for portability and easy installation. The package generation itself doesn't
guarantee any cross-platform compatibility.

It is recommended that a separate build and packaging be performed for each distribution and
CUDA version that needs to be supported.

The type of package (deb vs rpm) is determined automatically based on the platform the code
is being run on. To generate a package for the SDK -

```bash
make package
```

## Enable Unit Tests
To enable unit tests, add `-Dcga_enable_tests=ON` to the `cmake` command in the build step.

This builds GTest based unit tests for all applicable modules, and installs them under
`${CMAKE_INSTALL_PREFIX}/tests`. These tests are standalone binaries and can be executed
directly.
e.g.

```
cd $INSTALL_DIR
./tests/cudapoatests
```

## Enable Benchmarks
To enable benchmarks, add `-Dcga_enable_benchmarks=ON` to the `cmake` command in the build step.

This builds Google Benchmark based microbenchmarks for applicable modules. The built benchmarks
are installed under `${CMAKE_INSTALL_PREFIX}/benchmarks/<module>` and can be run directly.

e.g.
```
#INSTALL_DIR/benchmarks/cudapoa/multibatch
```

A description of each of the benchmarks is present in a README under the module's benchmark folder.

## Enable Doc Generation
To enable document generation for Clara Genomics Analysis, please install `Doxygen` on your system. Once
`Doxygen` has been installed, run the following to build documents.

```bash
make docs
```

Docs are also generated as part of the default `all` target when `Doxygen` is available on the system.

## Code Formatting

### C++ / CUDA
Clara Genomics Analysis makes use of `clang-format` to format it's source and header files. To make use of
auto-formatting, `clang-format` would have to be installed from the LLVM package (for latest builds,
best to refer to http://releases.llvm.org/download.html).

Once `clang-format` has been installed, make sure the binary is in your path.

To add a folder to the auto-formatting list, use the macro `cga_enable_auto_formatting(FOLDER)`. This
will add all cpp source/header files to the formatting list.

To auto-format, run the following in your build directory.

```bash
make format
```

To check if files are correct formatted, run the following in your build directory.

```bash
make check-format
```

### Python
Clara Genomics Analysis follows the PEP-8 style guidelines for all its Python code. The automated
CI system for Clara Genomics Analysis run `flake8` to check the style.

To run style check manually, simply run the following from the top level folder.
```
flake8 pyclaragenomics/
```
