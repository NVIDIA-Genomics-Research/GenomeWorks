# genomeworks

Python libraries and utilities for manipulating genomics data

## Installation

### Install from PyPI

A stable release of genomeworks can be installed from PyPI. Currently only CUDA 10.0 and CUDA 10.1 based packages are supported.
Both of those packages are available for CPython 3.5 and 3.6.

### Stable release (pre v0.5.0)

NOTE - v0.5.0 onwards our package name is changing. Please refer to `master` branch documentation
for details on python bindings for v0.4.0 and before.

### v0.5.0 onwards

```
pip install genomeworks-cuda10-0
```

or 

```
pip install genomeworks-cuda10-1
```

Details of the packages are available here -
- https://pypi.org/project/genomeworks-cuda-10-0
- https://pypi.org/project/genomeworks-cuda-10-1

### Testing installation

To test the installation execute:

```
cd test/
python -m pytest
```

### Install from source
```
pip install -r requirements.txt
python setup_genomeworks.py --build_output_folder BUILD_FOLDER
```

*Note* if you are developing genomeworks you should do a develop build instead, changes you make to the source code will then be picked up on immediately:

```
pip install -r requirements.txt
python setup_genomeworks.py --build_output_folder BUILD_FOLDER --develop
```

### Create a Wheel package

Use the following command in order to package genomeworks into a wheel. (without installing)

```
pip install -r requirements.txt
python setup_pygenomeworks.py --create_wheel_only
```

### Enable Doc Generation
`genomeworks` documentation generation is managed through `Sphinx`.

NOTE: `genomeworks` needs to be built completely in order for the
documentation to pick up docstrings for bindings.

```
pip install -r python-style-requirements.txt
./generate_docs
```

### Code Formatting

GenomeWorks follows the PEP-8 style guidelines for all its Python code. The automated
CI system for GenomeWorks run `flake8` to check the style.

To run style check manually, simply run the following from the top level folder.

```
pip install -r python-style-requirements.txt
./style_check
```

## Generating a simulated genome

A genome can be simulated without any parameters, to generate a 1Mbp reference with 20x coverage and median read length of 10kbp:

```
genome_simulator --snv_error_rate 0.01 --insertion_error_rate 0.005 --deletion_error_rate 0.005  --reference_length 1000000 --num_reads 2000 --median_read_length=10000
```

this will generate a 1Mbp reference genome with 20x coverage (default errors) reads in two files:

1. `ref.fasta` - the reference genome
2. `reads.fasta` - the corresponding reads
