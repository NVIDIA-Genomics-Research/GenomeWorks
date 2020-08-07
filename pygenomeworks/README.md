# genomeworks

Python libraries and utilities for manipulating genomics data

## Features

`genomeworks` python API exposes python bindings to the following CUDA-accelerated GenomeWorks
libraries -

1. cudapoa
2. cudaaligner

The `genomeworks` package also provides some utility tools for the development of mapping and alignment
algorithms. These tools are located under the `bin` folder of `pygenomeworks`.

1. `evaluate_paf` - A tool to compare two PAF files and generate precision, recall and F-1 metrics for their overlaps.
2. `genome_simulator` - A tool to generate a synthetic reference genome and extract reads from it with customizable error content.

## Installation

### Install from PyPI

A stable release of genomeworks can be installed from PyPI. Currently only CUDA 10.0 and CUDA 10.1 based packages are supported.
Both of those packages are available for CPython 3.5 and 3.6.

NOTE - v0.5.0 onwards our package name is changing. Please refer to older release tags for
details on python bindings for v0.4.0 and before.

#### v0.5.0+

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

### Install from source
```
pip install -r requirements.txt
python setup_pygenomeworks.py --build_output_folder BUILD_FOLDER
```

*Note* if you are developing genomeworks you should do a develop build instead, changes you make to the source code will then be picked up on immediately:

```
pip install -r requirements.txt
python setup_pygenomeworks.py --build_output_folder BUILD_FOLDER --develop
```

### Create a Wheel package

Use the following command in order to package genomeworks into a wheel. (without installing)

```
pip install -r requirements.txt
python setup_pygenomeworks.py --create_wheel_only
```
### Testing installation

To test the installation execute:

```
cd test/
python -m pytest
```

## Development Support

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
