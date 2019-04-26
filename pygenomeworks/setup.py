#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(name='pygenomeworks',
      version='0.1',
      description='NVIDIA genomics python libraries an utiliites',
      author='Mike Vella',
      author_email='mvella@nvidia.com',
      packages=find_packages(),
      scripts=['bin/genome_simulator',
               'bin/assembly_evaluator'])
