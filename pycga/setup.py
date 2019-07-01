#!/usr/bin/env python3
import os.path

from setuptools import setup, find_packages

setup(name='pyclaragenomics',
      version='0.1',
      description='NVIDIA genomics python libraries an utiliites',
      author='Mike Vella',
      author_email='mvella@nvidia.com',
      packages=find_packages(),
      scripts=[os.path.join('bin', 'genome_simulator'),
               os.path.join('bin', 'assembly_evaluator')])
