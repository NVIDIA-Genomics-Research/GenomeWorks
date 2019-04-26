#!/usr/bin/env python3

from distutils.core import setup

setup(name='pygenomeworks',
      version='0.1',
      description='NVIDIA genomics python libraries an utiliites',
      author='Mike Vella',
      author_email='mvella@nvidia.com',
      packages=['genomeworks',
                'genomeworks.simulators',
                'genomeworks.io'],
      scripts=['bin/genome_simulator',
               'bin/assembly_evaluator'])
