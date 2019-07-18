#!/usr/bin/env python3

#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

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
