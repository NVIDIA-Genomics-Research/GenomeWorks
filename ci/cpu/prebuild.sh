#!/usr/bin/env bash

# Note we still _BUILD_ for GPU, we just don't (can't) test on it
export BUILD_FOR_GPU=1
export TEST_ON_GPU=0
