#
# Copyright 2019-2020 NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import pytest


from genomeworks import cuda


@pytest.mark.gpu
def test_cuda_get_device():
    device_count = cuda.cuda_get_device_count()
    assert(device_count > 0)


@pytest.mark.gpu
def test_cuda_device_selection():
    device_count = cuda.cuda_get_device_count()
    if (device_count > 0):
        for device in range(device_count):
            cuda.cuda_set_device(device)
            assert(cuda.cuda_get_device() == device)


@pytest.mark.gpu
def test_cuda_memory_info():
    device_count = cuda.cuda_get_device_count()
    if (device_count > 0):
        for device in range(device_count):
            (free, total) = cuda.cuda_get_mem_info(device)
            assert(free > 0)
            assert(total > 0)
            assert(free < total)
