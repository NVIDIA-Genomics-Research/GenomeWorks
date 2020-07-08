

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
