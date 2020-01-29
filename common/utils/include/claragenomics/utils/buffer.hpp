/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once
#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/utils/allocator.hpp>

namespace claragenomics
{

template <typename T, typename Allocator>
class buffer
{
public:
    using value_type     = T;
    using size_type      = std::size_t;
    using iterator       = value_type*;
    using const_iterator = const value_type*;

    buffer(const buffer& other) = delete;

    buffer& operator=(const buffer& other) = delete;

    buffer(size_type n                          = 0,
           std::shared_ptr<Allocator> allocator = std::make_shared<Allocator>(),
           cudaStream_t stream                  = 0)
        : _size(n)
        , _capacity(n)
        , _data(nullptr)
        , _stream(stream)
        , _allocator(allocator)
    {
        if (_capacity > 0)
        {
            _data = static_cast<value_type*>(
                _allocator->allocate(_capacity * sizeof(value_type), _stream));
            CGA_CU_CHECK_ERR(cudaStreamSynchronize(_stream));
        }
    }

    // Should this be removed and force it to use allocator for a common interface?
    buffer(std::shared_ptr<Allocator> allocator, cudaStream_t stream = 0)
        : buffer(0, allocator, stream)
    {
    }

    // move constructor
    buffer(buffer&& r)
        : _size(std::exchange(r._size, 0))
        , _capacity(std::exchange(r._size, 0))
        , _data(std::exchange(r._data, nullptr))
        , _stream(std::exchange(r._stream, cudaStream_t(0)))
        , _allocator(std::exchange(r._allocator, std::shared_ptr<Allocator>(nullptr)))
    {
    }

    ~buffer()
    {
        if (nullptr != _data)
        {
            _allocator->deallocate(_data, _capacity * sizeof(value_type), _stream);
        }
    }

    value_type* data() { return _data; }

    const value_type* data() const { return _data; }

    size_type size() const { return _size; }

    void clear() { _size = 0; }

    iterator begin() { return _data; }

    const_iterator begin() const { return _data; }

    iterator end() { return _data + _size; }

    const_iterator end() const { return _data + _size; }

    void reserve(const size_type new_capacity, cudaStream_t stream = 0)
    {
        set_stream(stream);
        if (new_capacity > _capacity)
        {
            value_type* new_data = static_cast<value_type*>(
                _allocator->allocate(new_capacity * sizeof(value_type), _stream));
            if (_size > 0)
            {
                CGA_CU_CHECK_ERR(cudaMemcpyAsync(new_data, _data, _size * sizeof(value_type),
                                                 cudaMemcpyDefault, _stream));
            }
            if (nullptr != _data)
            {
                _allocator->deallocate(_data, _capacity * sizeof(value_type), _stream);
            }
            _data     = new_data;
            _capacity = new_capacity;
        }
    }

    void resize(const size_type new_size, cudaStream_t stream = 0)
    {
        reserve(new_size, stream);
        _size = new_size;
    }

    void swap(buffer& v)
    {
        std::swap(_data, v._data);
        std::swap(_size, v._size);
        std::swap(_capacity, v._capacity);
        std::swap(_stream, v._stream);
        std::swap(_allocator, v._allocator);
    }

    void shrink_to_fit(cudaStream_t stream = 0)
    {
        set_stream(stream);
        if (_capacity > _size)
        {
            value_type* new_data = static_cast<value_type*>(
                _allocator->allocate(_size * sizeof(value_type), _stream));
            if (_size > 0)
            {
                CGA_CU_CHECK_ERR(cudaMemcpyAsync(new_data, _data, _size * sizeof(value_type),
                                                 cudaMemcpyDefault, _stream));
            }
            if (nullptr != _data)
            {
                _allocator->deallocate(_data, _capacity * sizeof(value_type), _stream);
            }
            _data     = new_data;
            _capacity = _size;
        }
    }

    void free(cudaStream_t stream = 0)
    {
        set_stream(stream);
        if (nullptr != _data)
        {
            _allocator->deallocate(_data, _capacity * sizeof(value_type), _stream);
        }
        _data     = nullptr;
        _capacity = 0;
        _size     = 0;
    }

private:
    value_type* _data = nullptr;
    size_type _size;
    size_type _capacity;
    cudaStream_t _stream;
    std::shared_ptr<Allocator> _allocator;

    void set_stream(cudaStream_t stream)
    {
        if (_stream != stream)
        {
            cudaEvent_t event;
            CGA_CU_CHECK_ERR(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            CGA_CU_CHECK_ERR(cudaEventRecord(event, _stream));
            CGA_CU_CHECK_ERR(cudaStreamWaitEvent(stream, event, 0));
            _stream = stream;
            CGA_CU_CHECK_ERR(cudaEventDestroy(event));
        }
    }
};

} // namespace claragenomics
