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

#include <memory>
#include <claragenomics/utils/cudautils.hpp>

namespace claragenomics
{

/// @brief Container for a single object of type `T` in host or device memory.
///        `T` must be trivially copyable.
/// @tparam T The object's type
/// @tparam Allocator The allocator's type
template <typename T, typename Allocator>
class buffer
{
public:
    /// data type for the \b buffer allocation [supports only trivially copyable types].
    using value_type = T;

    /// signed integer type for defining size, count, and the global offsets.
    using size_type = std::ptrdiff_t;

    /// a pointer type to iterate/refer the \p buffer elements.
    using iterator = value_type*;

    /// a const pointer type to iterate/refer the \p buffer elements.
    using const_iterator = const value_type*;

    buffer(const buffer& other) = delete;

    buffer& operator=(const buffer& other) = delete;

    /// @brief This constructor creates \p buffer with the given size.
    /// @param n The number of elements to initially create.
    /// @param allocator The allocator to use by this buffer.
    /// @param stream The CUDA stream to be associated with this allocation. Default is stream 0.
    explicit buffer(size_type n                          = 0,
                    std::shared_ptr<Allocator> allocator = std::make_shared<Allocator>(),
                    cudaStream_t stream                  = 0)
        : _size(n)
        , _capacity(n)
        , _data(nullptr)
        , _stream(stream)
        , _allocator(allocator)
    {
        static_assert(std::is_trivially_copyable<value_type>::value, "buffer only supports trivially copyable types and classes, because destructors will not be called.");
        assert(_size >= 0);
        if (_capacity > 0)
        {
            _data = static_cast<value_type*>(
                _allocator->allocate(_capacity * sizeof(value_type), _stream));
            CGA_CU_CHECK_ERR(cudaStreamSynchronize(_stream));
        }
    }

    /// @brief This constructor creates an empty \p buffer.
    /// @param allocator The allocator to use by this buffer.
    /// @param stream The CUDA stream to be associated with this allocation. Default is stream 0.
    explicit buffer(std::shared_ptr<Allocator> allocator, cudaStream_t stream = 0)
        : buffer(0, allocator, stream)
    {
        static_assert(std::is_trivially_copyable<value_type>::value, "buffer only supports trivially copyable types and classes, because destructors will not be called.");
    }

    /// @brief Move constructor moves from another \p buffer.
    /// @param r The bufer to move.
    buffer(buffer&& r)
        : _size(std::exchange(r._size, 0))
        , _capacity(std::exchange(r._size, 0))
        , _data(std::exchange(r._data, nullptr))
        , _stream(std::exchange(r._stream, cudaStream_t(0)))
        , _allocator(std::exchange(r._allocator, std::shared_ptr<Allocator>(nullptr)))
    {
    }

    /// @brief Move assign operator moves from another \p buffer.
    /// @param r The buffer to move.
    buffer& operator=(buffer&& r)
    {
        _size      = std::exchange(r._size, 0);
        _capacity  = std::exchange(r._size, 0);
        _data      = std::exchange(r._data, nullptr);
        _stream    = std::exchange(r._stream, cudaStream_t(0));
        _allocator = std::exchange(r._allocator, std::shared_ptr<Allocator>(nullptr));
        return *this;
    }

    /// @brief This destructor releases the buffer allocation, returning it to the allocator.
    ~buffer()
    {
        if (nullptr != _data)
        {
            _allocator->deallocate(_data, _capacity * sizeof(value_type), _stream);
        }
    }

    /// @brief This method returns a pointer to this buffer's first element.
    /// @return A pointer to the first element of this buffer.
    value_type* data() { return _data; }

    /// @brief This method returns a const_pointer to this buffer's first element.
    /// @return A const pointer to the first element of this buffer.
    const value_type* data() const { return _data; }

    /// @brief This method returns the number of elements in this buffer.
    size_type size() const { return _size; }

    /// @brief This method resizes this buffer to 0.
    void clear() { _size = 0; }

    /// @brief This method returns an iterator pointing to the beginning of this buffer.
    iterator begin() { return _data; }

    /// @brief This method returns an const_iterator pointing to the beginning of this buffer.
    const_iterator begin() const { return _data; }

    /// @brief This method returns a iterator pointing to one element past the last of this buffer.
    /// @return begin() + size().
    iterator end() { return _data + _size; }

    /// @brief This method returns a const_iterator pointing to one element past the last of this buffer.
    /// @return begin() + size().
    const_iterator end() const { return _data + _size; }

    /// @brief This method reserves memory for the specified number of elements.
    ///        If new_capacity is less than or equal to _capacity, this call has no effect.
    ///        Otherwise, this method is a request for allocation of additional memory. If
    ///        the request is successful, then _capacity is equal to the new_capacity;
    ///        otherwise, _capacity is unchanged. In either case, size() is unchanged.
    /// @param new_capacity The number of elements to reserve.
    /// @param stream The CUDA stream to be associated with this method.
    void reserve(const size_type new_capacity, cudaStream_t stream)
    {
        assert(new_capacity >= 0);
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

    /// @brief This method reserves memory for the specified number of elements. It is implicitly
    ///        associated with the CUDA stream provided during the buffer creation.
    /// @param new_capacity The number of elements to reserve.
    void reserve(const size_type new_capacity)
    {
        reserve(new_capacity, _stream);
    }

    /// @brief This method resizes this buffer to the specified number of elements.
    /// @param new_size Number of elements this buffer should contain.
    /// @param stream The CUDA stream to be associated with this method.
    void resize(const size_type new_size, cudaStream_t stream)
    {
        reserve(new_size, stream);
        _size = new_size;
    }

    /// @brief This method resizes this buffer to the specified number of elements. It is
    ///        implicitly associated with the CUDA stream provided during the buffer creation.
    /// @param new_size Number of elements this buffer should contain.
    void resize(const size_type new_size)
    {
        resize(new_size, _stream);
    }

    /// @brief This method swaps the contents of this buffer with another buffer.
    /// @param v The buffer with which to swap.
    void swap(buffer& v)
    {
        std::swap(_data, v._data);
        std::swap(_size, v._size);
        std::swap(_capacity, v._capacity);
        std::swap(_stream, v._stream);
        std::swap(_allocator, v._allocator);
    }

    /// @brief This method shrinks the capacity of this buffer to exactly fit its elements.
    /// @param stream The CUDA stream to be associated with this method.
    void shrink_to_fit(cudaStream_t stream)
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

    /// @brief This method shrinks the capacity of this buffer to exactly fit its elements.
    ///        It is implicitly associated with the CUDA stream provided during the buffer creation.
    void shrink_to_fit()
    {
        shrink_to_fit(_stream);
    }

    /// @brief This method releases the memory allocated by this buffer, returning it to the allocator.
    /// @param stream The CUDA stream to be associated with this method.
    void free(cudaStream_t stream)
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

    /// @brief This method releases the memory allocated by this buffer, returning it to the allocator.
    ///        It is implicitly associated with the CUDA stream provided during the buffer creation.
    void free()
    {
        free(_stream);
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

/// @brief This method swaps the contents of buffer a with b.
template <typename T, typename Allocator>
void swap(buffer<T, Allocator>& a,
          buffer<T, Allocator>& b)
{
    a.swap(b);
}

} // namespace claragenomics
