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

/// @brief Container for an array of elements of type `T` in host or device memory.
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

    /// allocator type.
    using allocator_type = Allocator;

    buffer(const buffer& other) = delete;

    buffer& operator=(const buffer& other) = delete;

    /// @brief This constructor creates \p buffer with the given size.
    /// @param n The number of elements to initially create.
    /// @param allocator The allocator to use by this buffer.
    /// @param stream The CUDA stream to be associated with this allocation. Default is stream 0.
    /// @tparam AllocatorIn Type of input allocator. If AllocatorIn::value_type is different than T AllocatorIn will be converted to Allocator<T> if possible, compilation will fail otherwise
    template <typename AllocatorIn = Allocator>
    explicit buffer(size_type n           = 0,
                    AllocatorIn allocator = AllocatorIn(),
                    cudaStream_t stream   = 0)
        : _data(nullptr)
        , _size(n)
        , _capacity(n)
        , _stream(stream)
        , _allocator(allocator)
    {
        static_assert(std::is_trivially_copyable<value_type>::value, "buffer only supports trivially copyable types and classes, because destructors will not be called.");
        assert(_size >= 0);
        if (_capacity > 0)
        {
            _data = _allocator.allocate(_capacity, _stream);
            CGA_CU_CHECK_ERR(cudaStreamSynchronize(_stream));
        }
    }

    /// @brief This constructor creates an empty \p buffer.
    /// @param allocator The allocator to use by this buffer.
    /// @param stream The CUDA stream to be associated with this allocation. Default is stream 0.
    /// @tparam AllocatorIn Type of input allocator. If AllocatorIn::value_type is different than T AllocatorIn will be converted to Allocator<T> if possible, compilation will fail otherwise
    template <typename AllocatorIn, std::enable_if_t<std::is_class<AllocatorIn>::value, int> = 0> // for calls like buffer(5, stream) the other constructor should be used -> only enable if AllocatorIn is a class.
    explicit buffer(AllocatorIn allocator, cudaStream_t stream = 0)
        : buffer(0, allocator, stream)
    {
        static_assert(std::is_trivially_copyable<value_type>::value, "buffer only supports trivially copyable types and classes, because destructors will not be called.");
    }

    /// @brief Move constructor moves from another \p buffer.
    /// Buffer to move from (rhs) is left in an empty state (size = capacity = 0), with the original stream and allocator.
    /// @param rhs The bufer to move.
    buffer(buffer&& rhs)
        : _data(std::exchange(rhs._data, nullptr))
        , _size(std::exchange(rhs._size, 0))
        , _capacity(std::exchange(rhs._capacity, 0))
        , _stream(rhs._stream)
        , _allocator(rhs._allocator)
    {
    }

    /// @brief Move assign operator moves from another \p buffer.
    /// Buffer to move from (rhs) is left in an empty state (size = capacity = 0), with the original stream and allocator.
    /// @param rhs The buffer to move.
    /// @return refrence to this buffer.
    buffer& operator=(buffer&& rhs)
    {
        _data      = std::exchange(rhs._data, nullptr);
        _size      = std::exchange(rhs._size, 0);
        _capacity  = std::exchange(rhs._capacity, 0);
        _stream    = rhs._stream;
        _allocator = rhs._allocator;
        return *this;
    }

    /// @brief This destructor releases the buffer allocation, returning it to the allocator.
    ~buffer()
    {
        if (nullptr != _data)
        {
            _allocator.deallocate(_data, _capacity, _stream);
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

    /// @brief This method returns the largest number of elements that can be stored in this buffer before a reallocation is needed.
    size_type capacity() const { return _capacity; }

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

    /// @brief This method returns the allocator used to allocate memory for this buffer.
    /// @return allocator.
    allocator_type get_allocator() const { return _allocator; }

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
            value_type* new_data = _allocator.allocate(new_capacity, _stream);
            if (_size > 0)
            {
                CGA_CU_CHECK_ERR(cudaMemcpyAsync(new_data, _data, _size * sizeof(value_type),
                                                 cudaMemcpyDefault, _stream));
            }
            if (nullptr != _data)
            {
                _allocator.deallocate(_data, _capacity, _stream);
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

    /// @brief This method swaps the contents of two buffers.
    /// @param a One buffer.
    /// @param b The other buffer.
    friend void swap(buffer& a, buffer& b) noexcept
    {
        using std::swap;
        swap(a._data, b._data);
        swap(a._size, b._size);
        swap(a._capacity, b._capacity);
        swap(a._stream, b._stream);
        swap(a._allocator, b._allocator);
    }

    /// @brief This method shrinks the capacity of this buffer to exactly fit its elements.
    /// @param stream The CUDA stream to be associated with this method.
    void shrink_to_fit(cudaStream_t stream)
    {
        set_stream(stream);
        if (_capacity > _size)
        {
            value_type* new_data = _allocator.allocate(_size, _stream);
            if (_size > 0)
            {
                CGA_CU_CHECK_ERR(cudaMemcpyAsync(new_data, _data, _size * sizeof(value_type),
                                                 cudaMemcpyDefault, _stream));
            }
            if (nullptr != _data)
            {
                _allocator.deallocate(_data, _capacity, _stream);
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
            _allocator.deallocate(_data, _capacity, _stream);
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
    Allocator _allocator;

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
