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

namespace claraparabricks
{

namespace genomeworks
{

/// /brief Container for an array of elements of type `T` in host or device memory.
///        `T` must be trivially copyable.
/// /tparam T The object's type
/// /tparam Allocator The allocator's type
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

    /// /brief This constructor creates \p buffer with the given size.
    /// /param n The number of elements to create
    /// /param allocator The allocator to use by this buffer.
    /// /param stream The CUDA stream to be associated with this allocation. Default is stream 0.
    /// /tparam AllocatorIn Type of input allocator. If AllocatorIn::value_type is different than T AllocatorIn will be converted to Allocator<T> if possible, compilation will fail otherwise
    template <typename AllocatorIn = Allocator>
    explicit buffer(size_type n           = 0,
                    AllocatorIn allocator = AllocatorIn(),
                    cudaStream_t stream   = 0)
        : _data(nullptr)
        , _size(n)
        , _stream(stream)
        , _allocator(allocator)
    {
        static_assert(std::is_trivially_copyable<value_type>::value, "buffer only supports trivially copyable types and classes, because destructors will not be called.");
        assert(_size >= 0);
        if (_size > 0)
        {
            _data = _allocator.allocate(_size, _stream);
        }
    }

    /// /brief This constructor creates an empty \p buffer.
    /// /param allocator The allocator to use by this buffer.
    /// /param stream The CUDA stream to be associated with this allocation. Default is stream 0.
    /// /tparam AllocatorIn Type of input allocator. If AllocatorIn::value_type is different than T AllocatorIn will be converted to Allocator<T> if possible, compilation will fail otherwise
    template <typename AllocatorIn, std::enable_if_t<std::is_class<AllocatorIn>::value, int> = 0> // for calls like buffer(5, stream) the other constructor should be used -> only enable if AllocatorIn is a class.
    explicit buffer(AllocatorIn allocator, cudaStream_t stream = 0)
        : buffer(0, allocator, stream)
    {
        static_assert(std::is_trivially_copyable<value_type>::value, "buffer only supports trivially copyable types and classes, because destructors will not be called.");
    }

    /// /brief Move constructor moves from another \p buffer.
    /// Buffer to move from (rhs) is left in an empty state (size = capacity = 0), with the original stream and allocator.
    /// /param rhs The bufer to move.
    buffer(buffer&& rhs)
        : _data(std::exchange(rhs._data, nullptr))
        , _size(std::exchange(rhs._size, 0))
        , _stream(rhs._stream)
        , _allocator(rhs._allocator)
    {
    }

    /// /brief Move assign operator moves from another \p buffer.
    /// Buffer to move from (rhs) is left in an empty state (size = capacity = 0), with the original stream and allocator.
    /// /param rhs The buffer to move.
    /// /return refrence to this buffer.
    buffer& operator=(buffer&& rhs)
    {
        _data      = std::exchange(rhs._data, nullptr);
        _size      = std::exchange(rhs._size, 0);
        _stream    = rhs._stream;
        _allocator = rhs._allocator;
        return *this;
    }

    /// /brief This destructor releases the buffer allocation, returning it to the allocator.
    ~buffer()
    {
        if (nullptr != _data)
        {
            _allocator.deallocate(_data, _size);
        }
    }

    /// /brief This method returns a pointer to this buffer's first element.
    /// /return A pointer to the first element of this buffer.
    value_type* data() { return _data; }

    /// /brief This method returns a const_pointer to this buffer's first element.
    /// /return A const pointer to the first element of this buffer.
    const value_type* data() const { return _data; }

    /// /brief This method returns the number of elements in this buffer.
    size_type size() const { return _size; }

    /// /brief This method returns an iterator pointing to the beginning of this buffer.
    iterator begin() { return _data; }

    /// /brief This method returns an const_iterator pointing to the beginning of this buffer.
    const_iterator begin() const { return _data; }

    /// /brief This method returns a iterator pointing to one element past the last of this buffer.
    /// /return begin() + size().
    iterator end() { return _data + _size; }

    /// /brief This method returns a const_iterator pointing to one element past the last of this buffer.
    /// /return begin() + size().
    const_iterator end() const { return _data + _size; }

    /// /brief This method returns the associated stream.
    /// /return associated stream.
    cudaStream_t get_stream() const { return _stream; }

    /// /brief This method returns the allocator used to allocate memory for this buffer.
    /// /return allocator.
    allocator_type get_allocator() const { return _allocator; }

    /// /brief This method releases the memory and resizes the buffer to 0.
    void clear_and_free()
    {
        if (_size > 0)
        {
            _allocator.deallocate(_data, _size);
            _data = nullptr;
            _size = 0;
        }
    }

    /// /brief This method resizes buffer and discards old data
    ///
    /// If old data is still needed it is recommended to crete a new buffer and manually copy the data from the old buffer
    ///
    /// /param new_size Number of elements
    void clear_and_resize(const size_type new_size)
    {
        assert(new_size >= 0);

        if (new_size == _size)
        {
            return;
        }

        clear_and_free();
        _size = new_size;
        _data = _size > 0 ? _allocator.allocate(_size, _stream) : nullptr;
    }

    /// /brief This method swaps the contents of two buffers.
    /// /param a One buffer.
    /// /param b The other buffer.
    friend void swap(buffer& a, buffer& b) noexcept
    {
        using std::swap;
        swap(a._data, b._data);
        swap(a._size, b._size);
        swap(a._stream, b._stream);
        swap(a._allocator, b._allocator);
    }

private:
    value_type* _data;
    size_type _size;
    cudaStream_t _stream;
    Allocator _allocator;
};

} // namespace genomeworks

} // namespace claraparabricks
