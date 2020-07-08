/*
* Copyright 2019-2020 NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once

#include <vector>

#include <claraparabricks/genomeworks/utils/allocator.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace details
{

/// \brief Container for an array of elements of type `T` in host or device memory.
///        `T` must be trivially copyable.
/// \tparam T The object's type
/// \tparam Allocator The allocator's type
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

    /// \brief This constructor creates \p buffer with the given size.
    /// \param n The number of elements to create
    /// \param allocator The allocator to use by this buffer.
    /// \param streams The device memory provided by this buffer is guaranteed to live until all operations on these CUDA streams have completed. If no stream is specified default stream (0) is used.
    /// \tparam AllocatorIn Type of input allocator. If AllocatorIn::value_type is different than T AllocatorIn will be converted to Allocator<T> if possible, compilation will fail otherwise
    /// \tparam CudaStreamType All streams should be of type cudaStream_t
    template <typename AllocatorIn = Allocator, typename... CudaStreamType>
    explicit buffer(size_type n           = 0,
                    AllocatorIn allocator = AllocatorIn(),
                    CudaStreamType... streams)
        : data_(nullptr)
        , size_(n)
        , streams_({streams...})
        , allocator_(allocator)
    {
        // if no stream is passed use default stream
        if (streams_.empty())
        {
            streams_.push_back(0);
        }

        static_assert(std::is_trivially_copyable<value_type>::value, "buffer only supports trivially copyable types and classes, because destructors will not be called.");
        assert(size_ >= 0);
        if (size_ > 0)
        {
            data_ = allocator_.allocate(size_, streams_);
        }
    }

    /// \brief This constructor creates an empty \p buffer.
    /// \param allocator The allocator to use by this buffer.
    /// \param streams The device memory provided by this buffer is guaranteed to live until all operations on these CUDA streams have completed. If no stream is specified default stream (0) is used.
    /// \tparam AllocatorIn Type of input allocator. If AllocatorIn::value_type is different than T AllocatorIn will be converted to Allocator<T> if possible, compilation will fail otherwise
    /// \tparam CudaStreamType All streams should be of type cudaStream_t
    template <typename AllocatorIn, typename... CudaStreamType, std::enable_if_t<std::is_class<AllocatorIn>::value, int> = 0> // for calls like buffer(5, stream) the other constructor should be used -> only enable if AllocatorIn is a class.
    explicit buffer(AllocatorIn allocator,
                    CudaStreamType... streams)
        : buffer(0, allocator, streams...)
    {
        static_assert(std::is_trivially_copyable<value_type>::value, "buffer only supports trivially copyable types and classes, because destructors will not be called.");
    }

    /// \brief Move constructor moves from another \p buffer.
    /// Buffer to move from (rhs) is left in an empty state (size = capacity = 0), with the original stream and allocator.
    /// \param rhs The bufer to move.
    buffer(buffer&& rhs)
        : data_(std::exchange(rhs.data_, nullptr))
        , size_(std::exchange(rhs.size_, 0))
        , streams_(rhs.streams_)
        , allocator_(rhs.allocator_)
    {
    }

    /// \brief Move assign operator moves from another \p buffer.
    /// Buffer to move from (rhs) is left in an empty state (size = capacity = 0), with the original stream and allocator.
    /// \param rhs The buffer to move.
    /// \return refrence to this buffer.
    buffer& operator=(buffer&& rhs)
    {
        data_      = std::exchange(rhs.data_, nullptr);
        size_      = std::exchange(rhs.size_, 0);
        streams_   = rhs.streams_;
        allocator_ = rhs.allocator_;
        return *this;
    }

    /// \brief This destructor releases the buffer allocation, returning it to the allocator.
    ~buffer()
    {
        if (nullptr != data_)
        {
            allocator_.deallocate(data_, size_);
        }
    }

    /// \brief This method returns a pointer to this buffer's first element.
    /// \return A pointer to the first element of this buffer.
    value_type* data() { return data_; }

    /// \brief This method returns a const_pointer to this buffer's first element.
    /// \return A const pointer to the first element of this buffer.
    const value_type* data() const { return data_; }

    /// \brief This method returns the number of elements in this buffer.
    size_type size() const { return size_; }

    /// \brief This method returns an iterator pointing to the beginning of this buffer.
    iterator begin() { return data_; }

    /// \brief This method returns an const_iterator pointing to the beginning of this buffer.
    const_iterator begin() const { return data_; }

    /// \brief This method returns a iterator pointing to one element past the last of this buffer.
    /// \return begin() + size().
    iterator end() { return data_ + size_; }

    /// \brief This method returns a const_iterator pointing to one element past the last of this buffer.
    /// \return begin() + size().
    const_iterator end() const { return data_ + size_; }

    /// \brief This method returns the allocator used to allocate memory for this buffer.
    /// \return allocator.
    allocator_type get_allocator() const { return allocator_; }

    /// \brief This method releases the memory and resizes the buffer to 0.
    void free()
    {
        if (size_ > 0)
        {
            allocator_.deallocate(data_, size_);
            data_ = nullptr;
            size_ = 0;
        }
    }

    /// \brief This method resizes buffer and discards old data
    ///
    /// If old data is still needed it is recommended to crete a new buffer and manually copy the data from the old buffer
    ///
    /// \param new_size Number of elements
    void clear_and_resize(const size_type new_size)
    {
        assert(new_size >= 0);

        if (new_size == size_)
        {
            return;
        }

        free();
        data_ = new_size > 0 ? allocator_.allocate(new_size, streams_) : nullptr;
        assert(new_size == 0 || data_ != nullptr);
        size_ = new_size;
    }

    /// \brief This method swaps the contents of two buffers.
    /// \param a One buffer.
    /// \param b The other buffer.
    friend void swap(buffer& a, buffer& b) noexcept
    {
        using std::swap;
        swap(a.data_, b.data_);
        swap(a.size_, b.size_);
        swap(a.streams_, b.streams_);
        swap(a.allocator_, b.allocator_);
    }

private:
    value_type* data_;
    size_type size_;
    std::vector<cudaStream_t> streams_;
    Allocator allocator_;
};

} // namespace details

template <typename T>
#ifdef GW_ENABLE_CACHING_ALLOCATOR
using device_buffer = details::buffer<T, CachingDeviceAllocator<T, DevicePreallocatedAllocator>>;
#else
using device_buffer = details::buffer<T, CudaMallocAllocator<T>>;
#endif

} // namespace genomeworks

} // namespace claraparabricks
