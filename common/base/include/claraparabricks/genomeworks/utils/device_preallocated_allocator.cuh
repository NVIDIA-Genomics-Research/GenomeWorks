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

#include <algorithm>
#include <list>
#include <memory>
#include <mutex>
#include <vector>

#include <claraparabricks/genomeworks/utils/cudautils.hpp>

namespace claraparabricks
{

namespace genomeworks
{

/// \brief Allocator that preallocates one big buffer of device memory and assigns sections of it to allocation requests
/// Allocator allocates one big buffer of device memory during constructor and keeps it until destruction.
/// For every allocation request it linearly scans the preallocated memory and assigns first buffer of it that is big enough.
///
/// For example imagine 100000 bytes buffer has been preallocated and that sections between (10000, 19999), (30000, 39999)
/// and (60000, 79999) bytes have already been assigned:
///
/// 0 --- 10000 --- 20000 --- 30000 --- 40000 --- 50000 --- 60000 --- 70000 --- 80000 --- 90000 --- 100000
/// |  FREE |   USED  |  FREE   |  USED   |       FREE        |       USED        |        FREE        |
///
/// If an allocation request for 15000 bytes arrives allocator will starting looking for free block of at least 15000 bytes
/// looking from left to right and the first such block will be the one between (40000, 59999) bytes.
/// After the allocation the memory will look like this:
///
/// 0 --- 10000 --- 20000 --- 30000 --- 40000 ------ 55000 - 60000 --- 70000 --- 80000 --- 90000 --- 100000
/// |  FREE |   USED  |  FREE   |  USED   |    USED    | FREE  |       USED        |        FREE        |
///
/// Uppon returning a memory allocation its gets merged with any neighboring free arey, so if section (10000, 19999) is
/// returned the memory will look like this:
///
/// 0 ---------------------- 30000 --- 40000 ------ 55000 - 60000 --- 70000 --- 80000 --- 90000 --- 100000
/// |           FREE           |  USED   |    USED    | FREE  |       USED        |        FREE        |
///
/// Allocator aligns memory by 256 bytes
///
/// Note that this allocator heavily relies on iterating over lists of free and used memory and should thus
/// not be used for cases when there are millions of allocation at a time
class DevicePreallocatedAllocator
{
public:
    /// \brief Constructor
    /// Allocates the buffer
    /// \param buffer_size
    DevicePreallocatedAllocator(size_t buffer_size)
        : buffer_size_(buffer_size)
        , buffer_ptr_(create_buffer(buffer_size))
    {
        assert(buffer_size > 0);
        MemoryBlock whole_memory_block;
        whole_memory_block.begin = 0;
        whole_memory_block.size  = buffer_size;
        free_blocks_.push_back(whole_memory_block);
    }

    DevicePreallocatedAllocator()                                   = delete;
    DevicePreallocatedAllocator(const DevicePreallocatedAllocator&) = delete;
    DevicePreallocatedAllocator operator=(const DevicePreallocatedAllocator&) = delete;

    DevicePreallocatedAllocator(DevicePreallocatedAllocator&&) = delete;
    DevicePreallocatedAllocator operator=(DevicePreallocatedAllocator&&) = delete;

    ~DevicePreallocatedAllocator() = default;
    // ^^^^ buffer_ptr_'s destructor deallocates device memory

    /// \brief allocates memory (assigns part of the buffer)
    /// Memory allocation is aligned by 256 bytes
    /// \param ptr on return pointer to allocated memory, nullptr if allocation was not successful
    /// \param bytes_needed
    /// \param associated_streams on deallocation this memory block is guaranteed to live at least until all previously scheduled work in these streams has finished
    /// \return cudaSuccess if allocation was successful, cudaErrorMemoryAllocation otherwise
    cudaError_t DeviceAllocate(void** ptr,
                               size_t bytes_needed,
                               const std::vector<cudaStream_t>& associated_streams)
    {
        assert(!associated_streams.empty());

        std::lock_guard<std::mutex> mutex_lock_guard(memory_operation_mutex_);
        return allocate_memory_block(ptr,
                                     bytes_needed,
                                     associated_streams);
    }

    /// \brief deallocates memory (returns its part of buffer to the list of free parts)
    /// This function blocks until all work on associated_stream is done
    /// \param ptr
    /// \return error status
    cudaError_t DeviceFree(void* ptr)
    {
        cudaError_t status = cudaSuccess;

        if (nullptr != ptr)
        {
            std::lock_guard<std::mutex> mutex_lock_guard(memory_operation_mutex_);
            status = free_memory_block(ptr);
        }

        return status;
    }

    /// @brief returns the size of the largest free memory block
    int64_t get_size_of_largest_free_memory_block() const
    {
        size_t size = 0;
        for (auto& block : free_blocks_)
        {
            size = std::max(size, block.size);
        }
        return static_cast<int64_t>(size);
    }

private:
    /// \brief represents one part of the buffer, free or available
    struct MemoryBlock
    {
        // byte in buffer at which this block starts
        size_t begin;
        // number of bytes in this block
        size_t size;
        // this block will get freed only once all previously scheduled work on these streams has finished
        std::vector<cudaStream_t> associated_streams;
    };

    /// \brief allocates the underlying buffer
    /// \param buffer_size
    /// \return allocated shared_ptr
    static std::unique_ptr<char, void (*)(char*)> create_buffer(size_t buffer_size)
    {
        // shared_ptr creation packed in a function so it can be used in constructor's initilaization list
        void* ptr = nullptr;
        GW_CU_CHECK_ERR(cudaMalloc(&ptr, buffer_size));
        auto ret_val = std::unique_ptr<char, void (*)(char*)>(static_cast<char*>(ptr),
                                                              [](char* ptr) {
                                                                  GW_CU_ABORT_ON_ERR(cudaFree(ptr));
                                                              });
        return ret_val;
    }

    /// \brief finds a memory block of the given size
    /// \param ptr on return pointer to allocated memory, nullptr if allocation was not successful
    /// \param bytes_needed
    /// \param associated_streams on deallocation this memory block is guaranteed to live at least until all previously scheduled work in these streams has finished
    /// \return cudaSuccess if allocation was successful, cudaErrorMemoryAllocation otherwise
    cudaError_t allocate_memory_block(void** ptr,
                                      size_t bytes_needed,
                                      const std::vector<cudaStream_t>& associated_streams)
    {
        *ptr = nullptr;

        if (free_blocks_.empty())
        {
            return cudaErrorMemoryAllocation;
        }

        // ** All allocations should be alligned with 256 bytes
        // The easiest way to do this is to make all allocation request sizes divisible by 256
        if ((bytes_needed & 0xFF) != 0)
        {
            // bytes needed not divisible by 256, increase it to the next value divisible by 256
            bytes_needed = bytes_needed + (0x100 - (bytes_needed & 0xFF));
        }
        assert((bytes_needed & 0xFF) == 0);

        // ** look for first free block that can fit requested size
        auto block_to_get_memory_from_iter = std::find_if(std::begin(free_blocks_),
                                                          std::end(free_blocks_),
                                                          [bytes_needed](const MemoryBlock& memory_block) {
                                                              return memory_block.size >= bytes_needed;
                                                          });

        if (block_to_get_memory_from_iter == std::end(free_blocks_))
        {
            return cudaErrorMemoryAllocation;
        }

        MemoryBlock new_memory_block{block_to_get_memory_from_iter->begin,
                                     bytes_needed,
                                     associated_streams};

        // ** reduce the size of the block the memory is going to be taken from
        if (block_to_get_memory_from_iter->size == bytes_needed)
        {
            // this memory block is completely used, remove it
            free_blocks_.erase(block_to_get_memory_from_iter);
        }
        else
        {
            // there will still be some memory left, update free block size
            block_to_get_memory_from_iter->begin += new_memory_block.size;
            block_to_get_memory_from_iter->size -= new_memory_block.size;
        }

        // ** add new used memory block to the list of used blocks

        // look for the block right after the block that is to be added
        auto insert_used_block_before_iter = std::find_if(std::begin(used_blocks_),
                                                          std::end(used_blocks_),
                                                          [&new_memory_block](const MemoryBlock& memory_block) {
                                                              return memory_block.begin > new_memory_block.begin;
                                                          });

        // insert new block in the array
        used_blocks_.insert(insert_used_block_before_iter, new_memory_block);

        // set pointer to new location
        *ptr = static_cast<void*>(buffer_ptr_.get() + new_memory_block.begin);
        return cudaSuccess;
    }

    /// \brief returns the block starting at pointer
    /// This function blocks until all work on associated_streams is done
    /// \param pointer pointer at the begining of the block to be freed
    /// \return error status
    cudaError_t free_memory_block(void* pointer)
    {
        assert(static_cast<char*>(pointer) >= buffer_ptr_.get());
        const size_t block_start = static_cast<char*>(pointer) - buffer_ptr_.get();
        assert(block_start < buffer_size_);

        // ** look for pointer's memory block
        auto block_to_be_freed_iter = std::find_if(std::begin(used_blocks_),
                                                   std::end(used_blocks_),
                                                   [block_start](const MemoryBlock& memory_block) {
                                                       return memory_block.begin == block_start;
                                                   });
        assert(block_to_be_freed_iter != std::end(used_blocks_));

        // ** wait for all work on associated_streams to finish before freeing up this memory block
        for (cudaStream_t associated_stream : block_to_be_freed_iter->associated_streams)
        {
            // WARNING: The way and place this synchronization is done might change in the future, do not rely on this cudaStreamSynchronize() in the caller.
            // Guarantee that the memory will not be deallocated before all previously scheduled work on these streams will remain, but actual deallocation
            // (and synchronization) might happen for example only when this memory block is actually requested by another allocation, which would make any
            // code relying on an implicit synchronization here incorrect
            GW_CU_ABORT_ON_ERR(cudaStreamSynchronize(associated_stream));
        }

        // ** remove memory block from the list of used memory blocks
        const size_t number_of_bytes = block_to_be_freed_iter->size;
        used_blocks_.erase(block_to_be_freed_iter);

        // ** add the block back the list of free blocks (and merge with any neighbouring free blocks)

        // look for block immediately after the block that is to being freed
        auto insert_free_block_before_iter = std::find_if(std::begin(free_blocks_),
                                                          std::end(free_blocks_),
                                                          [block_start](const MemoryBlock& memory_block) {
                                                              return memory_block.begin > block_start;
                                                          });

        // * find the left neighbor and remove it if it is going to be be merged
        MemoryBlock block_to_the_left;
        if (std::begin(free_blocks_) == insert_free_block_before_iter)
        {
            // no left neighbor, create a virtual empty neighbor
            block_to_the_left.begin = block_start;
            block_to_the_left.size  = 0;
        }
        else
        {
            block_to_the_left = *std::prev(insert_free_block_before_iter);
            assert(block_to_the_left.begin + block_to_the_left.size <= block_start);
            if (block_to_the_left.begin + block_to_the_left.size == block_start)
            {
                // neighbor is "touching" this block and will be merged, remove it
                free_blocks_.erase(std::prev(insert_free_block_before_iter));
            }
            else
            {
                // neighbor won't be merged, create a virtual empty neighbor
                block_to_the_left.begin = block_start;
                block_to_the_left.size  = 0;
            }
        }

        // * find the right neighbor and remove if it is going to be be merged
        MemoryBlock block_to_the_right;
        if (std::end(free_blocks_) == insert_free_block_before_iter)
        {
            // no neighbor to the right, create a virtual empty neighbor
            block_to_the_right.begin = block_start + number_of_bytes;
            block_to_the_right.size  = 0;
        }
        else
        {
            block_to_the_right = *insert_free_block_before_iter;
            assert(block_start + number_of_bytes <= block_to_the_right.begin);
            if (block_start + number_of_bytes == block_to_the_right.begin)
            {
                // neighbor is "touching" this block and will be merged, remove it
                auto iter_past_right_neighbor = std::next(insert_free_block_before_iter);
                free_blocks_.erase(insert_free_block_before_iter);
                insert_free_block_before_iter = iter_past_right_neighbor;
            }
            else
            {
                // neighbor won't be merged, create a virtual empty neighbor
                block_to_the_right.begin = block_start + number_of_bytes;
                block_to_the_right.size  = 0;
            }
        }

        // create the new free memory block
        MemoryBlock new_memory_block;
        // block_to_the_left.begin corresponds to begin of the newly freed block if the left
        // neighbor should not be merged, begin of the actual left neighbor otherwise
        new_memory_block.begin = block_to_the_left.begin;
        // block_to_the_left.size and block_to_the_right.size have value 0 if left and right neighbors
        // should not be merged, number of elements in left and right neighbor otherwise
        new_memory_block.size = block_to_the_left.size + number_of_bytes + block_to_the_right.size;

        free_blocks_.insert(insert_free_block_before_iter, new_memory_block);

        return cudaSuccess;
    }

    /// buffer size
    const size_t buffer_size_;
    /// preallocated buffer
    const std::unique_ptr<char, void (*)(char*)> buffer_ptr_;
    /// (de)allocation mutex
    mutable std::mutex memory_operation_mutex_;

    /// list of free block, sorted by memory block beginning location
    std::list<MemoryBlock> free_blocks_;
    /// list of block in use, sorted by memory block beginning location
    std::list<MemoryBlock> used_blocks_;
};

} // namespace genomeworks

} // namespace claraparabricks
