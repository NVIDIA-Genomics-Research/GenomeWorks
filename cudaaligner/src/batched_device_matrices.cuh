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

#include "matrix_cpu.hpp"

#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/utils/limits.cuh>
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>
#include <claraparabricks/genomeworks/utils/device_buffer.hpp>
#include <claraparabricks/genomeworks/utils/pinned_host_vector.hpp>

#include <tuple>
#include <cassert>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

__device__ inline bool error(int32_t x, int32_t y)
{
    printf("assert: lhs=%d, rhs=%d\n", x, y);
    return false;
}

template <typename T>
class device_matrix_view
{
public:
    __device__ device_matrix_view()
    {
    }

    __device__ device_matrix_view(T* storage, int32_t n_rows, int32_t n_cols)
        : data_(storage)
        , n_rows_(n_rows)
        , n_cols_(n_cols)
    {
    }

    __device__ inline T const& operator()(int32_t i, int32_t j) const
    {
        assert(0 <= i || error(0, i));
        assert(i < n_rows_ || error(i, n_rows_));
        assert(0 <= j || error(0, j));
        assert(j < n_cols_ || error(j, n_cols_));
        return data_[i + n_rows_ * j];
    }
    __device__ inline T& operator()(int32_t i, int32_t j)
    {
        assert(0 <= i || error(0, i));
        assert(i < n_rows_ || error(i, n_rows_));
        assert(0 <= j || error(0, j));
        assert(j < n_cols_ || error(j, n_cols_));
        return data_[i + n_rows_ * j];
    }
    __device__ inline int32_t num_rows() const
    {
        return n_rows_;
    }
    __device__ inline int32_t num_cols() const
    {
        return n_cols_;
    }

private:
    T* data_        = nullptr;
    int32_t n_rows_ = 0;
    int32_t n_cols_ = 0;
};

template <typename T>
class batched_device_matrices
{
public:
    class device_interface
    {
    public:
        device_interface(T* storage, ptrdiff_t* offsets, int32_t n_matrices)
            : storage_(storage)
            , offsets_(offsets)
            , n_matrices_(n_matrices)
        {
            assert(storage_ != nullptr);
            assert(offsets_ != nullptr);
            assert(n_matrices >= 0);
        }

        __device__ device_matrix_view<T> get_matrix_view(int32_t id, int32_t n_rows, int32_t n_cols)
        {
            assert(storage_ != nullptr);
            assert(id < n_matrices_ || error(id, n_matrices_));
            const int32_t max_elements = get_max_elements_per_matrix(id);
            assert(n_rows * n_cols <= max_elements || error(n_rows * n_cols, max_elements));
            if (n_rows * n_cols > max_elements)
            {
                n_rows = 0;
                n_cols = 0;
            }
            return device_matrix_view<T>(storage_ + offsets_[id], n_rows, n_cols);
        }

        __device__ inline T* data(int32_t id)
        {
            assert(storage_ != nullptr);
            assert(id < n_matrices_);
            return storage_ + offsets_[id];
        }

        __device__ inline int32_t get_max_elements_per_matrix(int32_t id) const
        {
            assert(id < n_matrices_);
            assert(offsets_[id + 1] - offsets_[id] >= 0);
            assert(offsets_[id + 1] - offsets_[id] <= numeric_limits<int32_t>::max());
            return offsets_[id + 1] - offsets_[id];
        }

    private:
        T* storage_;
        ptrdiff_t* offsets_;
        int32_t n_matrices_;
        friend class batched_device_matrices<T>;
    };

    batched_device_matrices(int64_t max_elements, DefaultDeviceAllocator allocator, cudaStream_t stream)
        : storage_(max_elements, allocator, stream)
        , offsets_(allocator, stream)
        , dev_(1, allocator, stream)
        , offsets_host_(1, 0)
    {
    }

    batched_device_matrices(int32_t n_matrices, int32_t max_elements_per_matrix, DefaultDeviceAllocator allocator, cudaStream_t stream)
        : storage_(static_cast<size_t>(n_matrices) * static_cast<size_t>(max_elements_per_matrix), allocator, stream)
        , offsets_(n_matrices + 1, allocator, stream)
        , dev_(1, allocator, stream)
        , offsets_host_(n_matrices + 1)
    {
        assert(n_matrices >= 0);
        assert(max_elements_per_matrix >= 0);
        GW_CU_CHECK_ERR(cudaMemsetAsync(storage_.data(), 0, storage_.size() * sizeof(T), stream));

        for (int32_t i = 0; i < n_matrices + 1; ++i)
        {
            offsets_host_[i] = static_cast<ptrdiff_t>(max_elements_per_matrix) * i;
        }

        construct_device_matrices_async(stream);
    }

    ~batched_device_matrices() = default;

    batched_device_matrices(batched_device_matrices const&) = delete;
    batched_device_matrices& operator=(batched_device_matrices const&) = delete;
    batched_device_matrices(batched_device_matrices&&)                 = default;
    batched_device_matrices& operator=(batched_device_matrices&&) = default;

    device_interface* get_device_interface()
    {
        return dev_.data();
    }

    bool append_matrix(int32_t max_elements)
    {
        assert(offsets_host_.size() >= 1);
        const ptrdiff_t begin = offsets_host_.back();
        if (begin + max_elements > get_size(storage_))
        {
            return false;
        }
        offsets_host_.push_back(begin + max_elements);
        return true;
    }

    int32_t number_of_matrices() const
    {
        assert(offsets_host_.size() >= 1);
        return get_size<int32_t>(offsets_host_) - 1;
    }

    int64_t remaining_free_matrix_elements() const
    {
        assert(offsets_host_.size() >= 1);
        return get_size<int64_t>(storage_) - offsets_host_.back();
    }

    void clear()
    {
        offsets_host_.clear();
        offsets_host_.push_back(0);
    }

    void construct_device_matrices_async(cudaStream_t stream)
    {
        assert(offsets_host_.size() >= 1);
        offsets_.clear_and_resize(offsets_host_.size());
        dev_interface_host_.clear();
        dev_interface_host_.emplace_back(storage_.data(), offsets_.data(), get_size<int32_t>(offsets_host_) - 1);
        cudautils::device_copy_n(offsets_host_.data(), offsets_host_.size(), offsets_.data(), stream);
        cudautils::device_copy_n(dev_interface_host_.data(), 1, dev_.data(), stream);
    }

    matrix<T> get_matrix(int32_t id, int32_t n_rows, int32_t n_cols, cudaStream_t stream)
    {
        assert(id >= 0);
        assert(n_rows >= 0);
        assert(n_cols >= 0);

        if (id >= get_size(offsets_) - 1)
            throw std::runtime_error("Requested id is out of bounds.");

        matrix<T> m(n_rows, n_cols);
        if (n_rows * n_cols > offsets_host_[id + 1] - offsets_host_[id])
            throw std::runtime_error("Requested matrix size is larger than allocated memory on device.");
        cudautils::device_copy_n(storage_.data() + offsets_host_[id], n_rows * n_cols, m.data(), stream);
        GW_CU_CHECK_ERR(cudaStreamSynchronize(stream));
        return m;
    }

private:
    device_buffer<T> storage_;
    device_buffer<ptrdiff_t> offsets_;
    device_buffer<device_interface> dev_;
    pinned_host_vector<ptrdiff_t> offsets_host_;
    pinned_host_vector<device_interface> dev_interface_host_; // just to provide truly async copies
};

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
