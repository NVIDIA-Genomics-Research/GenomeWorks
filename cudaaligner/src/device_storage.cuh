#pragma once

#include <cudautils/cudautils.hpp>

namespace genomeworks {

template <typename T>
class device_storage {
  public:
    using value_type = T;
    device_storage() = delete;
    device_storage(size_t n_elements, uint32_t device_id)
    : size_(n_elements)
    , device_id_(device_id)
    {
        GW_CU_CHECK_ERR(cudaSetDevice(device_id_));
        GW_CU_CHECK_ERR( cudaMalloc(reinterpret_cast<void**>(&data_), size_*sizeof(T)) );
    }

    ~device_storage()
    {
        GW_CU_CHECK_ERR(cudaSetDevice(device_id_));
        GW_CU_CHECK_ERR( cudaFree(data_) );
    }

    T* data() { return data_; }
    T const* data() const { return data_; }
    size_t size() const { return size_; }

  private:
    T* data_;
    size_t size_;
    uint32_t device_id_;
};

} // end namespace genomeworks
