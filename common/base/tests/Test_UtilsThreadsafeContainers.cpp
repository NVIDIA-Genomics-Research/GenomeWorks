/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gtest/gtest.h"

#include <claragenomics/utils/threadsafe_containers.hpp>

#include <algorithm>
#include <mutex>
#include <cstdint>
#include <numeric>
#include <thread>

namespace claragenomics
{

void test_threadsafe_data_provider(const std::size_t number_of_elements,
                                   const std::size_t number_of_threads)
{
    std::vector<std::size_t> data(number_of_elements);
    std::iota(std::begin(data),
              std::end(data),
              0);

    ThreadsafeDataProvider<std::size_t> data_provider(std::move(data));

    std::mutex occurrenes_per_element_mutex;
    std::vector<std::size_t> occurrenes_per_element(number_of_elements, 0);

    std::vector<std::thread> threads;

    for (std::size_t thread_id = 0; thread_id < number_of_threads; ++thread_id)
    {
        threads.push_back(std::thread([&data_provider, &occurrenes_per_element, &occurrenes_per_element_mutex]() {
            while (true)
            {
                auto val = data_provider.next_element();

                if (!val.has_value()) // reached the end
                {
                    break;
                }
                else
                {
                    std::lock_guard<std::mutex> occurrenes_per_element_mutex_lock(occurrenes_per_element_mutex);
                    occurrenes_per_element[val.value()]++;
                }
            }
        }));
    }

    for (std::size_t thread_id = 0; thread_id < number_of_threads; ++thread_id)
    {
        threads[thread_id].join();
    }

    ASSERT_TRUE(std::all_of(std::begin(occurrenes_per_element),
                            std::end(occurrenes_per_element),
                            [](const std::size_t val) {
                                return val == 1;
                            }));
}

TEST(TestUtilsThreadsafeContainers, test_threadsafe_data_provider_with_data)
{
    test_threadsafe_data_provider(10'000'000, 1000);
}

TEST(TestUtilsThreadsafeContainers, test_threadsafe_data_provider_no_data)
{
    test_threadsafe_data_provider(0, 1000);
}

} // namespace claragenomics
