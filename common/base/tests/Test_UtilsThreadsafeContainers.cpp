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

// *** test ThreadsafeDataProvider ***

void test_threadsafe_data_provider(const std::int32_t number_of_elements,
                                   const std::int32_t number_of_threads)
{
    std::vector<std::int32_t> data(number_of_elements);
    std::iota(std::begin(data),
              std::end(data),
              0);

    ThreadsafeDataProvider<std::int32_t> data_provider(std::move(data));

    std::mutex occurrences_per_element_mutex; // using mutex instead of atomic as the test was failining when using more than 1'000'000 atomics
    std::vector<std::int32_t> occurrences_per_element(number_of_elements, 0);

    std::vector<std::thread> threads;

    for (std::int32_t thread_id = 0; thread_id < number_of_threads; ++thread_id)
    {
        threads.push_back(std::thread([&data_provider, &occurrences_per_element, &occurrences_per_element_mutex]() {
            while (true)
            {
                cga_optional_t<std::int32_t> val = data_provider.get_next_element();

                if (!val) // reached the end
                {
                    break;
                }
                else
                {
                    std::lock_guard<std::mutex> occurrences_per_element_mutex_lock(occurrences_per_element_mutex);
                    occurrences_per_element[val.value()]++;
                }
            }
        }));
    }

    for (std::thread& thread : threads)
    {
        thread.join();
    }

    ASSERT_TRUE(std::all_of(std::begin(occurrences_per_element),
                            std::end(occurrences_per_element),
                            [](const std::int32_t val) {
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

// *** ThreadsafeProducerConsumer ***

void test_test_threadsafe_producer_consumer(const std::int32_t number_of_elements,
                                            const std::int32_t number_of_producers,
                                            const std::int32_t number_of_consumers,
                                            const std::int32_t producers_sleep_for_ms) // give consumers some time to empty the queue)
{
    ASSERT_GT(number_of_elements, 0);
    ASSERT_GT(number_of_producers, 0);
    ASSERT_GT(number_of_consumers, 0);
    ASSERT_GT(producers_sleep_for_ms, 0);
    const std::int32_t number_of_elements_per_producer = number_of_elements / number_of_producers;
    const std::int32_t producers_sleep_after           = number_of_elements_per_producer / 10 * 3;
    ASSERT_GT(number_of_elements_per_producer, producers_sleep_after);

    ThreadsafeProducerConsumer<std::int32_t> producer_consumer;

    std::mutex occurrences_per_element_mutex; // using mutex instead of atomic as the test was failining when using more than 1'000'000 atomics
    std::vector<std::int32_t> occurrences_per_element(number_of_elements, 0);

    std::vector<std::thread> producer_threads;
    std::vector<std::thread> consumer_threads;

    for (std::int32_t producer_id = 0; producer_id < number_of_producers; ++producer_id)
    {
        producer_threads.push_back(std::thread([&producer_consumer, number_of_producers, producer_id, producers_sleep_after, producers_sleep_for_ms, number_of_elements_per_producer]() {
            const std::int32_t producer_offset = producer_id * number_of_elements_per_producer;
            for (std::int32_t i = 0; i < producers_sleep_after; ++i)
            {
                producer_consumer.add_new_element(producer_offset + i);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(producers_sleep_for_ms));
            for (std::int32_t i = producers_sleep_after; i < number_of_elements_per_producer; ++i)
            {
                producer_consumer.add_new_element(producer_offset + i);
            }
            if (number_of_producers == 1)
            {
                // if there is only one producer there is no need to wait for other (nonexistent) producers to finish writing
                producer_consumer.signal_pushed_last_element();
            }
        }));
    }

    for (std::int32_t consumer_id = 0; consumer_id < number_of_consumers; ++consumer_id)
    {
        consumer_threads.push_back(std::thread([&producer_consumer, &occurrences_per_element, &occurrences_per_element_mutex]() {
            while (true)
            {
                cga_optional_t<std::int32_t> val = producer_consumer.get_next_element();
                if (!val) // reached the end
                {
                    break;
                }
                else
                {
                    std::lock_guard<std::mutex> lg(occurrences_per_element_mutex); // if everything goes well every element will be written to exactly one, but if there is a bug several theads could access the same element
                    occurrences_per_element[val.value()]++;
                }
            }
        }));
    }

    for (std::thread& producer_thread : producer_threads)
    {
        producer_thread.join();
    }
    if (number_of_producers > 1)
    {
        producer_consumer.signal_pushed_last_element();
    }
    for (std::thread& consumer_thread : consumer_threads)
    {
        consumer_thread.join();
    }

    ASSERT_TRUE(std::all_of(std::begin(occurrences_per_element),
                            std::end(occurrences_per_element),
                            [](const std::int32_t val) {
                                return val == 1;
                            }));
}

TEST(TestUtilsThreadsafeContainers, test_threadsafe_producer_consumer_single_producer_single_consumer)
{
    ThreadsafeProducerConsumer<std::int32_t> producer_consumer;

    const std::int32_t number_of_elements     = 1'000'000;
    const std::int32_t number_of_producers    = 1;
    const std::int32_t number_of_consumers    = 1;
    const std::int32_t producers_sleep_for_ms = 1000;

    test_test_threadsafe_producer_consumer(number_of_elements,
                                           number_of_producers,
                                           number_of_consumers,
                                           producers_sleep_for_ms);
}

TEST(TestUtilsThreadsafeContainers, test_threadsafe_producer_consumer_multiple_producers_multiple_consumers)
{
    ThreadsafeProducerConsumer<std::int32_t> producer_consumer;

    const std::int32_t number_of_elements     = 10'000'000;
    const std::int32_t number_of_producers    = 100;
    const std::int32_t number_of_consumers    = 200;
    const std::int32_t producers_sleep_for_ms = 1000;

    test_test_threadsafe_producer_consumer(number_of_elements,
                                           number_of_producers,
                                           number_of_consumers,
                                           producers_sleep_for_ms);
}

TEST(TestUtilsThreadsafeContainers, test_threadsafe_producer_consumer_add_after_last)
{
    ThreadsafeProducerConsumer<std::int32_t> producer_consumer;

    producer_consumer.add_new_element(5);
    producer_consumer.signal_pushed_last_element();
    ASSERT_THROW(producer_consumer.add_new_element(10), std::logic_error);
}

TEST(TestUtilsThreadsafeContainers, test_threadsafe_producer_consumer_multiple_signals)
{
    ThreadsafeProducerConsumer<std::int32_t> producer_consumer;

    producer_consumer.add_new_element(5);
    producer_consumer.signal_pushed_last_element();
    ASSERT_THROW(producer_consumer.signal_pushed_last_element(), std::logic_error);
    ASSERT_THROW(producer_consumer.signal_pushed_last_element(), std::logic_error);
}

TEST(TestUtilsThreadsafeContainers, test_threadsafe_producer_consumer_rvalue_and_lvalue)
{
    ThreadsafeProducerConsumer<std::vector<std::int32_t>> producer_consumer;

    std::vector<std::int32_t> vect_0({0, 1, 2});
    producer_consumer.add_new_element(vect_0);
    producer_consumer.add_new_element({3, 4, 5});
    auto val_0 = producer_consumer.get_next_element();
    ASSERT_EQ(val_0.value(), vect_0);
    auto val_1 = producer_consumer.get_next_element();
    ASSERT_EQ(val_1.value(), std::vector<std::int32_t>({3, 4, 5}));
}

TEST(TestUtilsThreadsafeContainers, test_threadsafe_producer_consumer_signal_on_empty)
{
    ThreadsafeProducerConsumer<std::int32_t> producer_consumer;

    producer_consumer.signal_pushed_last_element();
    auto val = producer_consumer.get_next_element();
    ASSERT_FALSE(val);
}

} // namespace claragenomics
