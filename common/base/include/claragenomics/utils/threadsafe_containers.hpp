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

#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <vector>

#include <claragenomics/types.hpp>

namespace claragenomics
{

/// ThreadsafeDataProvider - wrapper around std::vector which gives elements one by one to multiple threads and signals when there are no elements left
template <typename T>
class ThreadsafeDataProvider
{
public:
    /// \brief default constructor
    ThreadsafeDataProvider()
        : data_()
        , counter_(0)
    {
    }

    /// \brief Constructor
    /// \param data
    ThreadsafeDataProvider(const std::vector<T>& data)
        : data_(data)
        , counter_(0)
    {
    }

    /// \brief Constructor
    /// \param data
    ThreadsafeDataProvider(std::vector<T>&& data)
        : data_(data)
        , counter_(0)
    {
    }

    /// \brief deleted copy constructor
    /// \param rhs
    ThreadsafeDataProvider(const ThreadsafeDataProvider& rhs) = delete;

    /// \brief deleted copy assignment operator
    /// \param rhs
    ThreadsafeDataProvider& operator==(const ThreadsafeDataProvider& rhs) = delete;

    /// \brief deleted move constructor
    /// \param rhs
    ThreadsafeDataProvider(ThreadsafeDataProvider&& rhs) = delete; // could be implemeted if needed

    /// \brief deleted move assignment operator
    /// \param rhs
    ThreadsafeDataProvider& operator==(ThreadsafeDataProvider&& rhs) = delete; // could be implemeted if needed

    /// \brief destructor
    ~ThreadsafeDataProvider() = default;

    /// \brief returns next available element or empty optional object if there are no elements left
    cga_optional_t<T> get_next_element()
    {
        size_t my_counter = counter_++;
        return my_counter < data_.size() ? cga_optional_t<T>(std::move(data_[my_counter])) : cga_nullopt;
    }

private:
    /// data to provide, whenever an element is provided it is moved from this value
    std::vector<T> data_;
    /// number of elements provided so far
    std::atomic<std::size_t> counter_;
};

/// ThreadsafeProducerConsumer - a threadsafe implementation of producer-consumer pattern with an option to signal that there will be no new elements
///
/// Producers add elements using add_new_element(), consumers consume elements in the order the were added using get_next_element().
/// If there is no available element get_next_element() blocks and waits for one.
///
/// One producer can signal that there are not going to be any new elements using signal_pushed_last_element(). After this is done and all elements
/// have been consumed get_next_element() returns empty optionals indicating that there is not going to be any new data and consumers can carry on
template <typename T>
class ThreadsafeProducerConsumer
{
public:
    /// \brief default constructor
    ThreadsafeProducerConsumer()
        : data_()
        , pushed_last_element_(false)
        , mutex_()
        , condition_variable_()
    {
    }

    /// \brief deleted copy constructor
    /// \param rhs
    ThreadsafeProducerConsumer(const ThreadsafeProducerConsumer& rhs) = delete;

    /// \brief deleted copy assignment operator
    /// \param rhs
    ThreadsafeProducerConsumer& operator==(const ThreadsafeProducerConsumer& rhs) = delete;

    /// \brief deleted move constructor
    /// \param rhs
    ThreadsafeProducerConsumer(ThreadsafeProducerConsumer&& rhs) = delete; // could be implemeted if needed

    /// \brief deleted move assignment operator
    /// \param rhs
    ThreadsafeProducerConsumer& operator==(ThreadsafeProducerConsumer&& rhs) = delete; // could be implemeted if needed

    /// \brief destructor
    ~ThreadsafeProducerConsumer() = default;

    /// \brief adds an element to the queue
    ///
    /// \param element element to add
    /// \throw std::logic_error if called after signal_pushed_last_element() has been called by any producer
    void add_new_element(const T& element) // not using reference in order to support both rvalue and lvalue
    {
        // consider adding max number of elements and waiting for consumer to process an element before adding a new one
        {
            std::lock_guard<std::mutex> lg(mutex_);
            if (pushed_last_element_)
            {
                throw std::logic_error("ThreadsafeProducerConsumer: pushed an element after signal_pushed_last_element() has been called");
            }
            data_.push_front(element);
        }
        condition_variable_.notify_one();
    }

    /// \brief adds an element to the queue
    ///
    /// \param element element to add
    /// \throw std::logic_error if called after signal_pushed_last_element() has been called by any producer
    void add_new_element(T&& element) // not using reference in order to support both rvalue and lvalue
    {
        // consider adding max number of elements and waiting for consumer to process an element before adding a new one
        {
            std::lock_guard<std::mutex> lg(mutex_);
            if (pushed_last_element_)
            {
                throw std::logic_error("ThreadsafeProducerConsumer: pushed an element after signal_pushed_last_element() has been called");
            }
            data_.push_front(std::move(element));
        }
        condition_variable_.notify_one();
    }

    /// \brief tells container that no new elements are going to be added
    ///
    /// After this method has been called and all elements have been consumed get_next_element() will be returning
    /// empty optionals to indicate that there will be no new elements and that consumers can finish processing
    ///
    /// \throw std::logic_error if called after any producer has already called this method
    void signal_pushed_last_element()
    {
        {
            std::lock_guard<std::mutex> lg(mutex_);
            if (pushed_last_element_)
            {
                throw std::logic_error("ThreadsafeProducerConsumer: called signal_pushed_last_element() more than once");
            }
            pushed_last_element_ = true;
        }
        condition_variable_.notify_all();
    }

    /// \brief gets the next element
    ///
    /// If no element is available waits for an element to become available
    /// If no element is available and signal_pushed_last_element() has already been called returns an empty optional
    ///
    /// \return optional, either with the value or empty if no element available and signal_pushed_last_element() has been called
    cga_optional_t<T> get_next_element()
    {
        std::unique_lock<std::mutex> ul(mutex_);

        while (data_.empty() && !pushed_last_element_)
        {
            condition_variable_.wait(ul);
        }

        const bool no_elements_left = pushed_last_element_ && data_.empty();
        if (no_elements_left)
        {
            return cga_nullopt;
        }
        else
        {
            cga_optional_t<T> res = std::move(data_.back());
            data_.pop_back();
            return res;
        }
    }

private:
    /// data
    std::deque<T> data_;
    /// if true no new calls to signal_pushed_last_element() is called
    bool pushed_last_element_;
    /// mutex for condition_variable_
    std::mutex mutex_;
    /// condition_variable to wait on if there are no available elements
    std::condition_variable condition_variable_;
};

} // namespace claragenomics
