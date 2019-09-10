/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "index_gpu.cuh"

namespace claragenomics {

namespace index_gpu {

namespace detail {

    std::vector<representation_t> generate_representation_buckets(const std::vector<std::vector<representation_t>>& arrays_of_representations,
                                                                  const std::uint64_t approximate_sketch_elements_per_bucket
                                                                 )
    {
        // The function samples every approximate_sketch_elements_per_bucket/number_of_arrays element of each array and sorts them by representation.
        // For the following input and approximate_sketch_elements_per_bucket = 7 this means sampling every second element:
        // (1 1 2 2 4 4 6 6 9 9)
        //  ^   ^   ^   ^   ^
        // (0 0 1 5 5 5 7 8 8 8)
        //  ^   ^   ^   ^   ^
        // (1 1 1 1 3 4 5 7 9 9)
        //  ^   ^   ^   ^   ^
        // Sorted: 0 1 1 1 1 2 3 4 5 5 6 7 8 9 9
        //
        // Number of samples that fit one bucket is approximate_sketch_elements_per_bucket/sample_size = 3
        // 0) add smallest representation
        //    0 1 1 1 1 2 3 4 5 5 6 7 8 9 9
        //    ^
        //    representation_buckets = 0
        // 1) move three samples
        //    0 1 1 1 1 2 3 4 5 5 6 7 8 9 9
        //    ^ ->  ^
        //    representation_buckets = 0, 1
        // 2) move three samples
        //    0 1 1 1 1 2 3 4 5 5 6 7 8 9 9
        //          ^ ->  ^
        //    representation_buckets = 0, 1, 3
        // 3) move three samples
        //    0 1 1 1 1 2 3 4 5 5 6 7 8 9 9
        //                ^ ->  ^
        //    representation_buckets = 0, 1, 3, 5
        // 4) move three samples
        //    0 1 1 1 1 2 3 4 5 5 6 7 8 9 9
        //                      ^ ->  ^
        //    representation_buckets = 0, 1, 3, 5, 8
        // 4) move three samples
        //    0 1 1 1 1 2 3 4 5 5 6 7 8 9 9
        //                            ^ ->  ^ -> end
        //    representation_buckets = 0, 1, 3, 5, 8
        //
        // Obtained buckets are:
        // 0: 0 0
        // 1: 1 1 1 1 1 1 1 1 2 2
        // 3: 3 4 4 4
        // 5: 5 5 5 5 6 6 7 7
        // 8: 8 8 8 9 9 9 9
        //
        // If something like this would happen
        // 0 1 1 1 1 1 1 1 2 3 4 5 5 6 7 8 9 9
        // ^ ->  ^
        // 0 1 1 1 1 1 1 1 2 3 4 5 5 6 7 8 9 9
        //       ^ ->  ^
        // 0 1 1 1 1 1 1 1 2 3 4 5 5 6 7 8 9 9
        //             ^ ->  ^
        // i.e. the same representation is encountered more than once those additional encounters should be skipped

        std::vector<representation_t> sampled_representations;

        const std::uint64_t sample_length = approximate_sketch_elements_per_bucket / arrays_of_representations.size();

        for (std::size_t array_index = 0; array_index < arrays_of_representations.size(); ++array_index) {
            for (std::size_t sample_index = 0; sample_index < arrays_of_representations[array_index].size(); sample_index += sample_length) {
                sampled_representations.push_back(arrays_of_representations[array_index][sample_index]);
            }
        }

        const std::uint64_t samples_in_one_bucket = approximate_sketch_elements_per_bucket / sample_length;
        std::vector<representation_t> representation_buckets;

        std::sort(std::begin(sampled_representations), std::end(sampled_representations));

        representation_buckets.push_back(sampled_representations[0]);
        for (std::size_t sample_index = samples_in_one_bucket; sample_index < sampled_representations.size(); sample_index += samples_in_one_bucket) {
            if(sampled_representations[sample_index] != representation_buckets.back()) {
                representation_buckets.push_back(sampled_representations[sample_index]);
            }
        }

        return representation_buckets;
    }

    std::vector<std::size_t> generate_representation_indices(const std::vector<std::vector<representation_t>>& arrays_of_representations,
                                                             const representation_t representation
                                                            )
    {
        std::vector<std::size_t> representation_indices;

        for (const auto& one_array_of_representations : arrays_of_representations) {
            auto representation_iterator = std::lower_bound(std::begin(one_array_of_representations),
                                                            std::end(one_array_of_representations),
                                                            representation
                                                           );
            representation_indices.push_back(representation_iterator - std::cbegin(one_array_of_representations));
        }

        return representation_indices;
    }

} // namespace index_gpu

} // namespace detail

} // namespace claragenomics
