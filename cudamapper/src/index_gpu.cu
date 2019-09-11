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

namespace details {

namespace index_gpu {

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

        if (sample_length == 0) {
            throw approximate_sketch_elements_per_bucket_too_small("approximate_sketch_elements_per_bucket is " + std::to_string(approximate_sketch_elements_per_bucket) +
                                                                   " but should be at least " + std::to_string(arrays_of_representations.size()));
        }

        // sample every sample_length representation
        for (std::size_t array_index = 0; array_index < arrays_of_representations.size(); ++array_index) {
            for (std::size_t sample_index = 0; sample_index < arrays_of_representations[array_index].size(); sample_index += sample_length) {
                sampled_representations.push_back(arrays_of_representations[array_index][sample_index]);
            }
        }

        // The number of samples whose sketch elements fit one bucket on the gpu when grouped together
        const std::uint64_t samples_in_one_bucket = approximate_sketch_elements_per_bucket / sample_length;
        std::vector<representation_t> representation_buckets;

        std::sort(std::begin(sampled_representations), std::end(sampled_representations));

        // Merge every samples_in_one_bucket samples into one bucket, skipping samples that have the same representation as the previosuly added sample
        // in order to avoid having representations split across multiple buckets
        representation_buckets.push_back(sampled_representations[0]);
        for (std::size_t sample_index = samples_in_one_bucket; sample_index < sampled_representations.size(); sample_index += samples_in_one_bucket) {
            if(sampled_representations[sample_index] != representation_buckets.back()) {
                representation_buckets.push_back(sampled_representations[sample_index]);
            } else {
                CGA_LOG_INFO("Representation {} does not fit one bucket", sampled_representations[sample_index]);
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

    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> generate_bucket_boundary_indices(const std::vector<std::vector<representation_t>>& arrays_of_representations,
                                                                                                   const std::vector<representation_t>& representation_buckets
                                                                                                  )
    {
        const std::size_t number_of_arrays = arrays_of_representations.size();
        const std::size_t number_of_buckets = representation_buckets.size();

        std::vector<std::vector<std::pair<std::size_t, std::size_t>>> bucket_boundary_indices(number_of_buckets);

        // all buckets start from 0
        std::vector<std::size_t> first_index_per_array(number_of_arrays, 0);
        // treat last bucket separately as its last representation is not saved in representation_buckets
        for (std::size_t bucket_index = 0; bucket_index < number_of_buckets - 1; ++bucket_index) {
            std::vector<std::size_t> last_index_per_array = generate_representation_indices(arrays_of_representations, representation_buckets[bucket_index + 1]);
            for (std::size_t array_index = 0; array_index < number_of_arrays; ++array_index) {
                bucket_boundary_indices[bucket_index].emplace_back(first_index_per_array[array_index],
                                                                   last_index_per_array[array_index]
                                                                  );
            }
            first_index_per_array = std::move(last_index_per_array);
        }
        // now deal with the last bucket (last bucket always goes up to the last element in the array)
        for (std::size_t array_index = 0; array_index < number_of_arrays; ++array_index) {
            bucket_boundary_indices.back().emplace_back(first_index_per_array[array_index],
                                                        arrays_of_representations[array_index].size()
                                                       );
        }

        return bucket_boundary_indices;
    }

} // namespace index_gpu

} // namespace details

} // namespace claragenomics
