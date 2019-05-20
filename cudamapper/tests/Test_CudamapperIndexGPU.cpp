#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "cudautils/cudautils.hpp"
#include "../src/index_gpu.hpp"
#include "../src/minimizer.hpp"

namespace genomeworks {

    class IndexGeneratorTest : public IndexGenerator {
    public:
        const std::unordered_multimap<std::uint64_t, std::unique_ptr<SketchElement>>& representation_sketch_element_mapping() const override {
            return data_;
        }

        void add_sketch_element(std::unique_ptr<SketchElement> sketch_element) {
            data_.insert(std::pair<std::uint64_t, std::unique_ptr<SketchElement>>(sketch_element->representation(), std::move(sketch_element)));
        }
    private:
        std::unordered_multimap<std::uint64_t, std::unique_ptr<SketchElement>> data_;
    };

    class TestCudamapperIndexGPU : public ::testing::Test {
    public:
        void SetUp() override {
            // for simplicity using all representations are in [0, largest_representation_], positions in [0, largest_position_] and sequence ids in [0, largest_sequence_id_]
            IndexGeneratorTest index_generator_test;
            // Minimizer(representation, position, direction, sequence_id)
            minimizers_.emplace_back(0, 10, SketchElement::DirectionOfRepresentation::FORWARD, 100);
            minimizers_.emplace_back(0, 11, SketchElement::DirectionOfRepresentation::REVERSE, 100);
            minimizers_.emplace_back(0, 12, SketchElement::DirectionOfRepresentation::REVERSE, 101);
            minimizers_.emplace_back(0, 16, SketchElement::DirectionOfRepresentation::FORWARD, 101);
            minimizers_.emplace_back(0, 20, SketchElement::DirectionOfRepresentation::FORWARD, 105);
            minimizers_.emplace_back(0, 13, SketchElement::DirectionOfRepresentation::REVERSE, 108);
            minimizers_.emplace_back(0, 15, SketchElement::DirectionOfRepresentation::FORWARD, 105);
            minimizers_.emplace_back(1, 22, SketchElement::DirectionOfRepresentation::REVERSE, 107);
            minimizers_.emplace_back(1, 26, SketchElement::DirectionOfRepresentation::REVERSE, 106);
            minimizers_.emplace_back(1, 23, SketchElement::DirectionOfRepresentation::REVERSE, 103);
            minimizers_.emplace_back(1, 20, SketchElement::DirectionOfRepresentation::FORWARD, 108);
            minimizers_.emplace_back(1, 27, SketchElement::DirectionOfRepresentation::FORWARD, 109);
            minimizers_.emplace_back(1, 29, SketchElement::DirectionOfRepresentation::FORWARD, 106);
            minimizers_.emplace_back(1, 28, SketchElement::DirectionOfRepresentation::REVERSE, 101);
            minimizers_.emplace_back(1, 24, SketchElement::DirectionOfRepresentation::REVERSE, 100);
            minimizers_.emplace_back(2, 35, SketchElement::DirectionOfRepresentation::FORWARD, 100);
            minimizers_.emplace_back(2, 33, SketchElement::DirectionOfRepresentation::REVERSE, 106);
            minimizers_.emplace_back(2, 39, SketchElement::DirectionOfRepresentation::FORWARD, 100);
            minimizers_.emplace_back(2, 34, SketchElement::DirectionOfRepresentation::REVERSE, 103);
            minimizers_.emplace_back(2, 32, SketchElement::DirectionOfRepresentation::FORWARD, 103);
            minimizers_.emplace_back(2, 30, SketchElement::DirectionOfRepresentation::REVERSE, 106);
            minimizers_.emplace_back(2, 38, SketchElement::DirectionOfRepresentation::REVERSE, 105);
            minimizers_.emplace_back(2, 37, SketchElement::DirectionOfRepresentation::FORWARD, 109);
            minimizers_.emplace_back(2, 31, SketchElement::DirectionOfRepresentation::FORWARD, 109);
            minimizers_.emplace_back(2, 36, SketchElement::DirectionOfRepresentation::FORWARD, 109);
            minimizers_.emplace_back(3, 48, SketchElement::DirectionOfRepresentation::FORWARD, 104);
            minimizers_.emplace_back(3, 42, SketchElement::DirectionOfRepresentation::REVERSE, 103);
            minimizers_.emplace_back(3, 40, SketchElement::DirectionOfRepresentation::REVERSE, 105);
            minimizers_.emplace_back(3, 41, SketchElement::DirectionOfRepresentation::REVERSE, 106);
            minimizers_.emplace_back(4, 56, SketchElement::DirectionOfRepresentation::FORWARD, 100);
            minimizers_.emplace_back(4, 55, SketchElement::DirectionOfRepresentation::FORWARD, 101);
            minimizers_.emplace_back(5, 62, SketchElement::DirectionOfRepresentation::REVERSE, 101);
            minimizers_.emplace_back(5, 60, SketchElement::DirectionOfRepresentation::FORWARD, 104);
            minimizers_.emplace_back(5, 67, SketchElement::DirectionOfRepresentation::FORWARD, 106);

            largest_representation_ = 0;
            largest_position_ = 0;
            largest_sequence_id_ = 0;
            for (const Minimizer& minimizer : minimizers_) {
                index_generator_test.add_sketch_element(std::make_unique<Minimizer>(minimizer));
                largest_representation_ = std::max(largest_representation_, minimizer.representation());
                largest_position_ = std::max(largest_position_, minimizer.position());
                largest_sequence_id_ = std::max(largest_sequence_id_, minimizer.sequence_id());
            }


            std::vector<std::uint32_t> sketch_elems_for_representation_local(largest_representation_+1, 0);
            for (const Minimizer& minimizer : minimizers_) {
                ++sketch_elems_for_representation_local[minimizer.representation()];
            }
            sketch_elems_for_representation_ = std::move(sketch_elems_for_representation_local);

            // generate index
            index_ = std::make_unique<IndexGPU>(index_generator_test);

            GW_CU_CHECK_ERR(cudaMallocHost((void**)&representations_h_, index_->representation_to_device_arrays().size()*sizeof(std::uint64_t)));
            GW_CU_CHECK_ERR(cudaMallocHost((void**)&sequence_ids_h_, minimizers_.size()*sizeof(std::uint64_t)));
            GW_CU_CHECK_ERR(cudaMallocHost((void**)&positions_h_, minimizers_.size()*sizeof(std::size_t)));
            GW_CU_CHECK_ERR(cudaMallocHost((void**)&directions_h_, minimizers_.size()*sizeof(SketchElement::DirectionOfRepresentation)));

            const std::uint64_t* representations_d = index_->representations_d().get();
            GW_CU_CHECK_ERR(cudaMemcpy(representations_h_, representations_d, index_->representation_to_device_arrays().size()*sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
            const std::uint64_t* sequence_ids_d = index_->sequence_ids_d().get();
            GW_CU_CHECK_ERR(cudaMemcpy(sequence_ids_h_, sequence_ids_d, minimizers_.size()*sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
            const std::size_t* positions_d = index_->positions_d().get();
            GW_CU_CHECK_ERR(cudaMemcpy(positions_h_, positions_d, minimizers_.size()*sizeof(std::size_t), cudaMemcpyDeviceToHost));
            const SketchElement::DirectionOfRepresentation* directions_d = index_->directions_d().get();
            GW_CU_CHECK_ERR(cudaMemcpy(directions_h_, directions_d, minimizers_.size()*sizeof(SketchElement::DirectionOfRepresentation), cudaMemcpyDeviceToHost));
        }

        void TearDown() override {
            cudaFreeHost(representations_h_);
            cudaFreeHost(sequence_ids_h_);
            cudaFreeHost(positions_h_);
            cudaFreeHost(directions_h_);
        }

    protected:
        std::unique_ptr<IndexGPU> index_;

        std::uint64_t largest_representation_;
        std::size_t largest_position_;
        std::uint64_t largest_sequence_id_;
        std::vector<std::uint32_t> sketch_elems_for_representation_;
        std::vector<Minimizer> minimizers_;

        std::uint64_t* representations_h_;
        std::uint64_t* sequence_ids_h_;
        std::size_t* positions_h_;
        SketchElement::DirectionOfRepresentation* directions_h_;
    };

    // Tests whether index contains a representation which was not in the original dataset
    TEST_F(TestCudamapperIndexGPU, UnexpectedRepresentation) {
        for (auto const& elem : index_->representation_to_device_arrays()) {
            const auto& representation = elem.first;
            EXPECT_LE(representation, largest_representation_);
        }
    }

    // Tests whether each each mapping has the correct representation
    TEST_F(TestCudamapperIndexGPU, CorrectRepresentations) {
        for (const auto& elem : index_->representation_to_device_arrays()) {
            const auto& representation = elem.first;
            const IndexGPU::MappingToDeviceArrays& mapping = (*index_->representation_to_device_arrays().find(representation)).second;
            EXPECT_EQ(representation, representations_h_[mapping.location_representation_]);
        }

        cudaFreeHost(representations_h_);
    }

    // Tests whether there is a correct number of sketch elements for each representation
    TEST_F(TestCudamapperIndexGPU, CorrectNumberOfRepresentations) {
        for (const auto & elem : index_->representation_to_device_arrays()) {
            const auto& representation = elem.first;
            const IndexGPU::MappingToDeviceArrays& data_for_representation = (*index_->representation_to_device_arrays().find(representation)).second;
            EXPECT_EQ(data_for_representation.block_size_, sketch_elems_for_representation_[representation]);
        }
    }

    // Tests whether each sketch element is present in the index
    TEST_F(TestCudamapperIndexGPU, ElementMatch) {
        const auto& mappings = index_->representation_to_device_arrays();

        for (std::size_t i = 0; i < minimizers_.size(); ++i) {
            const auto representation = minimizers_[i].representation();
            const auto& mapping_iter = mappings.find(representation);
            if (mapping_iter == mappings.end()) {
                ASSERT_TRUE(false) << "mapping not found for representation " << representation;
            }

            const IndexGPU::MappingToDeviceArrays& mapping = (*mapping_iter).second;
            EXPECT_EQ(representation, representations_h_[mapping.location_representation_]);

            bool minimizer_found = false;
            for (std::size_t location = mapping.location_first_in_block_; location < mapping.location_first_in_block_ + mapping.block_size_; ++location) {
                if (minimizers_[i].sequence_id() == sequence_ids_h_[location] && minimizers_[i].position() == positions_h_[location] && minimizers_[i].direction() == directions_h_[location]) {
                    minimizer_found = true;
                    break;
                }
            }
            EXPECT_TRUE(minimizer_found) << "minimizer " << i;
        }
    }

    // Tests whether all arrays have the right ammount of elements
    TEST_F(TestCudamapperIndexGPU, CorrectValues) {
        std::vector<std::vector<std::uint32_t>> sequence_id_occurrences_per_representation(largest_representation_+1, std::vector<std::uint32_t>(largest_sequence_id_+1, 0));
        std::vector<std::vector<std::uint32_t>> position_occurrences_per_representation(largest_representation_+1, std::vector<std::uint32_t>(largest_position_+1, 0));
        std::vector<std::vector<std::uint32_t>> direction_occurrences_per_representation(largest_representation_+1, std::vector<std::uint32_t>(2, 0));

        for (const Minimizer& minimizer : minimizers_) {
            ++sequence_id_occurrences_per_representation[minimizer.representation()][minimizer.sequence_id()];
            ++position_occurrences_per_representation[minimizer.representation()][minimizer.position()];
            SketchElement::DirectionOfRepresentation::FORWARD == minimizer.direction() ? ++direction_occurrences_per_representation[minimizer.representation()][0] : ++direction_occurrences_per_representation[minimizer.representation()][1];
        }

        for (const auto& elem : index_->representation_to_device_arrays()) {
            const auto& representation = elem.first;
            const IndexGPU::MappingToDeviceArrays& mapping = (*index_->representation_to_device_arrays().find(representation)).second;

            std::vector<std::uint32_t> sequence_id_occurrences_seen(largest_sequence_id_+1, 0);
            std::vector<std::uint32_t> position_occurrences_seen(largest_position_+1, 0);
            std::vector<std::uint32_t> direction_occurrences_seen(2, 0);

            for (std::size_t location = mapping.location_first_in_block_; location < mapping.location_first_in_block_ + mapping.block_size_; ++location) {
                ++sequence_id_occurrences_seen[sequence_ids_h_[location]];
                ++position_occurrences_seen[positions_h_[location]];
                SketchElement::DirectionOfRepresentation::FORWARD == directions_h_[location] ? ++direction_occurrences_seen[0] : ++direction_occurrences_seen[1];
            }

            ASSERT_EQ(sequence_id_occurrences_seen.size(), sequence_id_occurrences_per_representation[representation].size()) << "representation " << representation;
            for (int i = 0; i < sequence_id_occurrences_seen.size(); ++i) {
                EXPECT_EQ(sequence_id_occurrences_seen[i], sequence_id_occurrences_per_representation[representation][i]) << "representation " << representation << ", element " << i;
            }
            ASSERT_EQ(position_occurrences_seen.size(), position_occurrences_per_representation[representation].size()) << "representation " << representation;
            for (int i = 0; i < position_occurrences_seen.size(); ++i) {
                EXPECT_EQ(position_occurrences_seen[i], position_occurrences_per_representation[representation][i]) << "representation " << representation << ", element " << i;
            }
            EXPECT_EQ(direction_occurrences_seen[0], direction_occurrences_per_representation[representation][0]) << "representation " << representation;
            EXPECT_EQ(direction_occurrences_seen[1], direction_occurrences_per_representation[representation][1]) << "representation " << representation;
        }
    }
}
