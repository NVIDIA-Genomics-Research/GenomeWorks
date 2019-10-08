/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <numeric>
#include "gtest/gtest.h"
#include "cudamapper_file_location.hpp"
#include "../src/index_gpu.cuh"
#include "../src/matcher.hpp"
#include "../src/minimizer.hpp"

namespace claragenomics {
namespace cudamapper {

    TEST(TestCudamapperMatcher, OneReadOneMinimizer) {
        // >read_0
        // GATT

        // only one read -> no anchors
        std::vector<std::pair<std::uint64_t, std::uint64_t>> read_ranges;
        const std::pair<std::uint64_t, std::uint64_t> query_range {0, std::numeric_limits<std::uint64_t>::max()};
        read_ranges.push_back(query_range);

        std::unique_ptr<FastaParser> parser = create_fasta_parser(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/gatt.fasta");
        std::vector<FastaParser*> parsers;
        parsers.push_back(parser.get());
        IndexGPU<Minimizer> index(parsers, 4, 1, read_ranges);
        Matcher matcher(index, 0);

        const std::vector<Anchor>& anchors = matcher.anchors();
        ASSERT_EQ(anchors.size(), 0u);
    }

    TEST(TestCudamapperMatcher, TwoReadsMultipleMiniminizers) {
        // >read_0
        // CATCAAG
        // >read_1
        // AAGCTA

        // CATCAAG
        // Central minimizers:
        // CATC: CAT, ATG, <ATC>, GAT
        // ATCA: <ATC>, GAT, TCA, TGA
        // TCAA: TCA, TGA, <CAA>, TTG
        // CAAG: CAA, TTG, <AAG>, CTT
        // front end minimizers: CAT, <ATG>
        // beck end minimizers: none
        // All minimizers: ATC(1f), CAA(3f), AAG(4f), ATG(0r)

        // AAGCTA
        // Central minimizers:
        // AAGC: <AAG>, CTT, AGC, GCT
        // AGCT: <AGC>, GCT, GCT, <AGC>
        // GCTA: GCT, <AGC>, CTA, TAG
        // Front end minimizers: none
        // Back end miniminers: <CTA>, TAG
        // All minimizers: AAG(0f), AGC(1f), AGC(2r), CTA(3f)

        // Anchor r0p4 - r1p0

        std::vector<std::pair<std::uint64_t, std::uint64_t>> read_ranges;
        const std::pair<std::uint64_t, std::uint64_t> query_range {0, std::numeric_limits<std::uint64_t>::max()};
        read_ranges.push_back(query_range);

        std::unique_ptr<FastaParser> parser = create_fasta_parser(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/catcaag_aagcta.fasta");
        std::vector<FastaParser*> parsers;
        parsers.push_back(parser.get());
        IndexGPU<Minimizer> index(parsers, 3, 2, read_ranges);
        Matcher matcher(index, 0);

        const std::vector<Anchor>& anchors = matcher.anchors();
        ASSERT_EQ(anchors.size(), 1u);
        EXPECT_EQ(anchors[0].query_read_id_, 0u);
        EXPECT_EQ(anchors[0].target_read_id_, 1u);
        EXPECT_EQ(anchors[0].query_position_in_read_, 4u);
        EXPECT_EQ(anchors[0].target_position_in_read_, 0u);
    }

    class TestIndex : public Index{
    public:
        // getters
        const std::vector<position_in_read_t>& positions_in_reads() const override { return positions_in_reads_; }
        const std::vector<read_id_t>& read_ids() const override { return read_ids_; }
        const std::vector<SketchElement::DirectionOfRepresentation>& directions_of_reads() const override { return directions_of_reads_; }
        std::uint64_t number_of_reads() const override { return number_of_reads_; }
        const std::vector<std::string>& read_id_to_read_name() const override { return read_id_to_read_name_; }
        const std::vector<std::uint32_t>& read_id_to_read_length() const override { return read_id_to_read_length_; }
        const std::vector<std::vector<Index::RepresentationToSketchElements>>& read_id_and_representation_to_sketch_elements() const override { return read_id_and_representation_to_sketch_elements_; }

        // setters
        void positions_in_reads(const std::vector<position_in_read_t>& val) { positions_in_reads_ = val; }
        void read_ids(const std::vector<read_id_t>& val) { read_ids_ = val; }
        void directions_of_reads(const std::vector<SketchElement::DirectionOfRepresentation>& val) { directions_of_reads_ = val; }
        void number_of_reads(std::uint64_t val) { number_of_reads_ = val; }
        void read_id_to_read_name(const std::vector<std::string>& val) { read_id_to_read_name_ = val; }
        void read_id_and_representation_to_sketch_elements( const std::vector<std::vector<RepresentationToSketchElements>>& val ) { read_id_and_representation_to_sketch_elements_ = val; }
        std::uint64_t minimum_representation() const override {return 0;};
        std::uint64_t maximum_representation() const override {return 1 << (15 * 2);}; //kmer_size = 15
        bool reached_end_of_input() const override {return false;};

    private:
        std::vector<position_in_read_t> positions_in_reads_;
        std::vector<read_id_t> read_ids_;
        std::vector<SketchElement::DirectionOfRepresentation> directions_of_reads_;
        std::uint64_t number_of_reads_;
        std::vector<std::string> read_id_to_read_name_;
        std::vector<std::uint32_t> read_id_to_read_length_;
        std::vector<std::vector<RepresentationToSketchElements>> read_id_and_representation_to_sketch_elements_;
    };

    TEST(TestCudamapperMatcher, CustomIndexTwoReads) {
        // Two reads, all minimizers have the same representation
        TestIndex test_index;

        std::vector<position_in_read_t> positions_in_reads(100);
        std::iota(std::begin(positions_in_reads), std::end(positions_in_reads), 0);
        test_index.positions_in_reads(positions_in_reads);

        std::vector<read_id_t> read_ids(100);
        std::fill(std::begin(read_ids), std::next(std::begin(read_ids), 50), 0);
        std::fill(std::next(std::begin(read_ids), 50), std::end(read_ids), 1);
        test_index.read_ids(read_ids);

        // no need for directions yet

        test_index.number_of_reads(2);

        // no need for read_id_to_read_name

        // pointers
        std::vector<std::vector<Index::RepresentationToSketchElements>> read_id_and_representation_to_sketch_elements(2);
        read_id_and_representation_to_sketch_elements[0].emplace_back(Index::RepresentationToSketchElements{0x23, {0,50}, {0,100}});
        read_id_and_representation_to_sketch_elements[1].emplace_back(Index::RepresentationToSketchElements{0x23, {50,50}, {0,100}});
        test_index.read_id_and_representation_to_sketch_elements(read_id_and_representation_to_sketch_elements);

        Matcher matcher(test_index, 0);

        const std::vector<Anchor>& anchors = matcher.anchors();
        ASSERT_EQ(anchors.size(), 2500u);

        for (std::size_t read_0_sketch_element = 0; read_0_sketch_element < 50; ++read_0_sketch_element) {
            for (std::size_t read_1_sketch_element = 0; read_1_sketch_element < 50; ++read_1_sketch_element) {
                ASSERT_EQ(anchors[read_0_sketch_element*50 + read_1_sketch_element].query_read_id_, 0u) << read_0_sketch_element << " " <<read_1_sketch_element;
                ASSERT_EQ(anchors[read_0_sketch_element*50 + read_1_sketch_element].target_read_id_, 1u) << read_0_sketch_element << " " <<read_1_sketch_element;
                ASSERT_EQ(anchors[read_0_sketch_element*50 + read_1_sketch_element].query_position_in_read_, read_0_sketch_element) << read_0_sketch_element << " " <<read_1_sketch_element;
                ASSERT_EQ(anchors[read_0_sketch_element*50 + read_1_sketch_element].target_position_in_read_, read_1_sketch_element + 50u) << read_0_sketch_element << " " <<read_1_sketch_element;
            }
        }
    }

    TEST(TestCudamapperMatcher, CustomIndexFourReads) {
        // Read 0:
        // representation 0: elems 0 - 49 (50)
        // representation 2: elems 50 - 69 (20)
        // representation 3: elems 70 - 199 (130)
        // representation 5: elems 200 - 269 (70)
        // Read 1:
        // representation 2: elems 0 - 29 (30)
        // representation 3: elems 30 - 99 (70)
        // representation 4: elems 100 - 159 (60)
        // representation 5: elems 160 - 199 (40)
        // Read 2:
        // representation 3: elems 0 - 99 (100)
        // representation 4: elems 100 - 199 (100)
        // representation 5: elems 200 - 299 (100)
        // Read 3:
        // representation 1: elems 0 - 79 (80)
        // representation 3: elems 80 - 159 (80)
        // representation 5: elems 160 - 239 (80)
        // representation 7: elems 240 - 319 (80)
        //
        // Total sketch elements: 270 + 200 + 300 + 320 = 1090
        //
        //        read 0 | read 1 | read 2 | read 3 
        // read 0   X    |   X    |   X    |   X
        // read 1  2,3,5 |   X    |   X    |   X
        // read 2  3,5   |  3,4,5 |   X    |   X
        // read 3  3,5   |  3,5   |  3,5   |   X
        //
        // Total anchors:
        // 0-1:2   0-1:3    0-1:5   0-2:3     0-2:5    0-3:3    0-3:5   1-2:3    1-2:4    1-2:5    1-3:3   1-3:5   2-3:3    2-3:5
        // 20*30 + 130*70 + 70*40 + 130*100 + 70*100 + 130*80 + 70*80 + 70*100 + 60*100 + 40*100 + 70*80 + 40*80 + 100*80 + 100*80 = 90300

        TestIndex test_index;

        // positions
        std::vector<position_in_read_t> positions_in_reads(1090);
        std::iota(std::begin(positions_in_reads), std::next(std::begin(positions_in_reads), 50), 0); // rep 0, read 0
        std::iota(std::next(std::begin(positions_in_reads), 50), std::next(std::begin(positions_in_reads), 50+80), 0); // rep 1, read 3
        std::iota(std::next(std::begin(positions_in_reads), 130), std::next(std::begin(positions_in_reads),130+20), 50); // rep 2, read 0
        std::iota(std::next(std::begin(positions_in_reads), 150), std::next(std::begin(positions_in_reads),150+30), 0); // rep 2, read 1
        std::iota(std::next(std::begin(positions_in_reads), 180), std::next(std::begin(positions_in_reads),180+130), 70); // rep 3, read 0
        std::iota(std::next(std::begin(positions_in_reads), 310), std::next(std::begin(positions_in_reads),310+70), 30); // rep 3, read 1
        std::iota(std::next(std::begin(positions_in_reads), 380), std::next(std::begin(positions_in_reads),380+100), 0); // rep 3, read 2
        std::iota(std::next(std::begin(positions_in_reads), 480), std::next(std::begin(positions_in_reads),480+80), 80); // rep 3, read 3
        std::iota(std::next(std::begin(positions_in_reads), 560), std::next(std::begin(positions_in_reads),560+60), 100); // rep 4, read 1
        std::iota(std::next(std::begin(positions_in_reads), 620), std::next(std::begin(positions_in_reads),620+100), 100); // rep 4, read 2
        std::iota(std::next(std::begin(positions_in_reads), 720), std::next(std::begin(positions_in_reads),720+70), 200); // rep 5, read 0
        std::iota(std::next(std::begin(positions_in_reads), 790), std::next(std::begin(positions_in_reads),790+40), 160); // rep 5, read 1
        std::iota(std::next(std::begin(positions_in_reads), 830), std::next(std::begin(positions_in_reads),830+100), 200); // rep 5, read 2
        std::iota(std::next(std::begin(positions_in_reads), 930), std::next(std::begin(positions_in_reads),930+80), 160); // rep 5, read 3
        std::iota(std::next(std::begin(positions_in_reads), 1010), std::next(std::begin(positions_in_reads),1010+80), 240); // rep 7, read 3
        test_index.positions_in_reads(positions_in_reads);

        // read_ids
        std::vector<read_id_t> read_ids(1090);
        std::fill(std::begin(read_ids), std::next(std::begin(read_ids),50), 0); // rep 0, read 0
        std::fill(std::next(std::begin(read_ids), 50), std::next(std::begin(read_ids),50+80), 3); // rep 1, read 3
        std::fill(std::next(std::begin(read_ids), 130), std::next(std::begin(read_ids),130+20), 0); // rep 2, read 0
        std::fill(std::next(std::begin(read_ids), 150), std::next(std::begin(read_ids),150+30), 1); // rep 2, read 1
        std::fill(std::next(std::begin(read_ids), 180), std::next(std::begin(read_ids),180+130), 0); // rep 3, read 0
        std::fill(std::next(std::begin(read_ids), 310), std::next(std::begin(read_ids),310+70), 1); // rep 3, read 1
        std::fill(std::next(std::begin(read_ids), 380), std::next(std::begin(read_ids),380+100), 2); // rep 3, read 2
        std::fill(std::next(std::begin(read_ids), 480), std::next(std::begin(read_ids),480+80), 3); // rep 3, read 3
        std::fill(std::next(std::begin(read_ids), 560), std::next(std::begin(read_ids),560+60), 1); // rep 4, read 1
        std::fill(std::next(std::begin(read_ids), 620), std::next(std::begin(read_ids),620+100), 2); // rep 4, read 2
        std::fill(std::next(std::begin(read_ids), 720), std::next(std::begin(read_ids),720+70), 0); // rep 5, read 0
        std::fill(std::next(std::begin(read_ids), 790), std::next(std::begin(read_ids),790+40), 1); // rep 5, read 1
        std::fill(std::next(std::begin(read_ids), 830), std::next(std::begin(read_ids),830+100), 2); // rep 5, read 2
        std::fill(std::next(std::begin(read_ids), 930), std::next(std::begin(read_ids),930+80), 3); // rep 5, read 3
        std::fill(std::next(std::begin(read_ids), 1010), std::next(std::begin(read_ids),1010+80), 3); // rep 7, read 3
        test_index.read_ids(read_ids);

        // no need for directions yet

        test_index.number_of_reads(4);

        // no need for read_id_to_read_name

        // pointers
        std::vector<std::vector<Index::RepresentationToSketchElements>> read_id_and_representation_to_sketch_elements(4);
        read_id_and_representation_to_sketch_elements[0].emplace_back(Index::RepresentationToSketchElements{0, {0,   50},  {0,   50}});
        read_id_and_representation_to_sketch_elements[3].emplace_back(Index::RepresentationToSketchElements{1, {50,  80},  {50,  80}});
        read_id_and_representation_to_sketch_elements[0].emplace_back(Index::RepresentationToSketchElements{2, {130, 20},  {130, 50}});
        read_id_and_representation_to_sketch_elements[1].emplace_back(Index::RepresentationToSketchElements{2, {150, 30},  {130, 50}});
        read_id_and_representation_to_sketch_elements[0].emplace_back(Index::RepresentationToSketchElements{3, {180, 130}, {180, 380}});
        read_id_and_representation_to_sketch_elements[1].emplace_back(Index::RepresentationToSketchElements{3, {310, 70},  {180, 380}});
        read_id_and_representation_to_sketch_elements[2].emplace_back(Index::RepresentationToSketchElements{3, {380, 100}, {180, 380}});
        read_id_and_representation_to_sketch_elements[3].emplace_back(Index::RepresentationToSketchElements{3, {480, 80},  {180, 380}});
        read_id_and_representation_to_sketch_elements[1].emplace_back(Index::RepresentationToSketchElements{4, {560, 60},  {560, 160}});
        read_id_and_representation_to_sketch_elements[2].emplace_back(Index::RepresentationToSketchElements{4, {620, 100}, {560, 160}});
        read_id_and_representation_to_sketch_elements[0].emplace_back(Index::RepresentationToSketchElements{5, {720, 70},  {720, 290}});
        read_id_and_representation_to_sketch_elements[1].emplace_back(Index::RepresentationToSketchElements{5, {790, 40},  {720, 290}});
        read_id_and_representation_to_sketch_elements[2].emplace_back(Index::RepresentationToSketchElements{5, {830, 100}, {720, 290}});
        read_id_and_representation_to_sketch_elements[3].emplace_back(Index::RepresentationToSketchElements{5, {930, 80},  {720, 290}});
        read_id_and_representation_to_sketch_elements[3].emplace_back(Index::RepresentationToSketchElements{7, {1010,80},  {1010,80}});
        test_index.read_id_and_representation_to_sketch_elements(read_id_and_representation_to_sketch_elements);

        Matcher matcher(test_index, 0);

        const std::vector<Anchor>& anchors = matcher.anchors();
        ASSERT_EQ(anchors.size(), 90300u);

        // Anchors are grouped by query read id and within that by representation (both in increasing order).
        // Assume q0p4t2p8 means anchor of read id 0 at position 4 and read id 2 at position 8.
        // Assume read 0 has 30 sketch elements with certain representation, read 1 40 and read 2 50.
        // Anchors for read 0 as query and that represtnation looks like this:
        // q0p0t1p0, q0p0t1p1 .. q0p0t1p39, q0p0t2p0, q0p0t2p1 ... q0p0t2p49, q0p1t1p0, q0p1t1p1 ... q0p1t1p39, q0p1t2p0 .. q0p1t2p49, q0p2p1p0 ...

        // read 0 - rep 2: 20
        // read 1 - rep 2: 30
        for (std::size_t query = 0; query < 20; ++query) {
            for (std::size_t target = 0; target < 30; ++target) {
                ASSERT_EQ(anchors[0 + query*30 + target].query_read_id_, 0u) << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[0 + query*30 + target].target_read_id_, 1u) << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[0 + query*30 + target].query_position_in_read_, query + 50u) << "query: " << query << ", target: " << target; // position_in_read for read 0 rep 2 starts from 50
                ASSERT_EQ(anchors[0 + query*30 + target].target_position_in_read_, target + 0u) << "query: " << query << ", target: " << target; // position_in_read for read 1 rep 2 starts from 0
            }
        }

        // read 0 - rep 3: 130
        // read 1 - rep 3: 70
        // read 2 - rep 3: 100
        // read 3 - rep 3: 80
        for (std::size_t query = 0; query < 130; ++query) { // block starts from 20*30 = 600
            for (std::size_t target = 0; target < 70; ++target) { // read 1 - no shift
                ASSERT_EQ(anchors[600 + query*250 + target].query_read_id_, 0u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[600 + query*250 + target].target_read_id_, 1u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[600 + query*250 + target].query_position_in_read_, query + 70u)  << "query: " << query << ", target: " << target; // position_in_read for read 0 rep 3 starts from 70
                ASSERT_EQ(anchors[600 + query*250 + target].target_position_in_read_, target + 30u)  << "query: " << query << ", target: " << target; // position_in_read for read 1 rep 3 starts from 30
            }
            for (std::size_t target = 0; target < 100; ++target) { // read 2 - shift 70 due to read 1
                ASSERT_EQ(anchors[600 + 70 + query*250 + target].query_read_id_, 0u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[600 + 70 + query*250 + target].target_read_id_, 2u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[600 + 70 + query*250 + target].query_position_in_read_, query + 70u)  << "query: " << query << ", target: " << target; // position_in_read for read 0 rep 3 starts from 70
                ASSERT_EQ(anchors[600 + 70 + query*250 + target].target_position_in_read_, target + 0u)  << "query: " << query << ", target: " << target; // position_in_read for read 2 rep 3 starts from 0
            }
            for (std::size_t target = 0; target < 80; ++target) { // read 8 - shift 170 due to read 1 and read 2
                ASSERT_EQ(anchors[600 + 170 + query*250 + target].query_read_id_, 0u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[600 + 170 + query*250 + target].target_read_id_, 3u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[600 + 170 + query*250 + target].query_position_in_read_, query + 70u)  << "query: " << query << ", target: " << target; // position_in_read for read 0 rep 3 starts from 70
                ASSERT_EQ(anchors[600 + 170 + query*250 + target].target_position_in_read_, target + 80u)  << "query: " << query << ", target: " << target; // position_in_read for read 3 rep 3 starts from 80
            }
        }

        // read 0 - rep 5: 70
        // read 1 - rep 5: 40
        // read 2 - rep 5: 100
        // read 3 - rep 5: 80
        for (std::size_t query = 0; query < 70; ++query) { // block starts from 600 + 130*250 = 33100
            for (std::size_t target = 0; target < 40; ++target) { // read 1 - no shift
                ASSERT_EQ(anchors[33100 + query*220 + target].query_read_id_, 0u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[33100 + query*220 + target].target_read_id_, 1u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[33100 + query*220 + target].query_position_in_read_, query + 200u)  << "query: " << query << ", target: " << target; // position_in_read for read 0 rep 5 starts from 200
                ASSERT_EQ(anchors[33100 + query*220 + target].target_position_in_read_, target + 160u)  << "query: " << query << ", target: " << target; // position_in_read for read 1 rep 5 starts from 160
            }
            for (std::size_t target = 0; target < 100; ++target) { // read 2 - shift 40 due to read 1
                ASSERT_EQ(anchors[33100 + 40 + query*220 + target].query_read_id_, 0u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[33100 + 40 + query*220 + target].target_read_id_, 2u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[33100 + 40 + query*220 + target].query_position_in_read_, query + 200u)  << "query: " << query << ", target: " << target; // position_in_read for read 0 rep 5 starts from 200
                ASSERT_EQ(anchors[33100 + 40 + query*220 + target].target_position_in_read_, target + 200u)  << "query: " << query << ", target: " << target; // position_in_read for read 2 rep 5 starts from 200
            }
            for (std::size_t target = 0; target < 80; ++target) { // read 8 - shift 140 due to read 1 and read 2
                ASSERT_EQ(anchors[33100 + 140 + query*220 + target].query_read_id_, 0u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[33100 + 140 + query*220 + target].target_read_id_, 3u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[33100 + 140 + query*220 + target].query_position_in_read_, query + 200u)  << "query: " << query << ", target: " << target; // position_in_read for read 0 rep 5 starts from 200
                ASSERT_EQ(anchors[33100 + 140 + query*220 + target].target_position_in_read_, target + 160u)  << "query: " << query << ", target: " << target; // position_in_read for read 3 rep 5 starts from 160
            }
        }

        // read 1 - rep 3: 70
        // read 2 - rep 3: 100
        // read 3 - rep 3: 80
        for (std::size_t query = 0; query < 70; ++query) { // block starts from 33100 + 70 * 220 = 48500
            for (std::size_t target = 0; target < 100; ++target) { // read 2 - no shift
                ASSERT_EQ(anchors[48500 + query*180 + target].query_read_id_, 1u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[48500 + query*180 + target].target_read_id_, 2u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[48500 + query*180 + target].query_position_in_read_, query + 30u)  << "query: " << query << ", target: " << target; // position_in_read for read 1 rep 3 starts from 30
                ASSERT_EQ(anchors[48500 + query*180 + target].target_position_in_read_, target + 0u)  << "query: " << query << ", target: " << target; // position_in_read for read 2 rep 3 starts from 0
            }
            for (std::size_t target = 0; target < 80; ++target) { // read 3 - shift 100 due to read 2
                ASSERT_EQ(anchors[48500 + 100 + query*180 + target].query_read_id_, 1u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[48500 + 100 + query*180 + target].target_read_id_, 3u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[48500 + 100 + query*180 + target].query_position_in_read_, query + 30u)  << "query: " << query << ", target: " << target; // position_in_read for read 1 rep 3 starts from 30
                ASSERT_EQ(anchors[48500 + 100 + query*180 + target].target_position_in_read_, target + 80u)  << "query: " << query << ", target: " << target; // position_in_read for read 2 rep 3 starts from 80
            }
        }

        // read 1 - rep 4: 60
        // read 2 - rep 4: 100
        for (std::size_t query = 0; query < 60; ++query) { // block starts from 48500 + 70 * 180 = 61100
            for (std::size_t target = 0; target < 100; ++target) { // read 2 - no shift
                ASSERT_EQ(anchors[61100 + query*100 + target].query_read_id_, 1u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[61100 + query*100 + target].target_read_id_, 2u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[61100 + query*100 + target].query_position_in_read_, query + 100u)  << "query: " << query << ", target: " << target; // position_in_read for read 1 rep 4 starts from 100
                ASSERT_EQ(anchors[61100 + query*100 + target].target_position_in_read_, target + 100u)  << "query: " << query << ", target: " << target; // position_in_read for read 2 rep 4 starts from 100
            }
        }

        // read 1 - rep 5: 40
        // read 2 - rep 5: 100
        // read 3 - rep 5: 80
        for (std::size_t query = 0; query < 40; ++query) { // block starts from 61100 + 60 * 100 = 67100
            for (std::size_t target = 0; target < 100; ++target) { // read 2 - no shift
                ASSERT_EQ(anchors[67100 + query*180 + target].query_read_id_, 1u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[67100 + query*180 + target].target_read_id_, 2u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[67100 + query*180 + target].query_position_in_read_, query + 160u)  << "query: " << query << ", target: " << target; // position_in_read for read 1 rep 5 starts from 160
                ASSERT_EQ(anchors[67100 + query*180 + target].target_position_in_read_, target + 200u)  << "query: " << query << ", target: " << target; // position_in_read for read 2 rep 5 starts from 200
            }
            for (std::size_t target = 0; target < 80; ++target) { // read 3 - shift 100 due to read 2
                ASSERT_EQ(anchors[67100 + 100 + query*180 + target].query_read_id_, 1u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[67100 + 100 + query*180 + target].target_read_id_, 3u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[67100 + 100 + query*180 + target].query_position_in_read_, query + 160u)  << "query: " << query << ", target: " << target; // position_in_read for read 1 rep 5 starts from 160
                ASSERT_EQ(anchors[67100 + 100 + query*180 + target].target_position_in_read_, target + 160u)  << "query: " << query << ", target: " << target; // position_in_read for read 2 rep 5 starts from 160
            }
        }

        // read 2 - rep 3: 100
        // read 3 - rep 3: 80
        for (std::size_t query = 0; query < 100; ++query) { // block starts from 67100 + 40 * 180 = 74300
            for (std::size_t target = 0; target < 80; ++target) { // read 3 - no shift
                ASSERT_EQ(anchors[74300 + query*80 + target].query_read_id_, 2u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[74300 + query*80 + target].target_read_id_, 3u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[74300 + query*80 + target].query_position_in_read_, query + 0u)  << "query: " << query << ", target: " << target; // position_in_read for read 2 rep 3 starts from 0
                ASSERT_EQ(anchors[74300 + query*80 + target].target_position_in_read_, target + 80u)  << "query: " << query << ", target: " << target; // position_in_read for read 3 rep 3 starts from 80
            }
        }

        // read 2 - rep 5: 100
        // read 3 - rep 5: 80
        for (std::size_t query = 0; query < 100; ++query) { // block starts from 74300 + 100*800 = 82300
            for (std::size_t target = 0; target < 80; ++target) { // read 3 - no shift
                ASSERT_EQ(anchors[82300 + query*80 + target].query_read_id_, 2u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[82300 + query*80 + target].target_read_id_, 3u)  << "query: " << query << ", target: " << target;
                ASSERT_EQ(anchors[82300 + query*80 + target].query_position_in_read_, query + 200u)  << "query: " << query << ", target: " << target; // position_in_read for read 2 rep 5 starts from 200
                ASSERT_EQ(anchors[82300 + query*80 + target].target_position_in_read_, target + 160u)  << "query: " << query << ", target: " << target; // position_in_read for read 3 rep 5 starts from 160
            }
        }
    }
}
}
