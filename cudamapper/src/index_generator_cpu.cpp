/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <algorithm>
#include <deque>
#include <limits>
#include <string>
#include <iostream>
#include <utility>
#include "bioparser/bioparser.hpp"
#include "bioparser_sequence.hpp"
#include <claragenomics/logging/logging.hpp>
#include "index_generator_cpu.hpp"

namespace claragenomics {

    IndexGeneratorCPU::IndexGeneratorCPU(const std::string& query_filename, std::uint64_t minimizer_size, std::uint64_t window_size)
    : minimizer_size_(minimizer_size), window_size_(window_size), index_()
    {
        generate_index(query_filename);
    }

    std::uint64_t IndexGeneratorCPU::minimizer_size() const { return minimizer_size_; }

    std::uint64_t IndexGeneratorCPU::window_size() const { return window_size_; }

    const std::map<representation_t, std::vector<std::unique_ptr<SketchElement>>>& IndexGeneratorCPU::representations_to_sketch_elements() const { return index_; };

    const std::vector<std::string>& IndexGeneratorCPU::read_id_to_read_name() const { return read_id_to_read_name_; };

    const std::vector<std::uint32_t>& IndexGeneratorCPU::read_id_to_read_length() const { return read_id_to_read_length_; };

    std::uint64_t IndexGeneratorCPU::number_of_reads() const { return number_of_reads_; }

    void IndexGeneratorCPU::generate_index(const std::string &query_filename) {

        std::unique_ptr <bioparser::Parser<BioParserSequence>> query_parser = nullptr;

        auto is_suffix = [](const std::string &src, const std::string &suffix) -> bool {
            if (src.size() < suffix.size()) {
                return false;
            }
            return src.compare(src.size() - suffix.size(), suffix.size(), suffix) == 0;
        };

        if (is_suffix(query_filename, ".fasta") || is_suffix(query_filename, ".fa") ||
            is_suffix(query_filename, ".fasta.gz") || is_suffix(query_filename, ".fa.gz")) {
            CGA_LOG_INFO("Getting Query data");
            query_parser = bioparser::createParser<bioparser::FastaParser, BioParserSequence>(
                    query_filename);
        }

        //read the query file:
        std::vector <std::unique_ptr<BioParserSequence>> fasta_objects;
        query_parser->parse(fasta_objects, std::numeric_limits<std::uint64_t>::max());

        for (std::uint64_t read_id = 0; read_id < fasta_objects.size(); ++read_id) {
            read_id_to_read_name_.push_back(fasta_objects[read_id]->name());
            read_id_to_read_length_.push_back(fasta_objects[read_id]->data().length());
            add_read_to_index(*fasta_objects[read_id], read_id);
        }
        number_of_reads_ = fasta_objects.size();
    }

    void IndexGeneratorCPU::add_read_to_index(const Sequence& read, const read_id_t read_id) {
        // check if sequence fits at least one window
        if (read.data().size() < window_size_ + minimizer_size_ - 1) {
            // TODO: as long as the read fits at least one minimizer process it, but log it wasn't long enough
            return;
        }

        find_minimizers(read, read_id);
    }

    void IndexGeneratorCPU::find_minimizers(const Sequence& read, const read_id_t read_id) {
        representation_t current_minimizer_representation = std::numeric_limits<representation_t>::max();
        std::deque<Minimizer::RepresentationAndDirection> kmers_in_window; // values of all kmers in the current window
        std::deque<std::pair<position_in_read_t, Minimizer::DirectionOfRepresentation>> minimizer_pos; // positions of kmers that are minimzers in the current window and their directions
        const std::string& read_data = read.data();

        // Minimizers of first/last window migh be the same as some of end minimizers. These values will be needed for calculating the end minimizers
        representation_t first_window_minimizer_representation = 0;
        representation_t last_window_minimizer_representation = 0;

        // fill the initial window
        for (std::size_t kmer_in_window_index = 0; kmer_in_window_index < window_size_; ++kmer_in_window_index) {
            kmers_in_window.push_back(Minimizer::kmer_to_representation(read_data, kmer_in_window_index, minimizer_size_));
            if (kmers_in_window.back().representation_ == current_minimizer_representation) { // if this kmer is equeal to the current minimizer add it to the list of positions of that minimizer
                minimizer_pos.emplace_back(kmer_in_window_index, kmers_in_window.back().direction_);
            } else if (kmers_in_window.back().representation_ < current_minimizer_representation) { // if it is smaller than the current minimizer clear the list and make it the new minimizer
                current_minimizer_representation = kmers_in_window.back().representation_; // minimizer gets the value of the newest kmer as it is smaller than the previous minimizer
                minimizer_pos.clear(); // there is a new minimizer, clear the positions of the old minimizer
                minimizer_pos.emplace_back(kmer_in_window_index, kmers_in_window.back().direction_); // save the position of the new minimizer
            }
        }
        // add all position of the minimizer of the first window
        for (const std::pair<position_in_read_t, Minimizer::DirectionOfRepresentation>& pos : minimizer_pos) {
            index_[current_minimizer_representation].emplace_back(std::make_unique<Minimizer>(current_minimizer_representation, pos.first, pos.second, read_id));
        }
        first_window_minimizer_representation = current_minimizer_representation;

        // move the window by one basepair in each step
        // window 0 was processed in the previous step
        for (std::uint64_t window_num = 1; window_num <= read_data.size() - (window_size_ + minimizer_size_ - 1); ++window_num) {
            // remove the kmer which does not belong to the window anymore and add the new one
            kmers_in_window.pop_front();
            kmers_in_window.push_back(Minimizer::kmer_to_representation(read_data, window_num + window_size_ - 1, minimizer_size_)); // last kmer in that window
            // if the removed kmer was the minimizer find the new minimizer (unless another minimizer of the same value exists inside the window)
            if (minimizer_pos[0].first == window_num - 1) { // oldest kmer's index is always equal to current window_num - 1
                minimizer_pos.pop_front(); // remove the occurence of the minimizer
                if (minimizer_pos.empty()) { // removed kmer was the only minimzer -> find a new one
                    current_minimizer_representation = std::numeric_limits<std::uint64_t>::max();
                    // TODO: this happens fairly often, compare this solution with using a heap (which would have to updated for each added/removed kmer)
                    for (std::size_t i = 0; i < window_size_; ++i) {
                        if (kmers_in_window[i].representation_ == current_minimizer_representation) {
                            minimizer_pos.emplace_back(window_num + i, kmers_in_window[i].direction_);
                        } else if (kmers_in_window[i].representation_ < current_minimizer_representation) {
                            minimizer_pos.clear();
                            current_minimizer_representation = kmers_in_window[i].representation_;
                            minimizer_pos.emplace_back(window_num + i, kmers_in_window[i].direction_);
                        }
                    }
                    for (const std::pair<position_in_read_t, Minimizer::DirectionOfRepresentation>& pos : minimizer_pos) {
                        index_[current_minimizer_representation].emplace_back(std::make_unique<Minimizer>(current_minimizer_representation, pos.first, pos.second, read_id));
                    }
                } else { // there are other kmers with that value, proceed as if the oldest element was not not the smallest one
                    if (kmers_in_window.back().representation_ == current_minimizer_representation) {
                        minimizer_pos.emplace_back(window_num + minimizer_size_ - 1, kmers_in_window.back().direction_);
                        index_[current_minimizer_representation].emplace_back(std::make_unique<Minimizer>(current_minimizer_representation, window_num + window_size_ - 1, kmers_in_window.back().direction_, read_id));
                    } else if (kmers_in_window.back().representation_ < current_minimizer_representation) {
                        minimizer_pos.clear();
                        current_minimizer_representation = kmers_in_window.back().representation_;
                        minimizer_pos.emplace_back(window_num + minimizer_size_ - 1, kmers_in_window.back().direction_);
                        index_[current_minimizer_representation].emplace_back(std::make_unique<Minimizer>(current_minimizer_representation, window_num + window_size_ - 1, kmers_in_window.back().direction_, read_id));
                    }
                }
            } else {  // oldest kmer was not the minimizer
                if (kmers_in_window.back().representation_ == current_minimizer_representation) {
                    minimizer_pos.emplace_back(window_num + minimizer_size_ - 1, kmers_in_window.back().direction_);
                    index_[current_minimizer_representation].emplace_back(std::make_unique<Minimizer>(current_minimizer_representation, window_num + window_size_ - 1, kmers_in_window.back().direction_, read_id));
                } else if (kmers_in_window.back().representation_ < current_minimizer_representation) {
                    minimizer_pos.clear();
                    current_minimizer_representation = kmers_in_window.back().representation_;
                    minimizer_pos.emplace_back(window_num + minimizer_size_ - 1, kmers_in_window.back().direction_);
                    index_[current_minimizer_representation].emplace_back(std::make_unique<Minimizer>(current_minimizer_representation, window_num + window_size_ - 1, kmers_in_window.back().direction_, read_id));
                }
            }
        }
        last_window_minimizer_representation = current_minimizer_representation;

        find_end_minimizers(read, read_id, first_window_minimizer_representation, last_window_minimizer_representation);
    }

    void IndexGeneratorCPU::find_end_minimizers(const Sequence& read, const read_id_t read_id, const representation_t first_window_minimizer_representation, const representation_t last_window_minimizer_representation) {
        const std::string& read_data = read.data();

        // End minimizers are minimizers of windows with window_size_end = [1..window_size-1],
        // where each window starts at the begining/ends at the end of the read (front/back end miniminizers)
        // (discussion for front end minimizers, back end minimimizer are equivalent)
        // For window_size_end=1 minimizer is the representation of the first kmer.
        // For window_size_end=2 minimizer can either come from the first or the second kmer. If it comes from the first kmer
        // then it has already been added by window_size_end=1. If it comes from the second kmer then the representation of the second kmer
        // has to be smaller or equal than the representation of the first kmer.
        // This means that as the window_size_end is increased only the newly added kmer has to be checked and its representation added
        // if it is smaller or equal than the minimizer of the previous window.
        // "Central" minimizer for window_size has alread been added. If a kmer with representation equal to the minimizer of that window is found
        // that means that no kmer with representation smaller than that value will be found for window_size_end < window_size

        // "front" end minimizers
        representation_t current_minimizer_representation = std::numeric_limits<std::uint64_t>::max();
        for (std::size_t current_window_size = 1; current_window_size < window_size_; ++current_window_size) {
            Minimizer::RepresentationAndDirection new_kmer = Minimizer::kmer_to_representation(read_data, current_window_size-1, minimizer_size_);
            if (new_kmer.representation_ == first_window_minimizer_representation) { // minimizer already added by the first "central" minimizer window
                break;
            }
            if (new_kmer.representation_ <= current_minimizer_representation) {
                if (new_kmer.representation_ < current_minimizer_representation) {
                    current_minimizer_representation = new_kmer.representation_;
                }
                index_[current_minimizer_representation].emplace_back(std::make_unique<Minimizer>(current_minimizer_representation, current_window_size-1, new_kmer.direction_, read_id));
            }
        }


        current_minimizer_representation = std::numeric_limits<std::uint64_t>::max();
        for (std::size_t current_window_size = 1; current_window_size < window_size_; ++current_window_size) {
            std::size_t kmer_begin_position = read_data.size() - minimizer_size_ - current_window_size + 1;
            Minimizer::RepresentationAndDirection new_kmer = Minimizer::kmer_to_representation(read_data, kmer_begin_position, minimizer_size_);
            if (new_kmer.representation_ == last_window_minimizer_representation) { // minimizer already added by the first "central" minimizer window
                break;
            }
            if (new_kmer.representation_ <= current_minimizer_representation) {
                if (new_kmer.representation_ < current_minimizer_representation) {
                    current_minimizer_representation = new_kmer.representation_;
                }
                index_[current_minimizer_representation].emplace_back(std::make_unique<Minimizer>(current_minimizer_representation, kmer_begin_position, new_kmer.direction_, read_id));
            }
        }
    }
}
