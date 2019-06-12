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
#include <string>
#include <iostream>
#include <utility>
#include "bioparser/bioparser.hpp"
#include "bioparser_sequence.hpp"
#include <logging/logging.hpp>
#include "index_generator_cpu.hpp"

namespace genomeworks {

//    typedef std::pair<uint64_t, std::unique_ptr<Minimizer>> MinPair;

    IndexGeneratorCPU::IndexGeneratorCPU(const std::string& query_filename, std::uint64_t minimizer_size, std::uint64_t window_size)
    : minimizer_size_(minimizer_size), window_size_(window_size), index_()
    {
        generate_index(query_filename);
    }

    std::uint64_t IndexGeneratorCPU::minimizer_size() const { return minimizer_size_; }

    std::uint64_t IndexGeneratorCPU::window_size() const { return window_size_; }

    const std::map<representation_t, std::vector<std::unique_ptr<SketchElement>>>& IndexGeneratorCPU::representation_to_sketch_elements() const { return index_; };

    const std::vector<std::string>& IndexGeneratorCPU::read_id_to_read_name() const { return read_id_to_read_name_; };

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
            GW_LOG_INFO("Getting Query data");
            query_parser = bioparser::createParser<bioparser::FastaParser, BioParserSequence>(
                    query_filename);
        }

        //read the query file:
        std::vector <std::unique_ptr<BioParserSequence>> fasta_objects;
        query_parser->parse(fasta_objects, -1);

        for (std::uint64_t read_id = 0; read_id < fasta_objects.size(); ++read_id) {
            read_id_to_read_name_.push_back(fasta_objects[read_id]->name());
            add_read_to_index(*fasta_objects[read_id], read_id);
        }
    }

    void IndexGeneratorCPU::add_read_to_index(const Sequence& read, read_id_t read_id) {
        // check if sequence fits at least one window
        if (read.data().size() < window_size_ + minimizer_size_ - 1) {
            // TODO: as long as the read fits at least one minimizer process it, but log it wasn't long enough
            return;
        }

        find_central_minimizers(read, read_id);

        //find_end_minimizers(sequence, sequence_id);
    }

    void IndexGeneratorCPU::find_central_minimizers(const Sequence& sequence, std::uint64_t sequence_id) {

    }

/*    void IndexGeneratorCPU::find_central_minimizers(const Sequence& sequence, std::uint64_t sequence_id) {
        std::uint64_t minimizer = std::numeric_limits<std::uint64_t>::max(); // value of the current minimizer
        // These deques are going to be resized all the time. Think about using ring buffers if this limits the performance
        std::deque<Minimizer::RepresentationAndDirection> window; // values of all kmers in the current window
        std::deque<std::pair<std::size_t, Minimizer::DirectionOfRepresentation>> minimizer_pos; // positions of kmers that are minimzers in the current window and their directions
        const std::string& sequence_data = sequence.data();

        // fill the initial window
        for (std::size_t vector_pos = 0; vector_pos < window_size_; ++vector_pos) {
            window.push_back(Minimizer::kmer_to_integer_representation(sequence_data, vector_pos, minimizer_size_));
            if (window.back().representation_ == minimizer) { // if this kmer is equeal to the current minimizer add it to the list of positions of that minimizer
                minimizer_pos.emplace_back(vector_pos, window.back().direction_);
            } else if (window.back().representation_ < minimizer) { // if it is smaller than the current minimizer clear the list and make it the new minimizer
                minimizer = window.back().representation_; // minimizer gets the value of the newest kmer as it is smaller than the previous minimizer
                minimizer_pos.clear(); // there is a new minimizer, clear the positions of the old minimizer
                minimizer_pos.emplace_back(vector_pos, window.back().direction_); // save the position of the new minimizer
            }
        }
        // add all position of the minimizer of the first window
        for (const std::pair<std::uint64_t, Minimizer::DirectionOfRepresentation>& pos : minimizer_pos) {
            index_.emplace(MinPair(minimizer, std::make_unique<Minimizer>(minimizer, pos.first, pos.second, sequence_id)));
        }

        // move the window by one basepair in each step
        for (std::uint64_t window_num = 1; window_num <= sequence_data.size() - (window_size_ + minimizer_size_ - 1); ++window_num) {
            // remove the kmer which does not belong to the window anymore and add the new one
            window.pop_front();
            window.push_back(Minimizer::kmer_to_integer_representation(sequence_data, window_num + window_size_ - 1, minimizer_size_)); // last kmer in that window
            // if the removed kmer was the minimizer find the new minimizer (unless another minimizer of the same value exists inside the window)
            if (minimizer_pos[0].first == window_num - 1) { // oldest kmer's index is always equal to current window_num - 1
                minimizer_pos.pop_front(); // remove the occurence of the minimizer
                if (minimizer_pos.empty()) { // removed kmer was the only minimzer -> find a new one
                    minimizer = std::numeric_limits<std::uint64_t>::max();
                    for (std::size_t i = 0; i < window.size(); ++i) {
                        if (window[i].representation_ == minimizer) {
                            minimizer_pos.emplace_back(window_num + i, window[i].direction_);
                        } else if (window[i].representation_ < minimizer) {
                            minimizer_pos.clear();
                            minimizer = window[i].representation_;
                            minimizer_pos.emplace_back(window_num + i, window[i].direction_);
                        }
                    }
                    for (const std::pair<std::uint64_t, Minimizer::DirectionOfRepresentation>& pos : minimizer_pos) {
                        index_.emplace(MinPair(minimizer, std::make_unique<Minimizer>(minimizer, pos.first, pos.second, sequence_id)));
                    }
                } else { // there are other kmers with that value, proceed as if the oldest element was not not the smallest one
                    if (window.back().representation_ == minimizer) {
                        minimizer_pos.emplace_back(window_num + minimizer_size_ - 1, window.back().direction_);
                        index_.emplace(MinPair(minimizer, std::make_unique<Minimizer>(minimizer, window_num + minimizer_size_ - 1, window.back().direction_, sequence_id)));
                    } else if (window.back().representation_ < minimizer) {
                        minimizer_pos.clear();
                        minimizer = window.back().representation_;
                        minimizer_pos.emplace_back(window_num + minimizer_size_ - 1, window.back().direction_);
                        index_.emplace(MinPair(minimizer, std::make_unique<Minimizer>(minimizer, window_num + minimizer_size_ - 1, window.back().direction_, sequence_id)));
                    }
                }
            } else {  // oldest kmer was not the minimizer
                if (window.back().representation_ == minimizer) {
                    minimizer_pos.emplace_back(window_num + minimizer_size_ - 1, window.back().direction_);
                    index_.emplace(MinPair(minimizer, std::make_unique<Minimizer>(minimizer, window_num + minimizer_size_ - 1, window.back().direction_, sequence_id)));
                } else if (window.back().representation_ < minimizer) {
                    minimizer_pos.clear();
                    minimizer = window.back().representation_;
                    minimizer_pos.emplace_back(window_num + minimizer_size_ - 1, window.back().direction_);
                    index_.emplace(MinPair(minimizer, std::make_unique<Minimizer>(minimizer, window_num + minimizer_size_ - 1, window.back().direction_, sequence_id)));
                }
            }
        }
    }

    void IndexGeneratorCPU::find_end_minimizers(const Sequence& sequence, std::uint64_t sequence_id) {
        const std::string& sequence_data = sequence.data();

        // End minimizers are found by increasing the window size and keeping the same minimizer length.
        // This means that the window of size w+1 will either have the same or smaller minimizer than the window of size w
        // (or by induction any other smaller window)
        // It is thus necessary only to check new kmer. If it is a new minimizer it's guaraneed that that kmer is not present in the rest of the window

        // "front" end minimizers
        std::uint64_t minimizer = std::numeric_limits<std::uint64_t>::max();
        auto existing_positions = index_.equal_range(minimizer);
        for (std::size_t i = 0; i < window_size_; ++i) {
            Minimizer::RepresentationAndDirection new_kmer = Minimizer::kmer_to_integer_representation(sequence_data, i, minimizer_size_);
            if (new_kmer.representation_ <= minimizer) {
                if (new_kmer.representation_ < minimizer) {
                    minimizer = new_kmer.representation_;
                    existing_positions = index_.equal_range(minimizer);
                }
                // add minimizer (if not already present)
                if (std::none_of(existing_positions.first, existing_positions.second, [i, sequence_id](const decltype(index_)::value_type& a) {return a.second->position() == i && a.second->sequence_id() == sequence_id;})) {
                    index_.emplace(MinPair(minimizer, std::make_unique<Minimizer>(minimizer, i, new_kmer.direction_, sequence_id)));
                }
            }
        }

        // "back" end minimizers
        minimizer = std::numeric_limits<std::uint64_t>::max();
        existing_positions = index_.equal_range(minimizer);
        for(std::size_t i = 0; i < window_size_; ++i) {
            std::size_t kmer_position = sequence_data.size() - minimizer_size_ - i;
            Minimizer::RepresentationAndDirection new_kmer = Minimizer::kmer_to_integer_representation(sequence_data, kmer_position, minimizer_size_);
            if (new_kmer.representation_ <= minimizer) {
                if (new_kmer.representation_ < minimizer) {
                    minimizer = new_kmer.representation_;
                    existing_positions = index_.equal_range(minimizer);
                }
                // add minimizer (if not already present)
                if (std::none_of(existing_positions.first, existing_positions.second, [kmer_position, sequence_id](const decltype(index_)::value_type& a) {return a.second->position() == kmer_position && a.second->sequence_id() == sequence_id;})) {
                    index_.emplace(MinPair(minimizer, std::make_unique<Minimizer>(minimizer, kmer_position, new_kmer.direction_, sequence_id)));
                }
            }
        }
    }*/
}
