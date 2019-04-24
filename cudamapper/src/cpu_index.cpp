#include <algorithm>
#include <deque>
#include <string>
#include <iostream>
#include "bioparser/bioparser.hpp"
#include "bioparser_sequence.hpp"
#include "cpu_index.hpp"
#include "utils.cpp"

namespace genomeworks {

    typedef std::pair<uint64_t, Minimizer> MinPair;

    CPUIndex::CPUIndex(std::uint64_t minimizer_size, std::uint64_t window_size)
    : minimizer_size_(minimizer_size), window_size_(window_size)
    {}

    void CPUIndex::generate_index(const std::string &query_filename) {

        std::unique_ptr <bioparser::Parser<BioParserSequence>> query_parser = nullptr;

        auto is_suffix = [](const std::string &src, const std::string &suffix) -> bool {
            if (src.size() < suffix.size()) {
                return false;
            }
            return src.compare(src.size() - suffix.size(), suffix.size(), suffix) == 0;
        };

        if (is_suffix(query_filename, ".fasta") || is_suffix(query_filename, ".fa") ||
            is_suffix(query_filename, ".fasta.gz") || is_suffix(query_filename, ".fa.gz")) {
            std::cout << "Getting Query data" << std::endl;
            query_parser = bioparser::createParser<bioparser::FastaParser, BioParserSequence>(
                    query_filename);
        }

        //read the query file:
        std::vector <std::unique_ptr<BioParserSequence>> fasta_objects;
        query_parser->parse(fasta_objects, -1);

        for (std::uint64_t seq_id = 0; seq_id < fasta_objects.size(); ++seq_id) {
            add_sequence_to_index(*fasta_objects[seq_id], seq_id);
        }
    }

    void CPUIndex::add_sequence_to_index(const Sequence& sequence, std::uint64_t sequence_id) {
        // check if sequence fits at least one window
        if (sequence.data().size() < window_size_ + minimizer_size_ - 1) {
            return;
        }

        find_central_minimizers(sequence, sequence_id);

        find_end_minimizers(sequence, sequence_id);
    }

    void CPUIndex::find_central_minimizers(const Sequence& sequence, std::uint64_t sequence_id) {
        std::uint64_t minimizer = std::numeric_limits<std::uint64_t>::max(); // value of the current minimizer
        // These deques are going to be resized all the time. Think about using ring buffers if this limits the performance
        std::deque<std::uint64_t> window; // values of all kmers in the current window
        std::deque<std::size_t> minimizer_pos; // positions of kmers that are minimzers in the current window
        const std::string& sequence_data = sequence.data();

        // fill the initial window
        for (std::size_t vector_pos = 0; vector_pos < window_size_; ++vector_pos) {
            window.push_back(kmer_to_representation(sequence_data, vector_pos, minimizer_size_));
            if (window.back() == minimizer) { // if this kmer is equeal to the current minimizer add it to the list of positions of that minimizer
                minimizer_pos.push_back(vector_pos);
            } else if (window.back() < minimizer) { // if it is smaller than the current minimizer clear the list and make it the new minimizer
                minimizer = window.back(); // minimizer gets the value of the newest kmer as it is smaller than the previous minimizer
                minimizer_pos.clear(); // there is a new minimizer, clear the positions of the old minimizer
                minimizer_pos.push_back(vector_pos); // save the position of the new minimizer
            }
        }
        // add all position of the minimizer of the first window
        for (const std::size_t& pos : minimizer_pos) {
            index_.emplace(MinPair(minimizer, Minimizer(minimizer, pos, sequence_id)));
        }

        // move the window by one basepair in each step
        for (std::uint64_t window_num = 1; window_num <= sequence_data.size() - (window_size_ + minimizer_size_ - 1); ++window_num) {
            // remove the kmer which does not belong to the window anymore and add the new one
            window.pop_front();
            window.push_back(kmer_to_representation(sequence_data, window_num + window_size_ - 1, minimizer_size_)); // last kmer in that window
            // if the removed kmer was the minimizer find the new minimizer (unless another minimizer of the same value exists inside the window)
            if (minimizer_pos[0] == window_num - 1) { // oldest kmer's index is always equal to current window_num - 1
                minimizer_pos.pop_front(); // remove the occurence of the minimizer
                if (minimizer_pos.empty()) { // removed kmer was the only minimzer -> find a new one
                    minimizer = std::numeric_limits<std::uint64_t>::max();
                    for (std::size_t i = 0; i < window.size(); ++i) {
                        if (window[i] == minimizer) {
                            minimizer_pos.push_back(window_num + i);
                        } else if (window[i] < minimizer) {
                            minimizer_pos.clear();
                            minimizer = window[i];
                            minimizer_pos.push_back(window_num + i);
                        }
                    }
                    for (std::size_t m_pos : minimizer_pos) {
                        index_.emplace(MinPair(minimizer, Minimizer(minimizer, m_pos, sequence_id)));
                    }
                } else { // there are other kmers with that value, proceed as if the oldest element was not not the smallest one
                    if (window.back() == minimizer) {
                        minimizer_pos.push_back(window_num + minimizer_size_ - 1);
                        index_.emplace(MinPair(minimizer, Minimizer(minimizer, window_num + minimizer_size_ - 1, sequence_id)));
                    } else if (window.back() < minimizer) {
                        minimizer_pos.clear();
                        minimizer = window.back();
                        minimizer_pos.push_back(window_num + minimizer_size_ - 1);
                        index_.emplace(MinPair(minimizer, Minimizer(minimizer, window_num + minimizer_size_ - 1, sequence_id)));
                    }
                }
            } else {  // oldest kmer was not the minimizer
                if (window.back() == minimizer) {
                    minimizer_pos.push_back(window_num + minimizer_size_ - 1);
                    index_.emplace(MinPair(minimizer, Minimizer(minimizer, window_num + minimizer_size_ - 1, sequence_id)));
                } else if (window.back() < minimizer) {
                    minimizer_pos.clear();
                    minimizer = window.back();
                    minimizer_pos.push_back(window_num + minimizer_size_ - 1);
                    index_.emplace(MinPair(minimizer, Minimizer(minimizer, window_num + minimizer_size_ - 1, sequence_id)));
                }
            }
        }
    }

    void CPUIndex::find_end_minimizers(const Sequence& sequence, std::uint64_t sequence_id) {
        std::deque<std::uint64_t> window;
        const std::string& sequence_data = sequence.data();

        // End minimizers are found by increasing the window size and keeping the same minimizer length.
        // This means that the window of size w+1 will either have the same or smaller minimizer than the window of size w
        // (or by induction any other smaller window)
        // It is thus necessary only to check new kmer. If it is a new minimizer it's guaraneed that that kmer is not present in the rest of the window

        // "front" end minimizers
        std::uint64_t minimizer = std::numeric_limits<std::uint64_t>::max();
        auto existing_positions = index_.equal_range(minimizer);
        for (std::size_t i = 0; i < window_size_; ++i) {
            std::uint64_t new_minimizer = kmer_to_representation(sequence_data, i, minimizer_size_);
            if (new_minimizer <= minimizer) {
                if (new_minimizer < minimizer) {
                    minimizer = new_minimizer;
                    existing_positions = index_.equal_range(minimizer);
                }
                // add minimizer (if not already present)
                if (std::none_of(existing_positions.first, existing_positions.second, [i, sequence_id](const decltype(index_)::value_type& a) {return a.second.position() == i && a.second.sequence_id() == sequence_id;})) {
                    index_.emplace(MinPair(minimizer, Minimizer(minimizer, i, sequence_id)));
                }
            }
        }

        // "back" end minimizers
        minimizer = std::numeric_limits<std::uint64_t>::max();
        existing_positions = index_.equal_range(minimizer);
        for(std::size_t i = 0; i < window_size_; ++i) {
            std::size_t kmer_position = sequence_data.size() - minimizer_size_ - i;
            std::uint64_t new_minimizer = kmer_to_representation(sequence_data, kmer_position, minimizer_size_);
            if (new_minimizer <= minimizer) {
                if (new_minimizer < minimizer) {
                    minimizer = new_minimizer;
                    existing_positions = index_.equal_range(minimizer);
                }
                // add minimizer (if not already present)
                if (std::none_of(existing_positions.first, existing_positions.second, [kmer_position, sequence_id](const decltype(index_)::value_type& a) {return a.second.position() == kmer_position && a.second.sequence_id() == sequence_id;})) {
                    index_.emplace(MinPair(minimizer, Minimizer(minimizer, kmer_position, sequence_id)));
                }
            }
        }
    }

    std::uint64_t CPUIndex::minimizer_size() const { return minimizer_size_; }

    std::uint64_t CPUIndex::window_size() const { return window_size_; }

    const std::unordered_multimap<std::uint64_t, Minimizer>& CPUIndex::index() const { return index_; }
}
