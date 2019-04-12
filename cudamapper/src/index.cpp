#include <cudamapper/index.hpp>
#include <string>
#include <iostream>
#include "bioparser/bioparser.hpp"
#include "cudamapper/sequence.hpp"

namespace cudamapper {
    Index::Index() {}

    void Index::generate_index(char *fasta_filename) {

        std::string query_filename = std::string(fasta_filename);

        std::unique_ptr <bioparser::Parser<Sequence>> query_parser = nullptr;

        auto is_suffix = [](const std::string &src, const std::string &suffix) -> bool {
            if (src.size() < suffix.size()) {
                return false;
            }
            return src.compare(src.size() - suffix.size(), suffix.size(), suffix) == 0;
        };

        if (is_suffix(query_filename, ".fasta") || is_suffix(query_filename, ".fa") ||
            is_suffix(query_filename, ".fasta.gz") || is_suffix(query_filename, ".fa.gz")) {
            std::cout << "Getting Query data" << std::endl;
            query_parser = bioparser::createParser<bioparser::FastaParser, Sequence>(
                    query_filename);
        }

        //read the query file:
        std::vector <std::unique_ptr<Sequence>> fasta_objects;
        query_parser->parse(fasta_objects, -1);

        //We can read the data:
        for (int i = 0; i < fasta_objects.size(); i++) {
            std::cout << i << ": " << fasta_objects[i]->name() << std::endl;
            std::cout << i << ": " << fasta_objects[i]->data() << std::endl;
        }
    }
}