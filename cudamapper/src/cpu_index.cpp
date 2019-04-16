#include <string>
#include <iostream>
#include "bioparser/bioparser.hpp"
#include "bioparser_sequence.hpp"
#include "cpu_index.hpp"

namespace genomeworks {

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

        //We can read the data:
        for (int i = 0; i < fasta_objects.size(); i++) {
            std::cout << i << ": " << fasta_objects[i]->name() << std::endl;
            std::cout << i << ": " << fasta_objects[i]->data() << std::endl;
        }
    }
}
