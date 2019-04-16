#include "cudamapper/sequence.hpp"
#include "bioparser_sequence.hpp"


namespace genomeworks {
    std::unique_ptr<Sequence> Sequence::create_sequence(const char *name, uint32_t name_length, const char *data,
                                                        uint32_t data_length) {
        return std::make_unique<BioParserSequence>(name, name_length, data, data_length);
    }

    std::unique_ptr<Sequence> Sequence::create_sequence(const std::string &name, const std::string &data) {
        return std::make_unique<BioParserSequence>(name.c_str(), name.size(), data.c_str(), data.size());
    }

}
