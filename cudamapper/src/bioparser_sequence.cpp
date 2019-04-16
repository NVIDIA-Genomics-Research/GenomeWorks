#include <cctype>
#include "bioparser_sequence.hpp"

namespace genomeworks {

    BioParserSequence::BioParserSequence(const char *name, uint32_t name_length, const char *data,
                                         uint32_t data_length)
            : name_(name, name_length), data_(){

        data_.reserve(data_length);
        for (uint32_t i = 0; i < data_length; ++i) {
            data_ += std::toupper(data[i]);
        }
    }
}
