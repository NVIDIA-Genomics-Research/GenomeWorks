#pragma once

#include <memory>
#include <string>
#include "cudamapper/sequence.hpp"

namespace genomeworks {

    class BioParserSequence: public Sequence {
    public:
        ~BioParserSequence() = default;
        BioParserSequence() = default;

        const std::string &name() const {
            return name_;
        }

        const std::string &data() const {
            return data_;
        }

        BioParserSequence(const char *name, uint32_t name_length, const char *data,
                 uint32_t data_length);

    private:
        std::string name_;
        std::string data_;
    };
}