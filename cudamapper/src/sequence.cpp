#include "cudamapper/sequence.hpp"
#include "bioparser_sequence.hpp"

namespace genomeworks {
    std::unique_ptr<Sequence> Sequence::create_sequence() {
        return std::make_unique<BioParserSequence>();
    }
}
