#include "cudaaligner/aligner.hpp"
#include "aligner_global.hpp"

namespace genomeworks {

namespace cudaaligner {

    std::unique_ptr<Aligner> create_aligner(uint32_t max_query_length, uint32_t max_subject_length, uint32_t max_alignments, AlignmentType type, uint32_t device_id)
    {
        if (type == AlignmentType::global)
        {
            return std::make_unique<AlignerGlobal>(max_query_length, max_subject_length, max_alignments, device_id);
        }
        else
        {
            throw std::runtime_error("Aligner for specified type not implemented yet.");
        }
    }

}

}
