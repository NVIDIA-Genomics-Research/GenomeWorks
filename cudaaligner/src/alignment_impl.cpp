#include "alignment_impl.hpp"

namespace genomeworks {

namespace cudaaligner {

AlignmentImpl::AlignmentImpl(const char* query, uint32_t query_length, const char* target, uint32_t target_length)
    : query_(query, query + query_length)
    , target_(target, target + target_length)
    , status_(StatusType::uninitialized)
    , type_(AlignmentType::unset)
{
    // Initialize Alignment object.
}

AlignmentImpl::~AlignmentImpl()
{
    // Nothing to destroy right now.
}

std::string AlignmentImpl::convert_to_cigar() const
{
    throw std::runtime_error("Conversion to CIGAR not imlemented yet.");
}

FormattedAlignment AlignmentImpl::format_alignment() const
{
    std::string t_str = "";
    std::size_t t_pos = 0;
    std::string q_str = "";
    std::size_t q_pos = 0;
    for(std::size_t i = 0; i < alignment_.size(); i++)
    {
        switch(alignment_.at(i))
        {
            case AlignmentState::match:
            case AlignmentState::mismatch:
                t_str += target_[t_pos++];
                q_str += query_[q_pos++];
                break;
            case AlignmentState::insert_into_query:
                t_str += "-";
                q_str += query_[q_pos++];
                break;
            case AlignmentState::insert_into_target:
                t_str += target_[t_pos++];
                q_str += "-";
                break;
            default:
                throw std::runtime_error("Unknown alignment state");
        }
    }

    FormattedAlignment output(q_str, t_str);
    return output;
}

}

}
