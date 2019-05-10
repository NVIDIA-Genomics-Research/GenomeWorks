#include "alignment_impl.hpp"

namespace genomeworks {

namespace cudaaligner {

AlignmentImpl::AlignmentImpl(const char* query, uint32_t query_length, const char* subject, uint32_t subject_length)
    : query_(query, query + query_length)
    , subject_(subject, subject + subject_length)
    , status_(StatusType::uninitialized)
    , type_(AlignmentType::unset)
{
    // Initialize Alignment object.
}

AlignmentImpl::~AlignmentImpl()
{
    // Nothing to destroy right now.
}

std::string AlignmentImpl::alignment_state_to_cigar_state(AlignmentState s) const
{
    // CIGAR string format from http://bioinformatics.cvr.ac.uk/blog/tag/cigar-string/
    // Implementing a reduced set of CIGAR states, covering only the M, D and I characters.
    switch(s)
    {
        case AlignmentState::match:
        case AlignmentState::mismatch: return "M";
        case AlignmentState::insertion: return "D";
        case AlignmentState::deletion: return "I";
        default: throw std::runtime_error("Unrecognized alignment state.");
    }
}

std::string AlignmentImpl::convert_to_cigar() const
{
    if (alignment_.size() < 1)
    {
        return std::string("");
    }

    std::string cigar = "";
    std::string last_cigar_state = alignment_state_to_cigar_state(alignment_.at(0));
    uint32_t count_last_state = 1;
    for(std::size_t pos = 1; pos < alignment_.size(); pos++)
    {
        std::string cur_cigar_state = alignment_state_to_cigar_state(alignment_.at(pos));
        if (cur_cigar_state == last_cigar_state)
        {
            count_last_state++;
        }
        else
        {
            cigar += std::to_string(count_last_state) + last_cigar_state;
            count_last_state = 1;
            last_cigar_state = cur_cigar_state;
        }
    }
    cigar += std::to_string(count_last_state) + last_cigar_state;
    return cigar;
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
                t_str += subject_[t_pos++];
                q_str += query_[q_pos++];
                break;
            case AlignmentState::insertion:
                t_str += "-";
                q_str += query_[q_pos++];
                break;
            case AlignmentState::deletion:
                t_str += subject_[t_pos++];
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
