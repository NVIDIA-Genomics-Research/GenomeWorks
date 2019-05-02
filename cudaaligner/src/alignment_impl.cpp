#include "alignment_impl.hpp"

namespace genomeworks {

namespace cudaaligner {

AlignmentImpl::AlignmentImpl(const char* query, const char* target)
    : query_(query)
    , target_(target)
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

void AlignmentImpl::print_alignment() const
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
                t_str += target_.at(t_pos++);
                q_str += query_.at(q_pos++);
                break;
            case AlignmentState::insert_into_query:
                t_str += "-";
                q_str += query_.at(q_pos++);
                break;
            case AlignmentState::insert_into_target:
                t_str += target_.at(t_pos++);
                q_str += "-";
                break;
            default:
                throw std::runtime_error("Unknown alignment state");
        }
    }

    printf("%s\n%s\n", q_str.c_str(), t_str.c_str());
}

}

}
