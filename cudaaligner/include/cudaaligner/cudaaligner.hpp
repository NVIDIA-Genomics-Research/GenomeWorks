#pragma  once

/// \defgroup cudaaligner CUDA Aligner package
/// Base docs for the cudaaligner package (tbd)
/// \{

namespace genomeworks {

namespace cudaaligner {
    /// CUDA Aligner error type
    enum class StatusType {
        success = 0,
        uninitialized,
        exceeded_max_alignments,
        exceeded_max_length,
        generic_error
    };

    /// AlignmentType - Enum for storing type of alignment.
    enum class AlignmentType {
        global = 0,
        unset
    };

    /// AlignmentState - Enum for encoding each position in alignment.
    enum class AlignmentState {
        match = 0,
        mismatch,
        insertion, // Present in query, absent in subject
        deletion // Absent in query, present in subject
    };

    StatusType Init();
}

}
/// \}
