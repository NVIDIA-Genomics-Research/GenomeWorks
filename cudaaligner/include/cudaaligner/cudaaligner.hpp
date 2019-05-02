#pragma  once

/// \defgroup cudaaligner CUDA Aligner package
/// Base docs for the cudaaligner package (tbd)
/// \{

namespace genomeworks {

namespace cudaaligner {
    /// CUDA Aligner error type
    enum class StatusType {
        SUCCESS = 0,
        UNINITIALIZED,
        GENERIC_ERROR
    };

    StatusType Init();
}

}
/// \}
