#pragma  once

/// \defgroup cudapoa CUDA POA package
/// Base docs for the cudapoa package (tbd)
/// \{

namespace genomeworks {
    namespace cudapoa {
        /// CUDA POA error type
        enum class StatusType {
            success = 0,
            exceeded_maximum_poas,
            exceeded_maximum_sequence_size,
            exceeded_maximum_sequences_per_poa,
            generic_error
        };

        StatusType Init();
    }
}
/// \}
