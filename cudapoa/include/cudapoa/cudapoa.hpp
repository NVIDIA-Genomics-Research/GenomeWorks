#pragma  once

/// \defgroup cudapoa CUDA POA package
/// Base docs for the cudapoa package (tbd)
/// \ingroup cudapoa
/// \{

namespace genomeworks {
    namespace cudapoa {
        /// CUDA POA error type
        enum class StatusType {
            SUCCESS = 0,
            EXCEEDED_MAXIMUM_POAS,
            EXCEEDED_MAXIMUM_SEQUENCE_SIZE,
            EXCEEDED_MAXIMUM_SEQUENCES_PER_POA,
            GENERIC_ERROR
        };

        StatusType Init();
    };
};
/// \}
