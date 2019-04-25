#pragma  once

/// \defgroup cudamapper CUDA mapper package
/// Base docs for the cudamapper package (tbd)
/// \ingroup cudamapper
/// \{

namespace genomeworks {
    namespace cudamapper {
        enum class StatusType {
            SUCCESS = 0,
            GENERIC_ERROR
        };

        StatusType Init();
    };
};

/// \}
