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
            node_count_exceeded_maximum_graph_size,
            seq_len_exceeded_maximum_nodes_per_window,
            loop_count_exceeded_upper_bound,
            generic_error
        };

        StatusType Init();
    }
}
/// \}
