#include "cudapoa/batch.hpp"
#include "cudapoa_batch.hpp"


namespace genomeworks {

namespace cudapoa {

    std::unique_ptr<Batch> create_batch(uint32_t max_poas, 
                                               uint32_t max_sequences_per_poa,
                                               uint32_t device_id,
                                               int16_t gap_score,
                                               int16_t mismatch_score,
                                               int16_t match_score,
                                               bool cuda_banded_alignment)
    {
        return std::make_unique<CudapoaBatch>(max_poas, max_sequences_per_poa, device_id, gap_score, mismatch_score, match_score, cuda_banded_alignment);
    }

}

}
