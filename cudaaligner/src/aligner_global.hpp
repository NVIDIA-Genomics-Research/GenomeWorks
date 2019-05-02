#pragma once

#include "cudaaligner/aligner.hpp"

namespace genomeworks {

namespace cudaaligner {

class AlignerGlobal : public Aligner
{
    public:
        AlignerGlobal(uint32_t max_query_length, uint32_t max_target_length, uint32_t max_alignments);
        ~AlignerGlobal();

        virtual StatusType align_all() override;

        virtual StatusType add_alignment(const char* query, uint32_t query_length, const char* target, uint32_t target_length) override;

        virtual const std::vector<std::shared_ptr<Alignment>>& get_alignments() const override {
            return alignments_;
        }

        virtual uint32_t num_alignments() const {
            return num_alignments_;
        }

    private:
        virtual void update_alignments_with_results();

    private:
        uint32_t max_query_length_;
        uint32_t max_target_length_;
        uint32_t max_alignments_;
        uint32_t num_alignments_;
        std::vector<std::shared_ptr<Alignment>> alignments_;

        uint8_t* sequences_d_;
        uint8_t* sequences_h_;

        uint32_t* sequence_lengths_d_;
        uint32_t* sequence_lengths_h_;

        uint8_t* results_d_;
        uint8_t* results_h_;

        uint32_t* result_lengths_d_;
        uint32_t* result_lengths_h_;
};

}

}
