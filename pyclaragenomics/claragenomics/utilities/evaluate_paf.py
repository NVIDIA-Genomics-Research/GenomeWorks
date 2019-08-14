#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
"""Functions and tools for calculating the accuracy of overlap detection"""

import argparse
from claragenomics.io import pafio


def evaluate_paf(truth_paf_filepath, test_paf_filepath, pos_tolerance=500):
    """Given a truth and test set PAF file, count number of in/correctly detected, and non-detected overlaps
    Args:
       truth_paf_filepath (str): Path to truth set PAF file
       test_paf_filepath (str): Path to test set PAF file
       pos_tolerance (int): query and referennce positions within this range will be connsidered to be a matched overlap

    Returns: 3-tupe consisting of (rue_positive_count, false_positive_count, false_negative_count).
    """

    # Put the truth paf into a dictionary:
    truth_overlaps = {}

    for truth_overlap in pafio.read_paf(truth_paf_filepath):
        key = truth_overlap.query_sequence_name + truth_overlap.target_sequence_name
        truth_overlaps[key] = truth_overlap

    true_positive_count = 0
    false_positive_count = 0
    false_negative_count = 0

    for test_overlap in pafio.read_paf(test_paf_filepath):
        query_start_0 = test_overlap.query_start
        query_end_0 = test_overlap.query_end
        target_start_0 = test_overlap.target_start
        target_end_0 = test_overlap.target_end

        key = test_overlap.query_sequence_name + test_overlap.target_sequence_name
        key_reversed = test_overlap.target_sequence_name + test_overlap.query_sequence_name

        matched = False
        potential_match = False
        if key in truth_overlaps:
            potential_match = True
            truth_overlap = truth_overlaps[key]

            query_start_1 = truth_overlap.query_start
            query_end_1 = truth_overlap.query_end
            target_start_1 = truth_overlap.target_start
            target_end_1 = truth_overlap.target_end

        elif key_reversed in truth_overlaps:
            potential_match = True
            truth_overlap = truth_overlaps[key_reversed]

            query_start_1 = truth_overlap.target_start
            query_end_1 = truth_overlap.target_end
            target_start_1 = truth_overlap.query_start
            target_end_1 = truth_overlap.query_end

        matched = potential_match and \
            abs(query_start_0 - query_start_1) < pos_tolerance and \
            abs(query_end_0 - query_end_1) < pos_tolerance and \
            abs(target_start_0 - target_start_1) < pos_tolerance and \
            abs(target_end_0 - target_end_1) < pos_tolerance

        if matched:
            true_positive_count += 1
        else:
            false_positive_count += 1

    #  Now count the false negatives:
    num_true_overlaps = len(truth_overlaps)
    false_negative_count = num_true_overlaps - true_positive_count
    return(true_positive_count, false_positive_count, false_negative_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Given a truth (reference) and test set of overlaps in PAF format,\
         calculate precision and recall")
    parser.add_argument('--truth_paf',
                        type=str,
                        default='truth.paf')
    parser.add_argument('--test_paf',
                        type=str,
                        default='test.paf')

    args = parser.parse_args()

    true_positives, false_positives, false_negatives = evaluate_paf(args.truth_paf, args.test_paf)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    print("Precision = {}, Recall = {}".format(precision, recall))
