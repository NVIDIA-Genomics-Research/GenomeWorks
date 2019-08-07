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


def evaluate_paf(truth_paf_filepath, test_paf_filepath, pos_tolerance=100):
    """Given a truth and test set PAF file, count number of in/correctly detected, and non-detected overlaps
    Args:
       truth_paf_filepath (str): Path to truth set PAF file
       test_paf_filepath (str): Path to test set PAF file
       pos_tolerance (int): query and referennce positions within this range will be connsidered to be a matched overlap

    Returns: 3-tupe consisting of (rue_positive_count, false_positive_count, false_negative_count).
    """
    with open(truth_paf_filepath) as f:
        truth_paf = f.readlines()

    with open(test_paf_filepath) as f:
        test_paf = f.readlines()

    # Put the truth paf into a dictionary:
    truth_overlaps = {}

    for paf_entry in truth_paf:
        overlap = paf_entry.split('\t')
        key = overlap[0] + overlap[5]  # Unique key for this overlap
        truth_overlaps[key] = (overlap[2], overlap[3], overlap[7], overlap[8])

    true_positive_count = 0
    false_positive_count = 0
    false_negative_count = 0

    for paf_entry in test_paf:
        overlap = paf_entry.split('\t')

        query_start_0 = int(overlap[2])
        query_end_0 = int(overlap[3])
        target_start_0 = int(overlap[7])
        target_end_0 = int(overlap[8])

        key = overlap[0] + overlap[5]  # Unique key for this overlap
        key_reversed = overlap[5] + overlap[0]  # Unique key for this overlap, switching query and target

        matched = False
        if key in truth_overlaps:
            truth_overlap = truth_overlaps[key]

            query_start_1 = int(truth_overlap[0])
            query_end_1 = int(truth_overlap[1])
            target_start_1 = int(truth_overlap[2])
            target_end_1 = int(truth_overlap[3])

        elif key_reversed in truth_overlaps:
            truth_overlap = truth_overlaps[key]

            query_start_1 = int(truth_overlap[2])
            query_end_1 = int(truth_overlap[3])
            target_start_1 = int(truth_overlap[0])
            target_end_1 = int(truth_overlap[1])

        matched = abs(query_start_0 - query_start_1) < pos_tolerance and \
            abs(query_end_0 - query_end_1) < pos_tolerance and \
            abs(target_start_0 - target_start_1) < pos_tolerance and \
            abs(target_end_0 - target_end_1) < pos_tolerance

        if matched:
            true_positive_count += 1
        else:
            false_positive_count += 1

    #  Now count the false negatives:
    num_true_overlaps = len(truth_paf)
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
