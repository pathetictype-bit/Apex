"""Labeled splits for the in-memory manual-index transformation trial."""

from manual_retrieval_experiment import CHALLENGE_CASES, PRIMARY_CASES


TRAIN_EXAMPLES = tuple(
    (case["query"], case["expected_top"]) for case in PRIMARY_CASES
)
HELD_OUT_EXAMPLES = tuple(
    (case["query"], case["expected_top"]) for case in CHALLENGE_CASES
)
