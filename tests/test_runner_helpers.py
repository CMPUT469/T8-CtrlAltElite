import unittest

from harness.runner import (
    DATASETS,
    _compare_step_params,
    _find_subsequence_indices,
    _matched_prefix_length,
)


class MatchedPrefixLengthTests(unittest.TestCase):
    def test_empty_actual_matches_nothing(self):
        self.assertEqual(_matched_prefix_length([], ["a", "b"]), 0)

    def test_wrong_first_call_does_not_advance_progress(self):
        # A stray opening call should not move exposure to the next expected tool.
        self.assertEqual(_matched_prefix_length(["x"], ["a", "b"]), 0)

    def test_correct_first_call_advances_progress(self):
        self.assertEqual(_matched_prefix_length(["a"], ["a", "b"]), 1)

    def test_extra_middle_call_does_not_skip_expected_order(self):
        self.assertEqual(_matched_prefix_length(["a", "x"], ["a", "b"]), 1)

    def test_in_order_subsequence_advances_through_full_chain(self):
        self.assertEqual(_matched_prefix_length(["x", "a", "y", "b"], ["a", "b"]), 2)


class FindSubsequenceIndicesTests(unittest.TestCase):
    def test_returns_matching_indices_for_in_order_subsequence(self):
        self.assertEqual(
            _find_subsequence_indices(["x", "a", "y", "b"], ["a", "b"]),
            [1, 3],
        )

    def test_returns_none_when_full_sequence_is_missing(self):
        self.assertIsNone(_find_subsequence_indices(["a", "x"], ["a", "b"]))


class CompareStepParamsTests(unittest.TestCase):
    def test_single_step_params_use_first_matched_call(self):
        called = [{"ticker": "NOPE"}, {"ticker": "AAPL", "limit": 5}]
        matched = [1]
        self.assertTrue(
            _compare_step_params(called, {"ticker": "AAPL"}, matched_indices=matched)
        )

    def test_multi_step_params_align_to_matched_subsequence(self):
        called = [
            {"ticker": "NOPE"},
            {"ticker": "AAPL"},
            {"ticker": "MSFT", "limit": 4},
        ]
        matched = [1, 2]
        expected = [{"ticker": "AAPL"}, {"ticker": "MSFT", "limit": 4}]
        self.assertTrue(_compare_step_params(called, expected, matched_indices=matched))

    def test_multi_step_params_fail_when_subsequence_is_too_short(self):
        called = [{"ticker": "AAPL"}]
        expected = [{"ticker": "AAPL"}, {"ticker": "MSFT"}]
        self.assertFalse(_compare_step_params(called, expected, matched_indices=[0]))


class DatasetRegistryTests(unittest.TestCase):
    def test_jefferson_stage1_dataset_is_registered_for_all_levels(self):
        self.assertIn("jefferson_stage1", DATASETS)
        self.assertEqual(
            DATASETS["jefferson_stage1"]["tasks"],
            {
                "L1": "datasets/jefferson_stats_stage1/tasks_l1.jsonl",
                "L2": "datasets/jefferson_stats_stage1/tasks_l2.jsonl",
                "L3": "datasets/jefferson_stats_stage1/tasks_l3.jsonl",
            },
        )


if __name__ == "__main__":
    unittest.main()
