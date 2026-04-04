import unittest

from harness.model_client import _parse_fallback_json
from harness.runner import (
    DATASETS,
    _compare_params_exact,
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

    def test_single_step_params_fail_when_no_function_match(self):
        called = [{"ticker": "AAPL"}]
        self.assertFalse(
            _compare_step_params(called, {"ticker": "AAPL"}, matched_indices=[])
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


class StrictParamComparisonTests(unittest.TestCase):
    def test_exact_params_rejects_extra_keys(self):
        self.assertFalse(
            _compare_params_exact(
                {"ticker": "AAPL", "limit": 5, "period": "annual"},
                {"ticker": "AAPL", "limit": 5},
            )
        )

    def test_exact_params_accepts_same_keys_and_values(self):
        self.assertTrue(
            _compare_params_exact(
                {"ticker": "AAPL", "limit": 5},
                {"ticker": "AAPL", "limit": 5},
            )
        )

    def test_compare_step_params_strict_mode_enforces_exactness(self):
        called = [{"ticker": "AAPL", "limit": 5, "period": "annual"}]
        expected = {"ticker": "AAPL", "limit": 5}
        self.assertFalse(_compare_step_params(called, expected, strict=True))


class FallbackToolParsingTests(unittest.TestCase):
    def test_accepts_name_arguments_shape_for_allowed_tool(self):
        self.assertEqual(
            _parse_fallback_json(
                '{"name":"multiply","arguments":{"a":600,"b":0.2}}',
                {"multiply"},
            ),
            {"tool": "multiply", "args": {"a": 600, "b": 0.2}},
        )

    def test_accepts_single_item_list_shape_for_allowed_tool(self):
        self.assertEqual(
            _parse_fallback_json(
                '[{"name":"multiply","arguments":{"a":600,"b":0.2}}]',
                {"multiply"},
            ),
            {"tool": "multiply", "args": {"a": 600, "b": 0.2}},
        )

    def test_accepts_two_item_array_shape_for_allowed_tool(self):
        self.assertEqual(
            _parse_fallback_json(
                '["standard_deviation", {"numbers":[10,20,30,40,50]}]',
                {"standard_deviation"},
            ),
            {
                "tool": "standard_deviation",
                "args": {"numbers": [10, 20, 30, 40, 50]},
            },
        )

    def test_rejects_tool_not_in_allowed_set(self):
        self.assertIsNone(
            _parse_fallback_json(
                '{"name":"division","arguments":{"a":600,"b":0.2}}',
                {"multiply"},
            )
        )

    def test_rejects_prose_wrapped_json(self):
        self.assertIsNone(
            _parse_fallback_json(
                'Next step:\n{"name":"multiply","arguments":{"a":600,"b":0.2}}',
                {"multiply"},
            )
        )


class DatasetRegistryTests(unittest.TestCase):
    def test_jefferson_v2_dataset_is_registered_for_all_levels(self):
        self.assertIn("jefferson-v2", DATASETS)
        self.assertEqual(
            DATASETS["jefferson-v2"]["tasks"],
            {
                "L1": "datasets/jefferson_stats/tasks_l1_v2.jsonl",
                "L2": "datasets/jefferson_stats/tasks_l2_v2.jsonl",
                "L3": "datasets/jefferson_stats/tasks_l3_v2.jsonl",
            },
        )

    def test_finance_dataset_is_registered_for_all_levels(self):
        self.assertIn("finance", DATASETS)
        self.assertEqual(
            DATASETS["finance"]["tasks"],
            {
                "L1": "datasets/finance/tasks_l1.jsonl",
                "L2": "datasets/finance/tasks_l2.jsonl",
                "L3": "datasets/finance/tasks_l3.jsonl",
            },
        )

    def test_finance_v2_dataset_is_registered_for_all_levels(self):
        self.assertIn("finance-v2", DATASETS)
        self.assertEqual(
            DATASETS["finance-v2"]["tasks"],
            {
                "L1": "datasets/finance/tasks_l1_v2.jsonl",
                "L2": "datasets/finance/tasks_l2_v2.jsonl",
                "L3": "datasets/finance/tasks_l3_v2.jsonl",
            },
        )


if __name__ == "__main__":
    unittest.main()
