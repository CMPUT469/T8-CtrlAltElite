import sys
import types
import unittest


if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules["openai"] = openai_stub


from harness.incremental_sweep import _build_round_schedule


class IncrementalSweepScheduleTests(unittest.TestCase):
    def test_round_schedule_reassigns_exhausted_dataset_slots(self):
        tool_order = {
            "bfcl": [f"b{i}" for i in range(15)],
            "jefferson": [f"j{i}" for i in range(18)],
            "postgres": [f"p{i}" for i in range(7)],
            "finance": [f"f{i}" for i in range(18)],
        }

        schedule = _build_round_schedule(tool_order)

        self.assertEqual(
            schedule[0],
            {"bfcl": 2, "jefferson": 2, "postgres": 2, "finance": 2},
        )
        self.assertEqual(
            schedule[5],
            {"bfcl": 7, "jefferson": 7, "postgres": 7, "finance": 7},
        )
        self.assertEqual(
            schedule[6],
            {"bfcl": 8, "jefferson": 9, "postgres": 7, "finance": 8},
        )
        self.assertEqual(
            schedule[-1],
            {"bfcl": 15, "jefferson": 18, "postgres": 7, "finance": 18},
        )

    def test_total_tools_keep_advancing_by_dataset_slots_until_final_round(self):
        tool_order = {
            "bfcl": [f"b{i}" for i in range(15)],
            "jefferson": [f"j{i}" for i in range(18)],
            "postgres": [f"p{i}" for i in range(7)],
            "finance": [f"f{i}" for i in range(18)],
        }

        totals = [sum(counts.values()) for counts in _build_round_schedule(tool_order)]

        self.assertEqual(totals[:6], [8, 12, 16, 20, 24, 28])
        self.assertEqual(totals[6:], [32, 36, 40, 44, 48, 52, 56, 58])


if __name__ == "__main__":
    unittest.main()
