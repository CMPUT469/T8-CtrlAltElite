"""
Metrics for outcome-based evaluation.

Single metric: Weighted Outcome Score (WOS)

    WOS(task) = E(O, Ô) × (S_optimal / S_actual)

    • Wrong answer or no tool call  → 0.0
    • Correct, optimal path         → 1.0
    • Correct, one redundant call   → S_optimal / S_actual  (e.g. 0.67)

    Aggregate WOS = mean(WOS per task) × 100  → reported as a percentage.

This is TESR used directly as the outcome score. Binary outcome accuracy is
the special case where every task has optimal_steps == actual_steps == 1,
which gives WOS == outcome_accuracy. The weighting generalises it to
penalise inefficient multi-step solutions without needing a separate metric.

Tool choice is intentionally not scored. If the model used the wrong tool,
the result will be wrong and WOS captures that through E(O, Ô). The
`functions` field in task JSONL is a reference path for log transparency
only — it never influences scoring.
"""

from __future__ import annotations

import json
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Value comparison
# ──────────────────────────────────────────────────────────────────────────────

def compare_values(actual: Any, expected: Any, tolerance: float = 0.01) -> bool:
    """
    Deep equality with numeric tolerance.

    • Numeric strings coerced to float ("3.5" == 3.5)
    • Lists compared element-wise
    • Dicts: expected is treated as a required subset of actual —
      tasks only assert on fields they care about; extra keys returned
      by the tool (e.g. p_value on a regression result) are ignored.
    """
    if actual == expected:
        return True

    try:
        return abs(float(actual) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        pass

    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            compare_values(a, e, tolerance) for a, e in zip(actual, expected)
        )

    if isinstance(actual, dict) and isinstance(expected, dict):
        return all(
            k in actual and compare_values(actual[k], expected[k], tolerance)
            for k in expected
        )

    return False


def compare_params(actual: dict, expected: dict) -> bool:
    """
    Check that actual params satisfy expected params.
    Extra keys in actual are allowed; missing required keys → False.
    """
    for key, exp_val in expected.items():
        if key not in actual:
            return False
        act_val = actual[key]
        if isinstance(exp_val, list) and isinstance(act_val, list):
            if len(act_val) != len(exp_val):
                return False
            if not all(compare_values(a, e) for a, e in zip(act_val, exp_val)):
                return False
        elif not compare_values(act_val, exp_val):
            return False
    return True


def compare_outcome_across_steps(
    step_results: list[Any],
    expected_outcome: dict,
    tolerance: float = 0.01,
) -> bool:
    """
    Check whether all expected_outcome keys are satisfied across the
    union of step results — without caring which step produced which value.

    This supports any valid tool-call ordering the model chooses:
      {"mean": 60.0, "final_result": 1.778151}
    passes whether the model called mean→log or found another valid path,
    as long as both values appear somewhere in the step results.

    Strategy:
      1. Merge all step result dicts into a single pool (later steps win
         on key collision, so intermediate values are preserved too).
      2. For each expected key, check the pool first, then fall back to
         checking each step's scalar "result" key, then bare scalars.
    """
    # Build a merged pool of all key→value pairs emitted across steps
    merged: dict[str, Any] = {}
    for result in step_results:
        if isinstance(result, dict):
            merged.update(result)

    for exp_key, exp_val in expected_outcome.items():
        # 1. Direct key match in merged pool
        if exp_key in merged and compare_values(merged[exp_key], exp_val, tolerance):
            continue

        # 2. Scan each step result for a matching value
        found = False
        for result in step_results:
            if isinstance(result, dict):
                if "result" in result and compare_values(result["result"], exp_val, tolerance):
                    found = True
                    break
            elif compare_values(result, exp_val, tolerance):
                found = True
                break

        if not found:
            return False

    return True


def extract_result_value(tool_result: Any) -> Any:
    """
    Normalise an MCP tool result to a plain Python value.
    JSON round-trip converts Decimal/datetime/etc to JSON-safe primitives.
    """
    raw = None
    if hasattr(tool_result, "content"):
        content = tool_result.content
        if isinstance(content, list) and content:
            item = content[0]
            if hasattr(item, "text"):
                try:
                    raw = json.loads(item.text)
                except Exception:
                    return item.text
            else:
                raw = item
        else:
            raw = content
    elif hasattr(tool_result, "model_dump"):
        raw = tool_result.model_dump()
    else:
        raw = tool_result

    try:
        return json.loads(json.dumps(raw, default=str))
    except (TypeError, ValueError):
        return raw


def serialize_tool_result(tool_result: Any) -> str:
    """
    Stable string serialization of an MCP tool result for logging.

    Tries structured approaches first (model_dump, content attr),
    falls back to repr() as a last resort.
    """
    try:
        if hasattr(tool_result, "model_dump"):
            return json.dumps(tool_result.model_dump(), default=str)
    except Exception:
        pass
    try:
        content = getattr(tool_result, "content", None)
        if content is not None:
            return json.dumps(content, default=str)
    except Exception:
        pass
    try:
        return json.dumps(tool_result, default=str)
    except Exception:
        return repr(tool_result)


# ──────────────────────────────────────────────────────────────────────────────
# Weighted Outcome Score
# ──────────────────────────────────────────────────────────────────────────────

def wos(outcome: bool, optimal_steps: int, actual_steps: int) -> float:
    """
    Weighted Outcome Score for a single task.

    WOS = E(O, Ô) × (S_optimal / S_actual)
    """
    if not outcome:
        return 0.0
    return min(1.0, optimal_steps / max(actual_steps, 1))


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate metrics
# ──────────────────────────────────────────────────────────────────────────────

def calculate_metrics(details: list[dict], totals: dict) -> dict:
    """
    Compute WOS overall and per level, plus auxiliary diagnosis counts.

    totals keys : total_tests, correct_result, correct_function,
                  correct_params, no_tool_call, wrong_tool
    detail keys : correct_result, optimal_steps, actual_steps, level
    """
    n   = totals["total_tests"]
    ntc = totals["no_tool_call"]
    wt  = totals["wrong_tool"]

    # WOS — overall and per level
    scores: dict[str, list[float]] = {"L1": [], "L2": [], "L3": [], "all": []}
    for d in details:
        s = wos(
            outcome=d.get("correct_result", False),
            optimal_steps=d.get("optimal_steps", 1),
            actual_steps=d.get("actual_steps", 1),
        )
        level = d.get("level", "L1")
        scores[level].append(s)
        scores["all"].append(s)

    def _pct(lst: list) -> float:
        return round(sum(lst) / len(lst) * 100, 2) if lst else 0.0

    return {
        # ── Single primary metric ─────────────────────────────────────
        "wos":          _pct(scores["all"]),   # Weighted Outcome Score %
        "wos_l1":       _pct(scores["L1"]),
        "wos_l2":       _pct(scores["L2"]),
        "wos_l3":       _pct(scores["L3"]),

        # ── Counts (for diagnosing failure mode, not scored) ──────────
        "total_tasks":  n,
        "no_tool_call": ntc,   # model never tried → wos=0
        "wrong_tool":   wt,    # tried wrong tool  → wos=0
    }


def print_report(metrics: dict, model: str, dataset: str = "") -> None:
    label = f"{model}" + (f" on {dataset}" if dataset else "")
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"  WOS              : {metrics['wos']}%")
    print(f"  WOS  L1          : {metrics['wos_l1']}%")
    print(f"  WOS  L2          : {metrics['wos_l2']}%")
    print(f"  WOS  L3          : {metrics['wos_l3']}%")
    print(f"  total tasks      : {metrics['total_tasks']}")
    print(f"  no tool call     : {metrics['no_tool_call']}")
    print(f"  wrong tool       : {metrics['wrong_tool']}")
    print(sep + "\n")