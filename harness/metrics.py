"""
Metrics for outcome-based evaluation.
Primary metric: Weighted Outcome Score (WOS)
    WOS(task) = optimal_steps / (optimal_steps + |actual_steps - optimal_steps|)
- Wrong answer or no tool call -> 0.0
- Correct, optimal path        -> 1.0
- Correct, non-optimal path    -> optimal_steps / (optimal_steps + deviation)
    Aggregate WOS = mean(WOS per task) × 100  → reported as a percentage.
    
Unlike a simple outcome * (optimal / actual) ratio, this formulation penalises
both over-stepping (extra calls) and under-stepping (skipping required calls)
symmetrically via the absolute deviation from the golden path length.
 
Binary outcome accuracy is the special case where every task has
optimal_steps == actual_steps == 1, which gives WOS == outcome_accuracy.
The weighting generalises it to penalise inefficient solutions without needing
a separate metric.
 
Tool choice is intentionally not scored directly. If the model used the wrong
tool, the result will generally be wrong and WOS captures that through E(O, Ô).
For non-deterministic tasks, runner-level checks may use expected params as a
proxy for correctness before WOS is computed.
Aggregate WOS is reported as a percentage.
"""

from __future__ import annotations

import json
from typing import Any


def _is_schema_metadata_dict(value: Any) -> bool:
    return isinstance(value, dict) and {"schema_name", "owner", "has_usage"}.issubset(value)


def _compare_lists_unordered(
    actual: list[Any],
    expected: list[Any],
    tolerance: float,
) -> bool:
    """
    Compare lists as multisets while preserving strict value equality.

    This avoids failing deterministic tasks purely because a model returned
    the correct rows in a different order than the benchmark artifact.
    """
    if len(actual) != len(expected):
        return False

    unmatched_actual = list(actual)
    for exp_item in expected:
        match_index = next(
            (
                i
                for i, act_item in enumerate(unmatched_actual)
                if compare_values(act_item, exp_item, tolerance)
            ),
            None,
        )
        if match_index is None:
            return False
        unmatched_actual.pop(match_index)
    return True


def _compare_dict_values_by_position(
    actual: dict[Any, Any],
    expected: dict[Any, Any],
    tolerance: float,
) -> bool:
    """
    Compare row-like dicts by column position and values while ignoring the
    output alias text.

    Python preserves insertion order for dicts, so the row values reflect the
    output column order returned by the SQL execution engine.
    """
    if len(actual) != len(expected):
        return False
    return all(
        compare_values(act_val, exp_val, tolerance)
        for act_val, exp_val in zip(actual.values(), expected.values())
    )


def compare_values(actual: Any, expected: Any, tolerance: float = 0.01) -> bool:
    """
    Deep equality with numeric tolerance.

    - Numeric strings are coerced to float where possible.
    - Lists are compared element-wise.
    - Dicts treat expected as a required subset of actual.
    """
    if actual == expected:
        return True

    try:
        return abs(float(actual) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        pass

    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return False
        if all(compare_values(a, e, tolerance) for a, e in zip(actual, expected)):
            return True
        return _compare_lists_unordered(list(actual), list(expected), tolerance)

    if isinstance(actual, dict) and isinstance(expected, dict):
        # Schema owners can legitimately vary by local Postgres environment
        # (for example `postgres` vs the current cluster username). Keep the
        # comparison strict for schema name and privileges, but tolerate the
        # owner field when the benchmark expects the default `postgres` owner.
        if _is_schema_metadata_dict(actual) and _is_schema_metadata_dict(expected):
            comparable_expected = dict(expected)
            if comparable_expected.get("owner") == "postgres":
                comparable_expected.pop("owner", None)
            return all(
                k in actual and compare_values(actual[k], comparable_expected[k], tolerance)
                for k in comparable_expected
            )

        if all(
            k in actual and compare_values(actual[k], expected[k], tolerance)
            for k in expected
        ):
            return True

        # For SQL result rows, treat output aliases as presentation details:
        # if the row has the same number of columns and the column values
        # match in position, do not fail solely because alias text differs.
        return _compare_dict_values_by_position(actual, expected, tolerance)

    return False


def compare_params(actual: dict, expected: dict) -> bool:
    """
    Check that actual params satisfy expected params.
    Extra keys in actual are allowed.
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
    Normalize an MCP tool result to plain Python values.
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
    Stable string serialization for logging.
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


def wos(outcome: bool, optimal_steps: int, actual_steps: int) -> float:
    """
    Evaluates agent accuracy based on success and path efficiency.
    Penalizes both over-stepping (inefficiency) and under-stepping (skipping).
    """
    # If the task failed or no tools were used, the score is zero
    if not outcome or actual_steps <= 0:
        return 0.0
    
    # Calculate the absolute difference from the golden path
    deviation = abs(actual_steps - optimal_steps)
    
    # Using the optimal steps as the numerator ensures a score of 1.0 
    # when deviation is 0. As deviation increases, the score drops.
    accuracy_weight = optimal_steps / (optimal_steps + deviation)
    
    return round(accuracy_weight, 4)


def calculate_metrics(details: list[dict], totals: dict) -> dict:
    """
    Compute WOS overall/per-level and diagnostic counts.

    totals keys: total_tests, correct_result, no_tool_call,
                 wrong_tool, wrong_params
    """
    n = totals["total_tests"]
    ntc = totals["no_tool_call"]
    wt = totals["wrong_tool"]
    wp = totals.get("wrong_params", 0)

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

    def _pct(lst: list[float]) -> float:
        return round(sum(lst) / len(lst) * 100, 2) if lst else 0.0

    return {
        "wos": _pct(scores["all"]),
        "wos_l1": _pct(scores["L1"]),
        "wos_l2": _pct(scores["L2"]),
        "wos_l3": _pct(scores["L3"]),
        "total_tasks": n,
        "no_tool_call": ntc,
        "wrong_tool": wt,
        "wrong_params": wp,
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
    print(f"  wrong params     : {metrics.get('wrong_params', 0)}")
    print(sep + "\n")
