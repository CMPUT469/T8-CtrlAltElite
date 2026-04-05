"""
Incremental tool-count threshold evaluation.

Grows the tool set one tool at a time per dataset, adding each tool's
corresponding tasks as they become satisfiable.  This measures how
models cope with an increasing number of *real* tools — not random
distractors.

Usage:
    # Run one round (N tools per dataset)
    python -m harness.incremental_sweep --model qwen3:8b --round 3

    # Run full sweep (2 tools → all tools, stepping by 1)
    python -m harness.incremental_sweep --model qwen3:8b --sweep

    # Compare models from saved results
    python -m harness.incremental_sweep --compare \\
        --models qwen3:8b gpt-oss:20b mistral:12b llama3.1:8b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.metrics import (
    calculate_metrics,
    compare_outcome_across_steps,
    compare_values,
    extract_result_value,
    print_report,
    serialize_tool_result,
    wos,
)
from harness.model_client import ModelClient, ModelConfig, resolve_model_config
from harness.runner import DATASETS, load_tasks

TOOL_ORDER_PATH = Path("configs/tool_order.yaml")
RESULTS_DIR = Path("results/incremental")

# Map dataset names used in DATASETS to their tool_order.yaml keys.
# v2 variants share the same tool order as their base dataset.
_DATASET_TO_ORDER_KEY = {
    "bfcl": "bfcl",
    "bfcl-v2": "bfcl",
    "jefferson": "jefferson",
    "jefferson-v2": "jefferson",
    "postgres": "postgres",
    "postgres-v2": "postgres",
    "postgres_stage1": "postgres",
    "postgres_stage1-v2": "postgres",
    "finance": "finance",
    "finance-v2": "finance",
}

# Datasets evaluated in each round (one per domain).
SWEEP_DATASETS = ["bfcl", "jefferson", "postgres", "finance"]


# ──────────────────────────────────────────────────────────────────────────────
# Tool-order helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_tool_order() -> dict[str, list[str]]:
    with open(TOOL_ORDER_PATH) as f:
        return yaml.safe_load(f)


def _tools_for_round(tool_order: dict[str, list[str]], dataset: str, n: int) -> list[str]:
    """Return the first *n* tools for *dataset*."""
    key = _DATASET_TO_ORDER_KEY.get(dataset, dataset)
    all_tools = tool_order.get(key, [])
    return all_tools[:n]


def _max_round(tool_order: dict[str, list[str]]) -> int:
    """Longest tool list across all sweep datasets."""
    return max(len(tool_order.get(_DATASET_TO_ORDER_KEY[d], [])) for d in SWEEP_DATASETS)


def _filter_tasks_for_round(tasks: list[dict], available: set[str]) -> list[dict]:
    """Keep only tasks whose required tools are all in *available*."""
    out = []
    for task in tasks:
        required = task.get("functions") or (
            [task["function"]] if "function" in task else []
        )
        if required and all(f in available for f in required):
            out.append(task)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Single-round evaluation
# ──────────────────────────────────────────────────────────────────────────────

async def _run_round(
    round_num: int,
    model_cfg: ModelConfig,
    tool_order: dict[str, list[str]],
    prompt_template: Optional[str] = None,
) -> dict:
    """
    Run evaluation for one round across all SWEEP_DATASETS.

    For each dataset the model sees exactly the first *round_num* tools
    and only tasks that those tools can satisfy.
    """
    from harness.mcp_session import mcp_session

    all_details: list[dict] = []
    totals = {
        "total_tests": 0,
        "correct_result": 0,
        "no_tool_call": 0,
        "wrong_tool": 0,
        "wrong_params": 0,
    }
    per_dataset: dict[str, dict] = {}

    for dataset in SWEEP_DATASETS:
        round_tools = _tools_for_round(tool_order, dataset, round_num)
        if not round_tools:
            continue
        available = set(round_tools)

        # Load ALL tasks (L1-L3) and keep only satisfiable ones
        tasks = load_tasks(dataset, ["L1", "L2", "L3"], limit=None)
        tasks = _filter_tasks_for_round(tasks, available)
        if not tasks:
            per_dataset[dataset] = {"tools": len(round_tools), "tasks": 0, "wos": 0.0}
            continue

        print(f"\n{'─'*62}")
        print(f"  {dataset}  |  round {round_num}  |  {len(round_tools)} tools  |  {len(tasks)} tasks")
        print(f"{'─'*62}")

        server_script = DATASETS[dataset]["server"]
        strict = dataset in {"finance", "finance-v2"}

        async with mcp_session(server_script) as (session, openai_tools):
            # Filter the MCP tool list to only this round's tools
            exposed = [t for t in openai_tools if t["function"]["name"] in available]

            client = ModelClient(model_cfg)

            ds_details: list[dict] = []
            ds_totals = {
                "total_tests": 0,
                "correct_result": 0,
                "no_tool_call": 0,
                "wrong_tool": 0,
                "wrong_params": 0,
            }

            for i, task in enumerate(tasks, 1):
                task_id = task.get("id", f"task_{i}")
                query = task["query"]
                level = task.get("level", "L1")
                expected_params = task.get("expected_params")

                ref_functions = task.get("functions") or (
                    [task["function"]] if "function" in task else []
                )
                optimal_steps = len(ref_functions) if ref_functions else 1

                print(f"  [{i}/{len(tasks)}] {level} | {task_id}: {query[:60]}...")

                record: dict = {
                    "task_id": task_id,
                    "level": level,
                    "query": query,
                    "dataset": dataset,
                    "ref_functions": ref_functions,
                    "expected_params": expected_params,
                    "actual_functions": [],
                    "actual_params": None,
                    "actual_params_by_step": [],
                    "matched_ref_indices": None,
                    "tool_match": None,
                    "params_match": None,
                    "actual_result": None,
                    "correct_result": False,
                    "optimal_steps": optimal_steps,
                    "actual_steps": 0,
                    "error": None,
                    "call_source": "none",
                    "raw_model_output": None,
                    "tool_result": None,
                    "expected_outcome": None,
                }

                ds_totals["total_tests"] += 1

                sys_content = "You are a helpful assistant. Use the provided tools when needed."
                if prompt_template:
                    sys_content = prompt_template + "\n\n" + sys_content
                if dataset.startswith("postgres"):
                    sys_content += (
                        "\n\nFor Postgres tasks:"
                        "\n- Use exact schema, table, and column names returned by tools; never guess similar names."
                        "\n- If you inspect tables or relationships first, you must still finish with execute_query when the task asks for a computed result."
                        "\n- For execute_query, return one read-only SELECT statement with no trailing semicolon."
                        "\n- Match requested output column names when practical by using SQL aliases."
                        "\n- Do not change limit unless the task explicitly asks for a specific number of rows."
                        "\n- Use joins and JSON field access only as supported by the inspected schema and tool outputs."
                    )

                messages = [
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": query},
                ]

                max_steps = optimal_steps + 2
                step_results: list = []

                for step in range(max_steps):
                    try:
                        model_response = client.get_response(messages, exposed)
                        tool_call = model_response.tool_call
                        record["raw_model_output"] = model_response.raw_text
                    except Exception as exc:
                        record["error"] = f"Model call failed: {exc}"
                        ds_totals["no_tool_call"] += 1
                        break

                    if tool_call is None:
                        if step == 0:
                            record["error"] = "Model made no tool call"
                            ds_totals["no_tool_call"] += 1
                        break

                    record["actual_steps"] += 1
                    record["actual_functions"].append(tool_call.function_name)
                    record["actual_params"] = tool_call.arguments
                    record["actual_params_by_step"].append(tool_call.arguments)
                    record["call_source"] = tool_call.call_source

                    try:
                        raw_result = await session.call_tool(
                            tool_call.function_name, tool_call.arguments
                        )
                        record["tool_result"] = serialize_tool_result(raw_result)
                        result_value = extract_result_value(raw_result)
                        step_results.append(result_value)
                        record["actual_result"] = result_value
                    except Exception as exc:
                        record["error"] = f"Tool execution failed at step {step + 1}: {exc}"
                        ds_totals["no_tool_call"] += 1
                        break

                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": f"call_{step}",
                            "type": "function",
                            "function": {
                                "name": tool_call.function_name,
                                "arguments": json.dumps(tool_call.arguments),
                            },
                        }],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": f"call_{step}",
                        "content": json.dumps(result_value, default=str),
                    })

                    if optimal_steps == 1:
                        break

                # ── Outcome scoring ──────────────────────────────────
                expected_outcome = task.get("expected_outcome")
                record["expected_outcome"] = expected_outcome

                if expected_outcome is not None and step_results:
                    if optimal_steps == 1:
                        outcome_ok = compare_values(
                            step_results[-1], expected_outcome
                        )
                    else:
                        outcome_ok = compare_outcome_across_steps(
                            step_results, expected_outcome
                        )
                    if outcome_ok:
                        record["correct_result"] = True
                        ds_totals["correct_result"] += 1
                elif expected_outcome is None and record["actual_steps"] > 0:
                    # Non-deterministic: fall back to tool/param checks
                    from harness.runner import (
                        _compare_step_params,
                        _matched_prefix_length,
                    )
                    plen = _matched_prefix_length(
                        record["actual_functions"], ref_functions
                    )
                    record["tool_match"] = plen == len(ref_functions)
                    if not record["tool_match"]:
                        ds_totals["wrong_tool"] += 1

                    if expected_params and record["actual_params_by_step"]:
                        from harness.runner import _find_subsequence_indices
                        matched_indices = _find_subsequence_indices(
                            record["actual_functions"], ref_functions
                        )
                        record["matched_ref_indices"] = matched_indices
                        params_ok = _compare_step_params(
                            record["actual_params_by_step"],
                            expected_params,
                            matched_indices=matched_indices,
                            strict=strict,
                        )
                        record["params_match"] = params_ok
                        if not params_ok:
                            ds_totals["wrong_params"] += 1
                    else:
                        params_ok = True

                    tool_ok = record["tool_match"] if record["tool_match"] is not None else True
                    outcome_ok = (
                        record["actual_steps"] > 0
                        and record.get("error") is None
                        and bool(tool_ok)
                        and (expected_params is None or params_ok)
                    )
                    if outcome_ok:
                        record["correct_result"] = True
                        ds_totals["correct_result"] += 1

                ds_details.append(record)

                wos_val = wos(
                    outcome=record["correct_result"],
                    optimal_steps=record["optimal_steps"],
                    actual_steps=record["actual_steps"],
                )
                status = "OK" if record["correct_result"] else "X"
                actual_path = "→".join(record["actual_functions"]) or "none"
                ref_path = "→".join(ref_functions) or "?"
                print(
                    f"    {status}  actual={actual_path} | ref={ref_path} | "
                    f"steps={record['actual_steps']}/{record['optimal_steps']} | "
                    f"wos={wos_val:.2f}"
                )

            ds_metrics = calculate_metrics(ds_details, ds_totals)
            per_dataset[dataset] = {
                "tools": len(round_tools),
                "tools_exposed": round_tools,
                "tasks": ds_totals["total_tests"],
                "wos": ds_metrics["wos"],
                "metrics": ds_metrics,
            }

            all_details.extend(ds_details)
            for k in totals:
                totals[k] += ds_totals[k]

    metrics = calculate_metrics(all_details, totals)

    return {
        "model": model_cfg.name,
        "backend": model_cfg.backend,
        "base_url": model_cfg.base_url,
        "round": round_num,
        "timestamp": datetime.now().isoformat(),
        "per_dataset": per_dataset,
        "metrics": metrics,
        "details": all_details,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def _save(output: dict) -> Path:
    model_safe = output["model"].replace(":", "_").replace("/", "_")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"incremental_{model_safe}_round{output['round']}_{ts}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    return path


def _load_round(model: str, round_num: int) -> Optional[dict]:
    """Load the most recent result for a given model/round."""
    model_safe = model.replace(":", "_").replace("/", "_")
    pattern = f"incremental_{model_safe}_round{round_num}_*.json"
    matches = sorted(RESULTS_DIR.glob(pattern), reverse=True) if RESULTS_DIR.exists() else []
    for f in matches:
        try:
            data = json.loads(f.read_text())
            if data.get("round") == round_num:
                return data
        except Exception:
            continue
    return None


def _load_all_rounds(model: str, max_round: int) -> list[tuple[int, dict]]:
    """Load saved results for rounds 2..max_round."""
    results = []
    for n in range(2, max_round + 1):
        data = _load_round(model, n)
        if data:
            results.append((n, data))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Sweep
# ──────────────────────────────────────────────────────────────────────────────

def _sweep(
    model_cfg: ModelConfig,
    tool_order: dict,
    prompt_template: Optional[str] = None,
    from_round: int = 2,
    to_round: Optional[int] = None,
):
    max_n = _max_round(tool_order)
    start = max(from_round, 2)
    end = min(to_round, max_n) if to_round is not None else max_n
    results: list[tuple[int, dict]] = []

    for n in range(start, end + 1):
        # Skip if already on disk
        existing = _load_round(model_cfg.name, n)
        if existing:
            print(f"\n  [skip] round {n} already on disk")
            results.append((n, existing))
            continue

        print(f"\n{'='*62}")
        print(f"  Round {n}  ({n} tools per dataset)")
        print(f"{'='*62}")

        output = asyncio.run(_run_round(n, model_cfg, tool_order, prompt_template))
        path = _save(output)
        results.append((n, output))

        print(f"\n  Round {n} WOS: {output['metrics']['wos']}%  → saved {path}")

    _print_sweep_summary(results, model_cfg.name)


def _print_sweep_summary(results: list[tuple[int, dict]], model: str):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Incremental sweep summary — {model}")
    print(sep)

    header = f"  {'Round':<8} {'Tools':<8} {'Tasks':<8} {'WOS%':<12}"
    for d in SWEEP_DATASETS:
        header += f" {d:<12}"
    print(header)
    print("  " + "-" * 66)

    baseline_wos = None

    for n, output in results:
        m = output["metrics"]
        wos_pct = m["wos"]
        if baseline_wos is None:
            baseline_wos = wos_pct

        delta = (
            f"({wos_pct - baseline_wos:+.1f})" if n != results[0][0] else "(base)"
        )
        total_tasks = m["total_tasks"]

        # Per-dataset WOS
        pd = output.get("per_dataset", {})
        ds_cells = ""
        for d in SWEEP_DATASETS:
            info = pd.get(d, {})
            ds_wos = info.get("wos", 0.0)
            ds_cells += f" {ds_wos:<12.1f}"

        # Count total tools across datasets this round
        total_tools = sum(pd.get(d, {}).get("tools", 0) for d in SWEEP_DATASETS)

        print(
            f"  {n:<8} {total_tools:<8} {total_tasks:<8} "
            f"{wos_pct:<6.1f}% {delta:<8}"
            f"{ds_cells}"
        )

    print(sep + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Cross-model comparison
# ──────────────────────────────────────────────────────────────────────────────

def _compare_models(models: list[str]):
    tool_order = _load_tool_order()
    max_n = _max_round(tool_order)

    model_rounds: dict[str, list[tuple[int, dict]]] = {}
    for model in models:
        rounds = _load_all_rounds(model, max_n)
        if rounds:
            model_rounds[model] = rounds
        else:
            print(f"  [warn] No saved results for {model}, skipping")

    if not model_rounds:
        print("No data found for any model. Run --sweep first.")
        return

    _print_comparison_table(model_rounds, max_n)


def _print_comparison_table(
    model_rounds: dict[str, list[tuple[int, dict]]],
    max_n: int,
):
    models = list(model_rounds.keys())
    col_w = max(16, max(len(m) for m in models) + 4)

    # Build lookup: model -> round -> (wos, total_tasks, total_tools)
    lookup: dict[str, dict[int, tuple[float, int, int]]] = {}
    for model, rounds in model_rounds.items():
        lookup[model] = {}
        for n, output in rounds:
            m = output["metrics"]
            pd = output.get("per_dataset", {})
            total_tools = sum(pd.get(d, {}).get("tools", 0) for d in SWEEP_DATASETS)
            lookup[model][n] = (m["wos"], m["total_tasks"], total_tools)

    # Find decline threshold per model (first drop >= 10pp from baseline)
    DECLINE_PP = 10.0
    decline: dict[str, Optional[tuple[int, float, float]]] = {}
    for model in models:
        rounds_sorted = sorted(lookup[model].keys())
        baseline = lookup[model].get(rounds_sorted[0], (0, 0, 0))[0] if rounds_sorted else 0
        decline[model] = None
        for n in rounds_sorted[1:]:
            val = lookup[model][n][0]
            if (baseline - val) >= DECLINE_PP:
                decline[model] = (n, baseline, val)
                break

    sep = "=" * (22 + col_w * len(models))
    print(f"\n  Cross-model incremental sweep comparison")
    print(f"  {sep}")

    header = f"  {'Round':<7}{'Tools':<7}{'Tasks':<8}|"
    for m in models:
        header += f"  {m:<{col_w - 2}}"
    print(header)
    print(f"  {'-' * 20}+{'-' * (col_w * len(models))}")

    # Collect all round numbers present in any model
    all_rounds = sorted(set(n for m in models for n in lookup[m]))

    for n in all_rounds:
        # Use first model's data for tools/tasks columns
        first = None
        for m in models:
            if n in lookup[m]:
                first = lookup[m][n]
                break
        if first is None:
            continue
        _, tasks, tools = first

        row = f"  {n:<7}{tools:<7}{tasks:<8}|"
        for m in models:
            if n in lookup[m]:
                val = lookup[m][n][0]
                marker = " [*]" if (decline[m] and decline[m][0] == n) else ""
                row += f"  {val:<5.1f}%{marker:<{col_w - 8}}"
            else:
                row += f"  {'--':<{col_w - 2}}"
        print(row)

    print(f"  {sep}")
    print(f"\n  [*] = decline threshold (first drop >= {DECLINE_PP}pp from round 2 baseline)\n")

    print("  Decline thresholds:")
    for m in models:
        d = decline[m]
        if d:
            n, baseline, val = d
            print(f"    {m:<20}: round {n}  ({baseline:.1f}% → {val:.1f}%, {val - baseline:+.1f}pp)")
        else:
            rounds_sorted = sorted(lookup[m].keys())
            if rounds_sorted:
                baseline = lookup[m][rounds_sorted[0]][0]
                print(f"    {m:<20}: none detected  (baseline {baseline:.1f}%)")
            else:
                print(f"    {m:<20}: no data")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Incremental tool-count threshold evaluation",
        epilog=__doc__,
    )
    p.add_argument("--model", default=None,
                   help="Model identifier (required for --round / --sweep)")
    p.add_argument("--backend", default="ollama", choices=["ollama", "vllm", "openai"])
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--round", type=int, default=None,
                   help="Run a single round with N tools per dataset")
    p.add_argument("--sweep", action="store_true",
                   help="Run full sweep from 2 tools to max")
    p.add_argument("--from-round", type=int, default=2,
                   help="First round in sweep range (default: 2)")
    p.add_argument("--to-round", type=int, default=None,
                   help="Last round in sweep range (default: max tools)")
    p.add_argument("--compare", action="store_true",
                   help="Cross-model comparison from saved results")
    p.add_argument("--models", nargs="+", default=None,
                   help="Models to compare (used with --compare)")
    p.add_argument("--prompt-template", type=Path, default=None, metavar="FILE",
                   help="Path to a prompt template file")
    args = p.parse_args()

    # ── Compare mode (no model required) ──
    if args.compare:
        if not args.models:
            p.error("--compare requires --models <model1> <model2> ...")
        _compare_models(args.models)
        return

    # ── All other modes require --model ──
    if not args.model:
        p.error("--model is required for --round and --sweep modes")

    model_cfg = resolve_model_config(
        args.model,
        backend=args.backend,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    tool_order = _load_tool_order()

    prompt_template: Optional[str] = None
    if args.prompt_template:
        tpath = Path(args.prompt_template)
        if not tpath.exists():
            print(f"[error] prompt template not found: {tpath}")
            sys.exit(1)
        prompt_template = tpath.read_text(encoding="utf-8").strip()

    if args.round is not None:
        if args.round < 2:
            p.error("--round must be >= 2 (minimum 2 tools per dataset)")
        output = asyncio.run(_run_round(args.round, model_cfg, tool_order, prompt_template))
        path = _save(output)
        print_report(output["metrics"], model_cfg.name, "incremental")
        print(f"Saved → {path}")
        return

    if args.sweep:
        _sweep(model_cfg, tool_order, prompt_template,
               from_round=args.from_round, to_round=args.to_round)
        return

    p.print_help()


if __name__ == "__main__":
    main()
