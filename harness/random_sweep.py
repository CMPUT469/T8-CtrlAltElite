"""
Anchor-task threshold sweep.

Picks a fixed set of "anchor" L1 tasks per dataset, pre-exposes their
required tools, then adds the remaining tools one-by-one across all
datasets in a strict round-robin cycle (math -> stats -> finance -> sql
-> math -> ...) with each dataset's internal tool order randomly
shuffled per run.

At every step, the same anchor task set for the current dataset is
re-evaluated against the new (larger) exposed tool list. Because the
tasks are fixed, any change in WOS is attributable purely to the
growing tool count — that is the threshold signal.

Output: per-dataset baseline WOS (anchors only) and a step-by-step
WOS curve. The threshold is the step where WOS systematically drops
from baseline.

Usage
-----
# Random shuffle each run, default 5 anchors per dataset
python -m harness.random_sweep --model qwen2.5:7b

# Reproducible
python -m harness.random_sweep --model qwen2.5:7b --seed 42

# More anchors for a tighter signal
python -m harness.random_sweep --model qwen2.5:7b --anchors-per-dataset 10

# Resume / partial
python -m harness.random_sweep --model qwen2.5:7b --seed 42 --from-step 20 --to-step 40

# Subset of datasets
python -m harness.random_sweep --model qwen2.5:7b --datasets bfcl jefferson
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.runner import (
    DATASETS,
    AGENTIC_LOOP_FOOTER,
    load_tasks,
    _find_subsequence_indices,
    _compare_step_params,
)
from harness.metrics import (
    calculate_metrics,
    compare_outcome_across_steps,
    compare_values,
    extract_result_value,
    serialize_tool_result,
    wos,
)
from harness.mcp_session import mcp_session
from harness.model_client import ModelClient, ModelConfig, resolve_model_config


DATASET_CYCLE = ["bfcl", "jefferson", "finance", "postgres"]
RESULTS_DIR = Path("results/random_sweep")


# ──────────────────────────────────────────────────────────────────────────────
# Tool / task helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_dataset_tools(tasks: list[dict]) -> list[str]:
    """Collect unique tool names referenced by any task in this dataset."""
    tools: set[str] = set()
    for t in tasks:
        if "function" in t and isinstance(t["function"], str):
            tools.add(t["function"])
        for fn in t.get("functions", []) or []:
            if isinstance(fn, str):
                tools.add(fn)
    return sorted(tools)


def _task_required_tools(task: dict) -> list[str]:
    if "functions" in task and task["functions"]:
        return list(task["functions"])
    if "function" in task and task["function"]:
        return [task["function"]]
    return []


def _select_anchor_tasks(
    tasks: list[dict],
    count: int,
) -> list[dict]:
    """
    Pick anchor tasks deterministically from a dataset's task list.

    Strategy: walk the L1 tasks in load order and accept one task per
    distinct required tool until we have `count` tasks. Restricting to
    L1 keeps anchors single-tool and easy to reason about.
    """
    seen_tools: set[str] = set()
    anchors: list[dict] = []
    for t in tasks:
        if t.get("level") != "L1":
            continue
        required = _task_required_tools(t)
        if len(required) != 1:
            continue
        tool = required[0]
        if tool in seen_tools:
            continue
        seen_tools.add(tool)
        anchors.append(t)
        if len(anchors) >= count:
            break
    return anchors


def _anchor_tool_set(anchor_tasks: list[dict]) -> set[str]:
    """Union of required tools across all anchor tasks."""
    tools: set[str] = set()
    for t in anchor_tasks:
        for fn in _task_required_tools(t):
            tools.add(fn)
    return tools


def _build_round_robin_schedule(
    per_dataset_tools: dict[str, list[str]],
    excluded_tools: dict[str, set[str]],
    rng: random.Random,
    cycle: list[str],
) -> list[tuple[str, str]]:
    """
    Shuffle each dataset's non-excluded tools and interleave the shuffles
    in `cycle` order. When a dataset exhausts its remaining tools it is
    skipped; the cycle continues with the remaining datasets.

    `excluded_tools[ds]` is the set of tools to omit from the schedule
    (typically the anchor tools, which are pre-exposed).
    """
    queues: dict[str, list[str]] = {}
    for ds in cycle:
        excluded = excluded_tools.get(ds, set())
        tools = [t for t in per_dataset_tools.get(ds, []) if t not in excluded]
        rng.shuffle(tools)
        queues[ds] = tools

    schedule: list[tuple[str, str]] = []
    while any(queues[ds] for ds in cycle):
        for ds in cycle:
            if queues[ds]:
                schedule.append((ds, queues[ds].pop(0)))
    return schedule


# ──────────────────────────────────────────────────────────────────────────────
# Per-task evaluation (whitelist-based, adapted from runner.run_evaluation)
# ──────────────────────────────────────────────────────────────────────────────

def _build_system_prompt(dataset: str, allow_fallback: bool) -> str:
    sys_content = "You are a helpful assistant. Use the provided tools when needed."
    sys_content += "\n\n" + AGENTIC_LOOP_FOOTER
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
    if allow_fallback:
        sys_content += (
            "\n\nPrefer native tool calls whenever the endpoint supports them."
            "\nOnly fall back to plain-text JSON if the endpoint cannot emit a native tool call."
            '\nIf you must fall back, respond with ONLY one JSON tool request and no prose, '
            'using an exact provided tool name and exact parameter names.'
            '\nPreferred fallback shape: {"tool":"<name>","args":{...}}'
        )
    return sys_content


async def _evaluate_tasks(
    tasks: list[dict],
    session,
    client: ModelClient,
    exposed_tools: list[dict],
    dataset: str,
    allow_fallback: bool,
) -> tuple[list[dict], dict]:
    """
    Evaluate a list of tasks against a fixed `exposed_tools` whitelist.
    Returns (details, totals) ready to feed into calculate_metrics().
    """
    strict_tool_param_checks = dataset in {"finance", "finance-v2"}

    totals = dict(
        total_tests=len(tasks),
        correct_result=0,
        no_tool_call=0,
        wrong_tool=0,
        wrong_params=0,
    )
    details: list[dict] = []

    sys_prompt = _build_system_prompt(dataset, allow_fallback)

    for i, task in enumerate(tasks, 1):
        task_id = task.get("id", f"task_{i}")
        query = task["query"]
        level = task.get("level", "L1")
        expected_params = task.get("expected_params")

        ref_functions = task.get("functions") or (
            [task["function"]] if "function" in task else []
        )
        optimal_steps = len(ref_functions) if ref_functions else 1

        record: dict = {
            "task_id": task_id,
            "level": level,
            "query": query,
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

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query},
        ]

        max_steps = optimal_steps + 2
        step_results: list = []

        for step in range(max_steps):
            try:
                model_response = client.get_response(messages, exposed_tools)
                tool_call = model_response.tool_call
                record["raw_model_output"] = model_response.raw_text
            except Exception as exc:
                record["error"] = f"Model call failed: {exc}"
                totals["no_tool_call"] += 1
                break

            if tool_call is None:
                if step == 0:
                    record["error"] = "Model made no tool call"
                    totals["no_tool_call"] += 1
                    break
                steps_done = record["actual_steps"]
                if steps_done < optimal_steps:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Continue. You have completed {steps_done} of "
                            f"{optimal_steps} required steps. "
                            "Call the next required tool now."
                        ),
                    })
                    continue
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
                totals["no_tool_call"] += 1
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
                "content": json.dumps(result_value),
            })

            if optimal_steps == 1:
                break

        # ── Outcome scoring ───────────────────────────────────────────
        matched_indices: list[int] | None = None
        if ref_functions:
            matched_indices = _find_subsequence_indices(
                record["actual_functions"], ref_functions,
            )
            if matched_indices is None:
                matched_indices = []
        record["matched_ref_indices"] = matched_indices

        params_ok = True
        if expected_params is not None and record["actual_functions"]:
            params_ok = _compare_step_params(
                record["actual_params_by_step"],
                expected_params,
                matched_indices=matched_indices,
                strict=strict_tool_param_checks,
            )
            record["params_match"] = params_ok
            if not params_ok:
                totals["wrong_params"] += 1
        elif expected_params is not None:
            record["params_match"] = False

        tool_ok = True
        if ref_functions:
            if strict_tool_param_checks:
                tool_ok = record["actual_functions"] == ref_functions
            else:
                tool_ok = bool(matched_indices)
            record["tool_match"] = tool_ok

        if record["actual_functions"] and ref_functions and not tool_ok:
            totals["wrong_tool"] += 1

        if step_results:
            expected_outcome = task.get("expected_outcome")
            record["expected_outcome"] = expected_outcome

            if expected_outcome is not None:
                outcome_ok = (
                    compare_outcome_across_steps(step_results, expected_outcome)
                    if isinstance(expected_outcome, dict)
                    else compare_values(step_results[-1], expected_outcome)
                )
            else:
                if strict_tool_param_checks:
                    tool_ok_for_outcome = (
                        record["tool_match"]
                        if record["tool_match"] is not None
                        else True
                    )
                    outcome_ok = (
                        record["actual_steps"] > 0
                        and record.get("error") is None
                        and bool(tool_ok_for_outcome)
                        and (expected_params is None or params_ok)
                    )
                else:
                    outcome_ok = (
                        record["actual_steps"] > 0
                        and record.get("error") is None
                        and (expected_params is None or params_ok)
                    )

            if outcome_ok:
                record["correct_result"] = True
                totals["correct_result"] += 1

        details.append(record)

        wos_val = wos(
            outcome=record["correct_result"],
            optimal_steps=record["optimal_steps"],
            actual_steps=record["actual_steps"],
        )
        status = "OK" if record["correct_result"] else "X "
        actual_path = "→".join(record["actual_functions"]) or "none"
        ref_path = "→".join(ref_functions) or "?"
        print(
            f"    {status} {level} {task_id}: actual={actual_path} | "
            f"ref={ref_path} | steps={record['actual_steps']}/{record['optimal_steps']} | "
            f"wos={wos_val:.2f}"
        )

    return details, totals


# ──────────────────────────────────────────────────────────────────────────────
# Sweep orchestrator
# ──────────────────────────────────────────────────────────────────────────────

async def _sweep(
    model_cfg: ModelConfig,
    seed: Optional[int],
    from_step: Optional[int],
    to_step: Optional[int],
    allow_fallback: bool,
    output_path: Optional[Path],
    datasets: list[str],
    anchors_per_dataset: int,
) -> dict:
    if seed is None:
        seed = random.randrange(2**31)
    rng = random.Random(seed)

    cycle = [ds for ds in DATASET_CYCLE if ds in datasets]
    if not cycle:
        raise SystemExit(f"No valid datasets selected. Pick from {DATASET_CYCLE}.")

    # Load all L1/L2/L3 tasks per dataset and pick anchors
    per_dataset_tasks: dict[str, list[dict]] = {}
    per_dataset_tools: dict[str, list[str]] = {}
    anchor_tasks: dict[str, list[dict]] = {}
    anchor_tools: dict[str, set[str]] = {}
    for ds in cycle:
        tasks = load_tasks(ds, ["L1", "L2", "L3"], limit=None)
        per_dataset_tasks[ds] = tasks
        per_dataset_tools[ds] = _extract_dataset_tools(tasks)
        anchors = _select_anchor_tasks(tasks, anchors_per_dataset)
        anchor_tasks[ds] = anchors
        anchor_tools[ds] = _anchor_tool_set(anchors)
        print(
            f"  {ds:10s} {len(tasks):3d} tasks / {len(per_dataset_tools[ds]):2d} tools "
            f"| {len(anchors)} anchors → {sorted(anchor_tools[ds])}"
        )
        if not anchors:
            print(f"    [warn] no L1 anchors available for {ds}, dataset will be skipped")

    cycle = [ds for ds in cycle if anchor_tasks[ds]]
    if not cycle:
        raise SystemExit("No datasets with usable anchors. Aborting.")

    schedule = _build_round_robin_schedule(
        per_dataset_tools, anchor_tools, rng, cycle,
    )
    total_steps = len(schedule)

    print(f"\nSeed: {seed}")
    print(f"Sweep steps (non-anchor tools): {total_steps}")
    for ds in cycle:
        order = [t for d, t in schedule if d == ds]
        print(f"  {ds:10s} sweep order: {order}")
    print()

    client = ModelClient(model_cfg, allow_fallback=allow_fallback)

    # Each dataset starts with its anchor tools already exposed
    exposed_per_ds: dict[str, set[str]] = {ds: set(anchor_tools[ds]) for ds in cycle}
    step_records: list[dict] = []
    baseline_metrics: dict[str, dict] = {}

    async with AsyncExitStack() as stack:
        sessions: dict[str, tuple] = {}
        for ds in cycle:
            server_script = Path(DATASETS[ds]["server"])
            sess, all_tools = await stack.enter_async_context(mcp_session(server_script))
            sessions[ds] = (sess, all_tools)
            print(f"  MCP session up for {ds} ({len(all_tools)} tools available)")
        print()

        # ── Baseline: anchors only ────────────────────────────────────
        skip_baseline = bool(from_step and from_step > 1)
        if not skip_baseline:
            print("[Baseline] anchors only (no extra tools exposed)")
            for ds in cycle:
                session, all_tools = sessions[ds]
                exposed_tools = [
                    t for t in all_tools if t["function"]["name"] in exposed_per_ds[ds]
                ]
                print(
                    f"  {ds}: {len(anchor_tasks[ds])} anchors, "
                    f"{len(exposed_tools)} tools exposed"
                )
                t0 = time.perf_counter()
                details, totals = await _evaluate_tasks(
                    anchor_tasks[ds], session, client, exposed_tools, ds, allow_fallback,
                )
                elapsed = time.perf_counter() - t0
                metrics = calculate_metrics(details, totals)
                baseline_metrics[ds] = {
                    "metrics": metrics,
                    "details": details,
                    "exposed_tools": sorted(exposed_per_ds[ds]),
                }
                print(
                    f"    baseline {ds}: WOS={metrics['wos']}% "
                    f"({elapsed:.1f}s)\n"
                )

        # ── Sweep: add one non-anchor tool, re-eval anchors ───────────
        for step_num, (ds, new_tool) in enumerate(schedule, start=1):
            exposed_per_ds[ds].add(new_tool)

            if from_step and step_num < from_step:
                continue
            if to_step and step_num > to_step:
                break

            session, all_tools = sessions[ds]
            exposed_set = exposed_per_ds[ds]
            exposed_tools = [
                t for t in all_tools if t["function"]["name"] in exposed_set
            ]

            print(
                f"[Step {step_num}/{total_steps}] {ds} ← {new_tool}  "
                f"|  exposed: {len(exposed_set)} {ds} tools  "
                f"|  re-eval {len(anchor_tasks[ds])} anchors"
            )

            t0 = time.perf_counter()
            details, totals = await _evaluate_tasks(
                anchor_tasks[ds], session, client, exposed_tools, ds, allow_fallback,
            )
            elapsed = time.perf_counter() - t0
            metrics = calculate_metrics(details, totals)

            baseline_wos = (
                baseline_metrics.get(ds, {}).get("metrics", {}).get("wos")
                if baseline_metrics.get(ds) else None
            )
            delta_str = (
                f"  Δ={metrics['wos'] - baseline_wos:+.2f}pp"
                if baseline_wos is not None else ""
            )
            print(
                f"    anchor WOS={metrics['wos']}%{delta_str}  ({elapsed:.1f}s)\n"
            )

            step_records.append({
                "step": step_num,
                "dataset": ds,
                "tool_added": new_tool,
                "exposed_tools_in_dataset": sorted(exposed_set),
                "anchor_count": len(anchor_tasks[ds]),
                "metrics": metrics,
                "details": details,
            })

    # Build per-dataset WOS curves: baseline + each step that touched ds
    wos_curves: dict[str, list[dict]] = {}
    for ds in cycle:
        curve: list[dict] = []
        if ds in baseline_metrics:
            curve.append({
                "step": 0,
                "tool_added": None,
                "wos": baseline_metrics[ds]["metrics"]["wos"],
                "exposed_count": len(baseline_metrics[ds]["exposed_tools"]),
            })
        for rec in step_records:
            if rec["dataset"] == ds:
                curve.append({
                    "step": rec["step"],
                    "tool_added": rec["tool_added"],
                    "wos": rec["metrics"]["wos"],
                    "exposed_count": len(rec["exposed_tools_in_dataset"]),
                })
        wos_curves[ds] = curve

    output = {
        "model": model_cfg.name,
        "backend": model_cfg.backend,
        "base_url": model_cfg.base_url,
        "seed": seed,
        "datasets": cycle,
        "anchors_per_dataset": anchors_per_dataset,
        "anchor_tools_per_dataset": {ds: sorted(anchor_tools[ds]) for ds in cycle},
        "anchor_task_ids_per_dataset": {
            ds: [t.get("id", "?") for t in anchor_tasks[ds]] for ds in cycle
        },
        "from_step": from_step,
        "to_step": to_step,
        "tool_order_per_dataset": {
            ds: [t for d, t in schedule if d == ds] for ds in cycle
        },
        "schedule": [
            {"step": i + 1, "dataset": d, "tool": t}
            for i, (d, t) in enumerate(schedule)
        ],
        "total_steps": total_steps,
        "timestamp": datetime.now().isoformat(),
        "baseline_metrics": baseline_metrics,
        "steps": step_records,
        "wos_curves": wos_curves,
    }

    save_path = output_path or _default_output_path(model_cfg.name, seed)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved -> {save_path}")

    # Print compact threshold curves
    print("\n" + "=" * 62)
    print("  WOS curves (anchor tasks)")
    print("=" * 62)
    for ds in cycle:
        print(f"\n  {ds}:")
        for pt in wos_curves[ds]:
            label = "baseline" if pt["step"] == 0 else f"step {pt['step']:3d}"
            tool = f"+{pt['tool_added']}" if pt["tool_added"] else "(anchors only)"
            print(
                f"    {label} | {pt['exposed_count']:2d} tools | "
                f"WOS={pt['wos']:6.2f}% | {tool}"
            )
    print()

    return output


def _default_output_path(model_name: str, seed: int) -> Path:
    safe = model_name.replace(":", "_").replace("/", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"{safe}_seed{seed}_{ts}.json"


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Anchor-task threshold sweep with round-robin random tool addition.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", required=True,
                   help="Model identifier (e.g. qwen2.5:7b)")
    p.add_argument("--backend", default="ollama",
                   choices=["ollama", "vllm", "openai"])
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (omit for a fresh shuffle each run)")
    p.add_argument("--from-step", type=int, default=None,
                   help="Resume at this step (1-indexed)")
    p.add_argument("--to-step", type=int, default=None,
                   help="Stop after this step (inclusive)")
    p.add_argument("--datasets", nargs="+", default=DATASET_CYCLE,
                   choices=DATASET_CYCLE,
                   help=f"Datasets to include (default: {DATASET_CYCLE})")
    p.add_argument("--anchors-per-dataset", type=int, default=5,
                   help="Number of L1 anchor tasks per dataset (default: 5)")
    p.add_argument("--allow-fallback", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    return p


def main():
    args = _build_parser().parse_args()
    model_cfg = resolve_model_config(
        args.model,
        backend=args.backend,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    print("=" * 62)
    print(f"  anchor-task threshold sweep")
    print(f"  model    : {model_cfg.name}  [{model_cfg.backend}]")
    print(f"  endpoint : {model_cfg.base_url}")
    print(f"  datasets : {args.datasets}")
    print(f"  anchors  : {args.anchors_per_dataset} per dataset")
    print(f"  seed     : {args.seed if args.seed is not None else '(random)'}")
    if args.from_step or args.to_step:
        print(f"  steps    : {args.from_step or 1}..{args.to_step or 'end'}")
    print("=" * 62)

    output = asyncio.run(_sweep(
        model_cfg=model_cfg,
        seed=args.seed,
        from_step=args.from_step,
        to_step=args.to_step,
        allow_fallback=args.allow_fallback,
        output_path=args.output,
        datasets=args.datasets,
        anchors_per_dataset=args.anchors_per_dataset,
    ))

    if not output:
        sys.exit(1)


if __name__ == "__main__":
    main()
