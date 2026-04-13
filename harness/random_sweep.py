"""
Global growing-menu threshold sweep.

Starts with **one tool exposed** and **only that tool's tasks in the pool**,
then adds one more tool per step in strict round-robin order across datasets
(bfcl -> jefferson -> finance -> postgres -> bfcl -> ...) with each dataset's
internal tool order randomly shuffled per seed.

When tool K is added at step K, its tasks join the pool and the **entire
pool** is re-evaluated against the new K-tool menu. The task set is mixed
across datasets — one unified evaluation per step, no per-dataset isolation.

This measures a real degradation curve: the same tasks that succeeded at
step 5 may fail at step 40 because the menu got crowded.

Usage
-----
# Default: 5 tasks per tool, all 4 datasets, round-robin tool addition
python -m harness.random_sweep --model qwen2.5:7b

# Reproducible with a fixed seed
python -m harness.random_sweep --model qwen2.5:7b --seed 42

# Smaller/larger sample
python -m harness.random_sweep --model qwen2.5:7b --tasks-per-tool 3

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


def _load_dataset_l1_tasks(ds: str) -> list[dict]:
    """Load L1 tasks for one dataset: base file plus optional `tasks_l1_extra.jsonl`."""
    tasks = load_tasks(ds, ["L1"], limit=None)
    extra_path = Path(DATASETS[ds]["tasks"]["L1"]).parent / "tasks_l1_extra.jsonl"
    if extra_path.exists():
        with open(extra_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                extra = json.loads(line)
                extra.setdefault("level", "L1")
                tasks.append(extra)
    return tasks


def _collect_l1_by_tool(
    cycle: list[str],
    rng: random.Random,
    tasks_per_tool: int,
) -> tuple[dict[tuple[str, str], list[dict]], dict[str, list[str]]]:
    """
    For every dataset in `cycle`, load L1 tasks, group by (dataset, tool),
    seeded-subsample to `tasks_per_tool` per tool, and tag each task with
    its `_dataset`.

    Returns:
        by_tool : {(dataset, tool_name): [task, ...]}
        per_dataset_tools : {dataset: sorted list of tool names}
    """
    by_tool: dict[tuple[str, str], list[dict]] = {}
    per_dataset_tools: dict[str, list[str]] = {}

    for ds in cycle:
        tasks = _load_dataset_l1_tasks(ds)
        per_dataset_tools[ds] = _extract_dataset_tools(tasks)

        grouped: dict[str, list[dict]] = {}
        for t in tasks:
            required = _task_required_tools(t)
            if len(required) != 1:
                continue
            grouped.setdefault(required[0], []).append(t)

        for tool_name, tlist in grouped.items():
            if len(tlist) > tasks_per_tool:
                sampled = rng.sample(tlist, tasks_per_tool)
            else:
                sampled = list(tlist)
            for t in sampled:
                t["_dataset"] = ds
            by_tool[(ds, tool_name)] = sampled

    return by_tool, per_dataset_tools


def _build_round_robin_schedule(
    per_dataset_tools: dict[str, list[str]],
    rng: random.Random,
    cycle: list[str],
) -> list[tuple[str, str]]:
    """
    Shuffle each dataset's tool list and interleave in `cycle` order. When a
    dataset exhausts its tools it is skipped; the cycle continues with the
    rest.
    """
    queues: dict[str, list[str]] = {}
    for ds in cycle:
        tools = list(per_dataset_tools.get(ds, []))
        rng.shuffle(tools)
        queues[ds] = tools

    schedule: list[tuple[str, str]] = []
    while any(queues[ds] for ds in cycle):
        for ds in cycle:
            if queues[ds]:
                schedule.append((ds, queues[ds].pop(0)))
    return schedule


# ──────────────────────────────────────────────────────────────────────────────
# Per-task evaluation
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


async def _evaluate_pool(
    pool: list[dict],
    session,
    client: ModelClient,
    exposed_tools: list[dict],
    allow_fallback: bool,
) -> tuple[list[dict], dict]:
    """
    Evaluate a mixed-dataset task pool against one shared exposed tool list.
    Each task's system prompt is built from its own `_dataset` tag.
    """
    totals = dict(
        total_tests=len(pool),
        correct_result=0,
        no_tool_call=0,
        wrong_tool=0,
        wrong_params=0,
    )
    details: list[dict] = []

    for i, task in enumerate(pool, 1):
        task_id = task.get("id", f"task_{i}")
        query = task["query"]
        level = task.get("level", "L1")
        expected_params = task.get("expected_params")
        dataset = task.get("_dataset", "unknown")

        ref_functions = task.get("functions") or (
            [task["function"]] if "function" in task else []
        )
        optimal_steps = len(ref_functions) if ref_functions else 1
        strict_tool_param_checks = dataset in {"finance", "finance-v2"}

        record: dict = {
            "task_id": task_id,
            "dataset": dataset,
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

        sys_prompt = _build_system_prompt(dataset, allow_fallback)
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
            f"    {status} {dataset:9s} {task_id}: actual={actual_path} | "
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
    tasks_per_tool: int,
) -> dict:
    if seed is None:
        seed = random.randrange(2**31)
    rng = random.Random(seed)

    cycle = [ds for ds in DATASET_CYCLE if ds in datasets]
    if not cycle:
        raise SystemExit(f"No valid datasets selected. Pick from {DATASET_CYCLE}.")

    by_tool, per_dataset_tools = _collect_l1_by_tool(cycle, rng, tasks_per_tool)

    for ds in cycle:
        ds_tools = per_dataset_tools[ds]
        ds_task_count = sum(
            len(by_tool.get((ds, tool), [])) for tool in ds_tools
        )
        print(
            f"  {ds:10s} {len(ds_tools):2d} tools / {ds_task_count:3d} sampled L1 tasks "
            f"({tasks_per_tool}/tool)"
        )

    schedule = _build_round_robin_schedule(per_dataset_tools, rng, cycle)
    total_steps = len(schedule)

    print(f"\nSeed: {seed}")
    print(f"Sweep steps (total tools to add): {total_steps}")
    for ds in cycle:
        order = [t for d, t in schedule if d == ds]
        print(f"  {ds:10s} tool order: {order}")
    print()

    client = ModelClient(model_cfg, allow_fallback=allow_fallback)

    exposed: set[str] = set()
    pool: list[dict] = []
    step_records: list[dict] = []

    async with AsyncExitStack() as stack:
        server_script = Path(DATASETS[cycle[0]]["server"])
        session, all_tools = await stack.enter_async_context(mcp_session(server_script))
        print(f"  MCP session up ({len(all_tools)} tools available across all datasets)\n")

        for step_num, (ds, new_tool) in enumerate(schedule, start=1):
            exposed.add(new_tool)
            pool.extend(by_tool.get((ds, new_tool), []))

            if from_step and step_num < from_step:
                continue
            if to_step and step_num > to_step:
                break

            exposed_tools = [
                t for t in all_tools if t["function"]["name"] in exposed
            ]

            print(
                f"[Step {step_num}/{total_steps}] {ds} ← {new_tool}  "
                f"|  menu: {len(exposed_tools)} tools  "
                f"|  pool: {len(pool)} tasks"
            )

            t0 = time.perf_counter()
            details, totals = await _evaluate_pool(
                pool, session, client, exposed_tools, allow_fallback,
            )
            elapsed = time.perf_counter() - t0
            metrics = calculate_metrics(details, totals)
            print(
                f"    WOS={metrics['wos']}%  "
                f"correct={totals['correct_result']}/{totals['total_tests']}  "
                f"({elapsed:.1f}s)\n"
            )

            step_records.append({
                "step": step_num,
                "dataset": ds,
                "tool_added": new_tool,
                "exposed_tools": sorted(exposed),
                "exposed_count": len(exposed),
                "pool_size": len(pool),
                "metrics": metrics,
                "totals": totals,
                "details": details,
            })

    wos_curve = [
        {
            "step": rec["step"],
            "dataset": rec["dataset"],
            "tool_added": rec["tool_added"],
            "exposed_count": rec["exposed_count"],
            "pool_size": rec["pool_size"],
            "wos": rec["metrics"]["wos"],
            "correct": rec["totals"]["correct_result"],
            "wrong_tool": rec["totals"]["wrong_tool"],
            "wrong_params": rec["totals"]["wrong_params"],
            "no_tool_call": rec["totals"]["no_tool_call"],
        }
        for rec in step_records
    ]

    tasks_by_tool_ids = {
        f"{ds}:{tool}": [t.get("id", "?") for t in tlist]
        for (ds, tool), tlist in by_tool.items()
    }

    output = {
        "model": model_cfg.name,
        "backend": model_cfg.backend,
        "base_url": model_cfg.base_url,
        "seed": seed,
        "datasets": cycle,
        "tasks_per_tool": tasks_per_tool,
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
        "tasks_by_tool": tasks_by_tool_ids,
        "timestamp": datetime.now().isoformat(),
        "steps": step_records,
        "wos_curve": wos_curve,
    }

    save_path = output_path or _default_output_path(model_cfg.name, seed)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved -> {save_path}")

    print("\n" + "=" * 62)
    print("  Global WOS curve")
    print("=" * 62)
    for pt in wos_curve:
        print(
            f"  step {pt['step']:3d} | {pt['dataset']:9s} +{pt['tool_added']:35s} "
            f"| {pt['exposed_count']:2d} tools | {pt['pool_size']:4d} tasks | "
            f"WOS={pt['wos']:6.2f}%"
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
        description="Global growing-menu threshold sweep.",
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
    p.add_argument("--tasks-per-tool", type=int, default=5,
                   help="How many L1 tasks to sample per tool (default: 5)")
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
    print(f"  global growing-menu threshold sweep")
    print(f"  model         : {model_cfg.name}  [{model_cfg.backend}]")
    print(f"  endpoint      : {model_cfg.base_url}")
    print(f"  datasets      : {args.datasets}")
    print(f"  tasks/tool    : {args.tasks_per_tool}")
    print(f"  seed          : {args.seed if args.seed is not None else '(random)'}")
    if args.from_step or args.to_step:
        print(f"  steps         : {args.from_step or 1}..{args.to_step or 'end'}")
    print("=" * 62)

    output = asyncio.run(_sweep(
        model_cfg=model_cfg,
        seed=args.seed,
        from_step=args.from_step,
        to_step=args.to_step,
        allow_fallback=args.allow_fallback,
        output_path=args.output,
        datasets=args.datasets,
        tasks_per_tool=args.tasks_per_tool,
    ))

    if not output:
        sys.exit(1)


if __name__ == "__main__":
    main()
