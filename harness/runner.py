"""
Unified evaluation runner.

Replaces evaluate_jefferson.py and evaluate_bfcl.py with a single entry point.

Scoring is outcome-based only (MCPVerse-style WOS). The `functions` field in
task JSONL is a reference path logged for human transparency — it never
influences scoring. A model that reaches the correct answer via an alternative
tool sequence scores the same as one that follows the reference path.

Usage examples
──────────────
# Ollama (current setup)
python -m harness.runner \\
    --dataset jefferson \\
    --model qwen2.5:7b \\
    --level L1 L2 L3

# vLLM on Eureka (once endpoint is known — update configs/models.yaml)
python -m harness.runner \\
    --dataset jefferson \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --base-url http://eureka-node-01:8000/v1 \\
    --api-key token-abc123 \\
    --level L2 L3

# Oracle mode (only the reference tools exposed — no distractors)
python -m harness.runner --dataset bfcl --model qwen2.5:7b --oracle

# Limit to first 10 tasks for a quick smoke test
python -m harness.runner --dataset jefferson --model qwen2.5:7b --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Make sure the project root is on the path when run as a script
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
from harness.mcp_session import filter_tools_for_task, mcp_session
from harness.model_client import ModelClient, ModelConfig, resolve_model_config

# ──────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ──────────────────────────────────────────────────────────────────────────────

DATASETS: dict[str, dict] = {
    "jefferson": {
        "tasks": {
            "L1": "datasets/jefferson_stats/tasks_l1.jsonl",
            "L2": "datasets/jefferson_stats/tasks_l2.jsonl",
            "L3": "datasets/jefferson_stats/tasks_l3.jsonl",
        },
        "server": "mcp-server/main.py",
    },
    "bfcl": {
        "tasks": {
            "L1": "datasets/bfcl_math/tasks_l1.jsonl",
            "L2": "datasets/bfcl_math/tasks_l2.jsonl",
            "L3": "datasets/bfcl_math/tasks_l3.jsonl",
        },
        "server": "mcp-server/main.py",
    },
     "bfcl-v2": {
        "tasks": {
            "L1": "datasets/bfcl_math/tasks_l1_v2.jsonl",
            "L2": "datasets/bfcl_math/tasks_l2_v2.jsonl",
            "L3": "datasets/bfcl_math/tasks_l3_v2.jsonl",
        },
        "server": "mcp-server/main.py",
    },
    "postgres": {
        "tasks": {
            "L1": "datasets/postgres/tasks_l1.jsonl",
            "L2": "datasets/postgres/tasks_l2.jsonl",
            "L3": "datasets/postgres/tasks_l3.jsonl",
        },
        "server": "mcp-server/main.py",
    },
    "finance": {
        "tasks": {
            "L1": "datasets/finance/tasks_l1.jsonl",
            "L2": "datasets/finance/tasks_l2.jsonl",
            "L3": "datasets/finance/tasks_l3.jsonl",
        },
        "server": "mcp-server/main.py",
    },
}

RESULTS_DIR = Path("results")


# ──────────────────────────────────────────────────────────────────────────────
# Task loading
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_bfcl_task(task: dict) -> dict:
    """Convert BFCL task format to runner's standard format."""
    query = task["question"][0][0]["content"]
    expected_call = task["expected_call"]
    return {
        "id": task.get("id"),
        "level": task.get("level", "L1"),
        "query": query,
        "functions": [expected_call["name"]],
        "expected_params": expected_call.get("arguments", {}),
        "expected_outcome": {"result": task["expected_result"]},
    }


def load_tasks(dataset: str, levels: list[str], limit: Optional[int]) -> list[dict]:
    registry = DATASETS[dataset]["tasks"]
    tasks = []
    for level in levels:
        if level not in registry:
            print(f"  [warn] no {level} tasks registered for dataset '{dataset}', skipping")
            continue
        path = Path(registry[level])
        if not path.exists():
            print(f"  [warn] task file not found: {path}, skipping")
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    t = json.loads(line)
                    t.setdefault("level", level)
                    if dataset in ("bfcl", "postgres") and "question" in t:
                        t = _normalize_bfcl_task(t)
                    tasks.append(t)
    if limit:
        tasks = tasks[:limit]
    return tasks


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

async def run_evaluation(
    dataset: str,
    model_cfg: ModelConfig,
    levels: list[str],
    limit: Optional[int],
    num_distractors: Optional[int],
    allow_fallback: bool,
) -> dict:
    tasks = load_tasks(dataset, levels, limit)
    if not tasks:
        print("No tasks loaded - check dataset paths.")
        return {}

    print(f"\nLoaded {len(tasks)} tasks from {dataset} ({', '.join(levels)})")

    client = ModelClient(model_cfg, allow_fallback=allow_fallback)
    server_script = Path(DATASETS[dataset]["server"])

    totals = dict(
        total_tests=len(tasks),
        correct_result=0,
        no_tool_call=0,
    )
    details: list[dict] = []

    async with mcp_session(server_script) as (session, all_tools):
        print(f"MCP server ready - {len(all_tools)} tools available\n")

        for i, task in enumerate(tasks, 1):
            task_id      = task.get("id", f"task_{i}")
            query        = task["query"]
            level        = task.get("level", "L1")

            # Reference functions — for log transparency only, not scored
            ref_functions = task.get("functions") or (
                [task["function"]] if "function" in task else []
            )
            optimal_steps = len(ref_functions) if ref_functions else 1

            print(f"[{i}/{len(tasks)}] {level} | {task_id}: {query[:65]}...")

            record: dict = {
                "task_id":            task_id,
                "level":              level,
                "query":              query,
                # Reference path (transparency only)
                "ref_functions":      ref_functions,
                # What the model actually did
                "actual_functions":   [],
                "actual_params":      None,
                "actual_result":      None,
                "correct_result":     False,
                "optimal_steps":      optimal_steps,
                "actual_steps":       0,
                "error":              None,
                "call_source":        "none",
                "raw_model_output":   None,
                "tool_result":        None,
                "expected_outcome":   None,
            }

            sys_content = "You are a helpful assistant. Use the provided tools when needed."
            if allow_fallback:
                sys_content += (
                    '\n\nIf you cannot emit a native tool call, respond with ONLY valid JSON: '
                    '{"tool":"<name>","args":{...}}'
                )
            messages = [
                {"role": "system", "content": sys_content},
                {"role": "user",   "content": query},
            ]

            # Allow a couple of steps beyond optimal in case the model takes
            # a valid alternative path that happens to be slightly longer
            max_steps   = optimal_steps + 2
            step_results: list = []

            for step in range(max_steps):
                tools_for_step = filter_tools_for_task(
                    all_tools,
                    relevant_names=ref_functions or list({t["function"]["name"] for t in all_tools}),
                    num_distractors=num_distractors,
                )

                try:
                    model_response = client.get_response(messages, tools_for_step)
                    tool_call      = model_response.tool_call
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

                record["actual_steps"] += 1
                record["actual_functions"].append(tool_call.function_name)
                record["actual_params"]  = tool_call.arguments
                record["call_source"]    = tool_call.call_source

                try:
                    raw_result   = await session.call_tool(
                        tool_call.function_name, tool_call.arguments
                    )
                    record["tool_result"]  = serialize_tool_result(raw_result)
                    result_value           = extract_result_value(raw_result)
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
                        "id":   f"call_{step}",
                        "type": "function",
                        "function": {
                            "name":      tool_call.function_name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }],
                })
                messages.append({
                    "role":         "tool",
                    "tool_call_id": f"call_{step}",
                    "content":      json.dumps(result_value),
                })

                if level in ("L1", "L2"):
                    break

            # ── Outcome evaluation (no routing checks) ────────────────────
            if step_results:
                expected_outcome          = task.get("expected_outcome")
                record["expected_outcome"] = expected_outcome

                if expected_outcome is not None:
                    # Deterministic tasks: check values across all step results
                    outcome_ok = (
                        compare_outcome_across_steps(step_results, expected_outcome)
                        if isinstance(expected_outcome, dict)
                        else compare_values(step_results[-1], expected_outcome)
                    )
                else:
                    # Non-deterministic tasks (finance/postgres): a successful
                    # tool call with no error is a pass
                    outcome_ok = record["actual_steps"] > 0 and record.get("error") is None

                if outcome_ok:
                    record["correct_result"] = True
                    totals["correct_result"] += 1

            details.append(record)

            wos_val = wos(
                outcome=record["correct_result"],
                optimal_steps=record["optimal_steps"],
                actual_steps=record["actual_steps"],
            )
            status = "OK" if record["correct_result"] else "X"
            # Log actual vs reference path for human inspection
            actual_path = "→".join(record["actual_functions"]) or "none"
            ref_path    = "→".join(ref_functions) or "?"
            print(
                f"  {status}  actual={actual_path} | ref={ref_path} | "
                f"steps={record['actual_steps']}/{record['optimal_steps']} | "
                f"wos={wos_val:.2f}"
            )

    metrics = calculate_metrics(details, totals)
    return {
        "model":           model_cfg.name,
        "backend":         model_cfg.backend,
        "base_url":        model_cfg.base_url,
        "dataset":         dataset,
        "levels":          levels,
        "num_distractors": num_distractors,
        "timestamp":       datetime.now().isoformat(),
        "metrics":         metrics,
        "details":         details,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_results(output: dict) -> Path:
    model_safe  = output["model"].replace(":", "_").replace("/", "_")
    dataset     = output["dataset"]
    dataset_dir = RESULTS_DIR / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = dataset_dir / f"{dataset}_{model_safe}_{ts}.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified MCP evaluation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset",  required=True, choices=list(DATASETS))
    p.add_argument("--model",    required=True,
                   help="Model identifier (e.g. qwen2.5:7b or meta-llama/Llama-3.1-8B-Instruct)")
    p.add_argument("--backend",  default="ollama", choices=["ollama", "vllm", "openai"])
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key",  default=None)
    p.add_argument("--level",    nargs="+", default=["L1", "L2", "L3"],
                   choices=["L1", "L2", "L3"])
    p.add_argument("--limit",    type=int, default=None)
    p.add_argument("--oracle",   action="store_true",
                   help="Oracle mode: expose only the reference tools per task")
    p.add_argument("--num-distractors", type=int, default=None)
    p.add_argument("--allow-fallback",  action="store_true")
    p.add_argument("--output",   type=Path, default=None)
    return p


def main():
    args      = _build_parser().parse_args()
    model_cfg = resolve_model_config(
        args.model,
        backend=args.backend,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    num_distractors = (
        args.num_distractors if args.num_distractors is not None
        else (0 if args.oracle else None)
    )

    print("=" * 62)
    print(f"  dataset  : {args.dataset}")
    print(f"  model    : {model_cfg.name}  [{model_cfg.backend}]")
    print(f"  endpoint : {model_cfg.base_url}")
    print(f"  levels   : {args.level}")
    print(f"  mode     : {'oracle' if num_distractors == 0 else 'standard' if num_distractors is None else f'{num_distractors} distractors'}")
    print("=" * 62)

    output = asyncio.run(run_evaluation(
        dataset=args.dataset,
        model_cfg=model_cfg,
        levels=args.level,
        limit=args.limit,
        num_distractors=num_distractors,
        allow_fallback=args.allow_fallback,
    ))

    if not output:
        sys.exit(1)

    print_report(output["metrics"], model_cfg.name, args.dataset)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        path = args.output
    else:
        path = save_results(output)

    print(f"Results saved -> {path}\n")

    try:
        from harness.db_logger import log_run
        log_run(output, suite=args.dataset, num_distractors=num_distractors)
    except Exception:
        pass


if __name__ == "__main__":
    main()
