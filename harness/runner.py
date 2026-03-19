"""
Unified evaluation runner.

Replaces evaluate_jefferson.py and evaluate_bfcl.py with a single entry point.

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

# Oracle mode (only the correct tool exposed — no distractors)
python -m harness.runner --dataset bfcl --model qwen2.5:7b --oracle

# Limit to first 10 tasks for a quick smoke test
python -m harness.runner --dataset jefferson --model qwen2.5:7b --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Make sure the project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.metrics import (
    calculate_metrics,
    compare_params,
    compare_values,
    extract_result_value,
    print_report,
    wos,
)
from harness.mcp_session import filter_tools_for_task, mcp_session
from harness.model_client import ModelClient, ModelConfig

# ──────────────────────────────────────────────────────────────────────────────
# Dataset registry
# Each entry maps a dataset name to its JSONL path and the MCP server to use.
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
        },
        "server": "mcp-server/main.py",
    },
    "postgres": {
        "tasks": {
            "L1": "datasets/postgres/tasks_l1.jsonl",
        },
        "server": "mcp-server/main.py",
    },
}

RESULTS_DIR = Path("results")


# ──────────────────────────────────────────────────────────────────────────────
# Task loading
# ──────────────────────────────────────────────────────────────────────────────

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
        print("No tasks loaded — check dataset paths.")
        return {}

    print(f"\nLoaded {len(tasks)} tasks from {dataset} ({', '.join(levels)})")

    client = ModelClient(model_cfg, allow_fallback=allow_fallback)
    server_script = Path(DATASETS[dataset]["server"])

    # Running totals
    totals = dict(
        total_tests=len(tasks),
        correct_result=0,
        no_tool_call=0,
        wrong_tool=0,
    )
    details: list[dict] = []

    async with mcp_session(server_script) as (session, all_tools):
        print(f"MCP server ready — {len(all_tools)} tools available\n")

        for i, task in enumerate(tasks, 1):
            task_id = task.get("id", f"task_{i}")
            query = task["query"]
            level = task.get("level", "L1")
            # L3 tasks may list multiple functions; L1/L2 use a single string
            expected_fn = task.get("function") or task.get("functions", ["?"])[0]
            # For L3, optimal_steps = number of listed functions
            optimal_steps = len(task["functions"]) if "functions" in task else 1

            print(f"[{i}/{len(tasks)}] {level} | {task_id}: {query[:65]}...")

            record: dict = {
                "task_id": task_id,
                "level": level,
                "query": query,
                "expected_function": expected_fn,
                "actual_function": None,
                "actual_params": None,
                "actual_result": None,
                "correct_result": False,
                "optimal_steps": optimal_steps,
                "actual_steps": 0,
                "error": None,
                "call_source": "none",
            }

            # Build system prompt
            sys_content = "You are a helpful assistant. Use the provided tools when needed."
            if allow_fallback:
                sys_content += (
                    '\n\nIf you cannot emit a native tool call, respond with ONLY valid JSON: '
                    '{"tool":"<name>","args":{...}}'
                )
            messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": query},
            ]

            # ── Multi-step loop for L3 tasks ──────────────────────────────
            # For L1/L2 we still use the same loop — it just runs once.
            max_steps = optimal_steps + 2  # allow a couple of extra steps
            step_results = []

            for step in range(max_steps):
                tools_for_step = filter_tools_for_task(
                    all_tools,
                    relevant_name=expected_fn,
                    num_distractors=num_distractors,
                )

                try:
                    tool_call = client.get_tool_call(messages, tools_for_step)
                except Exception as exc:
                    record["error"] = f"Model call failed: {exc}"
                    totals["no_tool_call"] += 1
                    break

                if tool_call is None:
                    # Model stopped calling tools
                    if step == 0:
                        record["error"] = "Model made no tool call"
                        totals["no_tool_call"] += 1
                    break

                record["actual_steps"] += 1
                record["actual_function"] = tool_call.function_name
                record["actual_params"] = tool_call.arguments
                record["call_source"] = tool_call.call_source

                # Execute the tool
                try:
                    raw_result = await session.call_tool(
                        tool_call.function_name, tool_call.arguments
                    )
                    result_value = extract_result_value(raw_result)
                    step_results.append(result_value)
                    record["actual_result"] = result_value
                except Exception as exc:
                    record["error"] = f"Tool execution failed at step {step+1}: {exc}"
                    totals["no_tool_call"] += 1
                    break

                # Feed result back into conversation for multi-step tasks
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

                # For L1/L2, one step is enough — break after first tool call
                if level in ("L1", "L2"):
                    break

            # ── Outcome evaluation ────────────────────────────────────────
            if step_results:
                # Compare the final step result against expected_outcome
                expected_outcome = task.get("expected_outcome")
                if expected_outcome is not None:
                    outcome_ok = compare_values(step_results[-1], expected_outcome)
                else:
                    # No expected_outcome declared → pass if tool executed without error
                    outcome_ok = record.get("error") is None

                if outcome_ok:
                    record["correct_result"] = True
                    totals["correct_result"] += 1

                    exp_params = task.get("expected_params", {})
                    if record["actual_function"] != expected_fn:
                        totals["wrong_tool"] += 1

            details.append(record)
            status = "✓" if record["correct_result"] else "✗"
            wos_val = wos(
                outcome=record["correct_result"],
                optimal_steps=record["optimal_steps"],
                actual_steps=record["actual_steps"],
            )
            print(f"  {status}  fn={record['actual_function'] or 'none'} | "
                  f"steps={record['actual_steps']}/{record['optimal_steps']} | "
                  f"wos={wos_val:.2f}")

    metrics = calculate_metrics(details, totals)
    return {
        "model": model_cfg.name,
        "backend": model_cfg.backend,
        "dataset": dataset,
        "levels": levels,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "details": details,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_results(output: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_safe = output["model"].replace(":", "_").replace("/", "_")
    dataset = output["dataset"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{dataset}_{model_safe}_{ts}.json"
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
    p.add_argument("--dataset", required=True, choices=list(DATASETS),
                   help="Which dataset to evaluate")
    p.add_argument("--model", required=True,
                   help="Model identifier (e.g. qwen2.5:7b or meta-llama/Llama-3.1-8B-Instruct)")
    p.add_argument("--backend", default="ollama", choices=["ollama", "vllm", "openai"],
                   help="Provider backend (default: ollama)")
    p.add_argument("--base-url", default=None,
                   help="Override API base URL (default: http://localhost:11434/v1 for ollama)")
    p.add_argument("--api-key", default=None,
                   help="API key / bearer token for the endpoint")
    p.add_argument("--level", nargs="+", default=["L1", "L2", "L3"],
                   choices=["L1", "L2", "L3"],
                   help="Task levels to include (default: all)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap total tasks across all levels (useful for smoke tests)")
    p.add_argument("--oracle", action="store_true",
                   help="Oracle mode: expose only the correct tool per task")
    p.add_argument("--num-distractors", type=int, default=None,
                   help="Number of distractor tools (overrides --oracle)")
    p.add_argument("--allow-fallback", action="store_true",
                   help="Accept JSON-in-text tool calls from models without native support")
    p.add_argument("--output", type=Path, default=None,
                   help="Custom output path for results JSON")
    return p


_BACKEND_DEFAULTS = {
    "ollama": "http://localhost:11434/v1",
    "vllm":   "http://localhost:8000/v1",
    "openai": "https://api.openai.com/v1",
}


def main():
    args = _build_parser().parse_args()

    base_url = args.base_url or _BACKEND_DEFAULTS[args.backend]
    api_key  = args.api_key or os.environ.get("LLM_API_KEY", "none")

    model_cfg = ModelConfig(
        name=args.model,
        backend=args.backend,
        base_url=base_url,
        api_key=api_key,
    )

    num_distractors = args.num_distractors if args.num_distractors is not None \
                      else (0 if args.oracle else None)

    print("=" * 62)
    print(f"  dataset  : {args.dataset}")
    print(f"  model    : {args.model}  [{args.backend}]")
    print(f"  endpoint : {base_url}")
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

    print_report(output["metrics"], args.model, args.dataset)

    path = args.output or save_results(output)
    if not args.output:
        # save_results already wrote it
        pass
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(output, f, indent=2)

    print(f"Results saved → {path}\n")

    # Cloud logging — silently skipped if SUPABASE_URL/KEY not in .env
    try:
        from harness.db_logger import log_run
        log_run(output, suite=args.dataset, num_distractors=num_distractors)
    except Exception:
        pass


if __name__ == "__main__":
    main()