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
    serialize_tool_result,
    wos,
)
from harness.mcp_session import filter_tools_for_task, mcp_session
from harness.model_client import ModelClient, resolve_model_config

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
    "finance_stage0": {
        "tasks": {
            "L1": "datasets/finance_stage0/tasks_l1.jsonl",
            "L2": "datasets/finance_stage0/tasks_l2.jsonl",
            "L3": "datasets/finance_stage0/tasks_l3.jsonl",
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
        "function": expected_call["name"],
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


def _find_subsequence_indices(actual: list[str], expected: list[str]) -> list[int] | None:
    """
    Return indices in `actual` that match `expected` in-order as a subsequence.
    Returns None if no full subsequence match exists.
    """
    if not expected:
        return []
    indices: list[int] = []
    exp_i = 0
    for act_i, fn in enumerate(actual):
        if fn == expected[exp_i]:
            indices.append(act_i)
            exp_i += 1
            if exp_i == len(expected):
                return indices
    return None


def _compare_step_params(
    called_params: list[dict],
    expected_params: object,
    matched_indices: list[int] | None = None,
) -> bool:
    """
    Compare expected params for single-step and multi-step tasks.

    - dict: compare against the first matched call (or first call if no match passed)
    - list[dict]: compare expected per-step params against matched call indices
    """
    if expected_params is None:
        return True
    target_indices = matched_indices or []
    if isinstance(expected_params, dict):
        idx = target_indices[0] if target_indices else 0
        first = called_params[idx] if idx < len(called_params) else {}
        return compare_params(first, expected_params)
    if isinstance(expected_params, list):
        if not target_indices:
            target_indices = list(range(len(called_params)))
        if len(target_indices) < len(expected_params):
            return False
        for i, exp in enumerate(expected_params):
            if not isinstance(exp, dict):
                return False
            idx = target_indices[i]
            if idx >= len(called_params):
                return False
            if not compare_params(called_params[idx], exp):
                return False
        return True
    return False


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

    # Running totals
    totals = dict(
        total_tests=len(tasks),
        correct_result=0,
        no_tool_call=0,
        wrong_tool=0,
        wrong_params=0,
    )
    details: list[dict] = []

    async with mcp_session(server_script) as (session, all_tools):
        print(f"MCP server ready - {len(all_tools)} tools available\n")

        for i, task in enumerate(tasks, 1):
            task_id = task.get("id", f"task_{i}")
            query = task["query"]
            level = task.get("level", "L1")
            expected_functions = task.get("functions") or [task.get("function", "?")]
            expected_fn = expected_functions[0]
            optimal_steps = len(expected_functions)

            print(f"[{i}/{len(tasks)}] {level} | {task_id}: {query[:65]}...")

            record: dict = {
                "task_id": task_id,
                "level": level,
                "query": query,
                "expected_function": expected_fn,
                "expected_functions": expected_functions,
                "expected_params": task.get("expected_params"),
                "actual_function": None,
                "actual_params": None,
                "actual_result": None,
                "called_functions": [],
                "called_params": [],
                "params_ok": None,
                "correct_result": False,
                "optimal_steps": optimal_steps,
                "actual_steps": 0,
                "error": None,
                "call_source": "none",
                "raw_model_output": None,
                "tool_result": None,
                "expected_outcome": None,
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
                expected_step_fn = expected_functions[min(step, len(expected_functions) - 1)]
                tools_for_step = filter_tools_for_task(
                    all_tools,
                    relevant_name=expected_step_fn,
                    num_distractors=num_distractors,
                )

                try:
                    model_response = client.get_response(messages, tools_for_step)
                    tool_call = model_response.tool_call
                    record["raw_model_output"] = model_response.raw_text
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
                record["called_functions"].append(tool_call.function_name)
                record["called_params"].append(tool_call.arguments)
                record["call_source"] = tool_call.call_source

                # Execute the tool
                try:
                    raw_result = await session.call_tool(
                        tool_call.function_name, tool_call.arguments
                    )
                    record["tool_result"] = serialize_tool_result(raw_result)
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

                # For L1/L2, one step is enough - break after first tool call
                if level in ("L1", "L2"):
                    break

            # ── Outcome evaluation ────────────────────────────────────────
            matched_indices = _find_subsequence_indices(
                record.get("called_functions", []),
                expected_functions,
            )
            fn_sequence_ok = matched_indices is not None
            if record["actual_steps"] > 0 and not fn_sequence_ok:
                totals["wrong_tool"] += 1

            if step_results:
                # Compare the final step result against expected_outcome
                expected_outcome = task.get("expected_outcome")
                record["expected_outcome"] = expected_outcome
                expected_params = task.get("expected_params")
                fn_ok = fn_sequence_ok
                params_ok = _compare_step_params(
                    record.get("called_params", []),
                    expected_params,
                    matched_indices=matched_indices,
                )
                record["params_ok"] = params_ok
                if expected_params is not None and not params_ok:
                    totals["wrong_params"] += 1
                if expected_outcome is not None:
                    outcome_ok = (
                        compare_values(step_results[-1], expected_outcome)
                        and fn_ok
                        and params_ok
                        and record.get("error") is None
                    )
                else:
                    # No fixed outcome: score on correct tool routing + params.
                    outcome_ok = fn_ok and params_ok and record.get("error") is None

                if outcome_ok:
                    record["correct_result"] = True
                    totals["correct_result"] += 1

            details.append(record)
            status = "OK" if record["correct_result"] else "X"
            wos_val = wos(
                outcome=record["correct_result"],
                optimal_steps=record["optimal_steps"],
                actual_steps=record["actual_steps"],
            )
            chain = "->".join(record.get("called_functions", [])) or "none"
            print(f"  {status}  fn={chain} | "
                  f"steps={record['actual_steps']}/{record['optimal_steps']} | "
                  f"wos={wos_val:.2f}")

    metrics = calculate_metrics(details, totals)
    return {
        "model": model_cfg.name,
        "backend": model_cfg.backend,
        "base_url": model_cfg.base_url,
        "dataset": dataset,
        "levels": levels,
        "num_distractors": num_distractors,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "details": details,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_results(output: dict) -> Path:
    model_safe = output["model"].replace(":", "_").replace("/", "_")
    dataset = output["dataset"]
    dataset_dir = RESULTS_DIR / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
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


def main():
    args = _build_parser().parse_args()
    model_cfg = resolve_model_config(
        args.model,
        backend=args.backend,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    num_distractors = args.num_distractors if args.num_distractors is not None \
                      else (0 if args.oracle else None)

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

    path = args.output or save_results(output)
    if not args.output:
        # save_results already wrote it
        pass
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(output, f, indent=2)

    print(f"Results saved -> {path}\n")

    # Cloud logging — silently skipped if SUPABASE_URL/KEY not in .env
    try:
        from harness.db_logger import log_run
        log_run(output, suite=args.dataset, num_distractors=num_distractors)
    except Exception:
        pass


if __name__ == "__main__":
    main()
