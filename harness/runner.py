"""
Unified evaluation runner.

Replaces evaluate_jefferson.py and evaluate_bfcl.py with a single entry point.

Scoring is outcome-first (MCPVerse-style WOS).

- Deterministic tasks (`expected_outcome` present): tool path does not affect
  scoring as long as the expected outcome is reached.
- Non-deterministic tasks (`expected_outcome` absent): when `expected_params`
  is provided, pass/fail requires the called arguments to match it.

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
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Make sure the project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.metrics import (
    calculate_metrics,
    compare_outcome_across_steps,
    compare_params,
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

AGENTIC_LOOP_FOOTER = """
TOOL-CALLING DISCIPLINE
- Never describe or narrate a tool call in text — always execute it directly.
- After receiving a tool result, immediately determine whether another tool \
call is required and make it. Do not summarise intermediate results in text \
until all tool calls are complete.
- When a condition gates the next step (e.g. "if result > X call tool A, \
otherwise call tool B"), evaluate the condition from the previous tool result \
and call the correct next tool immediately.
- Complete all required tool calls before writing any final response.
""".strip()

DATASETS: dict[str, dict] = {
    "jefferson": {
        "tasks": {
            "L1": "datasets/jefferson_stats/tasks_l1.jsonl",
            "L2": "datasets/jefferson_stats/tasks_l2.jsonl",
            "L3": "datasets/jefferson_stats/tasks_l3.jsonl",
        },
        "server": "mcp-server/main.py",
    },
    "jefferson-v2": {
        "tasks": {
            "L1": "datasets/jefferson_stats/tasks_l1_v2.jsonl",
            "L2": "datasets/jefferson_stats/tasks_l2_v2.jsonl",
            "L3": "datasets/jefferson_stats/tasks_l3_v2.jsonl",
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
    "postgres-v2": {
        "tasks": {
            "L1": "datasets/postgres/tasks_l1_v2.jsonl",
            "L2": "datasets/postgres/tasks_l2_v2.jsonl",
            "L3": "datasets/postgres/tasks_l3_v2.jsonl",
        },
        "server": "mcp-server/main.py",
    },
    "postgres_stage1": {
        "tasks": {
            "L1": "datasets/postgres_stage1/tasks_l1.jsonl",
            "L2": "datasets/postgres_stage1/tasks_l2.jsonl",
            "L3": "datasets/postgres_stage1/tasks_l3.jsonl",
        },
        "server": "mcp-server/main.py",
    },
    "postgres_stage1-v2": {
        "tasks": {
            "L1": "datasets/postgres_stage1/tasks_l1_v2.jsonl",
            "L2": "datasets/postgres_stage1/tasks_l2_v2.jsonl",
            "L3": "datasets/postgres_stage1/tasks_l3_v2.jsonl",
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
    "finance-v2": {
        "tasks": {
            "L1": "datasets/finance/tasks_l1_v2.jsonl",
            "L2": "datasets/finance/tasks_l2_v2.jsonl",
            "L3": "datasets/finance/tasks_l3_v2.jsonl",
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
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    t = json.loads(line)
                    t.setdefault("level", level)
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


def _matched_prefix_length(actual: list[str], expected: list[str]) -> int:
    """
    Count how many expected functions have been matched in-order so far.

    This is used while the run is still in progress so tool exposure only
    advances after the model actually completes the current expected step.
    """
    exp_i = 0
    for fn in actual:
        if exp_i < len(expected) and fn == expected[exp_i]:
            exp_i += 1
    return exp_i


def _compare_values_exact(actual: object, expected: object) -> bool:
    """
    Deep equality with numeric tolerance, but no subset semantics.

    Dict keys must match exactly (including nested dicts/lists).
    """
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual.keys()) != set(expected.keys()):
            return False
        return all(_compare_values_exact(actual[k], expected[k]) for k in expected)

    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return False
        return all(_compare_values_exact(a, e) for a, e in zip(actual, expected))

    return compare_values(actual, expected)


def _compare_params_exact(actual: dict, expected: dict) -> bool:
    """
    Exact parameter equality for strict datasets.

    - Same keys only (no extras/missing keys)
    - Values compared recursively with numeric tolerance
    """
    if set(actual.keys()) != set(expected.keys()):
        return False
    return all(_compare_values_exact(actual[k], v) for k, v in expected.items())


def _compare_step_params(
    called_params: list[dict],
    expected_params: object,
    matched_indices: list[int] | None = None,
    strict: bool = False,
) -> bool:
    """
    Compare expected params for single-step and multi-step tasks.

    - dict: compare against the first matched call.
      If matched_indices is None, compare against the first call.
      If matched_indices is an empty list, fail (no aligned function call).
    - list[dict]: compare expected per-step params against matched call indices.
      If matched_indices is None, fall back to call order.
    """
    if expected_params is None:
        return True
    param_comparator = _compare_params_exact if strict else compare_params
    if isinstance(expected_params, dict):
        if matched_indices is None:
            idx = 0
        elif not matched_indices:
            return False
        else:
            idx = matched_indices[0]
        first = called_params[idx] if idx < len(called_params) else {}
        return param_comparator(first, expected_params)
    if isinstance(expected_params, list):
        target_indices = (
            list(range(len(called_params)))
            if matched_indices is None
            else matched_indices
        )
        if len(target_indices) < len(expected_params):
            return False
        for i, exp in enumerate(expected_params):
            if not isinstance(exp, dict):
                return False
            idx = target_indices[i]
            if idx >= len(called_params):
                return False
            if not param_comparator(called_params[idx], exp):
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
    prompt_template: Optional[str] = None,
    debug_timing: bool = False,
) -> dict:
    from harness.mcp_session import filter_tools_for_task, mcp_session
    from harness.model_client import ModelClient

    tasks = load_tasks(dataset, levels, limit)
    strict_tool_param_checks = dataset in {"finance", "finance-v2"}
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
        wrong_tool=0,
        wrong_params=0,
    )
    details: list[dict] = []

    async with mcp_session(server_script) as (session, all_tools):
        print(f"MCP server ready - {len(all_tools)} tools available\n")

        for i, task in enumerate(tasks, 1):
            task_id      = task.get("id", f"task_{i}")
            query        = task["query"]
            level        = task.get("level", "L1")
            expected_params = task.get("expected_params")

            # Reference functions: used for tool exposure and expected-param
            # alignment on non-deterministic tasks.
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
                "expected_params":    expected_params,
                # What the model actually did
                "actual_functions":   [],
                "actual_params":      None,
                "actual_params_by_step": [],
                "matched_ref_indices": None,
                "tool_match":         None,
                "params_match":       None,
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

            if prompt_template:
                sys_content = prompt_template + "\n\n" + sys_content

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
            messages = [
                {"role": "system", "content": sys_content},
                {"role": "user",   "content": query},
            ]

            # Allow a couple of steps beyond optimal in case the model takes
            # a valid alternative path that happens to be slightly longer
            max_steps   = optimal_steps + 2
            step_results: list = []

            for step in range(max_steps):
                # Advance the exposed "relevant" tool only after the matching
                # expected step has actually been completed in-order.
                tools_for_step = filter_tools_for_task(
                    all_tools,
                    relevant_names=ref_functions if ref_functions else [],
                    num_distractors=num_distractors,
                )

                try:
                    model_start = time.perf_counter()
                    if debug_timing:
                        print(
                            f"    [debug] step {step + 1}: model call start "
                            f"({len(tools_for_step)} tools)",
                            flush=True,
                        )
                    model_response = client.get_response(messages, tools_for_step)
                    model_elapsed = time.perf_counter() - model_start
                    tool_call      = model_response.tool_call
                    record["raw_model_output"] = model_response.raw_text
                    if debug_timing:
                        print(
                            f"    [debug] step {step + 1}: model call end "
                            f"({model_elapsed:.2f}s)",
                            flush=True,
                        )
                except Exception as exc:
                    record["error"] = f"Model call failed: {exc}"
                    totals["no_tool_call"] += 1
                    break

                if tool_call is None:
                    if step == 0:
                        record["error"] = "Model made no tool call"
                        totals["no_tool_call"] += 1
                        break
                    # Mid-chain stop: nudge the model to continue if steps remain
                    steps_done = record["actual_steps"]
                    steps_needed = optimal_steps
                    if steps_done < steps_needed:
                        messages.append({
                            "role": "user",
                            "content": (
                                f"Continue. You have completed {steps_done} of "
                                f"{steps_needed} required steps. "
                                "Call the next required tool now."
                            ),
                        })
                        continue  
                    break

                record["actual_steps"] += 1
                record["actual_functions"].append(tool_call.function_name)
                record["actual_params"] = tool_call.arguments
                record["actual_params_by_step"].append(tool_call.arguments)
                record["call_source"]    = tool_call.call_source

                try:
                    tool_start = time.perf_counter()
                    if debug_timing:
                        print(
                            f"    [debug] step {step + 1}: tool call start "
                            f"({tool_call.function_name})",
                            flush=True,
                        )
                    raw_result   = await session.call_tool(
                        tool_call.function_name, tool_call.arguments
                    )
                    tool_elapsed = time.perf_counter() - tool_start
                    record["tool_result"]  = serialize_tool_result(raw_result)
                    result_value           = extract_result_value(raw_result)
                    step_results.append(result_value)
                    record["actual_result"] = result_value
                    if debug_timing:
                        print(
                            f"    [debug] step {step + 1}: tool call end "
                            f"({tool_elapsed:.2f}s)",
                            flush=True,
                        )
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


                if optimal_steps == 1:
                    break

            # ── Outcome evaluation ────────────────────────────────────────
            matched_indices: list[int] | None = None
            if ref_functions:
                matched_indices = _find_subsequence_indices(
                    record["actual_functions"],
                    ref_functions,
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
                # Finance datasets require exact tool path; other datasets use
                # in-order subsequence matching.
                if strict_tool_param_checks:
                    tool_ok = record["actual_functions"] == ref_functions
                else:
                    tool_ok = bool(matched_indices)
                record["tool_match"] = tool_ok

            if record["actual_functions"] and ref_functions and not tool_ok:
                totals["wrong_tool"] += 1
                        
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
                    # tool call with no error is a pass, but when expected
                    # params are provided they must match.
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
    p.add_argument("--debug-timing", action="store_true",
                   help="Print per-step model and tool timing diagnostics")
    p.add_argument(
       "--prompt-template",
       type=Path,
       default=None,
       metavar="FILE",
       help="Path to a plain-text prompt template file. Its contents are "
            "prepended to the system message for every task in the run.",
    )
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

    prompt_template: Optional[str] = None
    if args.prompt_template:
       template_path = Path(args.prompt_template)
       if not template_path.exists():
           print(f"[error] prompt template not found: {template_path}")
           sys.exit(1)
       prompt_template = template_path.read_text(encoding="utf-8").strip()
       print(f"  template : {template_path}")

    output = asyncio.run(run_evaluation(
        dataset=args.dataset,
        model_cfg=model_cfg,
        levels=args.level,
        limit=args.limit,
        num_distractors=num_distractors,
        allow_fallback=args.allow_fallback,
        prompt_template=prompt_template,
        debug_timing=args.debug_timing,
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
        log_run(
            output,
            suite=args.dataset,
            num_distractors=num_distractors,
            prompt_template=args.prompt_template,
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
