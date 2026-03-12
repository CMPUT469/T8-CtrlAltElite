"""
Finance tool evaluation framework.

Modeled after evaluate_bfcl.py, but focused on the finance tools exposed by
mcp-server/main.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

# Add mcp-client to path for local project consistency with evaluate_bfcl.py
sys.path.append(str(Path(__file__).parent / "mcp-client"))

FINANCE_TOOLS = {
    "get_income_statements",
    "get_balance_sheets",
    "get_cash_flow_statements",
    "get_current_stock_price",
    "get_historical_stock_prices",
    "get_company_news",
    "get_available_crypto_tickers",
    "get_crypto_prices",
    "get_historical_crypto_prices",
    "get_current_crypto_price",
    "get_sec_filings",
    "getAnalystEstimates",
    "getFinancialMetrics",
    "getFinancialMetricsSnapshot",
    "getSegmentedRevenues",
    "getFilingItems",
    "getAvailableFilingItems",
    "getCompanyFacts",
}

DEFAULT_FINANCE_TESTS: List[Dict[str, Any]] = [
    {
        "id": "finance_001",
        "query": "Get annual income statements for Apple and return the latest 2 records.",
        "expected_function": "get_income_statements",
        "expected_params": {"ticker": "AAPL", "period": "annual", "limit": 2},
        "category": "finance",
    },
    {
        "id": "finance_002",
        "query": "Show quarterly balance sheets for Microsoft, limit to 3.",
        "expected_function": "get_balance_sheets",
        "expected_params": {"ticker": "MSFT", "period": "quarterly", "limit": 3},
        "category": "finance",
    },
    {
        "id": "finance_003",
        "query": "Fetch annual cash flow statements for Google, 2 results.",
        "expected_function": "get_cash_flow_statements",
        "expected_params": {"ticker": "GOOGL", "period": "annual", "limit": 2},
        "category": "finance",
    },
    {
        "id": "finance_004",
        "query": "What is the current stock price of Nvidia?",
        "expected_function": "get_current_stock_price",
        "expected_params": {"ticker": "NVDA"},
        "category": "finance",
    },
    {
        "id": "finance_005",
        "query": "Get Tesla daily stock prices from 2025-01-01 to 2025-01-31.",
        "expected_function": "get_historical_stock_prices",
        "expected_params": {
            "ticker": "TSLA",
            "start_date": "2025-01-01",
            "end_date": "2025-01-31",
            "interval": "day",
            "interval_multiplier": 1,
        },
        "category": "finance",
    },
    {
        "id": "finance_006",
        "query": "Get company news for Amazon.",
        "expected_function": "get_company_news",
        "expected_params": {"ticker": "AMZN"},
        "category": "finance",
    },
    {
        "id": "finance_007",
        "query": "List available crypto tickers.",
        "expected_function": "get_available_crypto_tickers",
        "expected_params": {},
        "category": "finance",
    },
    {
        "id": "finance_008",
        "query": "Get BTC-USD crypto prices from 2025-02-01 to 2025-02-07 with daily interval.",
        "expected_function": "get_crypto_prices",
        "expected_params": {
            "ticker": "BTC-USD",
            "start_date": "2025-02-01",
            "end_date": "2025-02-07",
            "interval": "day",
            "interval_multiplier": 1,
        },
        "category": "finance",
    },
    {
        "id": "finance_009",
        "query": "Get historical ETH-USD crypto prices from 2025-01-01 to 2025-01-10.",
        "expected_function": "get_historical_crypto_prices",
        "expected_params": {
            "ticker": "ETH-USD",
            "start_date": "2025-01-01",
            "end_date": "2025-01-10",
            "interval": "day",
            "interval_multiplier": 1,
        },
        "category": "finance",
    },
    {
        "id": "finance_010",
        "query": "What is the current price of BTC-USD?",
        "expected_function": "get_current_crypto_price",
        "expected_params": {"ticker": "BTC-USD"},
        "category": "finance",
    },
    {
        "id": "finance_011",
        "query": "Get the latest 5 10-K SEC filings for Apple.",
        "expected_function": "get_sec_filings",
        "expected_params": {"ticker": "AAPL", "limit": 5, "filing_type": "10-K"},
        "category": "finance",
    },
    {
        "id": "finance_012",
        "query": "Get annual analyst estimates for Apple and return the latest 2 records.",
        "expected_function": "getAnalystEstimates",
        "expected_params": {"ticker": "AAPL", "period": "annual", "limit": 2},
        "category": "finance",
    },
    {
        "id": "finance_013",
        "query": "Show quarterly financial metrics for Microsoft, limit to 3.",
        "expected_function": "getFinancialMetrics",
        "expected_params": {"ticker": "MSFT", "period": "quarterly", "limit": 3},
        "category": "finance",
    },
    {
        "id": "finance_014",
        "query": "What is the latest financial metrics snapshot for Nvidia?",
        "expected_function": "getFinancialMetricsSnapshot",
        "expected_params": {"ticker": "NVDA"},
        "category": "finance",
    },
    {
        "id": "finance_015",
        "query": "Get annual segmented revenues for Amazon with 2 records.",
        "expected_function": "getSegmentedRevenues",
        "expected_params": {"ticker": "AMZN", "period": "annual", "limit": 2},
        "category": "finance",
    },
    {
        "id": "finance_016",
        "query": "List the available filing items for 10-K filings.",
        "expected_function": "getAvailableFilingItems",
        "expected_params": {"filing_type": "10-K"},
        "category": "finance",
    },
    {
        "id": "finance_017",
        "query": "Get the Risk Factors section from Apple's 10-K filed on 2024-11-01.",
        "expected_function": "getFilingItems",
        "expected_params": {
            "ticker": "AAPL",
            "filing_type": "10-K",
            "filing_date": "2024-11-01",
            "item": "Risk Factors",
        },
        "category": "finance",
    },
    {
        "id": "finance_018",
        "query": "Get company facts for Tesla.",
        "expected_function": "getCompanyFacts",
        "expected_params": {"ticker": "TSLA"},
        "category": "finance",
    },
]


def _extract_query(example: Dict[str, Any]) -> str:
    """Extract user query from flexible test formats."""
    query = example.get("query")
    if isinstance(query, str):
        return query

    question = example.get("question")
    if isinstance(question, str):
        return question

    # BFCL-like nested question format: [[{"role":"user","content":"..."}]]
    if isinstance(question, list) and question:
        maybe_messages = question[0]
        if isinstance(maybe_messages, list):
            for message in maybe_messages:
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content", "")
                    if isinstance(content, str):
                        return content
    return ""


def load_finance_tests(test_file: Optional[str]) -> List[Dict[str, Any]]:
    """
    Load finance tests from file when available; otherwise use built-in defaults.

    Supported formats:
    - JSONL: one test per line
    - JSON: top-level list or {"test_cases": [...]}
    """
    if not test_file:
        print("No test file provided. Using built-in finance tests.")
        return DEFAULT_FINANCE_TESTS.copy()

    path = Path(test_file)
    if not path.exists():
        print(f"Test file not found: {path}")
        print("Falling back to built-in finance tests.")
        return DEFAULT_FINANCE_TESTS.copy()

    print(f"Loading finance tests from: {path}")
    try:
        if path.suffix.lower() == ".jsonl":
            raw_data = []
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw_data.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        print(f"Warning: Skipping malformed line {line_num}: {exc}")
        else:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and isinstance(loaded.get("test_cases"), list):
                raw_data = loaded["test_cases"]
            elif isinstance(loaded, list):
                raw_data = loaded
            else:
                print("Warning: Unsupported JSON format. Using built-in finance tests.")
                return DEFAULT_FINANCE_TESTS.copy()
    except Exception as exc:
        print(f"Error loading test file: {exc}")
        print("Falling back to built-in finance tests.")
        return DEFAULT_FINANCE_TESTS.copy()

    processed = normalize_finance_tests(raw_data)
    if not processed:
        print("No valid finance test cases found in file. Using built-in finance tests.")
        return DEFAULT_FINANCE_TESTS.copy()

    print(f"Loaded {len(processed)} finance test cases from file.")
    return processed


def normalize_finance_tests(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize different test schemas to a common format."""
    processed_tests: List[Dict[str, Any]] = []

    if not isinstance(data, list):
        return processed_tests

    for i, example in enumerate(data):
        if not isinstance(example, dict):
            continue

        query = _extract_query(example)
        if not query:
            continue

        expected_function = (
            example.get("expected_function")
            or example.get("expected_tool")
            or example.get("function")
        )

        if isinstance(expected_function, dict):
            expected_function = expected_function.get("name")

        # BFCL-style: function can be a list of function definitions
        if isinstance(expected_function, list) and expected_function:
            first = expected_function[0]
            if isinstance(first, dict):
                expected_function = first.get("name")

        if not isinstance(expected_function, str) or expected_function not in FINANCE_TOOLS:
            continue

        expected_params = example.get("expected_params", {})
        if not expected_params and isinstance(example.get("expected_call"), dict):
            expected_params = example["expected_call"].get("arguments", {})

        if not isinstance(expected_params, dict):
            expected_params = {}

        processed_tests.append(
            {
                "id": example.get("id", f"finance_test_{i}"),
                "query": query,
                "expected_function": expected_function,
                "expected_params": expected_params,
                "category": example.get("category", "finance"),
            }
        )

    return processed_tests


def _parse_tool_arguments(raw_args: Any) -> Dict[str, Any]:
    """Parse tool call arguments safely."""
    if isinstance(raw_args, dict):
        return raw_args

    if isinstance(raw_args, str):
        text = raw_args.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    return {}


def _compare_values(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True

    try:
        actual_num = float(actual)
        expected_num = float(expected)
        return abs(actual_num - expected_num) < 0.01
    except (ValueError, TypeError):
        return False


def _compare_params(actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    """
    Compare parameter dictionaries.

    Expected keys must be present in actual values and match.
    Extra keys in actual are allowed (except when expected is empty).
    """
    if not expected:
        return actual == {}

    for key, expected_val in expected.items():
        if key not in actual:
            return False

        actual_val = actual[key]
        if isinstance(expected_val, list) and isinstance(actual_val, list):
            if len(actual_val) != len(expected_val):
                return False
            for a, e in zip(actual_val, expected_val):
                if not _compare_values(a, e):
                    return False
        elif isinstance(expected_val, dict) and isinstance(actual_val, dict):
            if not _compare_params(actual_val, expected_val):
                return False
        else:
            if not _compare_values(actual_val, expected_val):
                return False

    return True


def _extract_tool_output(tool_result: Any) -> Any:
    """Normalize MCP tool return value for JSON serialization."""
    if hasattr(tool_result, "content"):
        content = tool_result.content
        if isinstance(content, list):
            extracted: List[Any] = []
            for item in content:
                if hasattr(item, "text"):
                    text = item.text
                    try:
                        extracted.append(json.loads(text))
                    except Exception:
                        extracted.append(text)
                elif hasattr(item, "model_dump"):
                    extracted.append(item.model_dump())
                else:
                    extracted.append(item)
            return extracted
        return content

    if hasattr(tool_result, "model_dump"):
        return tool_result.model_dump()

    return tool_result


async def evaluate_model(
    model: str,
    test_cases: List[Dict[str, Any]],
    limit: Optional[int] = None,
    num_distractors: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate model tool-calling on finance test cases."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating {model} on Finance MCP Tools")
    print(f"{'=' * 60}\n")

    if limit:
        test_cases = test_cases[:limit]
        print(f"Running {len(test_cases)} test cases (limited)\n")
    else:
        print(f"Running {len(test_cases)} test cases\n")

    ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).parent / "mcp-server" / "main.py")],
        env=None,
    )

    results: Dict[str, Any] = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(test_cases),
        "correct_function": 0,
        "correct_params": 0,
        "no_tool_call": 0,
        "wrong_tool": 0,
        "wrong_params": 0,
        "details": [],
    }

    rng = random.Random(42)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()
            all_tools = mcp_tools.tools
            print(f"Connected to MCP server with {len(all_tools)} total tools")

            all_openai_tools: List[Dict[str, Any]] = []
            for tool in all_tools:
                all_openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema or {},
                        },
                    }
                )

            available_tool_names = {tool["function"]["name"] for tool in all_openai_tools}
            missing_finance_tools = sorted(FINANCE_TOOLS - available_tool_names)
            if missing_finance_tools:
                print(f"Warning: Missing finance tools from MCP server: {missing_finance_tools}")

            for i, test in enumerate(test_cases, 1):
                print(f"Test {i}/{len(test_cases)}: {test['query'][:70]}...")
                expected_function = test["expected_function"]

                test_result: Dict[str, Any] = {
                    "test_id": test["id"],
                    "query": test["query"],
                    "expected_function": expected_function,
                    "expected_params": test["expected_params"],
                    "actual_function": None,
                    "actual_params": None,
                    "tool_output": None,
                    "tool_output_error": None,
                    "correct_function": False,
                    "correct_params": False,
                    "error": None,
                    "incorrect_output": None,
                }

                if expected_function not in available_tool_names:
                    test_result["error"] = f"Expected tool not present on server: {expected_function}"
                    results["wrong_tool"] += 1
                    results["details"].append(test_result)
                    print(f"  FAIL Function: none | Expected: {expected_function}")
                    continue

                try:
                    if num_distractors is not None:
                        relevant = [
                            t for t in all_openai_tools if t["function"]["name"] == expected_function
                        ]
                        distractors = [
                            t for t in all_openai_tools if t["function"]["name"] != expected_function
                        ]

                        if num_distractors == 0:
                            openai_tools = relevant
                        else:
                            sampled = rng.sample(
                                distractors, min(num_distractors, len(distractors))
                            )
                            openai_tools = relevant + sampled
                            rng.shuffle(openai_tools)
                    else:
                        openai_tools = all_openai_tools

                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a financial assistant. Use the provided tools when needed."
                            ),
                        },
                        {"role": "user", "content": test["query"]},
                    ]

                    response = ollama_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=openai_tools,
                        tool_choice="auto",
                    )

                    message = response.choices[0].message
                    tool_calls = getattr(message, "tool_calls", None)

                    if tool_calls and len(tool_calls) > 0:
                        tool_call = tool_calls[0]
                        actual_function = tool_call.function.name
                        actual_params = _parse_tool_arguments(tool_call.function.arguments)

                        test_result["actual_function"] = actual_function
                        test_result["actual_params"] = actual_params

                        # Capture tool output in saved JSON for every tool call.
                        try:
                            tool_result = await session.call_tool(actual_function, actual_params)
                            test_result["tool_output"] = _extract_tool_output(tool_result)
                        except Exception as exc:
                            test_result["tool_output_error"] = f"Tool execution failed: {exc}"

                        if actual_function == expected_function:
                            test_result["correct_function"] = True
                            results["correct_function"] += 1

                            if _compare_params(actual_params, test["expected_params"]):
                                test_result["correct_params"] = True
                                results["correct_params"] += 1
                            else:
                                results["wrong_params"] += 1
                                test_result["incorrect_output"] = {
                                    "called_function": actual_function,
                                    "actual_params": actual_params,
                                    "expected_params": test["expected_params"],
                                    "param_mismatch": True,
                                }
                        else:
                            results["wrong_tool"] += 1
                            test_result["incorrect_output"] = {
                                "called_function": actual_function,
                                "called_params": actual_params,
                                "expected_function": expected_function,
                            }
                    else:
                        results["no_tool_call"] += 1
                        test_result["error"] = "Model did not make a tool call"
                        test_result["incorrect_output"] = (
                            message.content if hasattr(message, "content") else None
                        )

                except Exception as exc:
                    test_result["error"] = f"Evaluation error: {exc}"
                    results["no_tool_call"] += 1

                results["details"].append(test_result)
                status = "PASS" if test_result["correct_params"] else "FAIL"
                actual = test_result["actual_function"] or "none"
                print(f"  {status} Function: {actual} | Expected: {expected_function}")

    return results


def calculate_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate F1/precision/recall and TSR metrics."""
    total = results["total_tests"]
    correct_function = results["correct_function"]
    correct_params = results["correct_params"]

    tsr_function = (correct_function / total * 100) if total > 0 else 0
    tsr_params = (correct_params / total * 100) if total > 0 else 0

    tools_called = total - results["no_tool_call"]
    precision = (correct_function / tools_called * 100) if tools_called > 0 else 0
    recall = (correct_function / total * 100) if total > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "f1_score": round(f1_score, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "tsr_function_selection": round(tsr_function, 2),
        "tsr_parameter_accuracy": round(tsr_params, 2),
        "total_tests": total,
        "correct_function": correct_function,
        "correct_params": correct_params,
        "no_tool_call": results["no_tool_call"],
        "wrong_tool": results["wrong_tool"],
        "wrong_params": results["wrong_params"],
    }


def print_report(metrics: Dict[str, Any], model: str) -> None:
    """Print evaluation summary."""
    print(f"\n{'=' * 60}")
    print(f"Finance Tool Evaluation Results - {model}")
    print(f"{'=' * 60}\n")

    print("Overall Metrics:")
    print(f"  F1 Score:              {metrics['f1_score']}%")
    print(f"  Precision:             {metrics['precision']}%")
    print(f"  Recall:                {metrics['recall']}%")
    print()

    print("Tool Selection Rate (TSR):")
    print(f"  Function Selection:    {metrics['tsr_function_selection']}%")
    print(f"  Parameter Accuracy:    {metrics['tsr_parameter_accuracy']}%")
    print()

    print("Breakdown:")
    print(f"  Total Tests:           {metrics['total_tests']}")
    print(f"  Correct Function:      {metrics['correct_function']}")
    print(f"  Correct Params:        {metrics['correct_params']}")
    print(f"  No Tool Call:          {metrics['no_tool_call']}")
    print(f"  Wrong Tool:            {metrics['wrong_tool']}")
    print(f"  Wrong Params:          {metrics['wrong_params']}")
    print()


def save_results(
    results: Dict[str, Any],
    metrics: Dict[str, Any],
    model: str,
    output_path: Optional[str],
) -> str:
    """Write evaluation output to disk."""
    output = {
        "model": model,
        "timestamp": results["timestamp"],
        "metrics": metrics,
        "raw_results": results,
    }

    if output_path:
        filename = output_path
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
    else:
        filename = f"finance_results_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {filename}")
    return filename


def _resolve_num_distractors(oracle_mode: bool, num_distractors: Optional[int]) -> Optional[int]:
    if num_distractors is not None:
        return max(0, num_distractors)
    return 0 if oracle_mode else None


async def main() -> None:
    parser = argparse.ArgumentParser(description="Finance tool evaluation for MCP models")
    parser.add_argument("--model", default="qwen2.5", help="Model to evaluate")
    parser.add_argument(
        "--test-file",
        default="finance_test_cases.jsonl",
        help="Path to finance test cases (.jsonl or .json)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tests")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Load and validate tests only, do not run evaluation",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument(
        "--oracle-mode",
        action="store_true",
        help="Only provide the relevant tool for each test (0 distractors)",
    )
    parser.add_argument(
        "--num-distractors",
        type=int,
        default=None,
        help="Number of distractor tools to include (0 = oracle mode)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Finance Tool Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Test file: {args.test_file}")
    print()

    test_cases = load_finance_tests(args.test_file)
    if args.limit:
        test_cases = test_cases[: args.limit]

    print(f"Prepared {len(test_cases)} finance test cases")
    if args.download_only:
        print("Validation complete (download-only mode).")
        return

    num_distractors = _resolve_num_distractors(args.oracle_mode, args.num_distractors)
    if num_distractors is None:
        print("Mode: STANDARD (all tools enabled)\n")
    elif num_distractors == 0:
        print("Mode: ORACLE (relevant tool only)\n")
    else:
        print(f"Mode: DISTRACTOR ({num_distractors} distractors per test)\n")

    results = await evaluate_model(
        model=args.model,
        test_cases=test_cases,
        limit=None,
        num_distractors=num_distractors,
    )
    metrics = calculate_metrics(results)
    print_report(metrics, args.model)
    output_file = save_results(results, metrics, args.model, args.output)

    print("Evaluation complete!")
    print("Next steps:")
    print(f"1. Review {output_file}")
    print("2. Add/expand finance_test_cases.jsonl for harder scenarios")
    print("3. Compare models by rerunning with --model <model_name>")


if __name__ == "__main__":
    asyncio.run(main())
