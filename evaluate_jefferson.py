#!/usr/bin/env python3
"""
Jefferson Stats Evaluation Script

Evaluates LLM function calling on statistical analysis tools from JeffersonStatsMCP.
Uses outcome-based evaluation E(O, Ô) ∈ {0, 1} following MCPVerse methodology.

Dataset: stats_test_cases.json (18 statistical functions)
Tools: 18 statistical analysis functions (basic, advanced, hypothesis, analysis)
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from inference_backend import (
    BackendConfig,
    add_backend_cli_args,
    create_openai_client,
    resolve_backend_config,
    run_chat_completion,
)

# Import shared evaluation framework
from evaluation_framework import (
    compare_values,
    compare_params,
    extract_result_value,
    calculate_metrics,
    mcp_tools_to_openai_tools,
    print_report,
    maybe_parse_fallback_tool_json,
    save_results
)

# Configuration
MCP_SERVER_PATH = Path(__file__).parent / "mcp-server" / "main.py"
TEST_CASES_PATH = Path(__file__).parent / "stats_test_cases.json"
RESULTS_DIR = Path("results") / "jefferson"


async def run_jefferson_evaluation(
    backend_config: BackendConfig,
    test_cases_path: Path,
    output_path: Path,
    limit: Optional[int] = None,
    allow_fallback: bool = False
):
    """
    Run evaluation on Jefferson statistical tools.
    
    Args:
        model: Model name (e.g., 'qwen2.5:7b')
        test_cases_path: Path to test cases JSON
        output_path: Path to save results
        limit: Optional limit on number of tests
        allow_fallback: Allow fallback JSON parsing for non-native tool models
    """
    print("=" * 60)
    print("Jefferson Stats Evaluation")
    print("=" * 60)
    print(f"Model: {backend_config.model}")
    print(f"Test Cases: {test_cases_path}")
    print(f"Provider: {backend_config.provider}")
    print(f"Base URL: {backend_config.openai_base_url}")
    print()
    
    # Load test cases
    with open(test_cases_path) as f:
        all_tests = json.load(f)
    
    if limit:
        all_tests = all_tests[:limit]
    
    print(f"Loaded {len(all_tests)} test cases\n")
    
    client = create_openai_client(backend_config)
    
    # Connect to MCP server
    server_params = StdioServerParameters(
        command="python",
        args=[str(MCP_SERVER_PATH)],
        env=None
    )
    
    print("Connecting to MCP server...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List available tools
            tools_list = await session.list_tools()
            print(f"Connected! Available tools: {len(tools_list.tools)}")
            
            # Convert to OpenAI format
            all_openai_tools = mcp_tools_to_openai_tools(tools_list)
            
            print(f"Running {len(all_tests)} tests...\n")
            
            # Initialize results
            results = {
                'model': backend_config.model,
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(all_tests),
                'correct_function': 0,
                'correct_params': 0,
                'correct_result': 0,
                'no_tool_call': 0,
                'wrong_tool': 0,
                'wrong_params': 0,
                'details': []
            }
            
            # Run tests
            for i, test in enumerate(all_tests, 1):
                print(f"Test {i}/{len(all_tests)}: {test['query'][:60]}...")
                
                test_result = {
                    'test_id': test['id'],
                    'query': test['query'],
                    'expected_function': test['expected_function'],
                    'expected_params': test['expected_params'],
                    'actual_function': None,
                    'actual_params': None,
                    'actual_result': None,
                    'correct_function': False,
                    'correct_params': False,
                    'correct_result': False,
                    'error': None,
                    'call_source': 'none'
                }
                
                try:
                    # Call model with tools
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant. Use the provided statistical analysis tools when needed."},
                        {"role": "user", "content": test['query']}
                    ]
                    
                    if allow_fallback:
                        messages[0]["content"] += (
                            "\n\nIf you want to call a tool but cannot emit a native tool call, "
                            'respond with ONLY valid JSON: {"tool":"<tool_name>","args":{...}}'
                        )
                    
                    response = run_chat_completion(
                        client,
                        backend_config,
                        messages,
                        all_openai_tools,
                        tool_choice="auto",
                    )
                    
                    message = response.choices[0].message
                    
                    # Extract tool call
                    actual_function = None
                    actual_params = None
                    
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        # Native tool call
                        tool_call = message.tool_calls[0]
                        actual_function = tool_call.function.name
                        actual_params = json.loads(tool_call.function.arguments)
                        test_result['call_source'] = 'native'
                    elif allow_fallback and hasattr(message, 'content') and message.content:
                        # Fallback JSON parsing
                        fallback = maybe_parse_fallback_tool_json(message.content)
                        if fallback:
                            actual_function = str(fallback["tool"])
                            actual_params = fallback["args"]
                            test_result['call_source'] = 'fallback'
                    
                    if actual_function:
                        test_result['actual_function'] = actual_function
                        test_result['actual_params'] = actual_params
                        
                        # Outcome-based evaluation: Execute and compare E(O, Ô)
                        try:
                            tool_result = await session.call_tool(actual_function, actual_params)
                            result_content = extract_result_value(tool_result)
                            test_result['actual_result'] = result_content
                            
                            # Get expected outcome
                            expected_outcome = test.get('expected_result')
                            
                            # E(O, Ô): Binary evaluation
                            outcome_matches = False
                            if expected_outcome is not None:
                                outcome_matches = compare_values(result_content, expected_outcome)
                            
                            if outcome_matches:
                                test_result['correct_result'] = True
                                results['correct_result'] += 1
                                
                                # Track auxiliary metrics
                                if actual_function == test['expected_function']:
                                    test_result['correct_function'] = True
                                    results['correct_function'] += 1
                                if compare_params(actual_params, test['expected_params']):
                                    test_result['correct_params'] = True
                                    results['correct_params'] += 1
                            else:
                                # Outcome mismatch - E(O, Ô) = 0
                                test_result['outcome_mismatch'] = {
                                    'expected': expected_outcome,
                                    'actual': result_content,
                                    'function_used': actual_function,
                                    'params_used': actual_params
                                }
                        
                        except Exception as e:
                            # Execution failure - E(O, Ô) = 0
                            test_result['error'] = f"Tool execution failed: {str(e)}"
                            results['no_tool_call'] += 1
                    else:
                        # No tool call - E(O, Ô) = 0
                        test_result['error'] = "Model did not make a tool call"
                        results['no_tool_call'] += 1
                
                except Exception as e:
                    test_result['error'] = f"Evaluation error: {str(e)}"
                    results['no_tool_call'] += 1
                
                results['details'].append(test_result)
                
                # Print status
                status = "✅ PASS" if test_result['correct_result'] else "❌ FAIL"
                print(f"  {status} Function: {test_result.get('actual_function', 'none')} | Expected: {test['expected_function']}")
            
            # Calculate metrics
            metrics = calculate_metrics(results)
            
            # Print report
            print()
            print_report(metrics, backend_config.model)
            
            # Save results
            save_results(output_path, backend_config.model, metrics, results)
            print(f"\n✅ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM function calling on Jefferson statistical tools"
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:7b",
        help="Ollama model name (default: qwen2.5:7b)"
    )
    parser.add_argument(
        "--test-cases",
        type=Path,
        default=TEST_CASES_PATH,
        help="Path to test cases JSON file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for results (default: results/jefferson/{model}_{timestamp}.json)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of test cases"
    )
    add_backend_cli_args(parser)
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow fallback JSON parsing for models without native tool support"
    )
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = args.model.replace(":", "_").replace("/", "_")
        args.output = RESULTS_DIR / f"{model_safe}_{timestamp}.json"

    backend_config = resolve_backend_config(
        model=args.model,
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    # Run evaluation
    asyncio.run(run_jefferson_evaluation(
        backend_config=backend_config,
        test_cases_path=args.test_cases,
        output_path=args.output,
        limit=args.limit,
        allow_fallback=args.allow_fallback
    ))


if __name__ == "__main__":
    main()
