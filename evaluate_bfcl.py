"""
BFCL (Berkeley Function Calling Leaderboard) Evaluation

Downloads BFCL dataset from HuggingFace and calculates F1/TSR metrics
for qwen2.5 using the 14 BFCL math tools.

Dataset: gorilla-llm/Berkeley-Function-Calling-Leaderboard
Contains: 16k+ real-world tool-use instructions with multi-step reasoning
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from datasets import load_dataset
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

# Add mcp-client to path
sys.path.append(str(Path(__file__).parent / "mcp-client"))

BFCL_DATASET = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"


def download_bfcl_dataset(category: str = "simple"):
    """
    Download BFCL dataset from HuggingFace.
    
    Categories:
    - 'simple' - Single function calls with basic parameters
    - 'multiple' - Multiple function calls
    - 'parallel' - Parallel function execution
    - 'composite' - Multi-step reasoning chains
    """
    print(f"Downloading BFCL dataset (category: {category})...")
    
    try:
        # BFCL dataset structure on HuggingFace
        dataset = load_dataset(BFCL_DATASET, category)
        print(f"Downloaded {len(dataset['test'])} test cases")
        return dataset
    except Exception as e:
        print(f"Error downloading BFCL dataset: {e}")
        print(f"Install datasets library: pip install datasets")
        return None


def filter_math_tests(dataset) -> List[Dict]:
    """
    Filter test cases that use math tools (our 14 BFCL tools).
    """
    math_tools = {
        "add", "subtract", "multiply", "divide", "power", 
        "square_root", "absolute_value", "round_number", "percentage",
        "sum_values", "mean", "min_value", "max_value", "standard_deviation"
    }
    
    filtered_tests = []
    
    for example in dataset['test']:
        # BFCL format: check if function name is in our math tools
        function_name = example.get('function', '')
        
        if function_name in math_tools:
            filtered_tests.append({
                'id': example.get('id', ''),
                'query': example.get('question', ''),
                'expected_function': function_name,
                'expected_params': example.get('parameters', {}),
                'expected_result': example.get('expected_output', None),
                'category': example.get('category', 'simple')
            })
    
    print(f"Filtered {len(filtered_tests)} math tool test cases")
    return filtered_tests


async def evaluate_model(model: str, test_cases: List[Dict], limit: Optional[int] = None):
    """
    Run BFCL test cases against the model and calculate F1/TSR.
    
    Args:
        model: Model name (e.g., 'qwen2.5')
        test_cases: List of BFCL test cases
        limit: Optional limit on number of tests to run
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model} on BFCL Math Tools")
    print(f"{'='*60}\n")
    
    if limit:
        test_cases = test_cases[:limit]
        print(f"Running {limit} test cases (limited)\n")
    else:
        print(f"Running {len(test_cases)} test cases\n")
    
    # Initialize Ollama client (OpenAI-compatible)
    ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    
    # Start MCP server
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).parent / "mcp-server" / "main.py")],
        env=None
    )
    
    results = {
        'model': model,
        'timestamp': datetime.now().isoformat(),
        'total_tests': len(test_cases),
        'correct_function': 0,
        'correct_params': 0,
        'correct_result': 0,
        'no_tool_call': 0,
        'wrong_tool': 0,
        'wrong_params': 0,
        'details': []
    }
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Get available tools
            mcp_tools = await session.list_tools()
            print(f"Connected to MCP server with {len(mcp_tools.tools)} tools\n")
            
            # Convert MCP tools to OpenAI format
            openai_tools = []
            for tool in mcp_tools.tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {}
                    }
                })
            
            # Run each test case
            for i, test in enumerate(test_cases, 1):
                print(f"Test {i}/{len(test_cases)}: {test['query'][:60]}...")
                
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
                    'error': None
                }
                
                try:
                    # Call Ollama model with tools
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant. Use the provided tools when needed."},
                        {"role": "user", "content": test['query']}
                    ]
                    
                    response = ollama_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=openai_tools,
                        tool_choice="auto"
                    )
                    
                    message = response.choices[0].message
                    tool_calls = getattr(message, 'tool_calls', None)
                    
                    if tool_calls and len(tool_calls) > 0:
                        # Extract first tool call
                        tool_call = tool_calls[0]
                        actual_function = tool_call.function.name
                        actual_params = json.loads(tool_call.function.arguments or "{}")
                        
                        test_result['actual_function'] = actual_function
                        test_result['actual_params'] = actual_params
                        
                        # Check function correctness
                        if actual_function == test['expected_function']:
                            test_result['correct_function'] = True
                            results['correct_function'] += 1
                            
                            # Check parameter correctness
                            params_match = _compare_params(actual_params, test['expected_params'])
                            if params_match:
                                test_result['correct_params'] = True
                                results['correct_params'] += 1
                                
                                # Execute tool and check result
                                try:
                                    tool_result = await session.call_tool(actual_function, actual_params)
                                    result_content = _extract_result_value(tool_result)
                                    test_result['actual_result'] = result_content
                                    
                                    # Compare with expected result
                                    if test.get('expected_result') is not None:
                                        if _compare_results(result_content, test['expected_result']):
                                            test_result['correct_result'] = True
                                            results['correct_result'] += 1
                                    else:
                                        # If no expected result, count as correct if executed
                                        test_result['correct_result'] = True
                                        results['correct_result'] += 1
                                        
                                except Exception as e:
                                    test_result['error'] = f"Tool execution failed: {str(e)}"
                            else:
                                results['wrong_params'] += 1
                        else:
                            results['wrong_tool'] += 1
                    else:
                        # No tool call made
                        results['no_tool_call'] += 1
                        test_result['error'] = "Model did not make a tool call"
                        
                except Exception as e:
                    test_result['error'] = f"Evaluation error: {str(e)}"
                    results['no_tool_call'] += 1
                
                results['details'].append(test_result)
                
                # Print quick status
                status = "PASS" if test_result['correct_result'] else "FAIL"
                print(f"  {status} Function: {test_result.get('actual_function', 'none')} | Expected: {test['expected_function']}")
    
    return results


def _compare_params(actual: Dict, expected: Dict) -> bool:
    """Compare parameter dictionaries, allowing for type coercion."""
    if set(actual.keys()) != set(expected.keys()):
        return False
    
    for key in expected.keys():
        actual_val = actual[key]
        expected_val = expected[key]
        
        # Handle list comparison
        if isinstance(expected_val, list) and isinstance(actual_val, list):
            if len(actual_val) != len(expected_val):
                return False
            for a, e in zip(actual_val, expected_val):
                if not _compare_values(a, e):
                    return False
        else:
            if not _compare_values(actual_val, expected_val):
                return False
    
    return True


def _compare_values(actual: Any, expected: Any) -> bool:
    """Compare two values with type coercion."""
    # Try direct comparison
    if actual == expected:
        return True
    
    # Try numeric comparison with tolerance
    try:
        actual_num = float(actual)
        expected_num = float(expected)
        return abs(actual_num - expected_num) < 0.01
    except (ValueError, TypeError):
        pass
    
    return False


def _compare_results(actual: Any, expected: Any) -> bool:
    """Compare result values with tolerance for floats."""
    return _compare_values(actual, expected)


def _extract_result_value(tool_result: Any) -> Any:
    """Extract the actual result value from MCP tool result."""
    # Handle different MCP result formats
    if hasattr(tool_result, 'content'):
        content = tool_result.content
        if isinstance(content, list) and len(content) > 0:
            item = content[0]
            if hasattr(item, 'text'):
                # Parse JSON from text
                try:
                    data = json.loads(item.text)
                    return data.get('result', data)
                except:
                    return item.text
            return item
        return content
    
    if hasattr(tool_result, 'model_dump'):
        return tool_result.model_dump()
    
    return tool_result


def calculate_metrics(results: Dict) -> Dict:
    """
    Calculate F1 score and TSR from evaluation results.
    
    F1 Score: Harmonic mean of precision and recall
    TSR (Tool Selection Rate): Percentage of correct tool selections
    """
    total = results['total_tests']
    correct_function = results['correct_function']
    correct_params = results['correct_params']
    correct_result = results['correct_result']
    
    # Tool Selection Rate (TSR)
    tsr_function = (correct_function / total * 100) if total > 0 else 0
    tsr_params = (correct_params / total * 100) if total > 0 else 0
    tsr_result = (correct_result / total * 100) if total > 0 else 0
    
    # F1 Score (for function selection)
    # Precision: correct tools / tools called
    tools_called = total - results['no_tool_call']
    precision = (correct_function / tools_called * 100) if tools_called > 0 else 0
    
    # Recall: correct tools / total tests
    recall = (correct_function / total * 100) if total > 0 else 0
    
    # F1 = 2 * (P * R) / (P + R)
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'f1_score': round(f1_score, 2),
        'precision': round(precision, 2),
        'recall': round(recall, 2),
        'tsr_function_selection': round(tsr_function, 2),
        'tsr_parameter_accuracy': round(tsr_params, 2),
        'tsr_result_accuracy': round(tsr_result, 2),
        'total_tests': total,
        'correct_function': correct_function,
        'correct_params': correct_params,
        'correct_result': correct_result,
        'no_tool_call': results['no_tool_call'],
        'wrong_tool': results['wrong_tool']
    }
    
    return metrics


def print_report(metrics: Dict, model: str):
    """Print evaluation report."""
    print(f"\n{'='*60}")
    print(f"BFCL Evaluation Results - {model}")
    print(f"{'='*60}\n")
    
    print(f"Overall Metrics:")
    print(f"  F1 Score:              {metrics['f1_score']}%")
    print(f"  Precision:             {metrics['precision']}%")
    print(f"  Recall:                {metrics['recall']}%")
    print()
    
    print(f"Tool Selection Rate (TSR):")
    print(f"  Function Selection:    {metrics['tsr_function_selection']}%")
    print(f"  Parameter Accuracy:    {metrics['tsr_parameter_accuracy']}%")
    print(f"  Result Accuracy:       {metrics['tsr_result_accuracy']}%")
    print()
    
    print(f"Breakdown:")
    print(f"  Total Tests:           {metrics['total_tests']}")
    print(f"  Correct Function:      {metrics['correct_function']}")
    print(f"  Correct Params:        {metrics['correct_params']}")
    print(f"  Correct Result:        {metrics['correct_result']}")
    print(f"  No Tool Call:          {metrics['no_tool_call']}")
    print(f"  Wrong Tool:            {metrics['wrong_tool']}")
    print()


def save_results(results: Dict, metrics: Dict, model: str):
    """Save evaluation results to JSON file."""
    output = {
        'model': model,
        'timestamp': results['timestamp'],
        'metrics': metrics,
        'raw_results': results
    }
    
    filename = f"bfcl_results_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {filename}")


async def main():
    """Main evaluation workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BFCL Evaluation for MCP Models")
    parser.add_argument("--model", default="qwen2.5", help="Model to evaluate")
    parser.add_argument("--category", default="simple", choices=["simple", "multiple", "parallel", "composite"],
                       help="BFCL test category")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tests")
    parser.add_argument("--download-only", action="store_true", help="Only download dataset, don't evaluate")
    
    args = parser.parse_args()
    
    print("="*60)
    print("BFCL Evaluation - Berkeley Function Calling Leaderboard")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Category: {args.category}")
    print(f"Dataset: {BFCL_DATASET}")
    print()
    
    # Download BFCL dataset
    dataset = download_bfcl_dataset(args.category)
    if not dataset:
        return
    
    # Filter to math tools
    test_cases = filter_math_tests(dataset)
    
    if args.download_only:
        print(f"\nDataset downloaded and filtered")
        print(f"   {len(test_cases)} math tool test cases ready")
        return
    
    # Run evaluation
    results = await evaluate_model(args.model, test_cases, args.limit)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print report
    print_report(metrics, args.model)
    
    # Save results
    save_results(results, metrics, args.model)
    
    print(f"\nEvaluation complete!")
    print(f"\nNext steps:")
    print(f"1. Review results in bfcl_results_{args.model}_*.json")
    print(f"2. Run for other models: python evaluate_bfcl.py --model llama3.2")
    print(f"3. Compare metrics across models")


if __name__ == "__main__":
    asyncio.run(main())
