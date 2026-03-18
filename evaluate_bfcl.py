"""
BFCL Evaluation Framework

Evaluates LLM function calling accuracy using Berkeley Function Calling Leaderboard.
Measures F1 score, precision, recall, and tool selection rate across multiple test categories.

Dataset: gorilla-llm/Berkeley-Function-Calling-Leaderboard
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
from huggingface_hub import hf_hub_download
import json
# Add mcp-client to path
sys.path.append(str(Path(__file__).parent / "mcp-client"))

BFCL_DATASET = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_RESULTS_DIR = Path("results") / "bfcl"
NATIVE_TOOLS_INCOMPATIBLE_MSG = (
    "Model {model} is not native tool-calling compatible under Ollama OpenAI endpoint. "
    "Use a tools-capable model (Tools category: Llama 3.1, etc.)."
)


def _maybe_parse_fallback_tool_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse fallback JSON for models that answer in text instead of native tool_calls.
    Expected format:
      {"tool":"<name>","args":{...}}
    """
    if not text:
        return None

    candidate = text.strip()

    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()
        else:
            candidate = candidate.strip("`").strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()

    if not (candidate.startswith("{") and candidate.endswith("}")):
        return None

    try:
        payload = json.loads(candidate)
    except Exception:
        return None

    if isinstance(payload, dict) and "tool" in payload and isinstance(payload.get("args"), dict):
        return payload

    return None


def _probe_native_tool_support(ollama_client: OpenAI, model: str) -> tuple[bool, Optional[str]]:
    """
    Send a minimal native tool-calling request to check model compatibility.
    """
    try:
        ollama_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Call the ping tool."}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "ping",
                        "description": "Return a simple ping payload.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "string"}
                            },
                            "required": ["value"]
                        },
                    },
                }
            ],
            tool_choice="auto"
        )
        return True, None
    except Exception as exc:
        message = str(exc)
        if "does not support tools" in message.lower():
            return False, NATIVE_TOOLS_INCOMPATIBLE_MSG.format(model=model)
        return True, None


def _build_incompatible_results(model: str, test_cases: List[Dict], message: str) -> Dict:
    """Build a BFCL result payload when model does not support native tool calling."""
    timestamp = datetime.now().isoformat()
    details = []
    for test in test_cases:
        details.append({
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
            'error': message,
            'incorrect_output': None,
            'call_source': 'none',
        })

    return {
        'model': model,
        'timestamp': timestamp,
        'total_tests': len(test_cases),
        'correct_function': 0,
        'correct_params': 0,
        'correct_result': 0,
        'no_tool_call': len(test_cases),
        'wrong_tool': 0,
        'wrong_params': 0,
        'details': details,
        'model_native_tools_supported': False,
        'allow_fallback': False,
    }


def download_bfcl_dataset(category: str = "simple"):
    """
    Load BFCL dataset - tries local math_test_cases.jsonl first, then HuggingFace.
    
    Categories (for HF):
    - 'simple' - Single function calls with basic parameters
    - 'multiple' - Multiple function calls
    - 'parallel' - Parallel function execution
    - 'composite' - Multi-step reasoning chains
    """
    
    # Try local math test cases first
    local_test_file = Path("math_test_cases.jsonl")
    if local_test_file.exists():
        print(f"Loading local math test cases from {local_test_file}...")
        try:
            with open(local_test_file, 'r') as f:
                data = [json.loads(line.strip()) for line in f if line.strip()]
            print(f"Loaded {len(data)} math test cases from local file")
            return data, "math"
        except Exception as e:
            print(f"Error loading local test file: {e}")
            print("Falling back to HuggingFace dataset...")
    
    # Fall back to HuggingFace BFCL dataset
    print(f"Downloading BFCL dataset from HuggingFace...")
    
    try:
        # Download the specific JSON file for the category
        filename = f"BFCL_v3_live_{category}.json"
        print(f"Downloading file: {filename}")
        
        file_path = hf_hub_download(
            repo_id=BFCL_DATASET,
            filename=filename,
            repo_type="dataset"
        )
        
        # Load JSON manually
        with open(file_path, 'r') as f:
            data = []
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    # Skip malformed lines
                    print(f"Warning: Skipping malformed line {line_num}: {e}")
                    continue
        
        print(f"Downloaded {len(data)} test cases")
        return data, category
        
    except Exception as e:
        print(f"Error downloading BFCL dataset: {e}")
        print(f"Tip: Generate local math tests with: python generate_math_tests.py")
        return None, None


def filter_math_tests(data, category: str = None) -> List[Dict]:
    """
    Process BFCL test cases to extract math tool tests.
    """
    math_tools = {
        "add", "subtract", "multiply", "divide", "power", 
        "square_root", "absolute_value", "round_number", "percentage",
        "sum_values", "mean", "min_value", "max_value", "standard_deviation"
    }
    
    processed_tests = []
    
    if not isinstance(data, list):
        print("Warning: Expected data to be a list")
        return []
    
    print(f"\nProcessing {len(data)} test cases...")
    
    for i, example in enumerate(data):
        try:
            if not isinstance(example, dict):
                continue
            
            # BFCL format: function is a LIST of function definitions
            func_list = example.get('function', [])
            if not func_list or not isinstance(func_list, list):
                # Try direct function name
                function_name = example.get('function', '')
                if isinstance(function_name, str) and function_name in math_tools:
                    # Simple format
                    query = example.get('question', '')
                    if isinstance(query, list) and len(query) > 0:
                        if isinstance(query[0], list) and len(query[0]) > 0:
                            query = query[0][0].get('content', '') if isinstance(query[0][0], dict) else str(query)
                    
                    processed_tests.append({
                        'id': example.get('id', f'test_{i}'),
                        'query': query,
                        'expected_function': function_name,
                        'expected_params': example.get('expected_call', {}).get('arguments', {}),
                        'category': category or 'math'
                    })
                continue
            
            # Extract the first function definition
            func_def = func_list[0] if func_list else {}
            function_name = func_def.get('name', '')
            
            if function_name not in math_tools:
                continue
            
            # Get the question (nested list in BFCL format)
            question_list = example.get('question', [[]])
            query = ''
            if question_list and isinstance(question_list, list):
                messages = question_list[0] if question_list else []
                if messages and isinstance(messages, list):
                    for msg in messages:
                        if isinstance(msg, dict) and msg.get('role') == 'user':
                            query = msg.get('content', '')
                            break
            
            if query and function_name:
                # Extract expected parameters from expected_call if available
                expected_params = {}
                if 'expected_call' in example:
                    expected_params = example['expected_call'].get('arguments', {})
                
                processed_tests.append({
                    'id': example.get('id', f'test_{i}'),
                    'query': query,
                    'expected_function': function_name,
                    'expected_params': expected_params,
                    'category': category or 'math'
                })
                
        except Exception as e:
            print(f"Warning: Error processing example {i}: {e}")
            continue
    
    print(f"Processed {len(processed_tests)} math tool test cases\n")
    return processed_tests


async def evaluate_model(
    model: str,
    test_cases: List[Dict],
    limit: Optional[int] = None,
    num_distractors: Optional[int] = None,
    allow_fallback: bool = False,
):
    """
    Run BFCL test cases against the model and calculate F1/TSR.
    
    Args:
        model: Model name (e.g., 'qwen2.5')
        test_cases: List of BFCL test cases
        limit: Optional limit on number of tests to run
        num_distractors: Number of distractor tools (0=oracle, None=all tools)
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
        'details': [],
        'model_native_tools_supported': True,
        'allow_fallback': allow_fallback,
    }
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            if not allow_fallback:
                supports_native_tools, compatibility_message = _probe_native_tool_support(ollama_client, model)
                results['model_native_tools_supported'] = supports_native_tools
                if not supports_native_tools:
                    print(compatibility_message)
                    return _build_incompatible_results(model, test_cases, compatibility_message)
            
            # Get available tools
            mcp_tools = await session.list_tools()
            all_tools = mcp_tools.tools
            print(f"Connected to MCP server with {len(all_tools)} tools")
            
            if num_distractors is not None:
                if num_distractors == 0:
                    print("Running in ORACLE MODE (0 distractors)\n")
                else:
                    print(f"Running with {num_distractors} DISTRACTORS ({num_distractors + 1} tools per test)\n")
            else:
                print(f"Running in STANDARD MODE (all {len(all_tools)} tools available)\n")
            
            # Convert MCP tools to OpenAI format
            all_openai_tools = []
            for tool in all_tools:
                all_openai_tools.append({
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
                    'error': None,
                    'incorrect_output': None,  # Captures actual model output when there's an error
                    'call_source': 'none',
                }
                
                try:
                    # Sample tools based on num_distractors
                    if num_distractors is not None:
                        # Get the relevant tool
                        relevant_tool = [t for t in all_openai_tools if t["function"]["name"] == test['expected_function']]
                        
                        if num_distractors == 0:
                            # Oracle mode: only relevant tool
                            openai_tools = relevant_tool
                        else:
                            # Sample N distractors
                            import random
                            distractors = [t for t in all_openai_tools if t["function"]["name"] != test['expected_function']]
                            sampled_distractors = random.sample(distractors, min(num_distractors, len(distractors)))
                            openai_tools = relevant_tool + sampled_distractors
                            random.shuffle(openai_tools)  # Randomize order
                    else:
                        # Use all tools
                        openai_tools = all_openai_tools
                    
                    # Call Ollama model with tools
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant. Use the provided tools when needed."},
                        {"role": "user", "content": test['query']}
                    ]
                    if allow_fallback:
                        messages[0]["content"] = (
                            "You are a helpful assistant. Use the provided tools when needed.\n\n"
                            "If you want to call a tool but cannot emit a native tool call, "
                            "respond with ONLY valid JSON in this exact format:\n"
                            '{"tool":"<tool_name>","args":{...}}'
                        )
                    
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
                        test_result['call_source'] = 'native'
                    elif allow_fallback:
                        raw_text = (message.content or "").strip() if hasattr(message, 'content') else ""
                        fallback = _maybe_parse_fallback_tool_json(raw_text)
                        if fallback:
                            actual_function = str(fallback["tool"])
                            actual_params = fallback["args"]
                            test_result['incorrect_output'] = raw_text
                            test_result['call_source'] = 'fallback'
                        else:
                            actual_function = None
                            actual_params = None
                    else:
                        actual_function = None
                        actual_params = None

                    if actual_function is not None:
                        test_result['actual_function'] = actual_function
                        test_result['actual_params'] = actual_params

                        # Outcome-based evaluation: E(O, Ô) ∈ {0, 1}
                        # Execute tool call and compare outcome with ground truth
                        try:
                            tool_result = await session.call_tool(actual_function, actual_params)
                            result_content = _extract_result_value(tool_result)
                            test_result['actual_result'] = result_content

                            # Get expected outcome from ground truth
                            expected_outcome = test.get('expected_result')
                            
                            # E(O, Ô): Binary evaluation based on outcome match
                            outcome_matches = False
                            if expected_outcome is not None:
                                outcome_matches = _compare_results(result_content, expected_outcome)
                            
                            # Binary score: 1 if outcomes match, 0 otherwise
                            if outcome_matches:
                                test_result['correct_result'] = True
                                results['correct_result'] += 1
                                # Track auxiliary metrics for analysis
                                if actual_function == test['expected_function']:
                                    test_result['correct_function'] = True
                                    results['correct_function'] += 1
                                if _compare_params(actual_params, test['expected_params']):
                                    test_result['correct_params'] = True
                                    results['correct_params'] += 1
                            else:
                                # Outcome mismatch - E(O, Ô) = 0
                                test_result['correct_result'] = False
                                test_result['outcome_mismatch'] = {
                                    'expected': expected_outcome,
                                    'actual': result_content,
                                    'function_used': actual_function,
                                    'params_used': actual_params
                                }

                        except Exception as e:
                            # Execution failure - E(O, Ô) = 0
                            test_result['correct_result'] = False
                            test_result['error'] = f"Tool execution failed: {str(e)}"
                            results['no_tool_call'] += 1
                    else:
                        # No tool call made - E(O, Ô) = 0
                        test_result['correct_result'] = False
                        results['no_tool_call'] += 1
                        test_result['error'] = "Model did not make a tool call"
                        # Capture the actual text content the model generated
                        test_result['incorrect_output'] = message.content if hasattr(message, 'content') else None
                        
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


def _compare_values(actual: Any, expected: Any, tolerance: float = 0.01) -> bool:
    """Compare two values with type coercion and tolerance for numerical values."""
    # Try direct comparison
    if actual == expected:
        return True
    
    # Handle dict vs primitive comparison (e.g., {"result": 1.803} vs 1.803)
    if isinstance(expected, dict) and not isinstance(actual, dict):
        # Extract value from dict (try common keys)
        for key in ['result', 'value', 'output']:
            if key in expected:
                expected = expected[key]
                break
    elif isinstance(actual, dict) and not isinstance(expected, dict):
        # Extract value from dict
        for key in ['result', 'value', 'output']:
            if key in actual:
                actual = actual[key]
                break
    
    # Try numeric comparison with tolerance
    try:
        actual_num = float(actual)
        expected_num = float(expected)
        return abs(actual_num - expected_num) < tolerance
    except (ValueError, TypeError):
        pass
    
    # Handle list/array comparison
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return False
        return all(_compare_values(a, e, tolerance) for a, e in zip(actual, expected))
    
    # Handle dict comparison recursively
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual.keys()) != set(expected.keys()):
            return False
        return all(_compare_values(actual[k], expected[k], tolerance) for k in actual.keys())
    
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
    Calculate evaluation metrics based on outcome-based evaluation: E(O, Ô) ∈ {0, 1}
    
    Primary Metric:
    - Outcome Accuracy: Percentage of tasks where final outcome matches ground truth
    
    Auxiliary Metrics (for analysis):
    - Function Selection: Which function was chosen
    - Parameter Accuracy: Correctness of parameters
    - Precision/Recall: Traditional metrics
    """
    total = results['total_tests']
    correct_result = results['correct_result']  # E(O, Ô) = 1 count
    correct_function = results['correct_function']
    correct_params = results['correct_params']
    
    # Primary metric: Outcome-based success rate
    outcome_accuracy = (correct_result / total * 100) if total > 0 else 0
    
    # Auxiliary metrics for trajectory analysis
    tsr_function = (correct_function / total * 100) if total > 0 else 0
    tsr_params = (correct_params / total * 100) if total > 0 else 0
    
    # Traditional F1/Precision/Recall (kept for comparison)
    tools_called = total - results['no_tool_call']
    precision = (correct_result / tools_called * 100) if tools_called > 0 else 0
    recall = (correct_result / total * 100) if total > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'outcome_accuracy': round(outcome_accuracy, 2),  # Primary: E(O, Ô)
        'f1_score': round(f1_score, 2),
        'precision': round(precision, 2),
        'recall': round(recall, 2),
        'tsr_function_selection': round(tsr_function, 2),
        'tsr_parameter_accuracy': round(tsr_params, 2),
        'total_tests': total,
        'correct_outcome': correct_result,  # E(O, Ô) = 1 count
        'correct_function': correct_function,
        'correct_params': correct_params,
        'no_tool_call': results['no_tool_call'],
        'wrong_tool': results['wrong_tool']
    }
    
    return metrics


def print_report(metrics: Dict, model: str):
    """Print evaluation report with outcome-based metrics."""
    print(f"\n{'='*60}")
    print(f"Outcome-Based Evaluation Results - {model}")
    print(f"{'='*60}\n")
    
    print(f"Primary Metric (E(O, Ô)):")
    print(f"  Outcome Accuracy:      {metrics['outcome_accuracy']}%")
    print(f"  Tasks Completed:       {metrics['correct_outcome']}/{metrics['total_tests']}")
    print()
    
    print(f"Auxiliary Metrics (Trajectory Analysis):")
    print(f"  Function Selection:    {metrics['tsr_function_selection']}%")
    print(f"  Parameter Accuracy:    {metrics['tsr_parameter_accuracy']}%")
    print()
    
    print(f"Traditional Metrics:")
    print(f"  F1 Score:              {metrics['f1_score']}%")
    print(f"  Precision:             {metrics['precision']}%")
    print(f"  Recall:                {metrics['recall']}%")
    print()
    
    print(f"Breakdown:")
    print(f"  Total Tests:           {metrics['total_tests']}")
    print(f"  Correct Outcome:       {metrics['correct_outcome']}")
    print(f"  Correct Function:      {metrics['correct_function']}")
    print(f"  Correct Params:        {metrics['correct_params']}")
    print(f"  No Tool Call:          {metrics['no_tool_call']}")
    print(f"  Wrong Tool:            {metrics['wrong_tool']}")
    print()


def save_results(results: Dict, metrics: Dict, model: str, output_path: Optional[str] = None):
    """Save evaluation results to JSON file."""
    output = {
        'model': model,
        'timestamp': results['timestamp'],
        'metrics': metrics,
        'raw_results': results
    }
    
    if output_path:
        filename = Path(output_path)
        # Create directory if needed
        filename.parent.mkdir(parents=True, exist_ok=True)
    else:
        safe_model = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in model)
        filename = BFCL_RESULTS_DIR / f"bfcl_results_{safe_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filename.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {filename}")


async def main():
    """Main evaluation workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BFCL Evaluation for MCP Models")
    parser.add_argument("--model", default="qwen2.5", help="Model to evaluate")
    parser.add_argument("--category", default="simple", 
                       choices=["simple", "multiple", "parallel", "composite", "all"],
                       help="BFCL test category (use 'all' to load all categories)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tests")
    parser.add_argument("--download-only", action="store_true", help="Only download dataset, don't evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output file path for results")
    parser.add_argument("--synthetic", type=str, default=None, help="Use synthetic test file instead of BFCL")
    parser.add_argument("--oracle-mode", action="store_true", help="Oracle mode: only provide the relevant tool (no distractors)")
    parser.add_argument("--num-distractors", type=int, default=None, help="Number of distractor tools to include (0 = oracle mode)")
    parser.add_argument(
        "--allow_fallback",
        action="store_true",
        help="Allow text-based fallback JSON tool requests. Default BFCL mode is native tool-calling only.",
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("BFCL Evaluation - Berkeley Function Calling Leaderboard")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Category: {args.category}")
    print()
    
    # Use synthetic tests if provided
    if args.synthetic:
        print(f"Loading synthetic tests from: {args.synthetic}")
        with open(args.synthetic, 'r') as f:
            test_cases = json.load(f)
        print(f"Loaded {len(test_cases)} synthetic test cases")
    else:
        print(f"Dataset: {BFCL_DATASET}")
        
        # Load multiple categories if 'all' is specified
        if args.category == "all":
            categories = ["simple", "multiple", "parallel"]
            all_test_cases = []
            for cat in categories:
                print(f"\nLoading category: {cat}")
                dataset, _ = download_bfcl_dataset(cat)
                if dataset:
                    cat_tests = filter_math_tests(dataset, cat)
                    all_test_cases.extend(cat_tests)
                    print(f"  Found {len(cat_tests)} math tests in {cat}")
            test_cases = all_test_cases
            print(f"\nTotal math tests from all categories: {len(test_cases)}")
        else:
            # Download single BFCL category
            dataset, category = download_bfcl_dataset(args.category)
            if not dataset:
                return
            
            print(f"\nDataset type: {type(dataset)}")
            print(f"Dataset length: {len(dataset) if dataset else 0}")
            
            # Filter to math tools
            test_cases = filter_math_tests(dataset, category)
    
    if args.download_only:
        print(f"\nDataset downloaded and filtered")
        print(f"   {len(test_cases)} math tool test cases ready")
        return
    
    # Run evaluation
    num_distractors = args.num_distractors if args.num_distractors is not None else (0 if args.oracle_mode else None)
    results = await evaluate_model(
        args.model,
        test_cases,
        args.limit,
        num_distractors=num_distractors,
        allow_fallback=args.allow_fallback,
    )
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print report
    print_report(metrics, args.model)
    
    # Save results
    save_results(results, metrics, args.model, args.output)

    # Log to cloud database (skipped silently if .env not configured)
    from db_logger import log_run
    log_run(results, metrics, model=args.model, suite="bfcl", num_distractors=num_distractors)

    print(f"\nEvaluation complete!")
    print(f"\nNext steps:")
    print(f"1. Review results in {BFCL_RESULTS_DIR}")
    print(f"2. Run for other models: python evaluate_bfcl.py --model llama3.2")
    print(f"3. Compare metrics across models")


if __name__ == "__main__":
    asyncio.run(main())
