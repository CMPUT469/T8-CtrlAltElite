"""
Shared Evaluation Framework for MCP Tool Calling

Provides common functionality for evaluating different MCP tool datasets:
- BFCL math tools
- Jefferson statistical tools
- Future datasets

Core Features:
- Outcome-based evaluation E(O, Ô) ∈ {0, 1}
- Numerical tolerance comparison
- MCP server connection management
- Result comparison with type coercion
"""

from pathlib import Path
from datetime import datetime
import json
from typing import Any, Dict, List, Optional


def json_dumps_safe(value: Any) -> str:
    """Serialize a value safely for logging or feeding back to the model."""
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return json.dumps(str(value), ensure_ascii=False)


def mcp_tools_to_openai_tools(mcp_tools: Any) -> List[Dict[str, Any]]:
    """Convert MCP tool definitions into OpenAI-compatible tools schema."""
    openai_tools: List[Dict[str, Any]] = []
    for tool in mcp_tools.tools:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema or {},
                },
            }
        )
    return openai_tools


def normalize_mcp_tool_result(result: Any) -> str:
    """Normalize MCP tool output into a compact string for model follow-up turns."""
    content = getattr(result, "content", None)
    if content is not None:
        return json_dumps_safe(content)

    if hasattr(result, "model_dump"):
        return json_dumps_safe(result.model_dump())

    return json_dumps_safe(result)


def compare_values(actual: Any, expected: Any, tolerance: float = 0.01) -> bool:
    """
    Compare two values with type coercion and tolerance for numerical values.
    
    Handles:
    - Dict vs primitive (extracts 'result', 'value', 'output' keys)
    - Numerical comparison with tolerance
    - Recursive list/dict comparison
    
    Args:
        actual: The actual result from tool execution
        expected: The expected result from ground truth
        tolerance: Numerical tolerance for float comparison (default: 0.01)
    
    Returns:
        bool: True if values match within tolerance
    """
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
        return all(compare_values(a, e, tolerance) for a, e in zip(actual, expected))
    
    # Handle dict comparison recursively
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual.keys()) != set(expected.keys()):
            return False
        return all(compare_values(actual[k], expected[k], tolerance) for k in actual.keys())
    
    return False


def compare_params(actual: Dict, expected: Dict) -> bool:
    """
    Compare parameter dictionaries, allowing for type coercion.
    
    Args:
        actual: Actual parameters used in tool call
        expected: Expected parameters from ground truth
        
    Returns:
        bool: True if parameters match (keys and values)
    """
    if set(actual.keys()) != set(expected.keys()):
        return False
    
    for key in actual.keys():
        try:
            if actual[key] != expected[key]:
                # Try numeric comparison
                try:
                    if abs(float(actual[key]) - float(expected[key])) > 0.0001:
                        return False
                except (ValueError, TypeError):
                    return False
        except:
            return False
    
    return True


def extract_result_value(tool_result: Any) -> Any:
    """
    Extract the actual result value from MCP tool result.
    
    Handles different MCP result formats:
    - Content with text field containing JSON
    - Direct result values
    - Model dumps
    
    Args:
        tool_result: Raw result from MCP tool execution
        
    Returns:
        Extracted result value
    """
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
    Calculate outcome-based evaluation metrics following MCPVerse methodology.
    
    Primary Metric: Outcome Accuracy E(O, Ô)
    Auxiliary Metrics: Function selection, parameter accuracy
    
    Args:
        results: Raw results containing counts and details
        
    Returns:
        Dict with calculated metrics
    """
    total = results['total_tests']
    correct_result = results['correct_result']
    correct_function = results['correct_function']
    correct_params = results['correct_params']
    
    # Primary Metric: Outcome Accuracy E(O, Ô) ∈ {0, 1}
    outcome_accuracy = (correct_result / total * 100) if total > 0 else 0
    
    # Auxiliary Metrics (for trajectory analysis)
    function_accuracy = (correct_function / total * 100) if total > 0 else 0
    param_accuracy = (correct_params / total * 100) if total > 0 else 0
    
    # Traditional metrics (for comparison)
    true_positive = correct_result
    false_positive = correct_function - correct_result  # Called right function but wrong outcome
    false_negative = results['no_tool_call'] + results['wrong_tool']
    
    precision = (true_positive / (true_positive + false_positive) * 100) if (true_positive + false_positive) > 0 else 0
    recall = (true_positive / (true_positive + false_negative) * 100) if (true_positive + false_negative) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    return {
        # Primary Metric (E(O, Ô))
        'outcome_accuracy': round(outcome_accuracy, 2),
        'correct_outcome': correct_result,
        
        # Auxiliary Metrics (Trajectory Analysis)
        'tsr_function_selection': round(function_accuracy, 2),
        'tsr_parameter_accuracy': round(param_accuracy, 2),
        
        # Traditional Metrics
        'f1_score': round(f1, 2),
        'precision': round(precision, 2),
        'recall': round(recall, 2),
        
        # Breakdown
        'total_tests': total,
        'correct_function': correct_function,
        'correct_params': correct_params,
        'no_tool_call': results['no_tool_call'],
        'wrong_tool': results['wrong_tool']
    }


def print_report(metrics: Dict, model: str):
    """
    Print outcome-based evaluation report.
    
    Args:
        metrics: Calculated metrics dictionary
        model: Model name being evaluated
    """
    print("\n" + "=" * 60)
    print(f"Outcome-Based Evaluation Results - {model}")
    print("=" * 60)
    
    print("\nPrimary Metric (E(O, Ô)):")
    print(f"  Outcome Accuracy:      {metrics['outcome_accuracy']}%")
    print(f"  Tasks Completed:       {metrics['correct_outcome']}/{metrics['total_tests']}")
    
    print("\nAuxiliary Metrics (Trajectory Analysis):")
    print(f"  Function Selection:    {metrics['tsr_function_selection']}%")
    print(f"  Parameter Accuracy:    {metrics['tsr_parameter_accuracy']}%")
    
    print("\nTraditional Metrics:")
    print(f"  F1 Score:              {metrics['f1_score']}%")
    print(f"  Precision:             {metrics['precision']}%")
    print(f"  Recall:                {metrics['recall']}%")
    
    print("\nBreakdown:")
    print(f"  Total Tests:           {metrics['total_tests']}")
    print(f"  Correct Outcome:       {metrics['correct_outcome']}")
    print(f"  Correct Function:      {metrics['correct_function']}")
    print(f"  Correct Params:        {metrics['correct_params']}")
    print(f"  No Tool Call:          {metrics['no_tool_call']}")
    print(f"  Wrong Tool:            {metrics['wrong_tool']}")


def maybe_parse_fallback_tool_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse fallback JSON for models that answer in text instead of native tool_calls.
    Expected format: {"tool":"<name>","args":{...}}
    
    Args:
        text: Text response from model
        
    Returns:
        Parsed tool call dict or None
    """
    if not text:
        return None

    candidate = text.strip()

    # Remove code block markers
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


def save_results(output_path: Path, model: str, metrics: Dict, raw_results: Dict):
    """
    Save evaluation results to JSON file.
    
    Args:
        output_path: Path to save results
        model: Model name
        metrics: Calculated metrics
        raw_results: Raw evaluation results with details
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "raw_results": raw_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
