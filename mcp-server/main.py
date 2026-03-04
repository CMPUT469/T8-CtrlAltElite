"""MCP server entrypoint for local stdio and HTTP transports."""

from __future__ import annotations

import argparse
import math
from typing import Dict, List

from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Gorilla BFCL Math Tools", json_response=True)


# ============================================================================
# REAL GORILLA BFCL TOOLS - Berkeley Function Calling Leaderboard
# Source Code: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval/func_source_code/math_api.py
# Dataset Info: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
# 
# These functions are adapted from BFCL's MathAPI executable validation backend.
# BFCL uses these to verify LLM function calling accuracy during evaluation.
# We've adapted them for MCP server deployment to test qwen2.5 threshold limits.
# ============================================================================

# -------------------- BASIC ARITHMETIC --------------------

@mcp.tool()
def add(a: float, b: float) -> Dict:
    """
    Add two numbers.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        result: Sum of the two numbers
    """
    try:
        return {"result": a + b}
    except TypeError:
        return {"error": "Both inputs must be numbers"}


@mcp.tool()
def subtract(a: float, b: float) -> Dict:
    """
    Subtract one number from another.
    
    Args:
        a: Number to subtract from
        b: Number to subtract
    
    Returns:
        result: Difference between the two numbers
    """
    try:
        return {"result": a - b}
    except TypeError:
        return {"error": "Both inputs must be numbers"}


@mcp.tool()
def multiply(a: float, b: float) -> Dict:
    """
    Multiply two numbers.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        result: Product of the two numbers
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return {"error": "Both inputs must be numbers"}
    try:
        return {"result": a * b}
    except TypeError:
        return {"error": "Both inputs must be numbers"}


@mcp.tool()
def divide(a: float, b: float) -> Dict:
    """
    Divide one number by another.
    
    Args:
        a: Numerator
        b: Denominator
    
    Returns:
        result: Quotient of the division
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return {"error": "Both inputs must be numbers"}
    if b == 0:
        return {"error": "Cannot divide by zero"}
    try:
        return {"result": a / b}
    except TypeError:
        return {"error": "Both inputs must be numbers"}


@mcp.tool()
def power(base: float, exponent: float) -> Dict:
    """
    Raise a number to the power of another.
    
    Args:
        base: Base number
        exponent: Exponent
    
    Returns:
        result: Result of base raised to exponent
    """
    if not isinstance(base, (int, float)) or not isinstance(exponent, (int, float)):
        return {"error": "Both inputs must be numbers"}
    try:
        return {"result": base ** exponent}
    except (TypeError, OverflowError) as e:
        return {"error": str(e)}


@mcp.tool()
def square_root(number: float, precision: int) -> Dict:
    """
    Calculate the square root of a number.
    
    Args:
        number: Number to calculate square root of
        precision: Number of decimal places
    
    Returns:
        result: Square root of the number
    """
    if not isinstance(number, (int, float)) or not isinstance(precision, int):
        return {"error": "Invalid input types"}
    if number < 0:
        return {"error": "Cannot calculate square root of negative number"}
    try:
        result = round(math.sqrt(number), precision)
        return {"result": result}
    except (TypeError, ValueError) as e:
        return {"error": str(e)}


@mcp.tool()
def absolute_value(number: float) -> Dict:
    """
    Calculate absolute value of a number.
    
    Args:
        number: Input number
    
    Returns:
        result: Absolute value
    """
    if not isinstance(number, (int, float)):
        return {"error": "Input must be a number"}
    try:
        return {"result": abs(number)}
    except TypeError:
        return {"error": "Input must be a number"}


@mcp.tool()
def round_number(number: float, decimal_places: int = 0) -> Dict:
    """
    Round a number to specified decimal places.
    
    Args:
        number: Number to round
        decimal_places: Number of decimal places (default: 0)
    
    Returns:
        result: Rounded number
    """
    if not isinstance(number, (int, float)) or not isinstance(decimal_places, int):
        return {"error": "Invalid input types"}
    try:
        return {"result": round(number, decimal_places)}
    except (TypeError, ValueError) as e:
        return {"error": str(e)}


@mcp.tool()
def percentage(part: float, whole: float) -> Dict:
    """
    Calculate what percentage one number is of another.
    
    Args:
        part: Part value
        whole: Whole value
    
    Returns:
        result: Percentage
    """
    if not isinstance(part, (int, float)) or not isinstance(whole, (int, float)):
        return {"error": "Both inputs must be numbers"}
    if whole == 0:
        return {"error": "Cannot calculate percentage with zero as whole"}
    try:
        return {"result": round((part / whole) * 100, 2)}
    except TypeError:
        return {"error": "Both inputs must be numbers"}


# -------------------- LIST OPERATIONS --------------------

@mcp.tool()
def sum_values(numbers: List[float]) -> Dict:
    """
    Calculate the sum of a list of numbers.
    
    Args:
        numbers: List of numbers to sum
    
    Returns:
        result: The sum of all numbers
    """
    if not numbers:
        return {"error": "Cannot calculate sum of an empty list"}
    try:
        return {"result": sum(numbers)}
    except TypeError:
        return {"error": "All elements in the list must be numbers"}


@mcp.tool()
def mean(numbers: List[float]) -> Dict:
    """
    Calculate the mean (average) of a list of numbers.
    
    Args:
        numbers: List of numbers
    
    Returns:
        result: Mean value
    """
    if not numbers:
        return {"error": "Cannot calculate mean of an empty list"}
    try:
        return {"result": sum(numbers) / len(numbers)}
    except TypeError:
        return {"error": "All elements in the list must be numbers"}


@mcp.tool()
def min_value(numbers: List[float]) -> Dict:
    """
    Find the minimum value in a list.
    
    Args:
        numbers: List of numbers
    
    Returns:
        result: Minimum value
    """
    if not numbers:
        return {"error": "Cannot find minimum of an empty list"}
    try:
        return {"result": min(numbers)}
    except TypeError:
        return {"error": "All elements in the list must be numbers"}


@mcp.tool()
def max_value(numbers: List[float]) -> Dict:
    """
    Find the maximum value in a list.
    
    Args:
        numbers: List of numbers
    
    Returns:
        result: Maximum value
    """
    if not numbers:
        return {"error": "Cannot find maximum of an empty list"}
    try:
        return {"result": max(numbers)}
    except TypeError:
        return {"error": "All elements in the list must be numbers"}


@mcp.tool()
def standard_deviation(numbers: List[float]) -> Dict:
    """
    Calculate the standard deviation of a list of numbers.
    
    Args:
        numbers: List of numbers
    
    Returns:
        result: Standard deviation
    """
    if not numbers:
        return {"error": "Cannot calculate standard deviation of an empty list"}
    try:
        mean_val = sum(numbers) / len(numbers)
        variance = sum((x - mean_val) ** 2 for x in numbers) / len(numbers)
        return {"result": math.sqrt(variance)}
    except TypeError:
        return {"error": "All elements in the list must be numbers"}


# ============================================================================
# PROMPTS & RESOURCES
# ============================================================================

@mcp.prompt()
def calculate_complex(operation: str, values: str) -> str:
    """Generate a prompt for complex calculations"""
    return f"Perform {operation} on these values: {values}. Show step-by-step calculation."


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MCP demo server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport to expose: stdio for local subprocess clients, or HTTP-based transports.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind for HTTP transports (ignored for stdio).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind for HTTP transports (ignored for stdio).",
    )
    parser.add_argument(
        "--mount-path",
        default=None,
        help="Optional ASGI mount path for HTTP transports.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.transport != "stdio":
        mcp.settings.host = args.host
        mcp.settings.port = args.port

    mcp.run(transport=args.transport, mount_path=args.mount_path)


if __name__ == "__main__":
    main()
