"""
BFCL Math Tools - Berkeley Function Calling Leaderboard
Source: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
Dataset: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard

Functions adapted from BFCL MathAPI for threshold testing with open-source LLMs.
"""

import math
from typing import Dict, List


def register_tools(mcp):
    """Register all BFCL math tools with the MCP server."""
    
    # ============================================================================
    # BASIC ARITHMETIC
    # ============================================================================

    @mcp.tool()
    def add(a: float, b: float) -> Dict:
        """Add two numbers."""
        try:
            return {"result": a + b}
        except TypeError:
            return {"error": "Both inputs must be numbers"}

    @mcp.tool()
    def subtract(a: float, b: float) -> Dict:
        """Subtract one number from another."""
        try:
            return {"result": a - b}
        except TypeError:
            return {"error": "Both inputs must be numbers"}

    @mcp.tool()
    def multiply(a: float, b: float) -> Dict:
        """Multiply two numbers."""
        try:
            return {"result": a * b}
        except TypeError:
            return {"error": "Both inputs must be numbers"}

    @mcp.tool()
    def divide(a: float, b: float) -> Dict:
        """Divide one number by another (with zero-check)."""
        try:
            if b == 0:
                return {"error": "Cannot divide by zero"}
            return {"result": a / b}
        except TypeError:
            return {"error": "Both inputs must be numbers"}

    @mcp.tool()
    def power(base: float, exponent: float) -> Dict:
        """Raise a number to a power."""
        try:
            return {"result": base ** exponent}
        except (TypeError, OverflowError) as e:
            return {"error": str(e)}

    @mcp.tool()
    def square_root(number: float, precision: int = 2) -> Dict:
        """Calculate square root with specified precision."""
        try:
            if number < 0:
                return {"error": "Cannot calculate square root of negative number"}
            result = math.sqrt(number)
            return {"result": round(result, precision)}
        except (TypeError, ValueError) as e:
            return {"error": str(e)}

    @mcp.tool()
    def absolute_value(number: float) -> Dict:
        """Get absolute value of a number."""
        try:
            return {"result": abs(number)}
        except TypeError:
            return {"error": "Input must be a number"}

    @mcp.tool()
    def round_number(number: float, decimal_places: int = 0) -> Dict:
        """Round a number to specified decimal places."""
        try:
            return {"result": round(number, decimal_places)}
        except (TypeError, ValueError) as e:
            return {"error": str(e)}

    @mcp.tool()
    def percentage(part: float, whole: float) -> Dict:
        """Calculate what percentage 'part' is of 'whole'."""
        try:
            if whole == 0:
                return {"error": "Whole cannot be zero"}
            return {"result": (part / whole) * 100}
        except TypeError:
            return {"error": "Both inputs must be numbers"}

    # ============================================================================
    # LIST OPERATIONS
    # ============================================================================

    @mcp.tool()
    def sum_values(numbers: List[float]) -> Dict:
        """Sum all numbers in a list."""
        try:
            if not numbers:
                return {"error": "List cannot be empty"}
            return {"result": sum(numbers)}
        except TypeError:
            return {"error": "All items must be numbers"}

    @mcp.tool()
    def mean(numbers: List[float]) -> Dict:
        """Calculate the mean (average) of a list of numbers."""
        try:
            if not numbers:
                return {"error": "List cannot be empty"}
            return {"result": sum(numbers) / len(numbers)}
        except TypeError:
            return {"error": "All items must be numbers"}

    @mcp.tool()
    def min_value(numbers: List[float]) -> Dict:
        """Find the minimum value in a list."""
        try:
            if not numbers:
                return {"error": "List cannot be empty"}
            return {"result": min(numbers)}
        except (TypeError, ValueError) as e:
            return {"error": str(e)}

    @mcp.tool()
    def max_value(numbers: List[float]) -> Dict:
        """Find the maximum value in a list."""
        try:
            if not numbers:
                return {"error": "List cannot be empty"}
            return {"result": max(numbers)}
        except (TypeError, ValueError) as e:
            return {"error": str(e)}

    @mcp.tool()
    def standard_deviation(numbers: List[float]) -> Dict:
        """Calculate the standard deviation of a list of numbers."""
        try:
            if not numbers:
                return {"error": "List cannot be empty"}
            if len(numbers) == 1:
                return {"result": 0.0}
            
            mean_val = sum(numbers) / len(numbers)
            variance = sum((x - mean_val) ** 2 for x in numbers) / (len(numbers) - 1)
            return {"result": math.sqrt(variance)}
        except (TypeError, ValueError) as e:
            return {"error": str(e)}
    
    print("Registered 14 BFCL Math Tools")
