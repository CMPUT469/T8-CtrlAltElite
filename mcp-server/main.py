"""MCP server entrypoint for local stdio and HTTP transports."""

from __future__ import annotations

import argparse
import math
from typing import Dict, List
import json
import os
import httpx
import logging
import sys
from urllib.parse import urlencode
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Gorilla BFCL Math Tools + Finance Tools", json_response=True)


# ============================================================================
# BFCL MATH TOOLS - Berkeley Function Calling Leaderboard
# Source: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
# Dataset: https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard
# 
# Functions adapted from BFCL MathAPI for threshold testing with open-source LLMs.
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


# -------------------- ADDITIONAL MATH TOOLS (THRESHOLD TESTING) --------------------

@mcp.tool()
def add_three(a: float, b: float, c: float) -> Dict:
    """
    Add three numbers together.
    
    Args:
        a: First number
        b: Second number
        c: Third number
    
    Returns:
        result: Sum of three numbers
    """
    if not all(isinstance(x, (int, float)) for x in [a, b, c]):
        return {"error": "All inputs must be numbers"}
    return {"result": a + b + c}


@mcp.tool()
def modulo(a: float, b: float) -> Dict:
    """
    Calculate modulo (remainder of division).
    
    Args:
        a: Dividend
        b: Divisor
    
    Returns:
        result: Remainder
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return {"error": "Both inputs must be numbers"}
    if b == 0:
        return {"error": "Cannot divide by zero"}
    return {"result": a % b}


@mcp.tool()
def factorial(n: int) -> Dict:
    """
    Calculate factorial of a number.
    
    Args:
        n: Non-negative integer
    
    Returns:
        result: Factorial value
    """
    if not isinstance(n, int):
        return {"error": "Input must be an integer"}
    if n < 0:
        return {"error": "Factorial not defined for negative numbers"}
    if n > 100:
        return {"error": "Number too large for factorial calculation"}
    result = 1
    for i in range(2, n + 1):
        result *= i
    return {"result": result}


@mcp.tool()
def median(numbers: List[float]) -> Dict:
    """
    Calculate median of a list of numbers.
    
    Args:
        numbers: List of numbers
    
    Returns:
        result: Median value
    """
    if not numbers:
        return {"error": "Cannot calculate median of empty list"}
    try:
        sorted_nums = sorted(numbers)
        n = len(sorted_nums)
        mid = n // 2
        if n % 2 == 0:
            return {"result": (sorted_nums[mid-1] + sorted_nums[mid]) / 2}
        return {"result": sorted_nums[mid]}
    except TypeError:
        return {"error": "All elements must be numbers"}


@mcp.tool()
def logarithm(number: float, base: float = 10) -> Dict:
    """
    Calculate logarithm of a number with specified base.
    
    Args:
        number: Number to calculate log of
        base: Logarithm base (default: 10)
    
    Returns:
        result: Logarithm value
    """
    if not isinstance(number, (int, float)) or not isinstance(base, (int, float)):
        return {"error": "Both inputs must be numbers"}
    if number <= 0:
        return {"error": "Cannot calculate log of non-positive number"}
    if base <= 0 or base == 1:
        return {"error": "Invalid logarithm base"}
    try:
        return {"result": math.log(number, base)}
    except (ValueError, ZeroDivisionError) as e:
        return {"error": str(e)}


@mcp.tool()
def ceiling(number: float) -> Dict:
    """
    Round number up to nearest integer.
    
    Args:
        number: Number to round up
    
    Returns:
        result: Ceiling value
    """
    if not isinstance(number, (int, float)):
        return {"error": "Input must be a number"}
    return {"result": math.ceil(number)}


@mcp.tool()
def floor(number: float) -> Dict:
    """
    Round number down to nearest integer.
    
    Args:
        number: Number to round down
    
    Returns:
        result: Floor value
    """
    if not isinstance(number, (int, float)):
        return {"error": "Input must be a number"}
    return {"result": math.floor(number)}


@mcp.tool()
def range_values(numbers: List[float]) -> Dict:
    """
    Calculate range (max - min) of a list.
    
    Args:
        numbers: List of numbers
    
    Returns:
        result: Range value
    """
    if not numbers:
        return {"error": "Cannot calculate range of empty list"}
    try:
        return {"result": max(numbers) - min(numbers)}
    except TypeError:
        return {"error": "All elements must be numbers"}


@mcp.tool()
def variance(numbers: List[float]) -> Dict:
    """
    Calculate variance of a list of numbers.
    
    Args:
        numbers: List of numbers
    
    Returns:
        result: Variance value
    """
    if not numbers:
        return {"error": "Cannot calculate variance of empty list"}
    try:
        mean_val = sum(numbers) / len(numbers)
        return {"result": sum((x - mean_val) ** 2 for x in numbers) / len(numbers)}
    except TypeError:
        return {"error": "All elements must be numbers"}


@mcp.tool()
def gcd(a: int, b: int) -> Dict:
    """
    Calculate greatest common divisor of two integers.
    
    Args:
        a: First integer
        b: Second integer
    
    Returns:
        result: GCD value
    """
    if not isinstance(a, int) or not isinstance(b, int):
        return {"error": "Both inputs must be integers"}
    return {"result": math.gcd(a, b)}


@mcp.tool()
def lcm(a: int, b: int) -> Dict:
    """
    Calculate least common multiple of two integers.
    
    Args:
        a: First integer
        b: Second integer
    
    Returns:
        result: LCM value
    """
    if not isinstance(a, int) or not isinstance(b, int):
        return {"error": "Both inputs must be integers"}
    return {"result": abs(a * b) // math.gcd(a, b) if math.gcd(a, b) != 0 else 0}


@mcp.tool()
def sine(angle: float, unit: str = "radians") -> Dict:
    """
    Calculate sine of an angle.
    
    Args:
        angle: Angle value
        unit: Unit of angle (radians or degrees)
    
    Returns:
        result: Sine value
    """
    if not isinstance(angle, (int, float)):
        return {"error": "Angle must be a number"}
    if unit == "degrees":
        angle = math.radians(angle)
    return {"result": math.sin(angle)}


@mcp.tool()
def cosine(angle: float, unit: str = "radians") -> Dict:
    """
    Calculate cosine of an angle.
    
    Args:
        angle: Angle value
        unit: Unit of angle (radians or degrees)
    
    Returns:
        result: Cosine value
    """
    if not isinstance(angle, (int, float)):
        return {"error": "Angle must be a number"}
    if unit == "degrees":
        angle = math.radians(angle)
    return {"result": math.cos(angle)}


@mcp.tool()
def tangent(angle: float, unit: str = "radians") -> Dict:
    """
    Calculate tangent of an angle.
    
    Args:
        angle: Angle value
        unit: Unit of angle (radians or degrees)
    
    Returns:
        result: Tangent value
    """
    if not isinstance(angle, (int, float)):
        return {"error": "Angle must be a number"}
    if unit == "degrees":
        angle = math.radians(angle)
    return {"result": math.tan(angle)}


@mcp.tool()
def exponential(x: float) -> Dict:
    """
    Calculate e raised to the power of x.
    
    Args:
        x: Exponent value
    
    Returns:
        result: e^x
    """
    if not isinstance(x, (int, float)):
        return {"error": "Input must be a number"}
    return {"result": math.exp(x)}


@mcp.tool()
def natural_log(x: float) -> Dict:
    """
    Calculate natural logarithm (base e) of x.
    
    Args:
        x: Input number
    
    Returns:
        result: ln(x)
    """
    if not isinstance(x, (int, float)):
        return {"error": "Input must be a number"}
    if x <= 0:
        return {"error": "Natural log undefined for non-positive numbers"}
    return {"result": math.log(x)}


@mcp.tool()
def log10(x: float) -> Dict:
    """
    Calculate logarithm base 10 of x.
    
    Args:
        x: Input number
    
    Returns:
        result: log10(x)
    """
    if not isinstance(x, (int, float)):
        return {"error": "Input must be a number"}
    if x <= 0:
        return {"error": "Log10 undefined for non-positive numbers"}
    return {"result": math.log10(x)}


@mcp.tool()
def cube_root(x: float) -> Dict:
    """
    Calculate cube root of a number.
    
    Args:
        x: Input number
    
    Returns:
        result: x^(1/3)
    """
    if not isinstance(x, (int, float)):
        return {"error": "Input must be a number"}
    return {"result": math.copysign(abs(x) ** (1/3), x)}


@mcp.tool()
def nth_root(x: float, n: int) -> Dict:
    """
    Calculate nth root of a number.
    
    Args:
        x: Input number
        n: Root degree
    
    Returns:
        result: x^(1/n)
    """
    if not isinstance(x, (int, float)) or not isinstance(n, int):
        return {"error": "Invalid input types"}
    if n == 0:
        return {"error": "Root degree cannot be zero"}
    if x < 0 and n % 2 == 0:
        return {"error": "Even root of negative number undefined"}
    return {"result": math.copysign(abs(x) ** (1/n), x) if x < 0 else x ** (1/n)}


@mcp.tool()
def hypotenuse(a: float, b: float) -> Dict:
    """
    Calculate hypotenuse of right triangle.
    
    Args:
        a: First side
        b: Second side
    
    Returns:
        result: sqrt(a^2 + b^2)
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return {"error": "Both inputs must be numbers"}
    return {"result": math.hypot(a, b)}


@mcp.tool()
def radians_to_degrees(radians: float) -> Dict:
    """
    Convert radians to degrees.
    
    Args:
        radians: Angle in radians
    
    Returns:
        result: Angle in degrees
    """
    if not isinstance(radians, (int, float)):
        return {"error": "Input must be a number"}
    return {"result": math.degrees(radians)}


@mcp.tool()
def degrees_to_radians(degrees: float) -> Dict:
    """
    Convert degrees to radians.
    
    Args:
        degrees: Angle in degrees
    
    Returns:
        result: Angle in radians
    """
    if not isinstance(degrees, (int, float)):
        return {"error": "Input must be a number"}
    return {"result": math.radians(degrees)}


@mcp.tool()
def permutation(n: int, r: int) -> Dict:
    """
    Calculate number of permutations: nPr = n!/(n-r)!
    
    Args:
        n: Total items
        r: Items to arrange
    
    Returns:
        result: Number of permutations
    """
    if not isinstance(n, int) or not isinstance(r, int):
        return {"error": "Both inputs must be integers"}
    if n < 0 or r < 0:
        return {"error": "Inputs must be non-negative"}
    if r > n:
        return {"error": "r cannot be greater than n"}
    return {"result": math.perm(n, r)}


@mcp.tool()
def combination(n: int, r: int) -> Dict:
    """
    Calculate number of combinations: nCr = n!/(r!(n-r)!)
    
    Args:
        n: Total items
        r: Items to choose
    
    Returns:
        result: Number of combinations
    """
    if not isinstance(n, int) or not isinstance(r, int):
        return {"error": "Both inputs must be integers"}
    if n < 0 or r < 0:
        return {"error": "Inputs must be non-negative"}
    if r > n:
        return {"error": "r cannot be greater than n"}
    return {"result": math.comb(n, r)}


@mcp.tool()
def is_prime(n: int) -> Dict:
    """
    Check if a number is prime.
    
    Args:
        n: Integer to check
    
    Returns:
        result: Boolean indicating primality
    """
    if not isinstance(n, int):
        return {"error": "Input must be an integer"}
    if n < 2:
        return {"result": False}
    if n == 2:
        return {"result": True}
    if n % 2 == 0:
        return {"result": False}
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return {"result": False}
    return {"result": True}


@mcp.tool()
def sum_of_squares(numbers: List[float]) -> Dict:
    """
    Calculate sum of squares of a list.
    
    Args:
        numbers: List of numbers
    
    Returns:
        result: Sum of squared values
    """
    if not numbers:
        return {"error": "Cannot calculate for empty list"}
    try:
        return {"result": sum(x**2 for x in numbers)}
    except TypeError:
        return {"error": "All elements must be numbers"}


@mcp.tool()
def product(numbers: List[float]) -> Dict:
    """
    Calculate product of all numbers in a list.
    
    Args:
        numbers: List of numbers
    
    Returns:
        result: Product of all values
    """
    if not numbers:
        return {"error": "Cannot calculate for empty list"}
    try:
        result = 1
        for x in numbers:
            result *= x
        return {"result": result}
    except TypeError:
        return {"error": "All elements must be numbers"}


@mcp.tool()
def geometric_mean(numbers: List[float]) -> Dict:
    """
    Calculate geometric mean of a list.
    
    Args:
        numbers: List of positive numbers
    
    Returns:
        result: Geometric mean
    """
    if not numbers:
        return {"error": "Cannot calculate for empty list"}
    try:
        if any(x <= 0 for x in numbers):
            return {"error": "All numbers must be positive"}
        prod = 1
        for x in numbers:
            prod *= x
        return {"result": prod ** (1/len(numbers))}
    except TypeError:
        return {"error": "All elements must be numbers"}


@mcp.tool()
def harmonic_mean(numbers: List[float]) -> Dict:
    """
    Calculate harmonic mean of a list.
    
    Args:
        numbers: List of non-zero numbers
    
    Returns:
        result: Harmonic mean
    """
    if not numbers:
        return {"error": "Cannot calculate for empty list"}
    try:
        if any(x == 0 for x in numbers):
            return {"error": "All numbers must be non-zero"}
        return {"result": len(numbers) / sum(1/x for x in numbers)}
    except TypeError:
        return {"error": "All elements must be numbers"}


@mcp.tool()
def clamp(value: float, min_val: float, max_val: float) -> Dict:
    """
    Clamp a value between minimum and maximum bounds.
    
    Args:
        value: Value to clamp
        min_val: Minimum bound
        max_val: Maximum bound
    
    Returns:
        result: Clamped value
    """
    if not all(isinstance(x, (int, float)) for x in [value, min_val, max_val]):
        return {"error": "All inputs must be numbers"}
    if min_val > max_val:
        return {"error": "Minimum cannot be greater than maximum"}
    return {"result": max(min_val, min(value, max_val))}


@mcp.tool()
def lerp(a: float, b: float, t: float) -> Dict:
    """
    Linear interpolation between two values.
    
    Args:
        a: Start value
        b: End value
        t: Interpolation factor (0 to 1)
    
    Returns:
        result: Interpolated value
    """
    if not all(isinstance(x, (int, float)) for x in [a, b, t]):
        return {"error": "All inputs must be numbers"}
    return {"result": a + (b - a) * t}


@mcp.tool()
def sign(x: float) -> Dict:
    """
    Return the sign of a number.
    
    Args:
        x: Input number
    
    Returns:
        result: -1 for negative, 0 for zero, 1 for positive
    """
    if not isinstance(x, (int, float)):
        return {"error": "Input must be a number"}
    if x > 0:
        return {"result": 1}
    elif x < 0:
        return {"result": -1}
    return {"result": 0}


# Finance Tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("financial-datasets-mcp")

# Constants
FINANCIAL_DATASETS_API_BASE = "https://api.financialdatasets.ai"


# Helper function to make API requests
async def make_request(url: str) -> dict[str, any] | None:
    """Make a request to the Financial Datasets API with proper error handling."""
    # Load environment variables from .env file
    load_dotenv()
    
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"Error": str(e)}


def build_api_url(path: str, **params: object) -> str:
    query = urlencode({key: value for key, value in params.items() if value is not None})
    return f"{FINANCIAL_DATASETS_API_BASE}{path}" + (f"?{query}" if query else "")


def dump_api_items(data: dict | None, field: str, error_message: str) -> str:
    if not data:
        return error_message

    items = data.get(field, [])
    if not items:
        return error_message

    return json.dumps(items, indent=2)


def dump_api_object(data: dict | None, field: str, error_message: str) -> str:
    if not data:
        return error_message

    item = data.get(field, {})
    if not item:
        return error_message

    return json.dumps(item, indent=2)


@mcp.tool()
async def get_income_statements(
    ticker: str,
    period: str = "annual",
    limit: int = 4,
) -> str:
    """Get income statements for a company.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
        period: Period of the income statement (e.g. annual, quarterly, ttm)
        limit: Number of income statements to return (default: 4)
    """
    # Fetch data from the API
    url = f"{FINANCIAL_DATASETS_API_BASE}/financials/income-statements/?ticker={ticker}&period={period}&limit={limit}"
    data = await make_request(url)

    # Check if data is found
    if not data:
        return "Unable to fetch income statements or no income statements found."

    # Extract the income statements
    income_statements = data.get("income_statements", [])

    # Check if income statements are found
    if not income_statements:
        return "Unable to fetch income statements or no income statements found."

    # Stringify the income statements
    return json.dumps(income_statements, indent=2)


@mcp.tool()
async def get_balance_sheets(
    ticker: str,
    period: str = "annual",
    limit: int = 4,
) -> str:
    """Get balance sheets for a company.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
        period: Period of the balance sheet (e.g. annual, quarterly, ttm)
        limit: Number of balance sheets to return (default: 4)
    """
    # Fetch data from the API
    url = f"{FINANCIAL_DATASETS_API_BASE}/financials/balance-sheets/?ticker={ticker}&period={period}&limit={limit}"
    data = await make_request(url)

    # Check if data is found
    if not data:
        return "Unable to fetch balance sheets or no balance sheets found."

    # Extract the balance sheets
    balance_sheets = data.get("balance_sheets", [])

    # Check if balance sheets are found
    if not balance_sheets:
        return "Unable to fetch balance sheets or no balance sheets found."

    # Stringify the balance sheets
    return json.dumps(balance_sheets, indent=2)


@mcp.tool()
async def get_cash_flow_statements(
    ticker: str,
    period: str = "annual",
    limit: int = 4,
) -> str:
    """Get cash flow statements for a company.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
        period: Period of the cash flow statement (e.g. annual, quarterly, ttm)
        limit: Number of cash flow statements to return (default: 4)
    """
    # Fetch data from the API
    url = f"{FINANCIAL_DATASETS_API_BASE}/financials/cash-flow-statements/?ticker={ticker}&period={period}&limit={limit}"
    data = await make_request(url)

    # Check if data is found
    if not data:
        return "Unable to fetch cash flow statements or no cash flow statements found."

    # Extract the cash flow statements
    cash_flow_statements = data.get("cash_flow_statements", [])

    # Check if cash flow statements are found
    if not cash_flow_statements:
        return "Unable to fetch cash flow statements or no cash flow statements found."

    # Stringify the cash flow statements
    return json.dumps(cash_flow_statements, indent=2)


@mcp.tool()
async def get_current_stock_price(ticker: str) -> str:
    """Get the current / latest price of a company.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
    """
    # Fetch data from the API
    url = f"{FINANCIAL_DATASETS_API_BASE}/prices/snapshot/?ticker={ticker}"
    data = await make_request(url)

    # Check if data is found
    if not data:
        return "Unable to fetch current price or no current price found."

    # Extract the current price
    snapshot = data.get("snapshot", {})

    # Check if current price is found
    if not snapshot:
        return "Unable to fetch current price or no current price found."

    # Stringify the current price
    return json.dumps(snapshot, indent=2)


@mcp.tool()
async def get_historical_stock_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "day",
    interval_multiplier: int = 1,
) -> str:
    """Gets historical stock prices for a company.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
        start_date: Start date of the price data (e.g. 2020-01-01)
        end_date: End date of the price data (e.g. 2020-12-31)
        interval: Interval of the price data (e.g. minute, hour, day, week, month)
        interval_multiplier: Multiplier of the interval (e.g. 1, 2, 3)
    """
    # Fetch data from the API
    url = f"{FINANCIAL_DATASETS_API_BASE}/prices/?ticker={ticker}&interval={interval}&interval_multiplier={interval_multiplier}&start_date={start_date}&end_date={end_date}"
    data = await make_request(url)

    # Check if data is found
    if not data:
        return "Unable to fetch prices or no prices found."

    # Extract the prices
    prices = data.get("prices", [])

    # Check if prices are found
    if not prices:
        return "Unable to fetch prices or no prices found."

    # Stringify the prices
    return json.dumps(prices, indent=2)


@mcp.tool()
async def get_company_news(ticker: str) -> str:
    """Get news for a company.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
    """
    # Fetch data from the API
    url = f"{FINANCIAL_DATASETS_API_BASE}/news/?ticker={ticker}"
    data = await make_request(url)

    # Check if data is found
    if not data:
        return "Unable to fetch news or no news found."

    # Extract the news
    news = data.get("news", [])

    # Check if news are found
    if not news:
        return "Unable to fetch news or no news found."
    return json.dumps(news, indent=2)


@mcp.tool()
async def get_available_crypto_tickers() -> str:
    """
    Gets all available crypto tickers.
    """
    # Fetch data from the API
    url = f"{FINANCIAL_DATASETS_API_BASE}/crypto/prices/tickers"
    data = await make_request(url)

    # Check if data is found
    if not data:
        return "Unable to fetch available crypto tickers or no available crypto tickers found."

    # Extract the available crypto tickers
    tickers = data.get("tickers", [])

    # Stringify the available crypto tickers
    return json.dumps(tickers, indent=2)


@mcp.tool()
async def get_crypto_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "day",
    interval_multiplier: int = 1,
) -> str:
    """
    Gets historical prices for a crypto currency.
    """
    # Fetch data from the API
    url = f"{FINANCIAL_DATASETS_API_BASE}/crypto/prices/?ticker={ticker}&interval={interval}&interval_multiplier={interval_multiplier}&start_date={start_date}&end_date={end_date}"
    data = await make_request(url)

    # Check if data is found
    if not data:
        return "Unable to fetch prices or no prices found."

    # Extract the prices
    prices = data.get("prices", [])

    # Check if prices are found
    if not prices:
        return "Unable to fetch prices or no prices found."

    # Stringify the prices
    return json.dumps(prices, indent=2)


@mcp.tool()
async def get_historical_crypto_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "day",
    interval_multiplier: int = 1,
) -> str:
    """Gets historical prices for a crypto currency.

    Args:
        ticker: Ticker symbol of the crypto currency (e.g. BTC-USD). The list of available crypto tickers can be retrieved via the get_available_crypto_tickers tool.
        start_date: Start date of the price data (e.g. 2020-01-01)
        end_date: End date of the price data (e.g. 2020-12-31)
        interval: Interval of the price data (e.g. minute, hour, day, week, month)
        interval_multiplier: Multiplier of the interval (e.g. 1, 2, 3)
    """
    # Fetch data from the API
    url = f"{FINANCIAL_DATASETS_API_BASE}/crypto/prices/?ticker={ticker}&interval={interval}&interval_multiplier={interval_multiplier}&start_date={start_date}&end_date={end_date}"
    data = await make_request(url)

    # Check if data is found
    if not data:
        return "Unable to fetch prices or no prices found."

    # Extract the prices
    prices = data.get("prices", [])

    # Check if prices are found
    if not prices:
        return "Unable to fetch prices or no prices found."

    # Stringify the prices
    return json.dumps(prices, indent=2)


@mcp.tool()
async def get_current_crypto_price(ticker: str) -> str:
    """Get the current / latest price of a crypto currency.

    Args:
        ticker: Ticker symbol of the crypto currency (e.g. BTC-USD). The list of available crypto tickers can be retrieved via the get_available_crypto_tickers tool.
    """
    # Fetch data from the API
    url = f"{FINANCIAL_DATASETS_API_BASE}/crypto/prices/snapshot/?ticker={ticker}"
    data = await make_request(url)

    # Check if data is found
    if not data:
        return "Unable to fetch current price or no current price found."

    # Extract the current price
    snapshot = data.get("snapshot", {})

    # Check if current price is found
    if not snapshot:
        return "Unable to fetch current price or no current price found."

    # Stringify the current price
    return json.dumps(snapshot, indent=2)


@mcp.tool()
async def get_sec_filings(
    ticker: str,
    limit: int = 10,
    filing_type: str | None = None,
) -> str:
    """Get all SEC filings for a company.

    Args:
        ticker: Ticker symbol of the company (e.g. AAPL, GOOGL)
        limit: Number of SEC filings to return (default: 10)
        filing_type: Type of SEC filing (e.g. 10-K, 10-Q, 8-K)
    """
    # Fetch data from the API
    url = f"{FINANCIAL_DATASETS_API_BASE}/filings/?ticker={ticker}&limit={limit}"
    if filing_type:
        url += f"&filing_type={filing_type}"
 
    # Call the API
    data = await make_request(url)

    # Extract the SEC filings
    filings = data.get("filings", [])

    # Check if SEC filings are found
    if not filings:
        return f"Unable to fetch SEC filings or no SEC filings found."

    # Stringify the SEC filings
    return json.dumps(filings, indent=2)


@mcp.tool()
async def getAnalystEstimates(
    ticker: str,
    period: str = "annual",
    limit: int = 4,
) -> str:
    """Get analyst consensus estimates for a company."""
    url = build_api_url(
        "/analyst-estimates/",
        ticker=ticker,
        period=period,
        limit=limit,
    )
    data = await make_request(url)
    return dump_api_items(
        data,
        "analyst_estimates",
        "Unable to fetch analyst estimates or no analyst estimates found.",
    )


@mcp.tool()
async def getFinancialMetrics(
    ticker: str,
    period: str = "annual",
    limit: int = 4,
) -> str:
    """Get historical financial metrics for a company."""
    url = build_api_url(
        "/financial-metrics/",
        ticker=ticker,
        period=period,
        limit=limit,
    )
    data = await make_request(url)
    return dump_api_items(
        data,
        "financial_metrics",
        "Unable to fetch financial metrics or no financial metrics found.",
    )


@mcp.tool()
async def getFinancialMetricsSnapshot(ticker: str) -> str:
    """Get the latest financial metrics snapshot for a company."""
    url = build_api_url("/financial-metrics/snapshot/", ticker=ticker)
    data = await make_request(url)
    return dump_api_object(
        data,
        "snapshot",
        "Unable to fetch financial metrics snapshot or no snapshot found.",
    )


@mcp.tool()
async def getSegmentedRevenues(
    ticker: str,
    period: str = "annual",
    limit: int = 4,
) -> str:
    """Get segmented revenue data for a company."""
    url = build_api_url(
        "/financials/segmented-revenues/",
        ticker=ticker,
        period=period,
        limit=limit,
    )
    data = await make_request(url)
    return dump_api_items(
        data,
        "segmented_revenues",
        "Unable to fetch segmented revenues or no segmented revenues found.",
    )


@mcp.tool()
async def getFilingItems(
    ticker: str,
    filing_type: str,
    filing_date: str,
    item: str,
) -> str:
    """Extract specific sections from a filing."""
    url = build_api_url(
        "/filings/items/",
        ticker=ticker,
        filing_type=filing_type,
        filing_date=filing_date,
        item=item,
    )
    data = await make_request(url)
    if not data:
        return "Unable to fetch filing items or no filing items found."

    filing_item = data.get("filing_item")
    if filing_item:
        return json.dumps(filing_item, indent=2)

    items = data.get("filing_items", [])
    if items:
        return json.dumps(items, indent=2)

    return "Unable to fetch filing items or no filing items found."


@mcp.tool()
async def getAvailableFilingItems(filing_type: str) -> str:
    """Get a list of extractable items for a filing type."""
    url = build_api_url("/filings/items/available/", filing_type=filing_type)
    data = await make_request(url)
    return dump_api_items(
        data,
        "items",
        "Unable to fetch available filing items or no filing items found.",
    )


@mcp.tool()
async def getCompanyFacts(ticker: str) -> str:
    """Get company details such as market cap, sector, industry, and exchange."""
    url = build_api_url("/company/facts/", ticker=ticker)
    data = await make_request(url)
    if not data:
        return "Unable to fetch company facts or no company facts found."

    company_facts = data.get("company_facts")
    if company_facts:
        return json.dumps(company_facts, indent=2)

    facts = data.get("facts")
    if facts:
        return json.dumps(facts, indent=2)

    return "Unable to fetch company facts or no company facts found."

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
