# Week 2 Progress: Real Gorilla BFCL Tools Implementation

**Date**: March 3, 2026  
**Status**: Complete - Ready for Threshold Testing

## Summary

Replaced custom financial tools with 14 actual tools from Berkeley Function Calling Leaderboard (BFCL).

**Source**: `github.com/ShishirPatil/gorilla/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval/func_source_code/math_api.py`

---

## The 14 Real Gorilla BFCL Tools

### Basic Arithmetic (9 tools)
1. **add**(a, b) - Add two numbers
2. **subtract**(a, b) - Subtract one number from another
3. **multiply**(a, b) - Multiply two numbers
4. **divide**(a, b) - Divide one number by another
5. **power**(base, exponent) - Raise to power
6. **square_root**(number, precision) - Calculate square root
7. **absolute_value**(number) - Get absolute value
8. **round_number**(number, decimal_places) - Round to decimals
9. **percentage**(part, whole) - Calculate percentage

### List Operations (5 tools)
10. **sum_values**(numbers: List) - Sum all numbers
11. **mean**(numbers: List) - Calculate average
12. **min_value**(numbers: List) - Find minimum
13. **max_value**(numbers: List) - Find maximum
14. **standard_deviation**(numbers: List) - Calculate std dev

---

## Verification

```bash
$ uv run python test_bfcl_tools.py

Server has 14 tools
add(42, 15) → 57.0
divide(100, 4) → 25.0  
square_root(144, 2) → 12.0
mean([10,20,30,40,50]) → 30.0
standard_deviation([2,4,6,8,10]) → 2.828...

All tests passed
```

## Why These Tools?

- Used in Gorilla's function calling evaluation benchmark
- Proper error handling (division by zero, negative square root, etc.)
- Simple enough for lightweight models, complex enough to test thresholds

---

## Week 2 Threshold Testing

Goal: Find the threshold limit of qwen2.5 model

### Tests to Run

1. **Tool Count Threshold**
   - Test with 3, 7, and 14 tools
   - Determine at what count accuracy degrades

2. **Parameter Complexity**
   - Simple: `add(a, b)` (2 params)
   - Medium: `square_root(number, precision)` (2 params + precision)
   - Complex: `mean([1,2,3,4,5])` (list params)

3. **Multi-Step Reasoning**
   - 2 steps: "Calculate 10+5, then divide by 3"
   - 3 steps: "Get mean of [2,4,6], square root it, round to 2 places"
   - Test maximum sequential tool call length

4. **Error Recovery**
   - Division by zero, negative square root, empty list operations

---

## Testing Commands

```bash
# Interactive client
cd mcp-client
uv run main.py --transport stdio --server ../mcp-server/main.py --model qwen2.5

# Automated threshold tests
uv run python test_threshold.py --model qwen2.5 --suite all

# Manual checklist
open docs/manual_threshold_tests.md
```

## Timeline

## Timeline

| Week | Dates | Deliverable | Status |
|------|-------|-------------|--------|
| Week 2 | Mar 2-6 | Connect MCP + Find qwen2.5 threshold | In Progress |
| Week 3 | Mar 9-13 | Test other models | Pending |
| Week 4 | Mar 16-20 | Full F1/TSR evaluation | Pending |
