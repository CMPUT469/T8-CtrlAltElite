# Real Gorilla BFCL Tools

**Date**: March 3, 2026  
**Status**: Ready for Threshold Testing

**Source**: Berkeley Function Calling Leaderboard (BFCL)  
`github.com/ShishirPatil/gorilla/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval/func_source_code/math_api.py`

---

## Quick Start for Week 2

```bash
# 1. Install dependencies
pip install -r requirements-eval.txt

# 2. Verify tools work
uv run python test_bfcl_tools.py

# 3. Run F1/TSR evaluation (Week 2 deliverable)
python evaluate_bfcl.py --model qwen2.5 --category simple --limit 50

# 4. For threshold testing
python test_threshold.py --model qwen2.5 --suite all

# 5. For interactive exploration
cd mcp-client
uv run main.py --transport stdio --server ../mcp-server/main.py --model qwen2.5
```

---

## Tools Implemented (14 total)

### Basic Arithmetic (9 tools)
```python
add(a, b)                          # Add two numbers
subtract(a, b)                     # Subtract one from another
multiply(a, b)                     # Multiply two numbers
divide(a, b)                       # Divide (with zero check)
power(base, exponent)              # Raise to power
square_root(number, precision)     # Calculate √ with precision
absolute_value(number)             # Get |x|
round_number(number, decimal_places)  # Round to decimals
percentage(part, whole)            # Calculate percentage
```

### List Operations (5 tools)
```python
sum_values(numbers: List)          # Sum all numbers
mean(numbers: List)                # Calculate average
min_value(numbers: List)           # Find minimum
max_value(numbers: List)           # Find maximum
standard_deviation(numbers: List)  # Calculate σ
```

---

## Verification: All Tools Working

```bash
$ uv run python test_bfcl_tools.py
All tests passed (add, divide, square_root, mean, standard_deviation)
```

---

## Week 2 Threshold Testing

Goal: Find the threshold limit of the qwen2.5 model

**Testing Areas:**
1. Tool Count - Test with 3, 7, 14 tools to find degradation point
2. Parameter Complexity - Simple (2 params), medium (precision), complex (lists)
3. Multi-Step Reasoning - 2-step, 3-step chains
4. Error Handling - Division by zero, negative square root, empty lists

### Setup (Do This First!)

```bash
# Install evaluation dependencies
pip install -r requirements-eval.txt

# Verify tools are working
uv run python test_bfcl_tools.py
```

### Three Testing Options

#### Option 1: Manual Interactive Testing (Recommended for Learning)

**Use when:** You want to understand the flow, see step-by-step what the model does, or explore edge cases interactively.

```bash
cd mcp-client
uv run main.py --transport stdio --server ../mcp-server/main.py --model qwen2.5
```

Follow the comprehensive testing guide in [`docs/manual_threshold_tests.md`](docs/manual_threshold_tests.md) which includes:
- Visual flow diagrams showing User → Client → Model → Tool → Result
- What to look for in terminal output (success/failure indicators)
- Fill-in-the-blank logging templates for recording results
- 5 test categories: Simple, Medium, Complex, Multi-Step, Tool Count Threshold

**Example queries:**
- "Add 42 and 15"
- "What is the square root of 144 with 2 decimal places?"
- "Calculate the mean of 10, 20, 30, 40, 50"
- "Add 25 and 15, then multiply the result by 3"

**Best for:** First-time testing, understanding model behavior, teaching others, exploring failures.

#### Option 2: Automated Threshold Testing

**Use when:** You want to run the full test suite quickly, compare multiple models systematically, or generate metrics for reports.

```bash
python test_threshold.py --model qwen2.5 --suite all
```

Results saved to: `threshold_results_qwen2.5_TIMESTAMP.json`

**Best for:** Week 3 model comparisons, generating quantitative metrics, batch testing.

#### Option 3: BFCL F1/TSR Evaluation (Week 2 Deliverable)

**Use when:** You need to calculate F1 score and TSR metrics using real BFCL dataset test cases.

```bash
# Download BFCL dataset and run evaluation
python evaluate_bfcl.py --model qwen2.5 --category simple

# Test with limited cases first
python evaluate_bfcl.py --model qwen2.5 --category simple --limit 50

# Just download dataset to inspect
python evaluate_bfcl.py --download-only
```

**Calculates:**
- F1 Score (function calling accuracy)
- TSR (Tool Selection Rate) - function/params/results
- Precision & Recall
- Detailed per-test results

Results saved to: `bfcl_results_qwen2.5_TIMESTAMP.json`

**Best for:** Week 2 client deliverable, comparing against BFCL leaderboard standards.

---

## Project Timeline

| Week | Dates | Deliverable | Status |
|------|-------|-------------|--------|
| **Week 2** | Mar 2-6 | **Connect MCP + Find qwen2.5 threshold** | **READY** |
| Week 3 | Mar 9-13 | Test other models   | Next |
| Week 4 | Mar 16-20 | Full F1/TSR evaluation with Gorilla dataset | Later |

## Notes

### Why BFCL Math Tools
- Deterministic results (√144 = 12, always)
- Easy to verify correctness automatically
- Tests core LLM reasoning without domain knowledge

### Implementation Details
- Proper error handling (division by zero, negative square root, etc.)
- Type checking on all parameters
- Consistent return format: `{"result": value}` or `{"error": message}`

## Week 2 Success Criteria

By March 6, document:
- [x] Tools Implemented - 14 real Gorilla BFCL tools
- [ ] Tool Count Threshold - Max tools before accuracy drops
- [ ] Parameter Complexity Limit - Can qwen2.5 handle list params?
- [ ] Multi-Step Limit - Max chain length (2, 3, 4+ steps?)
- [ ] Error Recovery - How does qwen2.5 handle tool errors?

### Week 2 Goals

- [ ] Find tool count threshold (how many tools before accuracy drops?)
- [ ] Test parameter complexity limits (simple vs medium vs complex)
- [ ] Measure multi-step reasoning (can it chain 2, 3, 4+ tool calls?)
- [ ] Document results in `WEEK2_THRESHOLD_RESULTS.md`
