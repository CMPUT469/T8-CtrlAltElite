# Manual Threshold Testing Guide - Week 2

## Overview

**Goal:** Find the threshold limits of qwen2.5 when using MCP tools  
**Method:** Manual interactive testing with graduated complexity  
**Output:** Quantitative metrics to compare against other models in Week 3

---

## Understanding the Testing Flow

### What We're Testing:
1. **Tool Selection Accuracy** - Does qwen2.5 pick the right tool?
2. **Parameter Extraction** - Does it extract correct values from natural language?
3. **Multi-Step Reasoning** - Can it chain multiple tool calls?
4. **Threshold Limits** - At what complexity does performance degrade?

### How It Works:

```
┌─────────────────┐
│  You type query │  "Add 15 and 27"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   MCP Client    │  Sends query + available tools to qwen2.5
│   (main.py)     │  
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  qwen2.5 Model  │  Analyzes query, decides which tool to use
│  (via Ollama)   │  Extracts parameters from natural language
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Tool Result   │  {"result": 42}
│   Displayed     │  
└─────────────────┘
```

### What to Look For:

**Success Indicators:**
- Model calls the correct tool (e.g., `add` not `multiply`)
- Parameters are correct (e.g., `a=15, b=27`)
- Result makes sense (e.g., `42` is correct sum)

**Failure Indicators:**
- Wrong tool selected
- Missing or incorrect parameters
- No tool call made (pure text response)
- Error in execution

---

## Setup Instructions

### Step 1: Start the MCP Client
```bash
cd mcp-client
uv run main.py --transport stdio --server ../mcp-server/main.py --model qwen2.5
```

### Step 2: Understand the Output

When you type a query, you'll see:

```
You: Add 15 and 27
────────────────────────────────────────────
🤖 Model Response:
add(a=15, b=27)

🔧 Tool Execution:
Tool: add
Arguments: {"a": 15, "b": 27}
Result: {"result": 42}

Final Answer:
The sum of 15 and 27 is 42.
```

### Step 3: How to Log Results

For each test below, fill in this template:

```
Tool Called: _____________    (e.g., "add", "multiply", "mean")
Parameters: _____________     (e.g., "a=15, b=27")
Result: _____________         (e.g., "42")
[x] Correct [ ] Wrong Tool [ ] Wrong Params [ ] No Tool Call

Notes: _________________________________
```

### Example Log Entry:

```
Test 1.1 - Addition
Query: "Add 15 and 27"

Tool Called: add              ✓ CORRECT
Parameters: a=15, b=27        ✓ CORRECT  
Result: 42                    ✓ CORRECT
[x] Correct [ ] Wrong Tool [ ] Wrong Params [ ] No Tool Call

Notes: Worked perfectly on first try. Model understood natural language.
```

---

## Available Tools Reference

**Basic Arithmetic (2 params):**
- `add(a, b)` - Add two numbers
- `subtract(a, b)` - Subtract b from a
- `multiply(a, b)` - Multiply two numbers
- `divide(a, b)` - Divide a by b
- `power(base, exponent)` - Raise base to exponent

**Single Number Operations (1-2 params):**
- `square_root(number, precision)` - Calculate square root
- `absolute_value(number)` - Get absolute value
- `round_number(number, decimal_places)` - Round to decimals
- `percentage(part, whole)` - Calculate percentage

**List Operations (1 param - array):**
- `sum_values(numbers)` - Sum all numbers in list
- `mean(numbers)` - Calculate average
- `min_value(numbers)` - Find minimum
- `max_value(numbers)` - Find maximum
- `standard_deviation(numbers)` - Calculate std dev

---

## Test 1: Simple Tools (2 parameters)

**Purpose:** Test basic tool selection with simple parameters

### Test 1.1 - Addition
**Query:** `Add 15 and 27`

**Expected:**
- Tool: `add`
- Params: `a=15, b=27`
- Result: `42`

**Log your results:**
```
Tool Called: _____________
Parameters: _____________
Result: _____________
[x] Correct [ ] Wrong Tool [ ] Wrong Params [ ] No Tool Call

Notes: _________________________________
```

---

### Test 1.2 - Subtraction
**Query:** `What is 100 minus 37?`

**Expected:**
- Tool: `subtract`
- Params: `a=100, b=37`
- Result: `63`

**Log your results:**
```
Tool Called: _____________
Parameters: _____________
Result: _____________
[x] Correct [ ] Wrong Tool [ ] Wrong Params [ ] No Tool Call

Notes: _________________________________
```

---

### Test 1.3 - Multiplication
**Query:** `Multiply 8 by 12`

**Expected:**
- Tool: `multiply`
- Params: `a=8, b=12`
- Result: `96`

**Log your results:**
```
Tool Called: _____________
Parameters: _____________
Result: _____________
[x] Correct [ ] Wrong Tool [ ] Wrong Params [ ] No Tool Call

Notes: _________________________________
```

---

### Test 1.4 - Division
**Query:** `Divide 144 by 12`

**Expected:**
- Tool: `divide`
- Params: `a=144, b=12`
- Result: `12`

**Log your results:**
```
Tool Called: _____________
Parameters: _____________
Result: _____________
[x] Correct [ ] Wrong Tool [ ] Wrong Params [ ] No Tool Call

Notes: _________________________________
```

---

### Test 1.5 - Absolute Value
**Query:** `What is the absolute value of -42?`

**Expected:**
- Tool: `absolute_value`
- Params: `number=-42`
- Result: `42`

**Log your results:**
```
Tool Called: _____________
Parameters: _____________
Result: _____________
[x] Correct [ ] Wrong Tool [ ] Wrong Params [ ] No Tool Call

Notes: _________________________________
```

**Test 1 Summary:**
```
Successful: ___ / 5
Accuracy: ___%
Common errors: _________________________________
```

---

## Test 2: Medium Complexity (Precision & Named Parameters)

**Purpose:** Test parameter complexity and precision handling

### Test 2.1 - Square Root with Precision
**Query:** `Calculate the square root of 144 with 2 decimal places`

**Expected:**
- Tool: `square_root`
- Params: `number=144, precision=2`
- Result: `12.00`

**Log your results:**
```
Tool Called: _____________
Parameters: _____________
Result: _____________
[x] Correct [ ] Wrong Tool [ ] Wrong Params [ ] No Tool Call

Did model extract precision correctly? [ ] Yes [ ] No
Notes: _________________________________
```

---

### Test 2.2 - Power
**Query:** `What is 5 raised to the power of 3?`

**Expected:**
- Tool: `power`
- Params: `base=5, exponent=3`
- Result: `125`

**Log your results:**
```
Tool Called: _____________
Parameters: _____________
Result: _____________
[x] Correct [ ] Wrong Tool [ ] Wrong Params [ ] No Tool Call

Notes: _________________________________
```

---

### Test 2.3 - Percentage
**Query:** `What percentage is 50 out of 200?`

**Expected:**
- Tool: `percentage`
- Params: `part=50, whole=200`
- Result: `25` (25%)

**Log your results:**
```
Tool Called: _____________
Parameters: _____________
Result: _____________
[x] Correct [ ] Wrong Tool [ ] Wrong Params [ ] No Tool Call

Notes: _________________________________
```

---

### Test 2.4 - Rounding
**Query:** `Round 3.14159 to 3 decimal places`

**Expected:**
- Tool: `round_number`
- Params: `number=3.14159, decimal_places=3`
- Result: `3.142`

**Log your results:**
```
Tool Called: _____________
Parameters: _____________
Result: _____________
[x] Correct [ ] Wrong Tool [ ] Wrong Params [ ] No Tool Call

Notes: _________________________________
```

**Test 2 Summary:**
```
Successful: ___ / 4
Accuracy: ___%
Issues with precision params? [ ] Yes [ ] No
Describe: _________________________________
```

---

## Test 3: Complex Tools (List/Array Parameters)

**Purpose:** Test handling of array/list parameters

### Test 3.1 - Mean (Average)
**Query:** `Calculate the average of 10, 20, 30, 40, and 50`

**Expected:**
- Tool: `mean`
- Params: `numbers=[10, 20, 30, 40, 50]`
- Result: `30`

**What to watch:**
- Does model format list correctly?
- Are all numbers included?

**Log your results:**
```
Tool Called: _____________
Parameters: _____________
Result: _____________
[x] Correct [ ] Wrong Tool [ ] Wrong Params [ ] No Tool Call

List formatted correctly? [ ] Yes [ ] No
All values included? [ ] Yes [ ] No
Notes: _________________________________
```

---

### Test 3.2 - Standard Deviation
**Query:** `Find the standard deviation of 2, 4, 6, 8, 10`

**Expected:**
- Tool: `standard_deviation`
- Params: `numbers=[2, 4, 6, 8, 10]`
- Result: ~`2.83`

**Log your results:**
```
Tool Called: _____________
Parameters: _____________
Result: _____________
[x] Correct [ ] Wrong Tool [ ] Wrong Params [ ] No Tool Call

Notes: _________________________________
```

---

### Test 3.3 - Minimum Value
**Query:** `What is the minimum value in the list: 15, 3, 27, 8, 19?`

**Expected:**
- Tool: `min_value`
- Params: `numbers=[15, 3, 27, 8, 19]`
- Result: `3`

**Log your results:**
```
Tool Called: _____________
Parameters: _____________
Result: _____________
[x] Correct [ ] Wrong Tool [ ] Wrong Params [ ] No Tool Call

Notes: _________________________________
```

---

### Test 3.4 - Sum
**Query:** `Add up all these numbers: 5, 10, 15, 20, 25`

**Expected:**
- Tool: `sum_values`
- Params: `numbers=[5, 10, 15, 20, 25]`
- Result: `75`

**Log your results:**
```
Tool Called: _____________
Parameters: _____________
Result: _____________
[x] Correct [ ] Wrong Tool [ ] Wrong Params [ ] No Tool Call

Notes: _________________________________
```

**Test 3 Summary:**
```
Successful: ___ / 4
Accuracy: ___%
List parameter issues? [ ] Yes [ ] No
Describe issues: _________________________________
```

---

## Test 4: Multi-Step Reasoning

**Purpose:** Test if model can chain multiple tool calls

### Test 4.1 - Two-Step Calculation
**Query:** `Add 15 and 27, then multiply the result by 3`

**Expected Steps:**
1. `add(a=15, b=27)` → `42`
2. `multiply(a=42, b=3)` → `126`

**What to watch:**
- Does model make both calls?
- Does it use first result in second call?
- Is final answer correct?

**Log your results:**
```
Step 1 Tool: _____________
Step 1 Result: _____________
Step 2 Tool: _____________
Step 2 Result: _____________
Final Answer: _____________

[x] Both steps completed [ ] Only one step [ ] No multi-step
Did model use first result in second call? [ ] Yes [ ] No

Notes: _________________________________
```

---

### Test 4.2 - Three-Step Chain
**Query:** `Calculate: (10 + 5) * 2 - 8`

**Expected Steps:**
1. `add(a=10, b=5)` → `15`
2. `multiply(a=15, b=2)` → `30`
3. `subtract(a=30, b=8)` → `22`

**Log your results:**
```
Step 1: _____________
Step 2: _____________
Step 3: _____________
Final Answer: _____________

Completed steps: ___ / 3
[x] All steps [ ] Partial [ ] Failed

Notes: _________________________________
```

---

### Test 4.3 - List Then Calculate
**Query:** `Find the average of 10, 20, 30, then multiply it by 2`

**Expected Steps:**
1. `mean(numbers=[10, 20, 30])` → `20`
2. `multiply(a=20, b=2)` → `40`

**Log your results:**
```
Step 1: _____________
Step 2: _____________
Final Answer: _____________

[x] Both steps [ ] Only one [ ] Failed

Notes: _________________________________
```

**Test 4 Summary:**
```
Successful: ___ / 3
2-step accuracy: ___%
3-step accuracy: ___%
Max reliable chain length: ___ steps
```

---

## Test 5: Tool Count Threshold

**Purpose:** Find if performance degrades with more available tools

### Setup Instructions:

You'll need to modify the server to expose different numbers of tools.

**Method:** Comment out unused `@mcp.tool()` functions in `mcp-server/main.py`

**3 Tools Test:** Enable only:
- `add`, `subtract`, `multiply`

**7 Tools Test:** Enable:
- `add`, `subtract`, `multiply`, `divide`, `power`, `square_root`, `absolute_value`

**14 Tools Test:** Enable all tools

### 3 Tools Available

**Restart server with only 3 tools enabled**

Run these queries:
1. "Add 10 and 20" → Expected: `add(10, 20)` = 30
2. "Subtract 50 from 100" → Expected: `subtract(100, 50)` = 50
3. "Multiply 6 by 7" → Expected: `multiply(6, 7)` = 42

```
Results: ___ / 3 correct
Accuracy: ___%
```

---

### 7 Tools Available

**Restart server with 7 tools enabled**

Run these queries:
1. "Divide 100 by 5"
2. "What is 2 to the power of 8?"
3. "Square root of 81 with 0 decimals"
4. "Absolute value of -15"
5. "Add 25 and 75"

```
Results: ___ / 5 correct
Accuracy: ___%
```

---

### 14 Tools Available (All)

**Restart server with all 14 tools**

Run these queries:
1. "Add 12 and 18"
2. "Average of 5, 10, 15, 20"
3. "Divide 200 by 8"
4. "Minimum of 7, 3, 9, 1, 5"
5. "What percentage is 30 out of 150?"

```
Results: ___ / 5 correct
Accuracy: ___%
```

**Test 5 Summary:**
```
3 tools: ___%
7 tools: ___%
14 tools: ___%

Performance degradation observed? [ ] Yes [ ] No
At what tool count: ___ tools
```

---

## Final Summary - Week 2 Findings

### qwen2.5 Threshold Metrics:

**Tool Selection Accuracy:**
- Simple tools (2 params): ___%
- Medium tools (precision): ___%
- Complex tools (lists): ___%
- Overall: ___%

**Multi-Step Reasoning:**
- 2-step chains: ___%
- 3-step chains: ___%
- Max reliable chain: ___ steps

**Tool Count Threshold:**
- Performance stable up to: ___ tools
- Degradation starts at: ___ tools

**Parameter Handling:**
- [ ] Handles numeric params well
- [ ] Handles precision params well
- [ ] Handles list/array params well
- [ ] Struggles with: _________________________________

**Common Failure Modes:**
1. _________________________________
2. _________________________________
3. _________________________________

### Key Observations:

**Strengths:**
- _________________________________
- _________________________________

**Weaknesses:**
- _________________________________
- _________________________________

**Context Window:**
- [ ] Good (handles full conversation)
- [ ] Moderate (loses context after ___ exchanges)
- [ ] Poor (frequent context loss)

### Recommendations for Week 3:

**Models to prioritize testing:**
- 

**Test focus areas:**
- [ ] Repeat failures from qwen2.5
- [ ] Compare multi-step reasoning
- [ ] Test tool count threshold

**Improvements needed:**
- _________________________________
- _________________________________

---

**Testing completed by:** _______________  
**Date:** _______________  
**Duration:** ___ hours
