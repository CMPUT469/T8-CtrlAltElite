# Manual Threshold Testing Guide - Week 2

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

📊 Final Answer:
The sum of 15 and 27 is 42.
```

### Step 3: Record Your Observations

For each test below:
1. **Copy the query** and paste it into the client
2. **Check the output** - did it call the right tool?
3. **Mark success/failure** in the checkbox
4. **Note any issues** in the observations section

### Example Log Entry:

```
Test 1.2 - Addition
Query: "Add 15 and 27"
✓ Tool Called: add (CORRECT)
✓ Parameters: a=15, b=27 (CORRECT)
✓ Result: 42 (CORRECT)
[x] Success
[ ] Failure

Notes: Worked perfectly on first try.
```

---

## Test 1: Simple Tools (1-2 params)

### Test 1.1 - Stock Price
**Query:** `What is the stock price of Apple?`
- **Expected Tool:** `get_stock_price`
- **Expected Params:** `{"ticker": "AAPL"}`
- **Success?** [ ] Yes [ ] No

### Test 1.2 - Addition
**Query:** `Add 15 and 27`
- **Expected Tool:** `add`
- **Expected Params:** `{"a": 15, "b": 27}`
- **Success?** [ ] Yes [ ] No

### Test 1.3 - Multiplication  
**Query:** `What is 50 multiplied by 8?`
- **Expected Tool:** `multiply`
- **Expected Params:** `{"a": 50, "b": 8}`
- **Success?** [ ] Yes [ ] No

### Test 1.4 - Index Value
**Query:** `Get the current value of NASDAQ index`
- **Expected Tool:** `get_index_value`
- **Expected Params:** `{"index": "NASDAQ"}`
- **Success?** [ ] Yes [ ] No

### Test 1.5 - Currency Conversion
**Query:** `Convert 1000 USD to EUR`
- **Expected Tool:** `convert_currency`
- **Expected Params:** `{"amount": 1000, "from_curr": "USD", "to_curr": "EUR"}`
- **Success?** [ ] Yes [ ] No

**Simple Test Accuracy: ___ / 5 = ___%**

---

## Test 2: Medium Complexity (3-4 params)

### Test 2.1 - Loan Payment
**Query:** `Calculate monthly payment for a $200,000 loan at 6% interest for 30 years`
- **Expected Tool:** `calculate_loan_payment`
- **Expected Params:** `{"principal": 200000, "rate": 6, "months": 360}`
- **Success?** [ ] Yes [ ] No

### Test 2.2 - Percentage
**Query:** `What is 20% of 500?`
- **Expected Tool:** `calculate_percentage`
- **Expected Params:** `{"value": 500, "percentage": 20}`
- **Success?** [ ] Yes [ ] No

### Test 2.3 - Stock Comparison
**Query:** `Compare stock prices of Apple and Microsoft`
- **Expected Tool:** `compare_stocks`
- **Expected Params:** `{"ticker1": "AAPL", "ticker2": "MSFT"}`
- **Success?** [ ] Yes [ ] No

### Test 2.4 - Simple Interest
**Query:** `Calculate the interest on $10,000 at 5% rate for 3 years`
- **Expected Tool:** `calculate_simple_interest`
- **Expected Params:** `{"principal": 10000, "rate": 5, "time": 3}`
- **Success?** [ ] Yes [ ] No

**Medium Test Accuracy: ___ / 4 = ___%**

---

## Test 3: Complex Tools (4+ params or parsing)

### Test 3.1 - Portfolio Value
**Query:** `Calculate my portfolio value: 10 shares of AAPL, 5 shares of GOOGL, and 8 shares of TSLA`
- **Expected Tool:** `portfolio_value`
- **Expected Params:** `{"holdings": "AAPL,GOOGL,TSLA", "quantities": "10,5,8"}`
- **Success?** [ ] Yes [ ] No

### Test 3.2 - Investment Return
**Query:** `If I invested $5000 and it grew to $8000 over 3 years, what's my return?`
- **Expected Tool:** `calculate_investment_return`
- **Expected Params:** `{"initial": 5000, "final": 8000, "years": 3}`
- **Success?** [ ] Yes [ ] No

**Complex Test Accuracy: ___ / 2 = ___%**

---

## Test 4: Multi-Step Reasoning

### Test 4.1 - Two Steps
**Query:** `What is the combined stock price of Apple and Google?`
- **Expected Steps:**
  1. `get_stock_price({"ticker": "AAPL"})`
  2. `get_stock_price({"ticker": "GOOGL"})`
  3. `add({"a": result1, "b": result2})`
- **Success?** [ ] Yes [ ] No
- **Completed Steps:** ___

### Test 4.2 - Three Steps
**Query:** `Get stock prices for AAPL, MSFT, and TSLA, then calculate the average`
- **Expected Steps:** 3 `get_stock_price` calls + arithmetic
- **Success?** [ ] Yes [ ] No
- **Completed Steps:** ___

### Test 4.3 - Chained Tools
**Query:** `Compare AAPL and GOOGL prices, then calculate what 15% of the difference would be`
- **Expected Steps:**
  1. `compare_stocks({"ticker1": "AAPL", "ticker2": "GOOGL"})`
  2. `calculate_percentage({"value": difference, "percentage": 15})`
- **Success?** [ ] Yes [ ] No
- **Completed Steps:** ___

**Multi-Step Test Accuracy: ___ / 3 = ___%**

---

## Test 5: Tool Count Threshold

Test with increasing numbers of available tools to find degradation point.

### 5 Tools Available
Run 5 simple tests with only these tools available:
- `get_stock_price`, `add`, `subtract`, `multiply`, `divide`

**Accuracy: ___ / 5 = ___%**

### 10 Tools Available
Run 5 simple tests with 10 tools available.

**Accuracy: ___ / 5 = ___%**

### 15 Tools Available
Run 5 simple tests with all 15+ tools available.

**Accuracy: ___ / 5 = ___%**

---

## Summary - Week 2 Findings

### qwen2.5 Thresholds:

**Simple Tools (1-2 params):** ___% accuracy
**Medium Tools (3-4 params):** ___% accuracy  
**Complex Tools (4+ params):** ___% accuracy
**Multi-Step (2-4 chains):** ___% accuracy

**Tool Count Limit:** Performance starts degrading after ___ tools

**Parameter Complexity Limit:** 
- [ ] Handles 1-2 params well
- [ ] Handles 3-4 params well
- [ ] Struggles with 5+ params

**Multi-Step Limit:** Can reliably chain ___ tool calls

**Context Window:** 
- [ ] Good (handles full conversation)
- [ ] Moderate (loses context after ___ exchanges)
- [ ] Poor (needs improvement)

### Observations:
- 
- 
- 

### Recommendations for Week 3:
- 
- 
- 

---

**Completed by:** _______________  
**Date:** _______________
