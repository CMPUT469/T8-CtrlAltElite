# Implementation Plan - Aligned with Course Timeline

## Current Status: Week 2 (2-6 March)

### Week 1 Completed (23-27 Feb)
- [x] GitHub repository setup
- [x] Basic MCP server with demo tools
- [x] MCP client with Ollama integration

---

## Week 2 (THIS WEEK): First LLM Connection & Threshold Testing

### Goals:
1. [x] Connect MCP server to qwen2.5 (ALREADY WORKING)
2. [ ] Find threshold limit of qwen2.5
3. [ ] Add diverse tools to test capabilities

### What is "Threshold Limit"?

Test qwen2.5's limits in:
1. **Number of tools** - How many tools before accuracy drops?
2. **Tool complexity** - Simple vs complex parameter schemas?
3. **Context length** - How many tool results can it handle?
4. **Multi-step reasoning** - Can it chain 2, 3, 4+ tool calls?

### Tasks for Week 2:

#### Monday-Tuesday (TODAY): Add Tools
**Add 10-15 diverse tools to test different capabilities:**

```python
# Simple tools (1-2 params)
get_weather(location: str)
get_stock_price(symbol: str)
calculate(expression: str)

# Medium complexity (3-4 params)
search_news(topic: str, date: str, country: str)
convert_currency(amount: float, from_curr: str, to_curr: str)

# Complex tools (5+ params, nested objects)
book_flight(origin: str, destination: str, date: str, passengers: int, class: str)
analyze_portfolio(stocks: List[str], timeframe: str, metrics: List[str])
```

#### Wednesday: Threshold Testing
**Test 1: Number of Tools**
- Start with 3 tools → measure accuracy
- Add 5 more → measure accuracy
- Add 10 more → measure accuracy
- Find the point where performance degrades

**Test 2: Parameter Complexity**
- Simple (1 param): "What's weather in NYC?" → `get_weather("NYC")`
- Medium (3 params): "Show me tech news from USA yesterday" → `search_news("tech", "2026-03-02", "USA")`
- Complex (5+ params): "Book 2 economy seats from LAX to JFK on March 15"

**Test 3: Multi-Step Chains**
- 1 tool: "What's Apple stock price?"
- 2 tools: "Compare Apple and Google stock prices"
- 3 tools: "Get weather in NYC, LA, and Chicago"
- 4+ tools: "Compare stock prices, get company sectors, and calculate average"

#### Thursday-Friday: Document Threshold Results
Create `WEEK2_THRESHOLD_RESULTS.md` with:
- Maximum tools before degradation
- Best parameter complexity
- Multi-step limit
- Context window observations

---

## Week 3 (9-13 March): Test Other Lightweight Models

### Same Threshold Tests for Each Model:
- Number of tools threshold
- Parameter complexity limit
- Multi-step reasoning capability
- Context window handling

### Deliverable: Comparison table

---

## Week 4 (16-20 March): Calculate Evaluation Metrics

### NOW we download Gorilla APIBench and calculate F1/TSR

```bash
# Download benchmark dataset
python evaluate_gorilla.py

# Run evaluation on each model
python run_evaluation.py --model qwen2.5
python run_evaluation.py --model llama3.2
python run_evaluation.py --model phi3
```

### Metrics to Calculate:

**F1 Score** - Tool selection accuracy
```
Precision = Correct tools / All tools called
Recall = Correct tools / All tools needed
F1 = 2 * (P * R) / (P + R)
```

**TSR (Tool Selection Rate)** - Success percentage
```
TSR = Correct selections / Total tests
```

**Additional Metrics:**
- Parameter accuracy
- Multi-step success rate
- Execution time per tool call

### Deliverable: 
- Baseline metrics for all models
- Identify which model performs best
- Document failure patterns

---

## Week 5 (23-27 March): Fine-Tuning

### Fine-tune best performing model on:
1. Tool selection accuracy
2. Parameter extraction from natural language
3. Multi-step planning

### Fine-tuning approaches:
- Few-shot examples in prompts
- Prompt engineering (system prompts)
- Model fine-tuning (if time/resources permit)

---

## Week 6 (30 March - 3 April): Final Evaluation

### Re-run all metrics on fine-tuned models
### Compare:
- Before fine-tuning vs After
- Model A vs Model B vs Model C
- Single-step vs Multi-step performance

---

## Week 7-8: Presentation & Report

---

## IMMEDIATE ACTION ITEMS (Week 2)

### Option 1: Quick Start (2 hours)
I can implement:
1. **10-15 diverse tools** in `mcp-server/main.py`
2. **Simple threshold test script** to measure performance
3. **Results logging** to track metrics

### Option 2: Full Setup (4 hours)
Everything in Option 1 PLUS:
1. **Automated threshold testing suite**
2. **Comparison framework** for Week 3
3. **Pre-configured test queries** for each tool

