# Test Model Prompt - Quick Start Guide

## Prerequisites

First, make sure gpt-oss:20b is installed:

```bash
# Check if model is installed
ollama list | grep gpt-oss

# If not installed, pull it
ollama pull gpt-oss:20b
```

## Basic Usage

### 1. Simple Text Prompt (No Tools)

```bash
# Single prompt test
python3 test_model_prompt.py --model gpt-oss:20b --prompt "What is the capital of France?"

# Interactive mode - enter multiple prompts
python3 test_model_prompt.py --model gpt-oss:20b
```

**Example:**
```
$ python3 test_model_prompt.py --model gpt-oss:20b --prompt "Explain quantum computing"

============================================================
Testing Model: gpt-oss:20b
============================================================

📝 Your Prompt:
Explain quantum computing

🤖 Model Response:
------------------------------------------------------------
Quantum computing is a type of computing that uses quantum-mechanical phenomena...
------------------------------------------------------------
✅ Success!
```

### 2. With Tools (Math & Statistics)

```bash
# Test with tools - single prompt
python3 test_model_prompt.py --model gpt-oss:20b --with-tools --prompt "Calculate the mean of [1, 2, 3, 4, 5]"

# Interactive mode with tools
python3 test_model_prompt.py --model gpt-oss:20b --with-tools
```

**Example:**
```
$ python3 test_model_prompt.py --model qwen2.5:7b --with-tools --prompt "Calculate skewness of [1,2,2,3,3,3,4,10,20]"

============================================================
Testing Model with Tools: qwen2.5:7b
============================================================

📝 Your Prompt:
Calculate skewness of [1,2,2,3,3,3,4,10,20]

🔧 Connecting to MCP server...
✅ Connected! 32 tools available

🛠️  Available Tools: add, subtract, multiply, divide, power...

🤖 Model Response:
------------------------------------------------------------
✅ Native Tool Call Detected!

🔧 Tool: calculate_skewness
📋 Arguments: {
  "collection": [1,2,2,3,3,3,4,10,20]
}

⚙️  Executing tool...
✅ Result: {
  "result": 1.8030165860570084
}
------------------------------------------------------------
✅ Test complete!
```

### 3. With Fallback Mode (for non-native tool models)

If gpt-oss:20b doesn't support native tool calling:

```bash
python3 test_model_prompt.py --model gpt-oss:20b --with-tools --fallback --prompt "Calculate mean of [1,2,3]"
```

This tells the model to respond with JSON format like: `{"tool":"calculate_mean","args":{"collection":[1,2,3]}}`

## Interactive Mode Examples

### Simple Chat
```bash
$ python3 test_model_prompt.py --model gpt-oss:20b

============================================================
🎯 Interactive Mode - gpt-oss:20b
============================================================
Enter your prompts below. Type 'quit' or 'exit' to stop.
============================================================

📝 Your prompt: What is 2+2?
🤖 Model Response:
------------------------------------------------------------
2+2 equals 4
------------------------------------------------------------
✅ Success!

📝 Your prompt: Tell me a joke
🤖 Model Response:
------------------------------------------------------------
Why did the chicken cross the road? To get to the other side!
------------------------------------------------------------
✅ Success!

📝 Your prompt: quit
👋 Goodbye!
```

### With Tools Interactive
```bash
$ python3 test_model_prompt.py --model qwen2.5:7b --with-tools

============================================================
🎯 Interactive Mode - qwen2.5:7b
============================================================
Enter your prompts below. Type 'quit' or 'exit' to stop.
============================================================

📝 Your prompt: What is the mean of [1, 2, 3, 4, 5]?
✅ Native Tool Call Detected!
🔧 Tool: mean
📋 Arguments: {"collection": [1,2,3,4,5]}
⚙️  Executing tool...
✅ Result: {"result": 3.0}
------------------------------------------------------------
✅ Test complete!

📝 Your prompt: Calculate variance of [2, 4, 6, 8, 10]
✅ Native Tool Call Detected!
🔧 Tool: calculate_variance
📋 Arguments: {"collection": [2,4,6,8,10]}
⚙️  Executing tool...
✅ Result: {"result": 8.0}
------------------------------------------------------------
✅ Test complete!

📝 Your prompt: exit
👋 Goodbye!
```

## Command Reference

### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--model MODEL` | Specify model name | `--model gpt-oss:20b` |
| `--prompt "TEXT"` | Single prompt (non-interactive) | `--prompt "What is AI?"` |
| `--with-tools` | Enable 32 math/stats tools | `--with-tools` |
| `--fallback` | Enable fallback JSON parsing | `--fallback` |

### Common Scenarios

**Test if model works:**
```bash
python3 test_model_prompt.py --model gpt-oss:20b --prompt "Hello"
```

**Test tool calling capability:**
```bash
python3 test_model_prompt.py --model gpt-oss:20b --with-tools --prompt "Calculate 5+3"
```

**Compare models:**
```bash
# Test gpt-oss:20b
python3 test_model_prompt.py --model gpt-oss:20b --with-tools --prompt "Calculate mean of [1,2,3]"

# Test qwen2.5:7b
python3 test_model_prompt.py --model qwen2.5:7b --with-tools --prompt "Calculate mean of [1,2,3]"
```

**Interactive exploration:**
```bash
python3 test_model_prompt.py --model gpt-oss:20b --with-tools
# Now you can type prompts and see responses in real-time
```

## Available Tools (32 total)

When using `--with-tools`, these tools are available:

**BFCL Math Tools (14):**
- add, subtract, multiply, divide
- power, square_root, absolute_value
- round_number, percentage
- sum_values, mean, min_value, max_value
- standard_deviation

**Jefferson Stats Tools (18):**
- Basic: calculate_median, calculate_mode, calculate_range, calculate_variance, calculate_quartiles, calculate_iqr
- Advanced: calculate_skewness, calculate_kurtosis, calculate_correlation, calculate_covariance, calculate_z_scores
- Hypothesis: perform_t_test, calculate_confidence_interval, detect_outliers, perform_normality_test
- Analysis: perform_linear_regression, calculate_moving_average, generate_descriptive_statistics

## Troubleshooting

### Model not found
```
❌ Error: model 'gpt-oss:20b' not found
```
**Solution:** Install the model: `ollama pull gpt-oss:20b`

### No tool calls despite --with-tools
The model returned text instead of calling a tool.

**Reasons:**
1. Model doesn't support native tool calling → Use `--fallback`
2. Prompt wasn't clear enough → Try: "Use the calculate_mean tool to find mean of [1,2,3]"
3. Model chose not to use tools → This is expected behavior for some prompts

### Connection errors
```
❌ Error: Connection refused
```
**Solution:** Make sure Ollama is running: `ollama serve` (in separate terminal)

## Tips

✅ **DO:**
- Use `--with-tools` when asking math/stats questions
- Use `--fallback` with gpt-oss:20b if native tools don't work
- Test in interactive mode first to explore model behavior
- Compare different models side-by-side

❌ **DON'T:**
- Forget to install the model first
- Use very long prompts (models have token limits)
- Expect all models to support native tool calling

## Next Steps

After testing your prompts manually:

1. **Run full evaluations:**
   ```bash
   # Jefferson stats
   python3 evaluate_jefferson.py --model gpt-oss:20b --allow-fallback
   
   # BFCL math
   python3 evaluate_bfcl.py --model gpt-oss:20b --category simple
   ```

2. **Compare models:**
   ```bash
   for model in qwen2.5:7b qwen2.5-coder:7b llama3.1:8b; do
     python3 evaluate_jefferson.py --model $model --output results/jefferson_${model//:/_}.json
   done
   ```

3. **Analyze results:**
   ```bash
   python3 -c "import json; print(json.load(open('results/jefferson_qwen2.5_7b.json'))['metrics'])"
   ```
