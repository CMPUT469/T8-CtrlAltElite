# Evaluation Scripts - Modular Architecture

This directory contains modular evaluation scripts for testing LLM function calling capabilities across different tool datasets.

## Architecture Overview

```
.
├── evaluation_framework.py    # Shared evaluation logic (outcome-based E(O, Ô))
├── evaluate_bfcl.py           # BFCL math tools evaluation
├── evaluate_jefferson.py      # Jefferson statistics tools evaluation
├── stats_test_cases.json      # Jefferson test cases (18 tests)
└── results/
    ├── bfcl/                  # BFCL evaluation results
    └── jefferson/             # Jefferson evaluation results
```

## Design Principles

### 1. **Separation of Concerns**
- **`evaluation_framework.py`**: Common evaluation logic shared across all datasets
- **`evaluate_bfcl.py`**: BFCL-specific evaluation (math tools, 46 functions)
- **`evaluate_jefferson.py`**: Jefferson-specific evaluation (stats tools, 18 functions)

### 2. **Outcome-Based Evaluation**
All scripts implement MCPVerse methodology: **E(O, Ô) ∈ {0, 1}**
- Primary Metric: Outcome Accuracy (did the task complete successfully?)
- Auxiliary Metrics: Function selection, parameter accuracy (for trajectory analysis)
- Allows multiple valid execution paths to same outcome

### 3. **Extensibility**
To add a new dataset:
1. Create test cases JSON with `expected_result` field
2. Create `evaluate_<dataset>.py` script
3. Import shared functions from `evaluation_framework.py`

## Usage

### Jefferson Statistics Evaluation

```bash
# Basic evaluation
python3 evaluate_jefferson.py --model qwen2.5:7b

# Limit test cases
python3 evaluate_jefferson.py --model qwen2.5:7b --limit 5

# Custom test cases
python3 evaluate_jefferson.py --model qwen2.5:7b --test-cases my_tests.json

# Custom output path
python3 evaluate_jefferson.py --model qwen2.5:7b --output results/my_run.json

# Allow fallback for non-native tool models
python3 evaluate_jefferson.py --model gpt-oss:20b --allow-fallback
```

### BFCL Math Evaluation

```bash
# Basic evaluation (all categories)
python3 evaluate_bfcl.py --model qwen2.5:7b --category all

# Specific category
python3 evaluate_bfcl.py --model qwen2.5:7b --category simple

# With synthetic test cases
python3 evaluate_bfcl.py --model qwen2.5:7b --synthetic math_test_cases.jsonl
```

## Evaluation Framework API

### Core Functions

#### `compare_values(actual, expected, tolerance=0.01)`
Compare results with numerical tolerance and type coercion.
- Handles dict vs primitive: `{"result": 1.803}` vs `1.803`
- Recursive list/dict comparison
- Numerical tolerance for float comparison

#### `compare_params(actual, expected)`
Compare parameter dictionaries with type coercion.

#### `extract_result_value(tool_result)`
Extract result from various MCP tool result formats.

#### `calculate_metrics(results)`
Calculate outcome-based metrics following MCPVerse:
- **Primary**: Outcome Accuracy E(O, Ô)
- **Auxiliary**: Function selection, parameter accuracy
- **Traditional**: F1, precision, recall

#### `print_report(metrics, model)`
Print formatted evaluation report emphasizing outcome accuracy.

#### `save_results(output_path, model, metrics, raw_results)`
Save results to JSON with timestamp and full details.

## Test Case Format

### Required Fields

```json
{
  "id": "stats_001",
  "query": "Calculate the skewness of [1, 2, 3]",
  "expected_function": "calculate_skewness",
  "expected_params": {"collection": [1, 2, 3]},
  "expected_result": {"result": 0.0},
  "category": "stats"
}
```

### Key Requirements
- **`expected_result`**: Required for outcome-based evaluation
  - Can be primitive: `1.803`
  - Or dict: `{"result": 1.803, "unit": "degrees"}`
  - Framework handles both formats automatically

## Results Format

```json
{
  "model": "qwen2.5:7b",
  "timestamp": "2026-03-12T00:49:54",
  "metrics": {
    "outcome_accuracy": 80.0,
    "correct_outcome": 4,
    "tsr_function_selection": 100.0,
    "tsr_parameter_accuracy": 80.0,
    "f1_score": 88.89,
    "precision": 100.0,
    "recall": 80.0,
    "total_tests": 5,
    "correct_function": 5,
    "correct_params": 4,
    "no_tool_call": 0,
    "wrong_tool": 0
  },
  "raw_results": {
    "details": [...]
  }
}
```

## Adding a New Dataset

Example: Adding "physics_tools" evaluation

1. **Create test cases** (`physics_test_cases.json`):
```json
[
  {
    "id": "physics_001",
    "query": "Calculate kinetic energy for mass=5kg, velocity=10m/s",
    "expected_function": "calculate_kinetic_energy",
    "expected_params": {"mass": 5, "velocity": 10},
    "expected_result": {"energy": 250, "unit": "J"},
    "category": "physics"
  }
]
```

2. **Create evaluation script** (`evaluate_physics.py`):
```python
#!/usr/bin/env python3
import asyncio
from pathlib import Path
from evaluation_framework import (
    compare_values,
    calculate_metrics,
    print_report,
    save_results
)

# Same structure as evaluate_jefferson.py
# Just change test cases path and output directory
```

3. **Run evaluation**:
```bash
python3 evaluate_physics.py --model qwen2.5:7b
```

## Threshold Testing

Test performance degradation with increasing tool count:

```bash
# Jefferson with different tool counts
for n in 0 5 10 15 20 25 30; do
  python3 evaluate_jefferson.py \
    --model qwen2.5:7b \
    --num-distractors $n \
    --output results/jefferson/threshold_${n}tools.json
done

# Analyze results
python3 analyze_threshold.py results/jefferson/threshold_*.json
```

## Best Practices

### ✅ Do
- Use `evaluation_framework.py` for shared logic
- Create dataset-specific scripts for unique evaluation needs
- Save results with descriptive filenames including model and timestamp
- Use outcome-based evaluation E(O, Ô) for primary metric
- Include expected_result in test cases

### ❌ Don't
- Mix multiple datasets in single evaluation script
- Hardcode evaluation logic in dataset-specific scripts
- Use trajectory-based metrics as primary evaluation criterion
- Skip numerical tolerance in comparison (causes false failures)

## Migration Guide

### Old Approach (Monolithic)
```python
# evaluate_bfcl.py had everything mixed together
if args.synthetic == 'stats_test_cases.json':
    # Jefferson logic
else:
    # BFCL logic
```

### New Approach (Modular)
```python
# evaluate_jefferson.py - dedicated to Jefferson stats
from evaluation_framework import compare_values, calculate_metrics

# evaluate_bfcl.py - dedicated to BFCL math
from evaluation_framework import compare_values, calculate_metrics
```

## Troubleshooting

### Issue: 5% outcome accuracy despite correct function calls
**Solution**: Check if `expected_result` format matches actual result.
- Framework automatically handles `{"result": 1.803}` vs `1.803`
- Adjust tolerance if needed: `compare_values(a, e, tolerance=0.1)`

### Issue: "Model does not support tools"
**Solution**: Use `--allow-fallback` flag for text-based tool responses:
```bash
python3 evaluate_jefferson.py --model gpt-oss:20b --allow-fallback
```

### Issue: Slow evaluation
**Solution**: Use `--limit` for quick tests:
```bash
python3 evaluate_jefferson.py --model qwen2.5:7b --limit 5
```

## References

- **MCPVerse Paper**: Outcome-based evaluation E(O, Ô) ∈ {0, 1}
- **BFCL**: Berkeley Function Calling Leaderboard
- **JeffersonStatsMCP**: Statistical analysis tools
