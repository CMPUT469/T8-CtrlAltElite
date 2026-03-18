#!/bin/bash

# Threshold Testing Script for qwen2.5:7b
# Tests model performance with different numbers of available tools

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "mcp-client/.venv/bin/activate" ]; then
  source "mcp-client/.venv/bin/activate"
fi

echo "Starting threshold tests for qwen2.5:7b..."
echo "Testing with different tool counts to find degradation point"
echo ""

# Test with all 46 tools (standard mode - no filtering)
echo "[1/7] Testing with ALL 46 tools..."
python3 evaluate_bfcl.py --model qwen2.5:7b --category all --output results/qwen2.5_46tools.json

# Test with 30 tools (29 distractors + 1 correct)
echo "[2/7] Testing with 30 tools..."
python3 evaluate_bfcl.py --model qwen2.5:7b --category all --num-distractors 29 --output results/qwen2.5_30tools.json

# Test with 20 tools (19 distractors + 1 correct)
echo "[3/7] Testing with 20 tools..."
python3 evaluate_bfcl.py --model qwen2.5:7b --category all --num-distractors 19 --output results/qwen2.5_20tools.json

# Test with 15 tools (14 distractors + 1 correct)
echo "[4/7] Testing with 15 tools..."
python3 evaluate_bfcl.py --model qwen2.5:7b --category all --num-distractors 14 --output results/qwen2.5_15tools.json

# Test with 10 tools (9 distractors + 1 correct)
echo "[5/7] Testing with 10 tools..."
python3 evaluate_bfcl.py --model qwen2.5:7b --category all --num-distractors 9 --output results/qwen2.5_10tools.json

# Test with 5 tools (4 distractors + 1 correct)
echo "[6/7] Testing with 5 tools..."
python3 evaluate_bfcl.py --model qwen2.5:7b --category all --num-distractors 4 --output results/qwen2.5_5tools.json

# Test with 1 tool (0 distractors - oracle mode)
echo "[7/7] Testing with 1 tool (oracle mode - baseline)..."
python3 evaluate_bfcl.py --model qwen2.5:7b --category all --num-distractors 0 --output results/qwen2.5_1tool.json

echo ""
echo "✓ All threshold tests complete!"
echo "Results saved in results/ directory"
echo ""
echo "Summary:"
grep -h "f1_score" results/qwen2.5_*tools.json | sort
