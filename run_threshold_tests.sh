#!/bin/bash

set -euo pipefail

MODEL="${1:-qwen2.5:7b}"
DATASET="${2:-bfcl}"

echo "Starting threshold sweep for model: ${MODEL}"
echo "Dataset: ${DATASET}"
echo ""

python -m harness.threshold_sweep \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --sweep
