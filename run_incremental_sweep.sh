#!/usr/bin/env bash
# run_incremental_sweep.sh
#
# Runs the incremental tool-count threshold sweep for all 4 models.
# Resumable: the sweep function skips rounds whose results already exist on disk.
#
# Usage:
#   bash run_incremental_sweep.sh              # run all 4 models sequentially
#   bash run_incremental_sweep.sh qwen3:8b     # run a single model

set -euo pipefail

PYTHON="${PYTHON:-python3}"
MODELS=("qwen3:8b" "gpt-oss:20b" "mistral:12b" "llama3.1:8b")

# If arguments are provided, use them as the model list
if [[ $# -gt 0 ]]; then
    MODELS=("$@")
fi

echo "=== Incremental Tool-Count Threshold Sweep ==="
echo "Models: ${MODELS[*]}"
echo ""

for model in "${MODELS[@]}"; do
    echo "──────────────────────────────────────────"
    echo "  Starting sweep: $model"
    echo "──────────────────────────────────────────"
    $PYTHON -m harness.incremental_sweep --model "$model" --sweep
    echo ""
done

echo "=== All sweeps complete ==="
echo ""
echo "To compare results across models:"
echo "  $PYTHON -m harness.incremental_sweep --compare --models ${MODELS[*]}"
