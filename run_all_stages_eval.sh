#!/bin/bash

set -uo pipefail

MODEL="${1:-llama3.1:8b}"

# Optional passthrough flags
BACKEND="${BACKEND:-ollama}"
BASE_URL="${BASE_URL:-}"
API_KEY="${API_KEY:-}"
ALLOW_FALLBACK="${ALLOW_FALLBACK:-0}"

MODEL_SAFE="$(echo "${MODEL}" | sed 's#[/:]#_#g')"
RUN_TS="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/${MODEL_SAFE}_${RUN_TS}.log"

mkdir -p "${LOG_DIR}"

if [ -f "venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "venv/bin/activate"
fi

run_one() {
  local stage="$1"
  local dataset="$2"
  local prompt_template="${3:-}"

  echo "" | tee -a "${LOG_FILE}"
  echo "======================================================================" | tee -a "${LOG_FILE}"
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] ${stage} | dataset=${dataset}" | tee -a "${LOG_FILE}"
  if [ -n "${prompt_template}" ]; then
    echo "prompt_template=${prompt_template}" | tee -a "${LOG_FILE}"
  fi
  echo "======================================================================" | tee -a "${LOG_FILE}"

  local cmd=(python -u -m harness.runner --dataset "${dataset}" --model "${MODEL}" --backend "${BACKEND}" --level L1 L2 L3)

  if [ -n "${BASE_URL}" ]; then
    cmd+=(--base-url "${BASE_URL}")
  fi
  if [ -n "${API_KEY}" ]; then
    cmd+=(--api-key "${API_KEY}")
  fi
  if [ "${ALLOW_FALLBACK}" = "1" ]; then
    cmd+=(--allow-fallback)
  fi
  if [ -n "${prompt_template}" ]; then
    cmd+=(--prompt-template "${prompt_template}")
  fi

  if "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"; then
    echo "[OK] ${stage} ${dataset}" | tee -a "${LOG_FILE}"
    return 0
  else
    local rc=$?
    echo "[FAIL] ${stage} ${dataset} (exit=${rc})" | tee -a "${LOG_FILE}"
    return ${rc}
  fi
}

failures=()

# Stage 0: OG datasets
stage0_datasets=(bfcl finance jefferson postgres)

# Stage 1: v2 datasets
stage1_datasets=(bfcl-v2 finance-v2 jefferson-v2 postgres-v2)

# Stage 2: prompt template on OG datasets ONLY (as requested)
stage2_datasets=(bfcl finance jefferson postgres)

get_template_for_dataset() {
  case "$1" in
    bfcl) echo "prompts/prompt_template_math.txt" ;;
    finance) echo "prompts/prompt_template_finance.txt" ;;
    jefferson) echo "prompts/prompt_template_jefferson_stats.txt" ;;
    postgres) echo "prompts/prompt_template_postgres.txt" ;;
    *) echo "" ;;
  esac
}

echo "Model: ${MODEL}" | tee -a "${LOG_FILE}"
echo "Backend: ${BACKEND}" | tee -a "${LOG_FILE}"
echo "Started: $(date +"%Y-%m-%d %H:%M:%S")" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"

for ds in "${stage0_datasets[@]}"; do
  run_one "stage0" "${ds}" || failures+=("stage0:${ds}")
done

for ds in "${stage1_datasets[@]}"; do
  run_one "stage1" "${ds}" || failures+=("stage1:${ds}")
done

for ds in "${stage2_datasets[@]}"; do
  tpl="$(get_template_for_dataset "${ds}")"
  run_one "stage2" "${ds}" "${tpl}" || failures+=("stage2:${ds}")
done

echo "" | tee -a "${LOG_FILE}"
echo "Finished: $(date +"%Y-%m-%d %H:%M:%S")" | tee -a "${LOG_FILE}"

if [ "${#failures[@]}" -eq 0 ]; then
  echo "All stage runs completed successfully." | tee -a "${LOG_FILE}"
  echo "Log saved at: ${LOG_FILE}"
  exit 0
fi

echo "Some runs failed:" | tee -a "${LOG_FILE}"
for f in "${failures[@]}"; do
  echo "  - ${f}" | tee -a "${LOG_FILE}"
done

echo "Log saved at: ${LOG_FILE}"
exit 1
