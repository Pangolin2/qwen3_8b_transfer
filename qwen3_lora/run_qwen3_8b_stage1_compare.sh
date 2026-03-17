#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_RESULT="${BASE_RESULT:-${ROOT_DIR}/stage1_eval_base_result.json}"
TUNED_RESULT="${TUNED_RESULT:-${ROOT_DIR}/stage1_eval_merged_result.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" scripts/qwen3_stage1_compare_eval.py \
  --base-result "${BASE_RESULT}" \
  --tuned-result "${TUNED_RESULT}"
