#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_FILE="${EVAL_FILE:-/mnt/hzl/10_qwen3_transfer/data/alpaca-gpt4-data/stage1_eval.json}"
CONFIG_PATH="${ROOT_DIR}/configs/qwen3/predict_qwen3_8b_stage1_eval.yaml"
MODEL_DIR="${MODEL_DIR:-/mnt/hzl/10_qwen3_transfer/model/Qwen3-8B}"
OUTPUT_FILE="${OUTPUT_FILE:-${ROOT_DIR}/stage1_eval_base_result.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" scripts/qwen3_stage1_eval.py \
  --eval-file "${EVAL_FILE}" \
  --config "${CONFIG_PATH}" \
  --pretrained-model-dir "${MODEL_DIR}" \
  --output-file "${OUTPUT_FILE}" \
  --python-bin "${PYTHON_BIN}"
