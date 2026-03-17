#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/configs/qwen3/finetune_qwen3_8b_lora_stage1.yaml"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/mnt/hzl/10_qwen3_transfer/data/alpaca-gpt4-data/stage1_train.json}"
MODEL_DIR="${MODEL_DIR:-/mnt/hzl/10_qwen3_transfer/model/Qwen3-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output_qwen3_8b_lora_stage1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${ROOT_DIR}"

echo "[INFO] config=${CONFIG_PATH}"
echo "[INFO] train_data=${TRAIN_DATA_PATH}"
echo "[INFO] model_dir=${MODEL_DIR}"
echo "[INFO] output_dir=${OUTPUT_DIR}"

"${PYTHON_BIN}" run_mindformer.py \
  --config "${CONFIG_PATH}" \
  --run_mode finetune \
  --output_dir "${OUTPUT_DIR}" \
  --options "train_dataset.data_loader.data_files=${TRAIN_DATA_PATH}" "pretrained_model_dir=${MODEL_DIR}"
