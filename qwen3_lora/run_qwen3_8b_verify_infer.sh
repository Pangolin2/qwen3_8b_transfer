#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/configs/qwen3/predict_qwen3_8b_verify.yaml"
MODEL_DIR="${MODEL_DIR:-/mnt/hzl/10_qwen3_transfer/model/Qwen3-8B}"
PROMPT="${1:-请用中文用三句话介绍你自己，并说明你能帮助用户完成哪些任务。}"

cd "${ROOT_DIR}"

echo "[INFO] root_dir=${ROOT_DIR}"
echo "[INFO] config=${CONFIG_PATH}"
echo "[INFO] model_dir=${MODEL_DIR}"
echo "[INFO] prompt=${PROMPT}"

python run_mindformer.py \
  --config "${CONFIG_PATH}" \
  --run_mode predict \
  --use_parallel False \
  --pretrained_model_dir "${MODEL_DIR}" \
  --predict_data "${PROMPT}"

echo "[INFO] verify inference finished"
echo "[INFO] check text_generation_result.txt for the generated text"
