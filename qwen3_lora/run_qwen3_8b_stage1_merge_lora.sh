#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_CKPT="${1:-${ROOT_DIR}/output_qwen3_8b_lora_stage1/checkpoint/qwen3_8b_lora_stage1-1_1.ckpt}"
DST_DIR="${2:-${ROOT_DIR}/output_qwen3_8b_lora_stage1/merged}"
LORA_SCALING="${LORA_SCALING:-2.0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${DST_DIR}"
cd "${ROOT_DIR}"

echo "[INFO] src_ckpt=${SRC_CKPT}"
echo "[INFO] dst_dir=${DST_DIR}"
echo "[INFO] lora_scaling=${LORA_SCALING}"

"${PYTHON_BIN}" mindformers/tools/transform_ckpt_lora.py \
  --src_ckpt_path_or_dir "${SRC_CKPT}" \
  --dst_ckpt_dir "${DST_DIR}" \
  --lora_scaling "${LORA_SCALING}" \
  --save_format ckpt

echo "[INFO] merged_ckpt=${DST_DIR}/merged_lora.ckpt"
