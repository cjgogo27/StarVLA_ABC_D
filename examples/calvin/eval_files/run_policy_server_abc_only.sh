#!/bin/bash
set -euo pipefail

export PYTHONPATH=$(pwd):${PYTHONPATH:-}

STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-/inspire/qb-ilm2/project/26summer-camp-10/public/four/miniconda3/envs/starvla/bin/python}
CKPT_PATH=${CKPT_PATH:-results/Checkpoints/qwen35vl4b_gr00t_calvin_abc_only/checkpoints/steps_30000_pytorch_model.pt}
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-5694}
IDLE_TIMEOUT=${IDLE_TIMEOUT:--1}

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH}" >&2
  echo "Set CKPT_PATH=/path/to/steps_xxx_pytorch_model.pt" >&2
  exit 1
fi

echo "Starting StarVLA policy server"
echo "CKPT_PATH=${CKPT_PATH}"
echo "GPU_ID=${GPU_ID}"
echo "PORT=${PORT}"

CUDA_VISIBLE_DEVICES=${GPU_ID} "${STAR_VLA_PYTHON}" deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT_PATH}" \
  --port "${PORT}" \
  --idle_timeout "${IDLE_TIMEOUT}" \
  --use_bf16
