#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false

CONFIG_YAML="${CONFIG_YAML:-./examples/LIBERO/train_files/starvla_pretrain_libero90_pi.yaml}"
BASE_VLM="${BASE_VLM:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/Model/Qwen3.5-4B}"
DATA_ROOT="${DATA_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/data/LIBERO-datasets/lerobot}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-./results/Checkpoints}"
RUN_ID="${RUN_ID:-qwen35vl4b_pi_libero90_stage1_head_only}"
NUM_PROCESSES="${NUM_PROCESSES:-$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)}"

if [[ "${NUM_PROCESSES}" == "0" ]]; then
  echo "No CUDA devices visible to PyTorch; aborting training." >&2
  exit 1
fi

mkdir -p "${RUN_ROOT_DIR}/${RUN_ID}"
cp "$0" "${RUN_ROOT_DIR}/${RUN_ID}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name QwenPI \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.data_mix libero_90 \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-8}" \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules qwen_vl_interface \
  --trainer.is_resume true \
  --trainer.learning_rate.base "${BASE_LR:-1.0e-05}" \
  --trainer.learning_rate.qwen_vl_interface "${QWEN_LR:-0.0}" \
  --trainer.learning_rate.action_model "${ACTION_LR:-1.0e-04}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS:-30000}" \
  --trainer.num_warmup_steps "${NUM_WARMUP_STEPS:-2000}" \
  --trainer.save_interval "${SAVE_INTERVAL:-5000}" \
  --trainer.eval_interval "${EVAL_INTERVAL:-500}" \
  --trainer.logging_frequency "${LOGGING_FREQUENCY:-50}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT:-starVLA_Libero90_PI}" \
  --wandb_entity "${WANDB_ENTITY:-your_wandb_entity}"
