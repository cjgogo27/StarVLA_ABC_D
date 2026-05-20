#!/bin/bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/miniconda/envs/starVLA/bin/python}
ACCELERATE_BIN=${ACCELERATE_BIN:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/miniconda/envs/starVLA/bin/accelerate}

NUM_PROCESSES=${NUM_PROCESSES:-$("${PYTHON_BIN}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)}

Framework_name=QwenGR00TMoE
base_vlm=/inspire/qb-ilm2/project/26summer-camp-10/public/two/Model/Qwen3.5-4B
config_yaml=./examples/calvin/train_files/starvla_train_calvin_gr00t_moe.yaml
calvin_data_root=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
data_mix=calvin_task_ABC_D
run_root_dir=./results/Checkpoints
run_id=qwen35vl4b_gr00t_moe_calvin_abc_d_from_gr00t
pretrained_checkpoint=./results/Checkpoints/qwen35vl4b_gr00t_calvin_abc_d/checkpoints/steps_30000_pytorch_model.pt

export WANDB_MODE=${WANDB_MODE:-disabled}

if [[ ! -f "${pretrained_checkpoint}" ]]; then
  echo "Pretrained GR00T checkpoint not found: ${pretrained_checkpoint}" >&2
  exit 1
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"

"${ACCELERATE_BIN}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --datasets.vla_data.data_root_dir "${calvin_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --trainer.pretrained_checkpoint "${pretrained_checkpoint}" \
  --trainer.is_resume false \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}"
