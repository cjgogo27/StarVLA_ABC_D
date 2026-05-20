#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

# Keep scheduler/container GPU mapping intact. If CUDA_VISIBLE_DEVICES is unset,
# leave it unset so PyTorch uses the runtime-visible devices.
NUM_PROCESSES=${NUM_PROCESSES:-$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)}

# === Please modify the following paths according to your environment ===
Framework_name=CosmoPredict2PI
freeze_module_list=''
cosmos_base_wm=/inspire/qb-ilm2/project/26summer-camp-10/public/two/Model/Cosmos-Predict2-2B-Video2World
calvin_data_root=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
data_mix=calvin_task_ABC_D
run_root_dir=./results/Checkpoints
run_id=cosmopredict2_2b_pi_calvin_abc_d
config_yaml=${run_root_dir}/${run_id}/config.full.yaml
expected_ckpt="${run_root_dir}/${run_id}/checkpoints/steps_30000_pytorch_model.pt"
max_train_steps=${MAX_TRAIN_STEPS:-100000}
export action_input_dim=2048
# === End of environment variable configuration ===

export WANDB_MODE=disabled

if [[ ! -f "${expected_ckpt}" ]]; then
  echo "Expected checkpoint not found: ${expected_ckpt}" >&2
  exit 1
fi

if [[ ! -f "${config_yaml}" ]]; then
  echo "Expected config not found: ${config_yaml}" >&2
  exit 1
fi

if [[ "${NUM_PROCESSES}" -lt 1 ]]; then
  echo "No CUDA devices visible; refusing to start training." >&2
  exit 1
fi

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
python - <<'PY'
import torch
n = torch.cuda.device_count()
print(f"torch.cuda.device_count()={n}")
for i in range(n):
    torch.cuda.set_device(i)
    print(f"cuda:{i} ok - {torch.cuda.get_device_name(i)}")
PY

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.world_model.base_wm "${cosmos_base_wm}" \
  --framework.qwenvl.base_vlm "${cosmos_base_wm}" \
  --datasets.vla_data.data_root_dir "${calvin_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.is_resume true \
  --trainer.max_train_steps "${max_train_steps}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_Calvin \
  --wandb_entity your_wandb_entity
