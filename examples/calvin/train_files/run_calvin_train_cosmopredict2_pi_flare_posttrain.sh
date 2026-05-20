#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

# Post-train CosmoPredict2PI from the 30k checkpoint with optional
# Future Latent Representation Alignment (FLARE-style) auxiliary loss.
NUM_PROCESSES=${NUM_PROCESSES:-$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)}

Framework_name=CosmoPredict2PI
freeze_module_list="${FREEZE_MODULES:-}"
cosmos_base_wm=/inspire/qb-ilm2/project/26summer-camp-10/public/two/Model/Cosmos-Predict2-2B-Video2World
calvin_data_root=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
data_mix=calvin_task_ABC_D
run_root_dir=./results/Checkpoints
source_run_id=cosmopredict2_2b_pi_calvin_abc_d
run_id=${RUN_ID:-cosmopredict2_2b_pi_calvin_abc_d_flare_posttrain}
config_yaml=${run_root_dir}/${source_run_id}/config.full.yaml
pretrained_checkpoint=${run_root_dir}/${source_run_id}/checkpoints/steps_30000_pytorch_model.pt

max_train_steps=${MAX_TRAIN_STEPS:-30000}
flare_weight=${FLARE_WEIGHT:-0.05}
flare_future_offset=${FLARE_FUTURE_OFFSET:-8}
flare_projection_dim=${FLARE_PROJECTION_DIM:-2048}
flare_loss_type=${FLARE_LOSS_TYPE:-cosine}

export action_input_dim=2048
export WANDB_MODE=disabled

if [[ ! -f "${pretrained_checkpoint}" ]]; then
  echo "Pretrained checkpoint not found: ${pretrained_checkpoint}" >&2
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
echo "pretrained_checkpoint=${pretrained_checkpoint}"
echo "run_id=${run_id}"
echo "FLARE_WEIGHT=${flare_weight}"
echo "FLARE_FUTURE_OFFSET=${flare_future_offset}"

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
  --datasets.vla_data.future_latent_alignment.enabled true \
  --datasets.vla_data.future_latent_alignment.weight "${flare_weight}" \
  --datasets.vla_data.future_latent_alignment.future_offset "${flare_future_offset}" \
  --datasets.vla_data.future_latent_alignment.projection_dim "${flare_projection_dim}" \
  --datasets.vla_data.future_latent_alignment.loss_type "${flare_loss_type}" \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.pretrained_checkpoint "${pretrained_checkpoint}" \
  --trainer.is_resume true \
  --trainer.max_train_steps "${max_train_steps}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_Calvin \
  --wandb_entity your_wandb_entity
