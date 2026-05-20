#!/bin/bash
set -euo pipefail

# Qwen3.5 + GR00T-MoE continuation from the strong-aug GR00T checkpoint.
# Mirrors run_calvin_train_qwen35_gr00t_strong_aug.sh while switching only
# the framework/config/run_id to the isolated MoE implementation.

NUM_PROCESSES=${NUM_PROCESSES:-$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)}

# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00TMoE
freeze_module_list=''
base_vlm=/inspire/qb-ilm2/project/26summer-camp-10/public/two/Model/Qwen3.5-4B
config_yaml=./examples/calvin/train_files/starvla_train_calvin_gr00t_moe.yaml
calvin_data_root=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
data_mix=calvin_task_ABC_D
run_root_dir=./results/Checkpoints
source_run_id=qwen35vl4b_gr00t_calvin_abc_d_strong_aug
run_id=qwen35vl4b_gr00t_moe_calvin_abc_d_strong_aug_from30000_new_lr_sched
pretrained_checkpoint=${run_root_dir}/${source_run_id}/checkpoints/steps_30000_pytorch_model.pt
export action_input_dim=2048
# === End of environment variable configuration ===

export WANDB_MODE=disabled

if [[ ! -f "${pretrained_checkpoint}" ]]; then
  echo "Pretrained GR00T checkpoint not found: ${pretrained_checkpoint}" >&2
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
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.action_model.action_model_type DiT-B \
  --datasets.vla_data.data_root_dir "${calvin_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.video_backend torchvision_av \
  --datasets.vla_data.visual_augmentation.enabled true \
  --datasets.vla_data.visual_augmentation.crop_scale_min 0.88 \
  --datasets.vla_data.visual_augmentation.brightness 0.20 \
  --datasets.vla_data.visual_augmentation.contrast 0.20 \
  --datasets.vla_data.visual_augmentation.saturation 0.15 \
  --datasets.vla_data.visual_augmentation.blur_prob 0.25 \
  --datasets.vla_data.visual_augmentation.blur_radius 0.8 \
  --datasets.vla_data.visual_augmentation.noise_prob 0.25 \
  --datasets.vla_data.visual_augmentation.noise_std 3.0 \
  --datasets.vla_data.visual_augmentation.jpeg_prob 0.25 \
  --datasets.vla_data.visual_augmentation.jpeg_quality_min 65 \
  --datasets.vla_data.visual_augmentation.jpeg_quality_max 92 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.pretrained_checkpoint "${pretrained_checkpoint}" \
  --trainer.is_resume true \
  --trainer.max_train_steps 60000 \
  --trainer.save_interval 1000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 250 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_Calvin \
  --wandb_entity your_wandb_entity
