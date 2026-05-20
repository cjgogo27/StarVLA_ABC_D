#!/bin/bash
set -euo pipefail

NUM_PROCESSES=${NUM_PROCESSES:-$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)}

Framework_name=QwenGR00T
freeze_module_list=''
base_vlm=/inspire/qb-ilm2/project/26summer-camp-10/public/two/Model/Qwen3.5-4B
config_yaml=./examples/calvin/train_files/starvla_train_calvin.yaml
shared_calvin_root=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
data_mix=calvin_task_ABC_only
run_root_dir=./results/Checkpoints
run_id=qwen35vl4b_gr00t_calvin_abc_only
accelerate_config=starVLA/config/deepseeds/deepspeed_zero2.yaml
export action_input_dim=2048
export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

log_file=${output_dir}/train.log
{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "run_id=${run_id}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "NUM_PROCESSES=${NUM_PROCESSES}"
  echo "data_mix=${data_mix}"
  echo "shared_calvin_root=${shared_calvin_root}"
  python examples/calvin/train_files/audit_calvin_abc_only.py \
    --shared-root "${shared_calvin_root}"
  python - <<'PY'
import torch
print(f"torch.__version__={torch.__version__}")
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"cuda:{i} {torch.cuda.get_device_name(i)}")
if torch.cuda.device_count() == 0:
    raise SystemExit("No CUDA devices are visible; refusing to launch 4B VLA training on CPU.")
PY

  accelerate launch \
    --config_file "${accelerate_config}" \
    --num_processes "${NUM_PROCESSES}" \
    starVLA/training/train_starvla.py \
    --config_yaml "${config_yaml}" \
    --framework.name "${Framework_name}" \
    --framework.qwenvl.base_vlm "${base_vlm}" \
    --framework.action_model.action_model_type DiT-B \
    --framework.action_model.hidden_size 768 \
    --framework.action_model.diffusion_model_cfg.num_layers 12 \
    --framework.action_model.diffusion_model_cfg.num_attention_heads 12 \
    --framework.action_model.diffusion_model_cfg.attention_head_dim 64 \
    --framework.action_model.diffusion_model_cfg.output_dim 768 \
    --framework.action_model.diffusion_model_cfg.dropout 0.15 \
    --framework.action_model.diffusion_model_cfg.final_dropout true \
    --framework.action_model.diffusion_model_cfg.interleave_self_attention true \
    --framework.action_model.diffusion_model_cfg.norm_type ada_norm \
    --framework.action_model.diffusion_model_cfg.positional_embeddings null \
    --framework.action_model.repeated_diffusion_steps 8 \
    --framework.action_model.num_inference_timesteps 4 \
    --datasets.vla_data.data_root_dir "${shared_calvin_root}" \
    --datasets.vla_data.data_mix "${data_mix}" \
    --datasets.vla_data.per_device_batch_size 8 \
    --datasets.vla_data.video_backend torchvision_av \
    --datasets.vla_data.visual_augmentation.enabled true \
    --datasets.vla_data.visual_augmentation.crop_scale_min 0.92 \
    --datasets.vla_data.visual_augmentation.brightness 0.15 \
    --datasets.vla_data.visual_augmentation.contrast 0.15 \
    --datasets.vla_data.visual_augmentation.saturation 0.10 \
    --datasets.vla_data.visual_augmentation.blur_prob 0.15 \
    --datasets.vla_data.visual_augmentation.blur_radius 0.6 \
    --datasets.vla_data.visual_augmentation.noise_prob 0.15 \
    --datasets.vla_data.visual_augmentation.noise_std 2.0 \
    --datasets.vla_data.visual_augmentation.jpeg_prob 0.15 \
    --datasets.vla_data.visual_augmentation.jpeg_quality_min 75 \
    --datasets.vla_data.visual_augmentation.jpeg_quality_max 95 \
    --trainer.freeze_modules "${freeze_module_list}" \
    --trainer.max_train_steps 30000 \
    --trainer.save_interval 5000 \
    --trainer.logging_frequency 100 \
    --trainer.eval_interval 250 \
    --trainer.learning_rate.base 1.0e-05 \
    --trainer.learning_rate.qwen_vl_interface 2.0e-06 \
    --trainer.learning_rate.action_model 1.0e-04 \
    --run_root_dir "${run_root_dir}" \
    --run_id "${run_id}" \
    --wandb_project starVLA_Calvin \
    --wandb_entity your_wandb_entity
} 2>&1 | tee "${log_file}"
