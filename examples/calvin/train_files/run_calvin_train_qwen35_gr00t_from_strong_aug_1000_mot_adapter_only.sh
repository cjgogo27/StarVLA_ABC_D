#!/bin/bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/miniconda/envs/starVLA2/bin/python}
ACCELERATE_BIN=${ACCELERATE_BIN:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/miniconda/envs/starVLA2/bin/accelerate}
MAIN_PROCESS_IP=${MAIN_PROCESS_IP:-127.0.0.1}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29501}

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "ACCELERATE_BIN=${ACCELERATE_BIN}"
echo "MAIN_PROCESS_IP=${MAIN_PROCESS_IP}"
echo "MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT}"

# Keep scheduler/container GPU mapping intact. If CUDA_VISIBLE_DEVICES is unset,
# leave it unset so PyTorch uses the runtime-visible devices.
NUM_PROCESSES=${NUM_PROCESSES:-$("${PYTHON_BIN}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)}

if [[ "${NUM_PROCESSES}" -lt 1 ]]; then
  echo "No CUDA devices visible; refusing to start training." >&2
  exit 1
fi

Framework_name=QwenGR00T
base_vlm=/inspire/qb-ilm2/project/26summer-camp-10/public/two/Model/Qwen3.5-4B
config_yaml=./examples/calvin/train_files/starvla_train_calvin.yaml
calvin_data_root=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
data_mix=calvin_task_ABC_D
run_root_dir=./results/Checkpoints

run_id=qwen35vl4b_gr00t_calvin_abc_d_strong_aug_from30000_new_lr_sched_steps1000_mot_adapter_only
pretrained_checkpoint=/inspire/qb-ilm2/project/26summer-camp-10/public/two/starVLA/results/Checkpoints/qwen35vl4b_gr00t_calvin_abc_d_strong_aug_from30000_new_lr_sched/checkpoints/steps_1000_pytorch_model.pt
IS_RESUME=${IS_RESUME:-false}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-150000}

# Stage 1: only train the newly added MOT adapter.
freeze_module_list=qwen_vl_interface,action_model

export action_input_dim=2048
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export MASTER_ADDR="${MAIN_PROCESS_IP}"

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "IS_RESUME=${IS_RESUME}"
echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
"${PYTHON_BIN}" - <<'PY'
import torch
n = torch.cuda.device_count()
print(f"torch.cuda.device_count()={n}")
for i in range(n):
    torch.cuda.set_device(i)
    print(f"cuda:{i} ok - {torch.cuda.get_device_name(i)}")
PY

"${ACCELERATE_BIN}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${calvin_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.include_state true \
  --datasets.vla_data.include_motion_history true \
  --datasets.vla_data.motion_history_window 4 \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.pretrained_checkpoint ${pretrained_checkpoint} \
  --trainer.is_resume ${IS_RESUME} \
  --trainer.learning_rate.base 1.0e-04 \
  --trainer.learning_rate.motion_adapter 1.0e-04 \
  --framework.motion_adapter.enabled true \
  --framework.motion_adapter.history_window 4 \
  --framework.motion_adapter.hidden_dim 512 \
  --framework.motion_adapter.num_layers 2 \
  --framework.motion_adapter.num_heads 8 \
  --framework.motion_adapter.dropout 0.1 \
  --framework.motion_adapter.state_dim 8 \
  --framework.motion_adapter.action_dim 7 \
  --framework.motion_adapter.pass_state_to_action_head false \
  --trainer.max_train_steps ${MAX_TRAIN_STEPS} \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 250 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Calvin \
  --wandb_entity your_wandb_entity
