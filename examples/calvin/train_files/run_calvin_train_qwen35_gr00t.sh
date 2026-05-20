#!/bin/bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/miniconda/envs/starVLA/bin/python}
ACCELERATE_BIN=${ACCELERATE_BIN:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/miniconda/envs/starVLA/bin/accelerate}

# Keep scheduler/container GPU mapping intact. If CUDA_VISIBLE_DEVICES is unset,
# leave it unset so PyTorch uses the runtime-visible devices.
NUM_PROCESSES=${NUM_PROCESSES:-$("${PYTHON_BIN}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)}

# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00T
freeze_module_list=''
base_vlm=/inspire/qb-ilm2/project/26summer-camp-10/public/two/Model/Qwen3.5-4B
config_yaml=./examples/calvin/train_files/starvla_train_calvin.yaml
calvin_data_root=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
data_mix=calvin_task_ABC_D
run_root_dir=./results/Checkpoints
source_run_id=qwen35vl4b_gr00t_calvin_abc_d
run_id=qwen35vl4b_gr00t_calvin_abc_d_from30000_mot_adapter
pretrained_checkpoint=${run_root_dir}/${source_run_id}/checkpoints/steps_30000_pytorch_model.pt
export action_input_dim=2048
# === End of environment variable configuration ===

export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

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
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${calvin_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.include_state true \
  --datasets.vla_data.include_motion_history true \
  --datasets.vla_data.motion_history_window 4 \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.pretrained_checkpoint ${pretrained_checkpoint} \
  --trainer.is_resume false \
  --framework.motion_adapter.enabled true \
  --framework.motion_adapter.history_window 4 \
  --framework.motion_adapter.hidden_dim 512 \
  --framework.motion_adapter.num_layers 2 \
  --framework.motion_adapter.num_heads 8 \
  --framework.motion_adapter.dropout 0.1 \
  --framework.motion_adapter.state_dim 8 \
  --framework.motion_adapter.action_dim 7 \
  --framework.motion_adapter.pass_state_to_action_head false \
  --trainer.learning_rate.motion_adapter 1.0e-04 \
  --trainer.max_train_steps 150000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 250 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Calvin \
  --wandb_entity your_wandb_entity \
  # --is_debug True



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
