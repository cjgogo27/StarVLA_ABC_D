#!/bin/bash
set -euo pipefail

# Continue qwen35vl4b_gr00t_calvin_abc_d from its latest checkpoint.
# Expected latest checkpoint before launch:
#   ./results/Checkpoints/qwen35vl4b_gr00t_calvin_abc_d/checkpoints/steps_30000_pytorch_model.pt

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN=${PYTHON_BIN:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/miniconda/envs/starVLA/bin/python}
ACCELERATE_BIN=${ACCELERATE_BIN:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/miniconda/envs/starVLA/bin/accelerate}

Framework_name=QwenGR00T
freeze_module_list=''
base_vlm=/inspire/qb-ilm2/project/26summer-camp-10/public/two/Model/Qwen3.5-4B
config_yaml=./examples/calvin/train_files/starvla_train_calvin.yaml
calvin_data_root=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
data_mix=calvin_task_ABC_D
run_root_dir=./results/Checkpoints
run_id=qwen35vl4b_gr00t_calvin_abc_d
expected_ckpt="${run_root_dir}/${run_id}/checkpoints/steps_30000_pytorch_model.pt"
export action_input_dim=2048
export WANDB_MODE=disabled

if [[ ! -f "${expected_ckpt}" ]]; then
  echo "Expected checkpoint not found: ${expected_ckpt}" >&2
  exit 1
fi

NUM_PROCESSES=${NUM_PROCESSES:-$("${PYTHON_BIN}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)}

if [[ "${NUM_PROCESSES}" -lt 1 ]]; then
  echo "No CUDA devices visible to ${PYTHON_BIN}; refusing to start training." >&2
  exit 1
fi

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
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
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.action_model.action_model_type DiT-B \
  --datasets.vla_data.data_root_dir "${calvin_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.is_resume true \
  --trainer.max_train_steps 150000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 250 \
  --trainer.learning_rate.base 2.0e-05 \
  --trainer.learning_rate.qwen_vl_interface 4.0e-06 \
  --trainer.learning_rate.action_model 2.0e-04 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_Calvin \
  --wandb_entity your_wandb_entity
