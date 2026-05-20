#!/bin/bash
set -euo pipefail

# Usage:
#   MODEL_KIND=cosmo bash examples/calvin/train_files/run_calvin_night_two_runs.sh
#   MODEL_KIND=gr00t  bash examples/calvin/train_files/run_calvin_night_two_runs.sh

MODEL_KIND=${MODEL_KIND:-cosmo}
NUM_PROCESSES=${NUM_PROCESSES:-$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)}

freeze_module_list=''
config_yaml=./examples/calvin/train_files/starvla_train_calvin.yaml
calvin_data_root=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_abc_d
data_mix=calvin_task_ABC_D
run_root_dir=./results/Checkpoints
export action_input_dim=2048
export WANDB_MODE=disabled

case "${MODEL_KIND}" in
  cosmo)
    Framework_name=CosmoPredict2PI
    run_id_base=cosmopredict2_2b_pi_calvin_abc_d
    model_args=(
      --framework.name "${Framework_name}"
      --framework.world_model.base_wm /inspire/qb-ilm2/project/26summer-camp-10/public/two/Model/Cosmos-Predict2-2B-Video2World
      --framework.qwenvl.base_vlm /inspire/qb-ilm2/project/26summer-camp-10/public/two/Model/Cosmos-Predict2-2B-Video2World
    )
    ;;
  gr00t)
    Framework_name=QwenGR00T
    run_id_base=qwen35vl4b_gr00t_calvin_abc_d
    model_args=(
      --framework.name "${Framework_name}"
      --framework.qwenvl.base_vlm /inspire/qb-ilm2/project/26summer-camp-10/public/two/Model/Qwen3.5-4B
      --framework.action_model.action_model_type DiT-B
    )
    ;;
  *)
    echo "Unknown MODEL_KIND=${MODEL_KIND}. Use cosmo or gr00t." >&2
    exit 2
    ;;
esac

common_args=(
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml
  --num_processes "${NUM_PROCESSES}"
  starVLA/training/train_starvla.py
  --config_yaml "${config_yaml}"
  --datasets.vla_data.data_root_dir "${calvin_data_root}"
  --datasets.vla_data.data_mix "${data_mix}"
  --datasets.vla_data.per_device_batch_size 8
  --datasets.vla_data.video_backend torchvision_av
  --trainer.freeze_modules "${freeze_module_list}"
  --trainer.max_train_steps 30000
  --trainer.save_interval 5000
  --trainer.logging_frequency 100
  --trainer.eval_interval 250
  --run_root_dir "${run_root_dir}"
  --wandb_project starVLA_Calvin
  --wandb_entity your_wandb_entity
)

check_cuda() {
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
}

run_one() {
  local run_id=$1
  shift

  local output_dir="${run_root_dir}/${run_id}"
  mkdir -p "${output_dir}"
  cp "$0" "${output_dir}/"

  echo "Starting ${run_id}"
  accelerate launch \
    "${common_args[@]}" \
    "${model_args[@]}" \
    --run_id "${run_id}" \
    "$@"
}

check_cuda

# Run 1: current baseline hyperparameters from YAML/script.
run_one "${run_id_base}"

# Run 2: conservative optimizer variant. Save/eval intervals intentionally unchanged.
run_one "${run_id_base}_lr5e5_warmup3k" \
  --trainer.learning_rate.action_model 5.0e-05 \
  --trainer.num_warmup_steps 3000
