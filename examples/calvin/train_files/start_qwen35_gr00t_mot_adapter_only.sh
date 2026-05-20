#!/bin/bash
set -euo pipefail

REPO_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/public/two/starVLA
RUN_ID=qwen35vl4b_gr00t_calvin_abc_d_strong_aug_from30000_new_lr_sched_steps1000_mot_adapter_only
LOG_DIR="${REPO_ROOT}/results/Checkpoints/${RUN_ID}/logs"
PID_FILE="${LOG_DIR}/train.pid"
TRAIN_SCRIPT="${REPO_ROOT}/examples/calvin/train_files/run_calvin_train_qwen35_gr00t_from_strong_aug_1000_mot_adapter_only.sh"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

if [[ -s "${PID_FILE}" ]]; then
  old_pid=$(cat "${PID_FILE}")
  if kill -0 "${old_pid}" 2>/dev/null; then
    echo "Training already appears to be running with PID ${old_pid}"
    echo "Log: ${LOG_DIR}/train.log"
    exit 0
  fi
fi

{
  echo "[launcher] start requested at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[launcher] cwd=${REPO_ROOT}"
  echo "[launcher] script=${TRAIN_SCRIPT}"
  echo "[launcher] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "[launcher] NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
  echo "[launcher] IS_RESUME=${IS_RESUME:-false}"
  echo "[launcher] MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-150000}"
} >> "${LOG_DIR}/train.log"

nohup /bin/bash "${TRAIN_SCRIPT}" >> "${LOG_DIR}/train.log" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${PID_FILE}"

echo "Started training with PID ${pid}"
echo "Log: ${LOG_DIR}/train.log"
echo "Tail with: tail -f ${LOG_DIR}/train.log"
