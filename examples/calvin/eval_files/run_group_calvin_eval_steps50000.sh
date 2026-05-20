#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/two}
REPO_ROOT=${REPO_ROOT:-${ROOT}/starVLA}
CKPT=${CKPT:-${REPO_ROOT}/results/Checkpoints/qwen35vl4b_gr00t_calvin_abc_d_strong_aug_from30000_new_lr_sched_steps1000_mot_adapter_only/checkpoints/steps_50000_pytorch_model.pt}

source "${ROOT}/miniconda/bin/activate"
conda activate starVLA2

export CKPT_PATH="${CKPT}"
export RUN_TAG=${RUN_TAG:-group_calvin_qwen35vl4b_gr00t_mot_adapter_only_steps50000}
export EVAL_ROOT=${EVAL_ROOT:-${REPO_ROOT}/results/calvin_eval/${RUN_TAG}_multigpu_$(date -u +%Y%m%d_%H%M%S)}
export CALVIN_DATASET_PATH=${CALVIN_DATASET_PATH:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d}
export CALVIN_ROOT=${CALVIN_ROOT:-${ROOT}/calvin}
export STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-${ROOT}/miniconda/envs/starVLA2/bin/python}
export CALVIN_PYTHON=${CALVIN_PYTHON:-${ROOT}/miniconda/envs/calvin_venv/bin/python}
export CALVIN_EXTRA_PYTHONPATH=${CALVIN_EXTRA_PYTHONPATH:-${REPO_ROOT}/.runtime/calvin_cv2_headless_overlay}

export NUM_GPUS=${NUM_GPUS:-8}
export START_GPU=${START_GPU:-0}
export PROCS_PER_GPU=${PROCS_PER_GPU:-1}
export NUM_WORKERS=${NUM_WORKERS:-$((NUM_GPUS * PROCS_PER_GPU))}
export NUM_SEQUENCES=${NUM_SEQUENCES:-1000}
export BASE_PORT=${BASE_PORT:-7614}
export CALVIN_EP_LEN=${CALVIN_EP_LEN:-240}
export CALVIN_NUM_DDIM_STEPS=${CALVIN_NUM_DDIM_STEPS:-5}
export CALVIN_USE_EGL=${CALVIN_USE_EGL:-0}
export SAVE_VIDEO=${SAVE_VIDEO:-0}
export STAGGER_SERVER_SECONDS=${STAGGER_SERVER_SECONDS:-12}
export SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-900}

cd "${REPO_ROOT}"

echo "image: starVLA2"
echo "repo: ${REPO_ROOT}"
echo "ckpt: ${CKPT_PATH}"
echo "eval_root: ${EVAL_ROOT}"

bash examples/calvin/eval_files/run_calvin_eval_multigpu_local.sh
