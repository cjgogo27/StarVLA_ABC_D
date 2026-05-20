#!/bin/bash
set -euo pipefail

REPO_ROOT=$(pwd)
CALVIN_ROOT=${CALVIN_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/calvin}
CALVIN_PYTHON=${CALVIN_PYTHON:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/miniconda/envs/calvin_venv/bin/python}
CALVIN_EXTRA_PYTHONPATH=${CALVIN_EXTRA_PYTHONPATH:-${REPO_ROOT}/.runtime/calvin_cv2_headless_overlay}
CKPT_PATH=${CKPT_PATH:-results/Checkpoints/qwen35vl4b_gr00t_calvin_abc_d/checkpoints/steps_30000_pytorch_model.pt}
CALVIN_DATASET_PATH=${CALVIN_DATASET_PATH:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d}
CALVIN_CONFIG_PATH=${CALVIN_CONFIG_PATH:-${CALVIN_ROOT}/calvin_models/conf}
EVAL_SEQUENCES_PATH=${EVAL_SEQUENCES_PATH:-${REPO_ROOT}/examples/calvin/eval_files/eval_sequences.json}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-5694}
UNNORM_KEY=${UNNORM_KEY:-franka}
NUM_SEQUENCES=${NUM_SEQUENCES:-1000}
EVAL_LOG_DIR=${EVAL_LOG_DIR:-results/calvin_eval/$(date -u +%Y%m%d_%H%M%S)}

export PYTHONPATH="${CALVIN_EXTRA_PYTHONPATH}:${REPO_ROOT}:${CALVIN_ROOT}/calvin_models:${CALVIN_ROOT}/calvin_env:${PYTHONPATH:-}"
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig_starvla_calvin_eval}
mkdir -p "${MPLCONFIGDIR}"

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH}" >&2
  echo "Set CKPT_PATH=/path/to/steps_xxx_pytorch_model.pt" >&2
  exit 1
fi

if [[ ! -d "${CALVIN_DATASET_PATH}/validation" ]]; then
  echo "CALVIN_DATASET_PATH must contain validation/: ${CALVIN_DATASET_PATH}" >&2
  echo "Set CALVIN_DATASET_PATH=/path/to/original/calvin/task_D_D" >&2
  exit 1
fi

if [[ ! -d "${CALVIN_CONFIG_PATH}/callbacks/rollout/tasks" ]]; then
  echo "CALVIN_CONFIG_PATH must point to calvin_models/conf: ${CALVIN_CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${EVAL_SEQUENCES_PATH}" ]]; then
  echo "EVAL_SEQUENCES_PATH not found: ${EVAL_SEQUENCES_PATH}" >&2
  exit 1
fi

mkdir -p "${EVAL_LOG_DIR}"
echo "Starting CALVIN ABC->D evaluation"
echo "CKPT_PATH=${CKPT_PATH}"
echo "CALVIN_DATASET_PATH=${CALVIN_DATASET_PATH}"
echo "CALVIN_CONFIG_PATH=${CALVIN_CONFIG_PATH}"
echo "EVAL_SEQUENCES_PATH=${EVAL_SEQUENCES_PATH}"
echo "CALVIN_EXTRA_PYTHONPATH=${CALVIN_EXTRA_PYTHONPATH}"
echo "NUM_SEQUENCES=${NUM_SEQUENCES}"
echo "EVAL_LOG_DIR=${EVAL_LOG_DIR}"

"${CALVIN_PYTHON}" examples/calvin/eval_files/eval_calvin.py \
  --args.pretrained-path "${CKPT_PATH}" \
  --args.unnorm-key "${UNNORM_KEY}" \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.dataset-path "${CALVIN_DATASET_PATH}" \
  --args.calvin-config-path "${CALVIN_CONFIG_PATH}" \
  --args.eval-sequences-path "${EVAL_SEQUENCES_PATH}" \
  --args.num-sequences "${NUM_SEQUENCES}" \
  --args.eval-log-dir "${EVAL_LOG_DIR}"
