#!/bin/bash
set -euo pipefail

REPO_ROOT=$(pwd)
CKPT_PATH=${CKPT_PATH:-results/Checkpoints/qwen35vl4b_gr00t_calvin_abc_d/checkpoints/steps_30000_pytorch_model.pt}
CALVIN_DATASET_PATH=${CALVIN_DATASET_PATH:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d}
CALVIN_ROOT=${CALVIN_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/calvin}
STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/miniconda/envs/starVLA2/bin/python}
CALVIN_PYTHON=${CALVIN_PYTHON:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/miniconda/envs/calvin_venv/bin/python}
CALVIN_EXTRA_PYTHONPATH=${CALVIN_EXTRA_PYTHONPATH:-${REPO_ROOT}/.runtime/calvin_cv2_headless_overlay}
NUM_WORKERS=${NUM_WORKERS:-8}
NUM_SEQUENCES=${NUM_SEQUENCES:-1000}
BASE_PORT=${BASE_PORT:-5694}
START_GPU=${START_GPU:-0}
UNNORM_KEY=${UNNORM_KEY:-franka}
EVAL_ROOT=${EVAL_ROOT:-results/calvin_eval/qwen35vl4b_gr00t_calvin_abc_d_$(date -u +%Y%m%d_%H%M%S)}

CALVIN_CONFIG_PATH=${CALVIN_CONFIG_PATH:-${CALVIN_ROOT}/calvin_models/conf}
EVAL_SEQUENCES_PATH=${EVAL_SEQUENCES_PATH:-${REPO_ROOT}/examples/calvin/eval_files/eval_sequences.json}

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH}" >&2
  exit 1
fi
if [[ ! -d "${CALVIN_DATASET_PATH}/validation" ]]; then
  echo "CALVIN_DATASET_PATH must contain validation/: ${CALVIN_DATASET_PATH}" >&2
  exit 1
fi

mkdir -p "${EVAL_ROOT}"

if ! PYTHONPATH="${CALVIN_EXTRA_PYTHONPATH}:${PYTHONPATH:-}" "${CALVIN_PYTHON}" -c "import websockets.sync.client, msgpack" >/dev/null 2>&1; then
  echo "The CALVIN Python env is missing compatible websocket dependencies." >&2
  echo "Run this once, then rerun this script:" >&2
  echo "${CALVIN_PYTHON} -m pip install 'websockets>=12,<13' 'msgpack>=1,<2'" >&2
  exit 1
fi

if ! PYTHONPATH="${CALVIN_EXTRA_PYTHONPATH}:${PYTHONPATH:-}" "${CALVIN_PYTHON}" -c "import cv2" >/dev/null 2>&1; then
  echo "The CALVIN Python env cannot import cv2, commonly because libGL.so.1 is missing." >&2
  echo "Fix one of these ways, then rerun this script:" >&2
  echo "  1) Install a system/conda OpenGL runtime such as libgl1/libgl, or" >&2
  echo "  2) Use headless OpenCV in the CALVIN env:" >&2
  echo "     ${CALVIN_PYTHON} -m pip uninstall -y opencv-python opencv-contrib-python" >&2
  echo "     ${CALVIN_PYTHON} -m pip install opencv-python-headless" >&2
  exit 1
fi

if ! "${STAR_VLA_PYTHON}" -c "import transformers; assert hasattr(transformers, 'Qwen3_5ForConditionalGeneration'), transformers.__version__" >/dev/null 2>&1; then
  echo "STAR_VLA_PYTHON does not provide Qwen3_5ForConditionalGeneration." >&2
  echo "Use the training env, e.g.:" >&2
  echo "STAR_VLA_PYTHON=/inspire/qb-ilm2/project/26summer-camp-10/public/two/miniconda/envs/starVLA2/bin/python" >&2
  exit 1
fi

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig_starvla_calvin_eval}
mkdir -p "${MPLCONFIGDIR}"

echo "CKPT_PATH=${CKPT_PATH}" | tee "${EVAL_ROOT}/run.info"
echo "CALVIN_DATASET_PATH=${CALVIN_DATASET_PATH}" | tee -a "${EVAL_ROOT}/run.info"
echo "EVAL_ROOT=${EVAL_ROOT}" | tee -a "${EVAL_ROOT}/run.info"
echo "CALVIN_EXTRA_PYTHONPATH=${CALVIN_EXTRA_PYTHONPATH}" | tee -a "${EVAL_ROOT}/run.info"
echo "NUM_WORKERS=${NUM_WORKERS}" | tee -a "${EVAL_ROOT}/run.info"
echo "NUM_SEQUENCES=${NUM_SEQUENCES}" | tee -a "${EVAL_ROOT}/run.info"

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT

for worker in $(seq 0 $((NUM_WORKERS - 1))); do
  gpu=$((START_GPU + worker))
  port=$((BASE_PORT + worker))
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=${gpu} "${STAR_VLA_PYTHON}" deployment/model_server/server_policy.py \
    --ckpt_path "${CKPT_PATH}" \
    --port "${port}" \
    --idle_timeout -1 \
    --use_bf16 \
    > "${EVAL_ROOT}/server_${worker}.log" 2>&1 &
  pids+=("$!")
done

sleep 20

eval_pids=()
for worker in $(seq 0 $((NUM_WORKERS - 1))); do
  port=$((BASE_PORT + worker))
  start=$((worker * NUM_SEQUENCES / NUM_WORKERS))
  end=$(((worker + 1) * NUM_SEQUENCES / NUM_WORKERS))
  worker_dir="${EVAL_ROOT}/worker_${worker}"
  mkdir -p "${worker_dir}"
  (
    export PYTHONPATH="${CALVIN_EXTRA_PYTHONPATH}:${REPO_ROOT}:${CALVIN_ROOT}/calvin_models:${CALVIN_ROOT}/calvin_env:${PYTHONPATH:-}"
    export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
    export MUJOCO_GL=${MUJOCO_GL:-osmesa}
    "${CALVIN_PYTHON}" examples/calvin/eval_files/eval_calvin.py \
      --args.pretrained-path "${CKPT_PATH}" \
      --args.unnorm-key "${UNNORM_KEY}" \
      --args.host 127.0.0.1 \
      --args.port "${port}" \
      --args.dataset-path "${CALVIN_DATASET_PATH}" \
      --args.calvin-config-path "${CALVIN_CONFIG_PATH}" \
      --args.eval-sequences-path "${EVAL_SEQUENCES_PATH}" \
      --args.num-sequences "$((end - start))" \
      --args.sequence-start "${start}" \
      --args.sequence-end "${end}" \
      --args.eval-log-dir "${worker_dir}"
  ) > "${EVAL_ROOT}/eval_${worker}.log" 2>&1 &
  eval_pids+=("$!")
done

for pid in "${eval_pids[@]}"; do
  wait "${pid}"
done

"${CALVIN_PYTHON}" examples/calvin/eval_files/aggregate_calvin_results.py "${EVAL_ROOT}" | tee "${EVAL_ROOT}/aggregate.log"

cleanup
trap - EXIT
echo "Evaluation complete: ${EVAL_ROOT}"
