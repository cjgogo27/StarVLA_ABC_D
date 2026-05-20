#!/bin/bash
set -euo pipefail

REPO_ROOT=$(pwd)
CKPT_PATH=${CKPT_PATH:-results/Checkpoints/qwen35vl4b_gr00t_calvin_abc_d/checkpoints/steps_30000_pytorch_model.pt}
CALVIN_DATASET_PATH=${CALVIN_DATASET_PATH:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d}
CALVIN_ROOT=${CALVIN_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/calvin}
STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/miniconda/envs/starVLA2/bin/python}
CALVIN_PYTHON=${CALVIN_PYTHON:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/miniconda/envs/calvin_venv/bin/python}
CALVIN_EXTRA_PYTHONPATH=${CALVIN_EXTRA_PYTHONPATH:-${REPO_ROOT}/.runtime/calvin_cv2_headless_overlay}
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-5714}
UNNORM_KEY=${UNNORM_KEY:-franka}
NUM_SEQUENCES=${NUM_SEQUENCES:-10}
SEQUENCE_START=${SEQUENCE_START:-0}
RUN_TAG=${RUN_TAG:-single_gpu}
EVAL_ROOT=${EVAL_ROOT:-results/calvin_eval/${RUN_TAG}_$(date -u +%Y%m%d_%H%M%S)}

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
if [[ ! -d "${CALVIN_CONFIG_PATH}/callbacks/rollout/tasks" ]]; then
  echo "CALVIN_CONFIG_PATH must point to calvin_models/conf: ${CALVIN_CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -d "${CALVIN_EXTRA_PYTHONPATH}/cv2" ]]; then
  echo "Missing CALVIN headless dependency overlay: ${CALVIN_EXTRA_PYTHONPATH}" >&2
  exit 1
fi

if ! PYTHONPATH="${CALVIN_EXTRA_PYTHONPATH}:${PYTHONPATH:-}" "${CALVIN_PYTHON}" -c "import cv2, websockets.sync.client, msgpack" >/dev/null 2>&1; then
  echo "CALVIN_PYTHON cannot import cv2/websockets/msgpack even with CALVIN_EXTRA_PYTHONPATH." >&2
  echo "CALVIN_PYTHON=${CALVIN_PYTHON}" >&2
  echo "CALVIN_EXTRA_PYTHONPATH=${CALVIN_EXTRA_PYTHONPATH}" >&2
  exit 1
fi

mkdir -p "${EVAL_ROOT}"
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig_starvla_calvin_eval}
mkdir -p "${MPLCONFIGDIR}"

echo "CKPT_PATH=${CKPT_PATH}" | tee "${EVAL_ROOT}/run.info"
echo "CALVIN_DATASET_PATH=${CALVIN_DATASET_PATH}" | tee -a "${EVAL_ROOT}/run.info"
echo "CALVIN_ROOT=${CALVIN_ROOT}" | tee -a "${EVAL_ROOT}/run.info"
echo "CALVIN_EXTRA_PYTHONPATH=${CALVIN_EXTRA_PYTHONPATH}" | tee -a "${EVAL_ROOT}/run.info"
echo "GPU_ID=${GPU_ID}" | tee -a "${EVAL_ROOT}/run.info"
echo "PORT=${PORT}" | tee -a "${EVAL_ROOT}/run.info"
echo "NUM_SEQUENCES=${NUM_SEQUENCES}" | tee -a "${EVAL_ROOT}/run.info"

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=${GPU_ID} "${STAR_VLA_PYTHON}" deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT_PATH}" \
  --port "${PORT}" \
  --idle_timeout -1 \
  --use_bf16 \
  > "${EVAL_ROOT}/server.log" 2>&1 &
server_pid=$!

for _ in $(seq 1 180); do
  if grep -q "server listening" "${EVAL_ROOT}/server.log" 2>/dev/null; then
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "Policy server exited before becoming ready. Last log lines:" >&2
    tail -n 120 "${EVAL_ROOT}/server.log" >&2 || true
    exit 1
  fi
  sleep 2
done

if ! grep -q "server listening" "${EVAL_ROOT}/server.log" 2>/dev/null; then
  echo "Timed out waiting for policy server. Last log lines:" >&2
  tail -n 120 "${EVAL_ROOT}/server.log" >&2 || true
  exit 1
fi

(
  export PYTHONPATH="${CALVIN_EXTRA_PYTHONPATH}:${REPO_ROOT}:${CALVIN_ROOT}/calvin_models:${CALVIN_ROOT}/calvin_env:${PYTHONPATH:-}"
  export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
  export MUJOCO_GL=${MUJOCO_GL:-osmesa}
  "${CALVIN_PYTHON}" examples/calvin/eval_files/eval_calvin.py \
    --args.pretrained-path "${CKPT_PATH}" \
    --args.unnorm-key "${UNNORM_KEY}" \
    --args.host 127.0.0.1 \
    --args.port "${PORT}" \
    --args.dataset-path "${CALVIN_DATASET_PATH}" \
    --args.calvin-config-path "${CALVIN_CONFIG_PATH}" \
    --args.eval-sequences-path "${EVAL_SEQUENCES_PATH}" \
    --args.num-sequences "${NUM_SEQUENCES}" \
    --args.sequence-start "${SEQUENCE_START}" \
    --args.eval-log-dir "${EVAL_ROOT}/worker_0"
) > "${EVAL_ROOT}/eval.log" 2>&1

"${CALVIN_PYTHON}" examples/calvin/eval_files/aggregate_calvin_results.py "${EVAL_ROOT}" | tee "${EVAL_ROOT}/aggregate.log"

cleanup
trap - EXIT
echo "Evaluation complete: ${EVAL_ROOT}"
