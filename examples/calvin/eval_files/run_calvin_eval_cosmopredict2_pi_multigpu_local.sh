#!/bin/bash
set -euo pipefail

# Multi-GPU CALVIN-D evaluation for CosmoPredict2PI checkpoints.
#
# Default behavior evaluates the two Cosmos+PI checkpoints below sequentially.
# You can override with either:
#   CKPT_PATH=/path/to/one.pt bash ...
# or:
#   CKPT_PATHS="/path/a.pt /path/b.pt" bash ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_CKPT_A="${REPO_ROOT}/results/Checkpoints/cosmopredict2_2b_pi_calvin_abc_d/checkpoints/steps_30000_pytorch_model.pt"
DEFAULT_CKPT_B="${REPO_ROOT}/results/Checkpoints/cosmopredict2_2b_pi_calvin_abc_d_lr5e5_warmup3k/checkpoints/steps_25000_pytorch_model.pt"

if [[ -n "${CKPT_PATH:-}" ]]; then
  CKPTS=("${CKPT_PATH}")
elif [[ -n "${CKPT_PATHS:-}" ]]; then
  # shellcheck disable=SC2206
  CKPTS=(${CKPT_PATHS})
else
  CKPTS=("${DEFAULT_CKPT_A}" "${DEFAULT_CKPT_B}")
fi

CALVIN_DATASET_PATH="${CALVIN_DATASET_PATH:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d}"
CALVIN_ROOT="${CALVIN_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/calvin}"
STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/miniconda/envs/starVLA2/bin/python}"
CALVIN_PYTHON="${CALVIN_PYTHON:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/miniconda/envs/calvin_venv/bin/python}"
CALVIN_EXTRA_PYTHONPATH="${CALVIN_EXTRA_PYTHONPATH:-${REPO_ROOT}/.runtime/calvin_cv2_headless_overlay}"

NUM_GPUS="${NUM_GPUS:-8}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
NUM_WORKERS="${NUM_WORKERS:-$((NUM_GPUS * PROCS_PER_GPU))}"
NUM_SEQUENCES="${NUM_SEQUENCES:-1000}"
BASE_PORT="${BASE_PORT:-19209}"
START_GPU="${START_GPU:-0}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
CALVIN_SEND_STATE="${CALVIN_SEND_STATE:-0}"
CALVIN_USE_EGL="${CALVIN_USE_EGL:-0}"
CALVIN_EP_LEN="${CALVIN_EP_LEN:-240}"
CALVIN_NUM_DDIM_STEPS="${CALVIN_NUM_DDIM_STEPS:-5}"
STAGGER_SERVER_SECONDS="${STAGGER_SERVER_SECONDS:-10}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
RUN_GROUP="${RUN_GROUP:-cosmopredict2_pi}"

CALVIN_CONFIG_PATH="${CALVIN_CONFIG_PATH:-${CALVIN_ROOT}/calvin_models/conf}"
EVAL_SEQUENCES_PATH="${EVAL_SEQUENCES_PATH:-${REPO_ROOT}/examples/calvin/eval_files/eval_sequences.json}"
EVAL_PY="${EVAL_PY:-examples/calvin/eval_files/eval_calvin_8gpu_local.py}"

if (( NUM_WORKERS <= 0 )); then
  echo "NUM_WORKERS must be > 0; got NUM_WORKERS=${NUM_WORKERS}" >&2
  exit 1
fi
if [[ ! -d "${CALVIN_DATASET_PATH}/validation" ]]; then
  echo "CALVIN_DATASET_PATH must contain validation/: ${CALVIN_DATASET_PATH}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_PY}" ]]; then
  echo "Eval python file not found: ${EVAL_PY}" >&2
  exit 1
fi

for ckpt in "${CKPTS[@]}"; do
  if [[ ! -f "${ckpt}" ]]; then
    echo "Checkpoint not found: ${ckpt}" >&2
    exit 1
  fi
done

if ! PYTHONPATH="${CALVIN_EXTRA_PYTHONPATH}:${PYTHONPATH:-}" "${CALVIN_PYTHON}" -c "import websockets.sync.client, msgpack, cv2" >/dev/null 2>&1; then
  echo "CALVIN_PYTHON is missing required runtime imports: websockets.sync.client, msgpack, or cv2." >&2
  echo "CALVIN_PYTHON=${CALVIN_PYTHON}" >&2
  exit 1
fi

if ! "${STAR_VLA_PYTHON}" -c "import torch; import transformers; print('torch_cuda_devices', torch.cuda.device_count())" >/dev/null 2>&1; then
  echo "STAR_VLA_PYTHON failed basic torch/transformers import check." >&2
  echo "STAR_VLA_PYTHON=${STAR_VLA_PYTHON}" >&2
  exit 1
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_starvla_calvin_eval}"
mkdir -p "${MPLCONFIGDIR}"

port_check() {
  local base_port="$1"
  local num_workers="$2"
  "${STAR_VLA_PYTHON}" - "${base_port}" "${num_workers}" <<'PY'
import socket
import sys

base_port = int(sys.argv[1])
num_workers = int(sys.argv[2])
busy = []
for port in range(base_port, base_port + num_workers):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError as exc:
        busy.append((port, str(exc)))
    finally:
        sock.close()

if busy:
    print("Some policy-server ports are already in use:", file=sys.stderr)
    for port, err in busy:
        print(f"  port {port}: {err}", file=sys.stderr)
    sys.exit(1)
PY
}

wait_for_policy_server() {
  local port="$1"
  PYTHONPATH="${REPO_ROOT}:${CALVIN_EXTRA_PYTHONPATH}:${PYTHONPATH:-}" "${CALVIN_PYTHON}" - "${port}" "${SERVER_READY_TIMEOUT}" <<'PY'
import sys
import time

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

port = int(sys.argv[1])
timeout = float(sys.argv[2])
deadline = time.time() + timeout
last_error = None

while time.time() < deadline:
    try:
        client = WebsocketClientPolicy("127.0.0.1", port)
        metadata = client.get_server_metadata()
        client.close()
        print(f"server ready on port {port}: {metadata}")
        raise SystemExit(0)
    except Exception as exc:
        last_error = exc
        time.sleep(2)

raise TimeoutError(f"server on port {port} was not ready within {timeout}s; last error: {last_error}")
PY
}

run_one_ckpt() {
  local ckpt="$1"
  local ckpt_base
  local ckpt_step
  local run_tag
  local eval_root

  ckpt_base="$(basename "$(dirname "$(dirname "${ckpt}")")")"
  ckpt_step="$(basename "${ckpt}" .pt)"
  run_tag="${ckpt_base}_${ckpt_step}"
  eval_root="${EVAL_ROOT:-results/calvin_eval/${RUN_GROUP}/${run_tag}_multigpu_$(date -u +%Y%m%d_%H%M%S)}"

  port_check "${BASE_PORT}" "${NUM_WORKERS}"
  mkdir -p "${eval_root}"

  {
    echo "CKPT_PATH=${ckpt}"
    echo "CALVIN_DATASET_PATH=${CALVIN_DATASET_PATH}"
    echo "CALVIN_CONFIG_PATH=${CALVIN_CONFIG_PATH}"
    echo "EVAL_SEQUENCES_PATH=${EVAL_SEQUENCES_PATH}"
    echo "EVAL_PY=${EVAL_PY}"
    echo "EVAL_ROOT=${eval_root}"
    echo "NUM_GPUS=${NUM_GPUS}"
    echo "PROCS_PER_GPU=${PROCS_PER_GPU}"
    echo "NUM_WORKERS=${NUM_WORKERS}"
    echo "NUM_SEQUENCES=${NUM_SEQUENCES}"
    echo "BASE_PORT=${BASE_PORT}"
    echo "START_GPU=${START_GPU}"
    echo "UNNORM_KEY=${UNNORM_KEY}"
    echo "CALVIN_SEND_STATE=${CALVIN_SEND_STATE}"
    echo "CALVIN_USE_EGL=${CALVIN_USE_EGL}"
    echo "CALVIN_EP_LEN=${CALVIN_EP_LEN}"
    echo "CALVIN_NUM_DDIM_STEPS=${CALVIN_NUM_DDIM_STEPS}"
    echo "STAGGER_SERVER_SECONDS=${STAGGER_SERVER_SECONDS}"
    echo "SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT}"
  } | tee "${eval_root}/run.info"

  local pids=()
  local eval_pids=()
  cleanup() {
    for pid in "${pids[@]:-}"; do
      kill "${pid}" 2>/dev/null || true
    done
    wait "${pids[@]:-}" 2>/dev/null || true
  }
  trap cleanup RETURN

  for worker in $(seq 0 $((NUM_WORKERS - 1))); do
    local gpu=$((START_GPU + worker / PROCS_PER_GPU))
    local port=$((BASE_PORT + worker))
    local start=$((worker * NUM_SEQUENCES / NUM_WORKERS))
    local end=$(((worker + 1) * NUM_SEQUENCES / NUM_WORKERS))
    local worker_dir="${eval_root}/worker_${worker}"
    mkdir -p "${worker_dir}"

    PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${gpu}" \
      "${STAR_VLA_PYTHON}" deployment/model_server/server_policy.py \
        --ckpt_path "${ckpt}" \
        --port "${port}" \
        --idle_timeout -1 \
        --use_bf16 \
      > "${eval_root}/server_${worker}_gpu_${gpu}.log" 2>&1 &
    pids+=("$!")
    echo "Started policy server ${worker}: GPU ${gpu}, port ${port}, log ${eval_root}/server_${worker}_gpu_${gpu}.log"

    wait_for_policy_server "${port}"

    (
      export PYTHONPATH="${CALVIN_EXTRA_PYTHONPATH}:${REPO_ROOT}:${CALVIN_ROOT}/calvin_models:${CALVIN_ROOT}/calvin_env:${PYTHONPATH:-}"
      export PYTHONUNBUFFERED=1
      export CALVIN_SEND_STATE
      export CALVIN_USE_EGL
      export CALVIN_EP_LEN
      if [[ "${CALVIN_USE_EGL}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
        export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
        export MUJOCO_GL="${MUJOCO_GL:-egl}"
      else
        export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
        export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
      fi
      "${CALVIN_PYTHON}" "${EVAL_PY}" \
        --args.pretrained-path "${ckpt}" \
        --args.unnorm-key "${UNNORM_KEY}" \
        --args.host 127.0.0.1 \
        --args.port "${port}" \
        --args.dataset-path "${CALVIN_DATASET_PATH}" \
        --args.calvin-config-path "${CALVIN_CONFIG_PATH}" \
        --args.eval-sequences-path "${EVAL_SEQUENCES_PATH}" \
        --args.num-ddim-steps "${CALVIN_NUM_DDIM_STEPS}" \
        --args.num-sequences "$((end - start))" \
        --args.sequence-start "${start}" \
        --args.sequence-end "${end}" \
        --args.eval-log-dir "${worker_dir}"
    ) > "${eval_root}/eval_${worker}.log" 2>&1 &
    eval_pids+=("$!")
    echo "Started eval worker ${worker}: sequences [${start}, ${end}), port ${port}, log ${eval_root}/eval_${worker}.log"

    if (( worker < NUM_WORKERS - 1 )); then
      sleep "${STAGGER_SERVER_SECONDS}"
    fi
  done

  for pid in "${eval_pids[@]}"; do
    if ! wait "${pid}"; then
      cleanup
      return 1
    fi
  done

  "${CALVIN_PYTHON}" examples/calvin/eval_files/aggregate_calvin_results.py "${eval_root}" | tee "${eval_root}/aggregate.log"
  cleanup
  trap - RETURN
  echo "Evaluation complete: ${eval_root}"
}

for ckpt in "${CKPTS[@]}"; do
  run_one_ckpt "${ckpt}"
  # Give ports/GPU memory a moment to clear before the next checkpoint.
  sleep "${BETWEEN_CKPT_SECONDS:-10}"
done
