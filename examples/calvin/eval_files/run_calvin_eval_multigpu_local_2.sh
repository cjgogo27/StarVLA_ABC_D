#!/bin/bash
set -euo pipefail


#  source /inspire/qb-ilm2/project/26summer-camp-10/public/two/miniconda/bin/activate
# conda activate starVLA2
# cd /inspire/qb-ilm2/project/26summer-camp-10/public/two/starVLA
# Multi-GPU CALVIN evaluation:
# - starts one or more policy servers per GPU
# - splits eval sequences across workers
# - writes each run to a timestamped EVAL_ROOT to avoid overwriting results

REPO_ROOT=$(pwd)

# 
CKPT_PATH=${CKPT_PATH:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/starVLA/results/Checkpoints/qwen35vl4b_gr00t_calvin_abc_d_strong_aug_from30000_new_lr_sched/checkpoints/steps_12000_pytorch_model.pt}
CALVIN_DATASET_PATH=${CALVIN_DATASET_PATH:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d}
CALVIN_ROOT=${CALVIN_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/calvin}
STAR_VLA_PYTHON=${STAR_VLA_PYTHON:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/miniconda/envs/starVLA2/bin/python}
CALVIN_PYTHON=${CALVIN_PYTHON:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/miniconda/envs/calvin_venv/bin/python}
CALVIN_EXTRA_PYTHONPATH=${CALVIN_EXTRA_PYTHONPATH:-${REPO_ROOT}/.runtime/calvin_cv2_headless_overlay}

NUM_GPUS=${NUM_GPUS:-8}
PROCS_PER_GPU=${PROCS_PER_GPU:-1}
NUM_WORKERS=${NUM_WORKERS:-$((NUM_GPUS * PROCS_PER_GPU))}
NUM_SEQUENCES=${NUM_SEQUENCES:-1000}
BASE_PORT=${BASE_PORT:-6714}
START_GPU=${START_GPU:-0}
UNNORM_KEY=${UNNORM_KEY:-franka}
EVAL_STAGGER_SECONDS=${EVAL_STAGGER_SECONDS:-0}
CALVIN_USE_EGL=${CALVIN_USE_EGL:-0}
CALVIN_EP_LEN=${CALVIN_EP_LEN:-240}
CALVIN_NUM_DDIM_STEPS=${CALVIN_NUM_DDIM_STEPS:-5}
CALVIN_STATE_DIM=${CALVIN_STATE_DIM:-7}
STAGGER_SERVER_SECONDS=${STAGGER_SERVER_SECONDS:-12}
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-900}
SAVE_VIDEO=${SAVE_VIDEO:-0}
#v
RUN_TAG=${RUN_TAG:-$(basename "$(dirname "$(dirname "${CKPT_PATH}")")")_$(basename "${CKPT_PATH}" .pt)}
#
EVAL_ROOT=${EVAL_ROOT:-results/calvin_eval/${RUN_TAG}_multigpu_$(date -u +%Y%m%d_%H%M%S)}

CALVIN_CONFIG_PATH=${CALVIN_CONFIG_PATH:-${CALVIN_ROOT}/calvin_models/conf}
EVAL_SEQUENCES_PATH=${EVAL_SEQUENCES_PATH:-${REPO_ROOT}/examples/calvin/eval_files/eval_sequences.json}
EVAL_PY=${EVAL_PY:-examples/calvin/eval_files/eval_calvin_8gpu_local.py}

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH}" >&2
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

PORT_CHECK_PY=${PORT_CHECK_PY:-${STAR_VLA_PYTHON}}
"${PORT_CHECK_PY}" - "${BASE_PORT}" "${NUM_WORKERS}" <<'PY'
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
    print("Set BASE_PORT to a free range or stop the old eval/server process.", file=sys.stderr)
    sys.exit(1)
PY

mkdir -p "${EVAL_ROOT}"

if ! PYTHONPATH="${CALVIN_EXTRA_PYTHONPATH}:${PYTHONPATH:-}" "${CALVIN_PYTHON}" -c "import websockets.sync.client, msgpack, cv2" >/dev/null 2>&1; then
  echo "CALVIN_PYTHON is missing required runtime imports: websockets.sync.client, msgpack, or cv2." >&2
  echo "CALVIN_PYTHON=${CALVIN_PYTHON}" >&2
  exit 1
fi

if ! "${STAR_VLA_PYTHON}" -c "import transformers; assert hasattr(transformers, 'Qwen3_5ForConditionalGeneration'), transformers.__version__" >/dev/null 2>&1; then
  echo "STAR_VLA_PYTHON does not provide Qwen3_5ForConditionalGeneration." >&2
  echo "STAR_VLA_PYTHON=${STAR_VLA_PYTHON}" >&2
  exit 1
fi

export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig_starvla_calvin_eval}
mkdir -p "${MPLCONFIGDIR}"

{
  echo "CKPT_PATH=${CKPT_PATH}"
  echo "CALVIN_DATASET_PATH=${CALVIN_DATASET_PATH}"
  echo "CALVIN_CONFIG_PATH=${CALVIN_CONFIG_PATH}"
  echo "EVAL_SEQUENCES_PATH=${EVAL_SEQUENCES_PATH}"
  echo "EVAL_PY=${EVAL_PY}"
  echo "EVAL_ROOT=${EVAL_ROOT}"
  echo "NUM_GPUS=${NUM_GPUS}"
  echo "PROCS_PER_GPU=${PROCS_PER_GPU}"
  echo "NUM_WORKERS=${NUM_WORKERS}"
  echo "NUM_SEQUENCES=${NUM_SEQUENCES}"
  echo "BASE_PORT=${BASE_PORT}"
  echo "START_GPU=${START_GPU}"
  echo "EVAL_STAGGER_SECONDS=${EVAL_STAGGER_SECONDS}"
  echo "CALVIN_USE_EGL=${CALVIN_USE_EGL}"
  echo "CALVIN_EP_LEN=${CALVIN_EP_LEN}"
  echo "CALVIN_NUM_DDIM_STEPS=${CALVIN_NUM_DDIM_STEPS}"
  echo "CALVIN_STATE_DIM=${CALVIN_STATE_DIM}"
  echo "STAGGER_SERVER_SECONDS=${STAGGER_SERVER_SECONDS}"
  echo "SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT}"
  echo "SAVE_VIDEO=${SAVE_VIDEO}"
} | tee "${EVAL_ROOT}/run.info"

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT

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

start_eval_worker() {
  local worker="$1"
  local port="$2"
  local start="$3"
  local end="$4"
  local worker_dir="${EVAL_ROOT}/worker_${worker}"
  mkdir -p "${worker_dir}"
  (
    export PYTHONPATH="${CALVIN_EXTRA_PYTHONPATH}:${REPO_ROOT}:${CALVIN_ROOT}/calvin_models:${CALVIN_ROOT}/calvin_env:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1
    export CALVIN_USE_EGL
    export CALVIN_EP_LEN
    export CALVIN_STATE_DIM
    export CALVIN_NUM_DDIM_STEPS
    export SAVE_VIDEO
    if [[ "${CALVIN_USE_EGL}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
      export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
      export MUJOCO_GL=${MUJOCO_GL:-egl}
    else
      export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
      export MUJOCO_GL=${MUJOCO_GL:-osmesa}
    fi
    eval_cmd=(
      "${CALVIN_PYTHON}"
      "${EVAL_PY}"
      --args.pretrained-path "${CKPT_PATH}"
      --args.unnorm-key "${UNNORM_KEY}"
      --args.host 127.0.0.1
      --args.port "${port}"
      --args.dataset-path "${CALVIN_DATASET_PATH}"
      --args.calvin-config-path "${CALVIN_CONFIG_PATH}"
      --args.eval-sequences-path "${EVAL_SEQUENCES_PATH}"
      --args.num-ddim-steps "${CALVIN_NUM_DDIM_STEPS}"
      --args.num-sequences "$((end - start))"
      --args.sequence-start "${start}"
      --args.sequence-end "${end}"
      --args.eval-log-dir "${worker_dir}"
    )
    if [[ "${SAVE_VIDEO}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
      eval_cmd+=(--args.debug)
    fi
    "${eval_cmd[@]}"
  ) > "${EVAL_ROOT}/eval_${worker}.log" 2>&1 &
  eval_pids+=("$!")
  echo "Started eval worker ${worker}: sequences [${start}, ${end}), port ${port}, log ${EVAL_ROOT}/eval_${worker}.log"
}

eval_pids=()
for worker in $(seq 0 $((NUM_WORKERS - 1))); do
  gpu=$((START_GPU + worker / PROCS_PER_GPU))
  port=$((BASE_PORT + worker))
  start=$((worker * NUM_SEQUENCES / NUM_WORKERS))
  end=$(((worker + 1) * NUM_SEQUENCES / NUM_WORKERS))
  server_cmd=(
    "${STAR_VLA_PYTHON}"
    deployment/model_server/server_policy.py
    --ckpt_path "${CKPT_PATH}"
    --port "${port}"
    --idle_timeout -1
    --use_bf16
  )
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=${gpu} "${server_cmd[@]}" \
    > "${EVAL_ROOT}/server_${worker}_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
  echo "Started policy server ${worker}: GPU ${gpu}, port ${port}, log ${EVAL_ROOT}/server_${worker}_gpu_${gpu}.log"
  wait_for_policy_server "${port}"
  start_eval_worker "${worker}" "${port}" "${start}" "${end}"
  if (( worker < NUM_WORKERS - 1 )); then
    sleep "${STAGGER_SERVER_SECONDS}"
  fi
done

for pid in "${eval_pids[@]}"; do
  wait "${pid}"
done

"${CALVIN_PYTHON}" examples/calvin/eval_files/aggregate_calvin_results.py "${EVAL_ROOT}" | tee "${EVAL_ROOT}/aggregate.log"

cleanup
trap - EXIT
echo "Evaluation complete: ${EVAL_ROOT}"
