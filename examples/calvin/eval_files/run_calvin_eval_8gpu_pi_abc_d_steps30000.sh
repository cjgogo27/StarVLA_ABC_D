#!/bin/bash
set -euo pipefail

# Independent 8-GPU CALVIN D eval for QwenPI ABC->D step 30000.
# Results go to a timestamped directory and will not overwrite other eval runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TAG=qwen35vl4b_pi_calvin_abc_d_steps30000

export CKPT_PATH=${CKPT_PATH:-results/Checkpoints/qwen35vl4b_pi_calvin_abc_d/checkpoints/steps_30000_pytorch_model.pt}
export BASE_PORT=${BASE_PORT:-5694}
export EVAL_ROOT=${EVAL_ROOT:-results/calvin_eval/${RUN_TAG}_$(date -u +%Y%m%d_%H%M%S)}

bash "${SCRIPT_DIR}/run_calvin_eval_8gpu.sh"
