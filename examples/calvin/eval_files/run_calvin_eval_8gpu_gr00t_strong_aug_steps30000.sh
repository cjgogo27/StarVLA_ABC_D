#!/bin/bash
set -euo pipefail

# Independent 8-GPU CALVIN D eval for QwenGR00T strong-aug ABC->D step 30000.
# Results go to a timestamped directory and will not overwrite other eval runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TAG=qwen35vl4b_gr00t_calvin_abc_d_strong_aug_steps30000

export CKPT_PATH=${CKPT_PATH:-results/Checkpoints/qwen35vl4b_gr00t_calvin_abc_d_strong_aug/checkpoints/steps_30000_pytorch_model.pt}
export BASE_PORT=${BASE_PORT:-5704}
export EVAL_ROOT=${EVAL_ROOT:-results/calvin_eval/${RUN_TAG}_$(date -u +%Y%m%d_%H%M%S)}

bash "${SCRIPT_DIR}/run_calvin_eval_8gpu.sh"
