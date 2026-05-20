#!/bin/bash

###########################################################################################
# === Please modify the following paths according to your environment ===
export PYTHONPATH=$(pwd):${PYTHONPATH} # let Calvin client find websocket tools from main repo
export calvin_python=${calvin_python:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/miniconda/envs/calvin_venv/bin/python}
export CALVIN_EXTRA_PYTHONPATH=${CALVIN_EXTRA_PYTHONPATH:-$(pwd)/.runtime/calvin_cv2_headless_overlay}
export CALVIN_ROOT=${CALVIN_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/calvin}
export PYTHONPATH=${CALVIN_EXTRA_PYTHONPATH}:$(pwd):${CALVIN_ROOT}/calvin_models:${CALVIN_ROOT}/calvin_env:${PYTHONPATH}

host="127.0.0.1"
base_port=5694
unnorm_key="franka"
your_ckpt=/inspire/qb-ilm2/project/26summer-camp-10/public/two/starVLA/results/Checkpoints/qwen35vl4b_pi_calvin_abc_d/checkpoints/steps_30000_pytorch_model.pt
calvin_config_path=${calvin_config_path:-${CALVIN_ROOT}/calvin_models/conf}
eval_sequences_path=${eval_sequences_path:-/inspire/qb-ilm2/project/26summer-camp-10/public/two/starVLA/examples/calvin/eval_files/eval_sequences.json}

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/calvin_eval_$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}

${calvin_python} ./examples/calvin/eval_files/eval_calvin_local.py \
    --args.pretrained-path ${your_ckpt} \
    --args.unnorm-key ${unnorm_key} \
    --args.host "$host" \
    --args.port $base_port \
    --args.dataset_path /inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d/ \
    --args.calvin-config-path "${calvin_config_path}" \
    --args.eval-sequences-path "${eval_sequences_path}" \
    --args.eval-log-dir "${LOG_DIR}" \
    --args.num_sequences "${NUM_SEQUENCES:-1000}"
