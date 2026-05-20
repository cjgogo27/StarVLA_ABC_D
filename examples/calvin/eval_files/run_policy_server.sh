#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export star_vla_python=/inspire/qb-ilm2/project/26summer-camp-10/public/two/miniconda/envs/starVLA2/bin/python
your_ckpt=/inspire/qb-ilm2/project/26summer-camp-10/public/two/starVLA/results/Checkpoints/qwen35vl4b_pi_calvin_abc_d/checkpoints/steps_30000_pytorch_model.pt
gpu_id=0
port=5694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
