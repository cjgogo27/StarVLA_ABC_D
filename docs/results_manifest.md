# Results Manifest

This file records the lightweight artifacts included in the GitHub submission.
Large model checkpoints are intentionally not committed. The selected inference
checkpoint is hosted on Hugging Face.

## Selected Model

Model: `Qwen3.5-4B + GR00T + MOT adapter-only`

Local checkpoint path:

```text
/inspire/qb-ilm2/project/26summer-camp-10/public/two/starVLA/results/Checkpoints/qwen35vl4b_gr00t_calvin_abc_d_strong_aug_from30000_new_lr_sched_steps1000_mot_adapter_only/checkpoints/steps_50000_pytorch_model.pt
```

Hugging Face checkpoint:

```text
https://huggingface.co/cjgogo/qwen35vl4b_gr00t_calvin_abc_d_strong_aug_from30000_new_lr_sched_steps1000_mot_adapter_only
```

Uploaded file:

```text
steps_50000_pytorch_model.pt
```

## Lightweight Training Artifacts

These files are committed because they are small and reproduce the training
configuration without including checkpoint weights:

```text
results/Checkpoints/qwen35vl4b_gr00t_calvin_abc_d_strong_aug_from30000_new_lr_sched_steps1000_mot_adapter_only/config.yaml
results/Checkpoints/qwen35vl4b_gr00t_calvin_abc_d_strong_aug_from30000_new_lr_sched_steps1000_mot_adapter_only/config.full.yaml
results/Checkpoints/qwen35vl4b_gr00t_calvin_abc_d_strong_aug_from30000_new_lr_sched_steps1000_mot_adapter_only/dataset_statistics.json
results/Checkpoints/qwen35vl4b_gr00t_calvin_abc_d_strong_aug_from30000_new_lr_sched_steps1000_mot_adapter_only/summary.jsonl
results/Checkpoints/qwen35vl4b_gr00t_calvin_abc_d_strong_aug_from30000_new_lr_sched_steps1000_mot_adapter_only/run_calvin_train_qwen35_gr00t_from_strong_aug_1000_mot_adapter_only.sh
```

Excluded from Git:

```text
results/Checkpoints/**/checkpoints/*.pt
results/Checkpoints/**/tensorboard/
results/Checkpoints/**/logs/train.log
```

## Main Code Entrypoints

Training:

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/public/two/starVLA
bash examples/calvin/train_files/run_calvin_train_qwen35_gr00t_from_strong_aug_1000_mot_adapter_only.sh
```

Evaluation smoke test:

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/public/two/starVLA

CKPT_PATH=/inspire/qb-ilm2/project/26summer-camp-10/public/two/starVLA/results/Checkpoints/qwen35vl4b_gr00t_calvin_abc_d_strong_aug_from30000_new_lr_sched_steps1000_mot_adapter_only/checkpoints/steps_50000_pytorch_model.pt \
RUN_TAG=qwen35vl4b_gr00t_mot_adapter_only_steps50000_smoke \
NUM_GPUS=1 \
START_GPU=0 \
NUM_SEQUENCES=4 \
BASE_PORT=7814 \
CALVIN_EP_LEN=60 \
bash examples/calvin/eval_files/run_calvin_eval_multigpu_local.sh
```

Full evaluation:

```bash
cd /inspire/qb-ilm2/project/26summer-camp-10/public/two/starVLA

CKPT_PATH=/inspire/qb-ilm2/project/26summer-camp-10/public/two/starVLA/results/Checkpoints/qwen35vl4b_gr00t_calvin_abc_d_strong_aug_from30000_new_lr_sched_steps1000_mot_adapter_only/checkpoints/steps_50000_pytorch_model.pt \
RUN_TAG=qwen35vl4b_gr00t_mot_adapter_only_steps50000 \
NUM_GPUS=8 \
START_GPU=0 \
NUM_SEQUENCES=1000 \
BASE_PORT=7814 \
bash examples/calvin/eval_files/run_calvin_eval_multigpu_local.sh
```

## Evaluation Summary

The consolidated leaderboard-style summary is committed at:

```text
docs/eval_runs_summary.md
```

The selected MOT result reported there:

```text
SR@1: 77.3%
SR@2: 57.5%
SR@3: 44.5%
SR@4: 34.2%
SR@5: 24.5%
Average successful sequence length: 2.380
```
