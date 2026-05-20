# Code Overview

This document summarizes the main code added or modified for the StarVLA ABC-D project. The README intentionally stays concise; use this page when you need to inspect implementation details.

## 1. Main Policy Route

### `starVLA/model/framework/VLM4A/QwenGR00T.py`

Main Qwen3.5-VL + GR00T-style policy framework for CALVIN ABC->D. It wires together the multimodal backbone, robot state inputs, action expert, inference options, and serving-time generation path.

### `starVLA/model/modules/vlm/QWen3_5.py`

Qwen3.5-VL interface adjustments for this project. This is the backbone-side entry point used by the QwenGR00T policy route.

### `starVLA/model/framework/WM4A/CosmoPredict2PI.py`

Cosmos-Predict2 world-model route used for comparison with VLM-backed policies. It corresponds to the paper draft's world-model baseline direction.

## 2. Action-Head Variants

### `starVLA/model/modules/action_model/GR00T_MoE_ActionHeader.py`

Mixture-of-experts action header. This supports the progress-aware remedy direction described in the paper draft.

### `starVLA/model/framework/VLM4A/QwenGR00TMoE.py`

Qwen3.5 + GR00T MoE framework variant for action-head scaling experiments.

### `starVLA/model/framework/VLM4A/QwenGR00TMoEProgress.py`

Progress-aware MoE framework variant. This connects directly to the project's key diagnosis: push-style skills need a stronger signal for subtask progress.

## 3. CALVIN Data and Training

### `starVLA/dataloader/gr00t_lerobot/datasets.py`

CALVIN/GR00T LeRobot data loading, modality processing, action-state handling, and augmentation path.

### `starVLA/dataloader/gr00t_lerobot/mixtures.py`

Registers CALVIN data mixtures used by the training configs.

### Main configs

- `examples/calvin/train_files/starvla_train_calvin.yaml`
- `examples/calvin/train_files/starvla_train_calvin_gr00t_moe.yaml`
- `examples/calvin/train_files/starvla_train_calvin_gr00t_moe_progress.yaml`

### Main training launchers

- `examples/calvin/train_files/run_calvin_train_qwen35_gr00t.sh`
- `examples/calvin/train_files/run_calvin_train_qwen35_gr00t_strong_aug.sh`
- `examples/calvin/train_files/run_calvin_train_qwen35_gr00t_abc_only.sh`
- `examples/calvin/train_files/run_calvin_train_qwen35_gr00t_mot_state_adapter_from_strong_aug.sh`
- `examples/calvin/train_files/run_calvin_train_qwen35_gr00t_from_strong_aug_1000_mot_adapter_only.sh`
- `examples/calvin/train_files/run_calvin_train_qwen35_gr00t_moe_from_gr00t.sh`
- `examples/calvin/train_files/run_calvin_train_qwen35_gr00t_moe_progress.sh`
- `examples/calvin/train_files/run_calvin_train_cosmopredict2_pi.sh`
- `examples/calvin/train_files/run_calvin_train_cosmopredict2_pi_flare_posttrain.sh`

## 4. Evaluation and Result Aggregation

### `examples/calvin/eval_files/eval_calvin_8gpu_local.py`

Local multi-GPU CALVIN evaluator. It supports worker slicing, configurable episode/DDIM settings, policy-server interaction, and rollout GIF saving.

### `examples/calvin/eval_files/eval_calvin_local.py`

Smaller local evaluator variant for development and debugging.

### `examples/calvin/eval_files/run_calvin_eval_multigpu_local.sh`

Primary multi-GPU ABC->D evaluation launcher.

### `examples/calvin/eval_files/run_calvin_eval_multigpu_local_2.sh`

Alternate launcher used for repeat/checkpoint-specific runs.

### `examples/calvin/eval_files/run_group_calvin_eval_steps50000.sh`

Convenience launcher for evaluating the 50k MoT adapter checkpoint used by the strongest reported run. The script keeps the local defaults used in this project but exposes `ROOT`, `REPO_ROOT`, `CKPT`, `NUM_SEQUENCES`, `BASE_PORT`, and dataset paths as environment-variable overrides.

### `examples/calvin/eval_files/aggregate_calvin_results.py`

Aggregates per-worker `sequence_results.json` and `results.json` files into average chain length and Task 1-5 chain success rates.

## 5. Serving and Client Updates

### `deployment/model_server/tools/websocket_policy_client.py`

Adds more robust websocket-client compatibility and server connection handling.

### `examples/LIBERO/eval_files/model2libero_interface.py`

Passes the current step into policy inference requests so evaluation clients can support time/progress-aware logic.

## 6. Documentation and Website

- `README.md`: concise project overview, figures, results, reproduction commands, and reflections.
- `docs/CALVIN_ABC_D_REPORT.md`: technical report aligned with `file.md`.
- `docs/index.html`: GitHub Pages project page.
- `index.html`: root redirect for GitHub Pages configurations that publish from repository root.
- `file.md`: paper draft source.
- `figures/f1.png`, `figures/f2.png`: baseline framework figures used in README.
- `docs/assets/calvin_abc_d/*.gif`: selected CALVIN rollout GIFs used by the project page.

## 7. Minimal Reproduction Path

```bash
# Train
bash examples/calvin/train_files/run_calvin_train_qwen35_gr00t.sh

# Evaluate
bash examples/calvin/eval_files/run_calvin_eval_multigpu_local.sh

# Aggregate
python examples/calvin/eval_files/aggregate_calvin_results.py \
  --eval-root results/calvin_eval/qwen35vl4b_gr00t_calvin_abc_d_steps_30000_pytorch_model_multigpu_20260519_130220
```
