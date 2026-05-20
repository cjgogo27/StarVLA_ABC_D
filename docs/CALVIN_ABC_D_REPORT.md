# StarVLA ABC-D Technical Report

This report is the repository-facing companion to the paper draft in [`../file.md`](../file.md). It explains what was implemented, how the code maps to the empirical study, what has been evaluated so far, and which parts of the broader paper plan remain experimental directions.

## 1. Task and Paper Alignment

The project studies Vision-Language-Action models on CALVIN ABC->D. The policy is trained on CALVIN environments A/B/C and evaluated in unseen environment D. Each rollout contains up to five chained language instructions, so a failure early in the chain prevents later subtasks from being attempted.

The paper draft organizes the work around three axes:

1. **Baseline ablations:** backbone, action head, visual augmentation, and action-head pre-training.
2. **Failure case analysis:** residual errors are concentrated on push-style skills.
3. **Structural remedies:** add progress information through history conditioning, MoE + progress heads, future-frame conditioning, and RL post-training.

The repository mirrors that organization in code, README, and project page.

## 2. Implemented Baseline Routes

### Qwen3.5-VL + GR00T

The main implemented route uses Qwen3.5-VL as the visual-language backbone and a GR00T-style dual-system action expert for continuous action prediction.

Key files:

- `starVLA/model/framework/VLM4A/QwenGR00T.py`
- `starVLA/model/modules/vlm/QWen3_5.py`
- `starVLA/model/modules/action_model/GR00T_ActionHeader.py`
- `examples/calvin/train_files/starvla_train_calvin.yaml`
- `examples/calvin/train_files/run_calvin_train_qwen35_gr00t.sh`

### Cosmos-Predict2 Route

The paper draft compares general-purpose VLM pre-training with world-model pre-training. The repository includes the Cosmos-Predict2 policy route for this comparison:

- `starVLA/model/framework/WM4A/CosmoPredict2PI.py`
- `examples/calvin/train_files/run_calvin_train_cosmopredict2_pi.sh`
- `examples/calvin/eval_files/run_calvin_eval_cosmopredict2_pi_multigpu_local.sh`

### Action-Head Variants

The paper draft contrasts PI-style unified action prediction with GR00T-style dual-system prediction. The repository contains the GR00T route as the main path and keeps PI/Cosmos launchers for comparison experiments.

## 3. Data, Augmentation, and Pre-training

CALVIN training data is expected in LeRobot format. Evaluation uses the original CALVIN format with a `validation/` directory.

Relevant files:

- `starVLA/dataloader/gr00t_lerobot/datasets.py`
- `starVLA/dataloader/gr00t_lerobot/mixtures.py`
- `examples/calvin/train_files/starvla_train_calvin.yaml`
- `examples/calvin/train_files/run_calvin_train_qwen35_gr00t_strong_aug.sh`

The paper draft describes mild and aggressive visual augmentation schedules: random crop, brightness/contrast/saturation jitter, blur, noise, and JPEG artifacts. The repository includes the data-loader and launch-script hooks used for these augmentation-oriented training runs.

For action-head pre-training, the draft discusses LIBERO Spatial/Object/Goal/Long pre-training before CALVIN fine-tuning. This repository contains LIBERO-related hooks and shared model-client updates, but the committed public result focuses on the CALVIN ABC-D run.

## 4. Current Evaluation Snapshot

The best public result is reported from [`eval_runs_summary.md`](eval_runs_summary.md).

Saved run:

```text
results/calvin_eval/baseline_strong_aug_steps30000_statefix1000_multigpu_20260520_031141
```

Evaluation setup:

- Split: CALVIN ABC->D validation chains in environment D.
- Sequences: 1000.
- Model: Qwen3.5-4B + GR00T + MoT/state-fix variant.
- Metric: `SR@k` means the fraction of sequences completing at least the first `k` tasks in a five-task chain.

| Model | #Seq | SR@1 | SR@2 | SR@3 | SR@4 | SR@5 | Average chain length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B+GR00T (no aug) | 1000 | 69.5% | 45.7% | 31.7% | 21.0% | 14.5% | 1.824 |
| Qwen3.5-4B+PI | 1000 | 62.6% | 41.1% | 26.3% | 17.2% | 11.8% | 1.590 |
| Qwen3.5-4B+GR00T (strong aug) | 1000 | 75.2% | 54.0% | 39.3% | 28.9% | 19.3% | 2.167 |
| Qwen3.5-4B+GR00T+MoE | 1000 | 72.1% | 49.4% | 33.9% | 22.8% | 15.0% | 1.932 |
| Cosmos-Predict2-2B+PI | 1000 | 36.0% | 11.2% | 3.0% | 0.5% | 0.1% | 0.508 |
| Cosmos-Predict2-2B+PI+flare | 1000 | 41.6% | 14.8% | 3.7% | 1.5% | 0.6% | 0.622 |
| **Qwen3.5-4B+GR00T+MoT** | **1000** | **77.3%** | **57.5%** | **44.5%** | **34.2%** | **24.5%** | **2.380** |

The strongest current result is Qwen3.5-4B+GR00T+MoT with 24.5% full-chain success and 2.380 average chain length. The available rollout GIFs come from an earlier video-logging run; the best 1000-sequence run was retained as JSON statistics.

## 5. Failure Analysis

The paper draft identifies push-style tasks as the dominant residual failure family:

- `push_left`
- `push_right`
- `push_into_drawer`

The key hypothesis is missing **progress awareness**. Pick-and-place and articulated-object tasks often expose clear visual milestones, such as an open drawer or an object placed inside a receptacle. Push tasks may look similar before and after partial progress, so the policy can oscillate, under-push, or stop early.

Repository evidence and tooling:

- Rollout GIFs: `docs/assets/calvin_abc_d/*.gif`
- Result aggregation: `examples/calvin/eval_files/aggregate_calvin_results.py`
- Local evaluator with GIF logging: `examples/calvin/eval_files/eval_calvin_8gpu_local.py`

Failure categories tracked in this repository:

| Category | Example | Likely cause | Targeted remedy |
| --- | --- | --- | --- |
| Push progress ambiguity | push block left/right | no explicit completion signal | progress head, history context |
| Contact precision | place in slider, stack block | small pose errors compound | better action refinement, failure-aware data |
| Long-horizon compounding | later chain steps | early drift changes state distribution | re-planning, RL post-training |
| Visual domain shift | ABC->D textures/positions | unseen environment D | stronger but controlled augmentation |

## 6. Structural Remedies Implemented or Scaffolded

### Past-action / MOT-style context

The paper draft proposes feeding past action chunks back into the policy so the model can infer whether it is making progress. The repository includes motion-adapter and MOT-style training launchers:

- `examples/calvin/train_files/run_calvin_train_qwen35_gr00t_mot_state_adapter_from_strong_aug.sh`
- `examples/calvin/train_files/run_calvin_train_qwen35_gr00t_from_strong_aug_1000_mot_adapter_only.sh`
- `examples/calvin/train_files/start_qwen35_gr00t_mot_adapter_only.sh`
- `examples/calvin/train_files/start_qwen35_gr00t_mot_state_adapter.sh`

### MoE action expert with progress head

The paper draft describes sparsely gated experts plus an auxiliary progress prediction signal. The repository includes:

- `starVLA/model/modules/action_model/GR00T_MoE_ActionHeader.py`
- `starVLA/model/framework/VLM4A/QwenGR00TMoE.py`
- `starVLA/model/framework/VLM4A/QwenGR00TMoEProgress.py`
- `examples/calvin/train_files/starvla_train_calvin_gr00t_moe.yaml`
- `examples/calvin/train_files/starvla_train_calvin_gr00t_moe_progress.yaml`

### Future-frame conditioning with Cosmos

The paper draft proposes using Cosmos-Predict2 to imagine future frames and inject their features into the action expert. The repository includes the Cosmos-Predict2 PI route and evaluation launcher:

- `starVLA/model/framework/WM4A/CosmoPredict2PI.py`
- `examples/calvin/train_files/run_calvin_train_cosmopredict2_pi.sh`
- `examples/calvin/eval_files/run_calvin_eval_cosmopredict2_pi_multigpu_local.sh`

### RL post-training with RLinf-VLA

The paper draft discusses GRPO/RLinf-style post-training using outcome rewards. The public repository records this as a documented direction; large RL rollouts and checkpoints are not committed.

## 7. Reproduction Commands

The reproduction path covers two Qwen3.5 action heads: PI-style unified action prediction and GR00T-style continuous action prediction.

Train the Qwen3.5-PI policy:

```bash
bash examples/calvin/train_files/run_calvin_train.sh
```

Evaluate the Qwen3.5-PI checkpoint:

```bash
bash examples/calvin/eval_files/run_calvin_eval_8gpu_pi_abc_d_steps30000.sh
```

Train the Qwen3.5-GR00T policy:

```bash
bash examples/calvin/train_files/run_calvin_train_qwen35_gr00t.sh
```

Run multi-GPU local evaluation:

```bash
bash examples/calvin/eval_files/run_calvin_eval_multigpu_local.sh
```

Aggregate result files:

```bash
python examples/calvin/eval_files/aggregate_calvin_results.py \
  --eval-root results/calvin_eval/baseline_strong_aug_steps30000_statefix1000_multigpu_20260520_031141
```

Aggregate the reported Qwen3.5-PI run:

```bash
python examples/calvin/eval_files/aggregate_calvin_results.py \
  --eval-root results/calvin_eval/qwen35vl4b_pi_calvin_abc_d_steps_30000_pytorch_model_multigpu_20260519_070133
```

## 8. Website

The GitHub Pages source mirrors this report:

- `docs/index.html`
- `index.html`
- `.nojekyll`
- `docs/assets/calvin_abc_d/*.gif`

Expected URL after GitHub Pages is enabled:

```text
https://cjgogo27.github.io/StarVLA_ABC_D/
```

Use GitHub Settings -> Pages -> Deploy from a branch -> `main` -> `/docs`.
