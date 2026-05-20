# CALVIN ABC->D Eval Summary

统计项：`SR@k` 表示五任务链中至少完成前 `k` 个任务的比例；`SR@5` 即完整五任务链成功率。`平均成功链长` 由每条 sequence 实际完成的 task 数取平均。最差五个动作按 primitive success rate 从低到高排序。

## Overall

| Model | #Seq | SR@1 | SR@2 | SR@3 | SR@4 | SR@5 | 平均成功链长 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B+GR00T (no aug) | 1000 | 69.5% | 45.7% | 31.7% | 21.0% | 14.5% | 1.824 |
| Qwen3.5-4B+PI | 1000 | 62.6% | 41.1% | 26.3% | 17.2% | 11.8% | 1.590 |
| Qwen3.5-4B+GR00T (strong aug) | 1000 | 75.2% | 54.0% | 39.3% | 28.9% | 19.3% | 2.167 |
| Qwen3.5-4B+GR00T+MoE | 1000 | 72.1% | 49.4% | 33.9% | 22.8% | 15.0% | 1.932 |
| Cosmos-Predict2-2B+PI | 1000 | 36.0% | 11.2% | 3.0% | 0.5% | 0.1% | 0.508 |
| Cosmos-Predict2-2B+PI+flare | 1000 | 41.6% | 14.8% | 3.7% | 1.5% | 0.6% | 0.622 |
| Qwen3.5-4B+GR00T+MoT | 1000 | 77.3% | 57.5% | 44.5% | 34.2% | 24.5% | 2.380 |

## 1. Qwen3.5-4B+GR00T (no aug)

- Path: `starVLA/results/calvin_eval/qwen35vl4b_gr00t_calvin_abc_d_steps_30000_pytorch_model_multigpu_20260519_144802`
- Sequences: 1000
- 完整五任务链成功率 SR@5: **14.5%**
- 平均成功链长: **1.824 / 5**

| Worst action | Success | Total | SR |
| --- | ---: | ---: | ---: |
| `push_pink_block_left` | 10 | 66 | 15.2% |
| `rotate_pink_block_right` | 10 | 62 | 16.1% |
| `rotate_red_block_left` | 9 | 55 | 16.4% |
| `push_red_block_right` | 14 | 64 | 21.9% |
| `move_slider_left` | 48 | 167 | 28.7% |

## 2. Qwen3.5-4B+PI

- Path: `starVLA/results/calvin_eval/qwen35vl4b_pi_calvin_abc_d_steps_30000_pytorch_model_multigpu_20260519_070133`
- Sequences: 1000
- 完整五任务链成功率 SR@5: **11.8%**
- 平均成功链长: **1.590 / 5**
- Note: Aggregated from 8 worker shards, each evaluating 125 sequences.

| Worst action | Success | Total | SR |
| --- | ---: | ---: | ---: |
| `move_slider_left` | 7 | 166 | 4.2% |
| `rotate_pink_block_right` | 7 | 56 | 12.5% |
| `push_pink_block_left` | 10 | 62 | 16.1% |
| `push_red_block_right` | 14 | 59 | 23.7% |
| `rotate_pink_block_left` | 11 | 42 | 26.2% |

## 3. Qwen3.5-4B+GR00T (strong aug)

- Path: `starVLA/results/calvin_eval/qwen35vl4b_gr00t_calvin_abc_d_strong_aug_steps_30000_pytorch_model_multigpu_20260519_083818`
- Sequences: 1000
- 完整五任务链成功率 SR@5: **19.3%**
- 平均成功链长: **2.167 / 5**
- Note: Recovered full statistics from `analysis/calvin_failure_report_gr00t_strong_aug_20260519_083818` because some worker JSON files are empty.

| Worst action | Success | Total | SR |
| --- | ---: | ---: | ---: |
| `push_pink_block_left` | 10 | 68 | 14.7% |
| `push_red_block_right` | 15 | 65 | 23.1% |
| `rotate_pink_block_left` | 11 | 47 | 23.4% |
| `push_pink_block_right` | 17 | 58 | 29.3% |
| `push_blue_block_left` | 18 | 60 | 30.0% |

## 4. Qwen3.5-4B+GR00T+MoE

- Path: `starVLA/results/calvin_eval/qwen35vl4b_gr00t_moe_calvin_abc_d_strong_aug_from30000_new_lr_sched_steps_5000_pytorch_model_multigpu_20260520_024717`
- Sequences: 1000
- 完整五任务链成功率 SR@5: **15.0%**
- 平均成功链长: **1.932 / 5**

| Worst action | Success | Total | SR |
| --- | ---: | ---: | ---: |
| `move_slider_left` | 32 | 173 | 18.5% |
| `push_pink_block_left` | 12 | 63 | 19.0% |
| `push_red_block_right` | 17 | 59 | 28.8% |
| `push_blue_block_left` | 18 | 58 | 31.0% |
| `rotate_pink_block_left` | 15 | 47 | 31.9% |

## 5. Cosmos-Predict2-2B+PI

- Path: `starVLA/results/calvin_eval/cosmopredict2_pi/cosmopredict2_2b_pi_calvin_abc_d_steps_30000_pytorch_model_multigpu_20260519_075747`
- Sequences: 1000
- 完整五任务链成功率 SR@5: **0.1%**
- 平均成功链长: **0.508 / 5**

| Worst action | Success | Total | SR |
| --- | ---: | ---: | ---: |
| `push_blue_block_right` | 0 | 36 | 0.0% |
| `push_pink_block_right` | 0 | 35 | 0.0% |
| `lift_pink_block_drawer` | 0 | 1 | 0.0% |
| `rotate_pink_block_right` | 1 | 43 | 2.3% |
| `lift_red_block_slider` | 1 | 42 | 2.4% |

## 6. Cosmos-Predict2-2B+PI+flare

- Path: `starVLA/results/calvin_eval/cosmopredict2_pi_flare_w002/cosmopredict2_2b_pi_calvin_abc_d_flare_w002_steps_40000_pytorch_model_multigpu_20260520_060711`
- Sequences: 1000
- 完整五任务链成功率 SR@5: **0.6%**
- 平均成功链长: **0.622 / 5**

| Worst action | Success | Total | SR |
| --- | ---: | ---: | ---: |
| `lift_blue_block_slider` | 0 | 42 | 0.0% |
| `push_blue_block_right` | 0 | 35 | 0.0% |
| `push_red_block_right` | 1 | 45 | 2.2% |
| `lift_red_block_slider` | 1 | 38 | 2.6% |
| `push_pink_block_right` | 1 | 34 | 2.9% |

## 7. Qwen3.5-4B+GR00T+MoT

- Path: `starVLA/results/calvin_eval/baseline_strong_aug_steps30000_statefix1000_multigpu_20260520_031141`
- Sequences: 1000
- 完整五任务链成功率 SR@5: **24.5%**
- 平均成功链长: **2.380 / 5**

| Worst action | Success | Total | SR |
| --- | ---: | ---: | ---: |
| `push_pink_block_left` | 12 | 68 | 17.6% |
| `push_red_block_right` | 15 | 64 | 23.4% |
| `push_blue_block_right` | 16 | 57 | 28.1% |
| `push_pink_block_right` | 18 | 58 | 31.0% |
| `rotate_pink_block_left` | 15 | 48 | 31.2% |
