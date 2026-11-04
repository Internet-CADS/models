# 3rd run — Matched 4M-frame highway-v0 ablation (Vanilla DQN vs. SA-DQN vs. WocaR-DQN)

Each model was trained for `4,000,000` frames on the exact same `highway-v0` configuration (`seed: 1`, 4 lanes, 10 vehicles, 50-dim normalized `Kinematics`, `collision_reward: -2.0`, `normalize_reward: false`, `offroad_terminal: true`) on single-threaded CPU (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) after fixing the WocaR `bound_forward`, `worst_action_select`, state-importance weighting $w(s)$, and deterministic `mini_test` seeding bugs.

## Summary of Best Checkpoints

| Model | Folder | Best Checkpoint | Frame | `mini_test` Return | `mini_test` Crash Rate | `mini_test` Safety Score |
|---|---|---|---|---|---|---|
| **Vanilla DQN** | `highway-v0_vanilla/` | `best_frame_4000000.pth` (`frame_4000000.pth`) | `4,000,000` | **18.54** | 0.08 | **18.14** |
| **SA-DQN** | `highway-v0_sadqn/` | `best_frame_2450000.pth` (`frame_2450000.pth`) | `2,450,000` | 18.41 | **0.04** | **18.21** |
| **WocaR-DQN** | `highway-v0_wocar/` | `best_frame_3850000.pth` (`frame_3850000.pth`) | `3,850,000` | 18.00 | 0.06 | 17.70 |

## 100-Episode Evaluation Summary (`02_wocar_evaluation.ipynb`, seeds `0..99`, $\epsilon_{\text{train}} = 0.0025$)

| Model | Clean Return | Clean Crash | Random Noise ($\epsilon=0.0025$) | 10-step PGD ($\epsilon=0.0025$) | MinBest ($\epsilon=0.0025$) | Worst-Action PGD ($\epsilon=0.0025$) |
|---|---|---|---|---|---|---|
| **Vanilla DQN** | **16.67 ± 5.65** | 19.0% | 16.20 ± 6.12 (22.0% crash) | 15.33 ± 6.39 (30.0% crash) | 16.01 ± 6.04 (24.0% crash) | 14.75 ± 7.04 (31.0% crash) |
| **SA-DQN** | 16.46 ± 6.02 | 17.0% | 16.30 ± 6.04 (20.0% crash) | 13.86 ± 7.91 (33.0% crash) | 14.89 ± 7.31 (26.0% crash) | 14.66 ± 7.37 (28.0% crash) |
| **WocaR-DQN (Baseline)** | 16.60 ± 5.65 | **14.0%** | **16.98 ± 5.16 (11.0% crash)** | **16.13 ± 5.64 (18.0% crash)** | **16.03 ± 5.88 (19.0% crash)** | **15.86 ± 5.96 (21.0% crash)** |
