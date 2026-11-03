# 3rd run — Matched 4M-frame highway-v0 ablation (Vanilla DQN vs. SA-DQN vs. WocaR-DQN)

Each model is trained for `4,000,000` frames on the exact same `highway-v0` configuration (`seed: 1`, 4 lanes, 10 vehicles, 50-dim normalized `Kinematics`, `collision_reward: -2.0`, `normalize_reward: false`, `offroad_terminal: true`) after fixing the WocaR `bound_forward`, `worst_action_select`, state-importance weighting $w(s)$, and deterministic `mini_test` seeding bugs.

## Folders

- `highway-v0_vanilla/` — Vanilla DQN (`config_highway_vanilla.json`, `robust: false, worst: false`)
- `highway-v0_sadqn/` — SA-DQN (`config_highway_sadqn.json`, `robust: true, worst: false`)
- `highway-v0_wocar/` — WocaR-DQN (`config_highway_wocar.json`, `robust: true, worst: true`)
