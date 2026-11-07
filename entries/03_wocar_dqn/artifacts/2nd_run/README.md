# 2nd run — 4M-frame WocaR highway training

Full robust run on GPU (`NVIDIA GeForce RTX 4060`), `entries/02_wocar_dqn/train.py`
with `config_highway_wocar.json` + override `training_config:num_frames=4000000`.
Started 2026-10-29 15:03, finished 2026-10-30 12:18 (~21h). `torch 2.5.1+cu121`.

No `best_frame_*.pth` was ever saved: the `mini_test` safety gate
(`crash_rate <= 0.05`, `safety_score = test_reward - 5 * crash_rate`,
`train.py` mini-test every 50k frames, 50 episodes) was never met —
crash rate bottomed out at 0.06. The three strongest checkpoints were
kept here instead, picked from the `mini_test` rows in `train.log`:

| file | frame | test reward | crash rate | safety score | why kept |
|---|---|---|---|---|---|
| `frame_600000.pth` | 600,000 | 17.13 | **0.06** | 16.83 | lowest crash of the whole run (just over the 0.05 gate) |
| `frame_2200000.pth` | 2,200,000 | 17.81 | 0.08 | 17.41 | second-highest safety score |
| `frame_4000000.pth` | 4,000,000 (final) | **18.07** | 0.08 | **17.67** | highest reward and safety score |

Each `.pth` holds `current_model.features.state_dict()` (`torch.save`).
Load with the same `model_setup()` / `QNetwork` from `entries/02_wocar_dqn/`.

## Logs in this folder

- `train.log` — full training log (per-1000-frame scalars + every `mini_test`)
- `rewards_losses_so_far.pdf` — final reward/loss curves (`utils.plot`)
- `frame_{600000,2200000,4000000}_{losses,rewards}.npy` — raw curves up to that frame
- `tensorboard/` — event file with `episode/*`, `train/*`, `test/*` scalars;
  view with `tensorboard --logdir tensorboard`
