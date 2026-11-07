# 3rd run — 4M-frame SA-DQN highway training (`highway-v0_sadqn`)

Full state-adversarial DQN (`robust: true, worst: false`) run on single-threaded CPU (`CUDA_VISIBLE_DEVICES="-1"`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`), `entries/02_wocar_dqn/train.py` with `config_highway_sadqn.json` (`num_frames=4000000`).
Started 2026-11-03 16:30, finished 2026-11-04 07:02 (~14.5h, 52,302s). `torch 2.5.1+cu121`.

During the 4M-frame run, `mini_test` (every 50k frames, 50 deterministic episodes seeded `1000..1049`) evaluated the policy with `safety_max_crash_rate = 0.10` and `safety_score = test_reward - 5 * crash_rate`. Fifteen `best_frame_*.pth` milestones were reached, peaking at frame `2,450,000` (`crash_rate = 0.04`, `test_reward = 18.41`, `safety_score = 18.21`). The three key checkpoints kept here are:

| file | frame | test reward | crash rate | safety score | why kept |
|---|---|---|---|---|---|
| `frame_2350000.pth` (`best_frame_2350000.pth`) | 2,350,000 | 18.18 | 0.08 | 17.78 | second-highest safety score and reward |
| `frame_2450000.pth` (`best_frame_2450000.pth`) | 2,450,000 | **18.41** | **0.04** | **18.21** | lowest crash rate, highest reward, and highest safety score |
| `frame_4000000.pth` | 4,000,000 (final) | 17.32 | 0.14 | 16.62 | final checkpoint |

Each `.pth` holds `current_model.features.state_dict()` (`torch.save`).
Load with the same `model_setup()` / `QNetwork` from `entries/02_wocar_dqn/`.

## Logs in this folder

- `train.log` — full training log (per-1000-frame scalars + every `mini_test`)
- `rewards_losses_so_far.pdf` — final reward/loss curves (`utils.plot`)
- `frame_{2350000,2450000,4000000}_{losses,rewards}.npy` — raw curves up to that frame
- `tensorboard/` — event file with `episode/*`, `train/*`, `test/*` scalars;
  view with `tensorboard --logdir tensorboard`
