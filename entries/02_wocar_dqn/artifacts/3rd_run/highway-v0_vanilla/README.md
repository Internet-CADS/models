# 3rd run — 4M-frame Vanilla DQN highway training (`highway-v0_vanilla`)

Full non-robust Vanilla DQN (`robust: false, worst: false`) run on single-threaded CPU (`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`), `entries/02_wocar_dqn/train.py` with `config_highway_vanilla.json` (`num_frames=4000000`).
Started 2026-11-04 00:15, finished 2026-11-04 09:05 (~8.8h, 31,797s).

During the 4M-frame run, `mini_test` (every 50k frames, 50 deterministic episodes seeded `1000..1049`) evaluated the policy with `safety_max_crash_rate = 0.10` and `safety_score = test_reward - 5 * crash_rate`. The three key checkpoints kept here are:

| file | frame | test reward | crash rate | safety score | why kept |
|---|---|---|---|---|---|
| `frame_950000.pth` (`best_frame_950000.pth`) | 950,000 | 17.84 | **0.06** | 17.54 | lowest crash rate |
| `frame_2400000.pth` (`best_frame_2400000.pth`) | 2,400,000 | 18.26 | 0.08 | 17.86 | second-highest safety score and reward |
| `frame_4000000.pth` (`best_frame_4000000.pth`) | 4,000,000 (final) | **18.54** | 0.08 | **18.14** | highest reward, highest safety score, and final checkpoint |

Each `.pth` holds `current_model.features.state_dict()` (`torch.save`).
Load with the same `model_setup()` / `QNetwork` from `entries/02_wocar_dqn/`.

## Logs in this folder

- `train.log` — full training log (per-1000-frame scalars + every `mini_test`)
- `rewards_losses_so_far.pdf` — final reward/loss curves (`utils.plot`)
- `frame_{950000,2400000,4000000}_{losses,rewards}.npy` — raw curves up to that frame
- `tensorboard/` — event file with `episode/*`, `train/*`, `test/*` scalars;
  view with `tensorboard --logdir tensorboard`
