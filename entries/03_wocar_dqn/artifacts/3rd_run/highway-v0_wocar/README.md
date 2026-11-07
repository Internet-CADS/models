# 3rd run — 4M-frame WocaR-DQN highway training (`highway-v0_wocar`)

Full Worst-Case-Aware Robust DQN (`robust: true, worst: true, worst_rate: 4, worst_ratio: 0.1, worst_kappa_end: 1.0`) run on single-threaded CPU (`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`), `entries/02_wocar_dqn/train.py` with `config_highway_wocar.json` (`num_frames=4000000`).
Started 2026-11-04 00:15, finished 2026-11-04 11:35 (~11.3h, 40,760s).

During the 4M-frame run, `mini_test` (every 50k frames, 50 deterministic episodes seeded `1000..1049`) evaluated the policy with `safety_max_crash_rate = 0.10` and `safety_score = test_reward - 5 * crash_rate`. The key checkpoints kept here are:

| file | frame | test reward | crash rate | safety score | why kept |
|---|---|---|---|---|---|
| `frame_1350000.pth` (`best_frame_1350000.pth`) | 1,350,000 | 17.41 | **0.06** | 17.11 | tied lowest crash rate |
| `frame_2300000.pth` (`best_frame_2300000.pth`) | 2,300,000 | 17.60 | 0.08 | 17.20 | second-highest safety score and reward |
| `frame_3850000.pth` (`best_frame_3850000.pth`) | 3,850,000 | **18.00** | **0.06** | **17.70** | lowest crash rate, highest reward, and highest safety score |
| `frame_4000000.pth` | 4,000,000 (final) | 16.42 | 0.18 | 15.52 | final checkpoint |

Each `.pth` holds `current_model.features.state_dict()` (`torch.save`).
Load with the same `model_setup()` / `QNetwork` from `entries/02_wocar_dqn/`.

## Logs in this folder

- `train.log` — full training log (per-1000-frame scalars + every `mini_test`)
- `rewards_losses_so_far.pdf` — final reward/loss curves (`utils.plot`)
- `frame_{1350000,2300000,3850000,4000000}_{losses,rewards}.npy` — raw curves up to that frame
- `tensorboard/` — event file with `episode/*`, `train/*`, `test/*` scalars;
  view with `tensorboard --logdir tensorboard`
