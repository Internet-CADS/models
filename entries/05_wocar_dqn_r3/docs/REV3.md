# Rev3 — what changed relative to `04_wocar_dqn_r2`

Goal: a setup where robust training shows a visible gap to vanilla DQN, within a smaller compute budget.
Reward definition is **unchanged** (collision −2.0, right lane 0.1, high speed 0.4, `normalize_reward: false`).
Only the three `config_highway_*.json` files and two constants in `run_orchestrator.py` differ from r2 (this folder
is a copy of r2's code; `artifacts/` and the notebook were not copied).

| Setting | r2 | r3 | Config key | Files |
|---|---|---|---|---|
| Training frames | 4,000,000 | 2,000,000 (40 × 50,000) | `num_frames` | all |
| Seeds per group | 6 | 3 | `N_RUNS` in `run_orchestrator.py` | — |
| Other vehicles (env) | 10 | 25 | `env_config.vehicles_count` (observation still shows the 10 nearest) | all |
| Episode length | 40 | 60 | `env_config.duration` | all |
| Training epsilon | 0.0025 | 0.005 | `epsilon` | sadqn, wocar |
| kappa | 0.001 | 0.02 | `kappa` | sadqn, wocar |
| Epsilon ramp | frames 50k–250k | 100k–600k | `schedule_start`, `schedule_length` | sadqn, wocar |
| Worst-critic mixing ratio | 0.1 | 0.3 | `worst_ratio` | wocar |
| Worst epsilon end | 0.0025 | 0.005 | `worst_eps_end` | wocar |
| Worst ramp length | 200k | 1,000,000 | `worst_schedule` | wocar |
| Worst-case loss weight (`worst_kappa_end`) | 1.0 (off) | 0.5 | `worst_kappa_end` | wocar |

Notes
- `train.py` ramps `worst_eps`, `worst_ratio` and `worst_kappa` together: `coeff = (frame − schedule_start) / worst_schedule`.
  They therefore start with the epsilon ramp (frame 100k) and reach their end values at frame 1.1M.
- Checkpoint and mini-test interval stay at 50,000 (unchanged from r2), so the final checkpoint `frame_2000000.pth` is also the last mini-test. 40 checkpoints per run.
- Episode length (60) is not aligned with frames on purpose: crashes end episodes early, so training episodes vary in length.
- Planned gates: pilot of 1 seed vanilla + SA-DQN at ~600k frames (commands in README); watch `cert reg loss` falling after the ramp; if WocaR's
  worst-Q diverges or clean return collapses by ~400k, restart that run with `worst_kappa_end = 0.9` (or `worst_ratio = 0`).
- Evaluation scripts were adapted for r3: `evaluate.py` defaults to seeds 1–3, `--frames final` (2,000,000, `--final-frame` overrides), epsilons 0/0.0025/0.005/0.0075/0.01, attacks PGD + worst-Q;
  `report.py` uses `TRAIN_EPS = 0.005`, 2M frames, 3 seeds, shows the final-checkpoint robustness section first and skips attacks that were not evaluated.
  Fixed along the way: `evaluate.py` crashed (after finishing the job) when the run folder was outside the entry dir.
- Launcher additions: `run_orchestrator.py --groups --seeds --concurrent`; `monitoring.py` uses 9 runs × 2M frames. `smoke_test.py` (rev2 gate tooling) was not copied.
