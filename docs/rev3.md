# Rev3 — changes, status and next steps (`05_wocar_dqn_r3`)

Goal: a setup where robust training shows a visible gap to vanilla DQN, within a smaller compute budget.
Reward definition is **unchanged** (collision −2.0, right lane 0.1, high speed 0.4, `normalize_reward: false`).
Only the three `config_highway_*.json` files and two constants in `run_orchestrator.py` differ from r2 (this folder
is a copy of r2's code; `artifacts/` and the notebook were not copied).

| Setting | r2 | r3 | Config key | Files |
|---|---|---|---|---|
| Training frames | 4,000,000 | 2,000,000 (40 × 50,000) | `num_frames` | all |
| Seeds per group | 6 | 4 (12 single-threaded runs fill the desktop's 12 threads) | `N_RUNS` in `run_orchestrator.py` | — |
| Other vehicles (env) | 10 | 25 | `env_config.vehicles_count` (observation still shows the 10 nearest) | all |
| Episode length | 40 | 60 | `env_config.duration` | all |
| Training epsilon | 0.0025 | 0.005 | `epsilon` | sadqn, wocar |
| kappa | 0.001 | 0.02 | `kappa` | sadqn, wocar |
| Epsilon ramp | frames 50k–250k | 100k–600k | `schedule_start`, `schedule_length` | sadqn, wocar |
| Worst-critic mixing ratio | 0.1 | 0.1 (ramped with the eps ramp; authors use 0.01–0.1, RA-DQN 0.05 constant) | `worst_ratio` | wocar |
| Worst epsilon end (bound radius of the worst critic) | 0.0025 | 0.001 (RA-DQN uses a fixed 1e-3) | `worst_eps_end` | wocar |
| Worst ramp length | 200k | 1,000,000 | `worst_schedule` | wocar |
| Worst-case loss weight (`worst_kappa_end`) | 1.0 (off) | 0.5 | `worst_kappa_end` | wocar |

Notes
- `train.py` ramps `worst_eps`, `worst_ratio` and `worst_kappa` together: `coeff = (frame − schedule_start) / worst_schedule`.
  They therefore start with the epsilon ramp (frame 100k) and reach their end values at frame 1.1M.
- Checkpoint and mini-test interval stay at 50,000 (unchanged from r2), so the final checkpoint `frame_2000000.pth` is also the last mini-test. 40 checkpoints per run.
- Episode length (60) is not aligned with frames on purpose: crashes end episodes early, so training episodes vary in length.
- Evaluation scripts were adapted for r3: `evaluate.py` defaults to seeds 1–4, `--frames final` (2,000,000, `--final-frame` overrides), epsilons 0/0.0025/0.005/0.0075/0.01, attacks PGD + worst-Q;
  `report.py` uses `TRAIN_EPS = 0.005`, 2M frames, 4 seeds, shows the final-checkpoint robustness section first and skips attacks that were not evaluated.
  Fixed along the way: `evaluate.py` crashed (after finishing the job) when the run folder was outside the entry dir.
- Launcher additions: `run_orchestrator.py --groups --seeds --concurrent`; `monitoring.py` uses 12 runs × 2M frames. `smoke_test.py` (rev2 gate tooling) was not copied.

## Status (prepared, not yet trained)
- Committed on branch `wocar_r3`. Folder = copy of the r2 code with 3 changed configs, 9-run launcher/monitor constants, new orchestrator flags (`--groups --seeds --concurrent`), and rev3 defaults in `evaluate.py` / `report.py`.
- Verified on macOS: `run_orchestrator.py --dry-run`; 4,000-frame smoke runs of all three configs (WocaR with the ramps pulled forward so the worst-case-loss path runs: finite losses); the full pipeline smoke (`--smoke`, 9 runs × 3,000 frames, monitor shows 9/9 done); `evaluate.py` + `report.py` end to end on the smoke checkpoints; `visualize_perturbation.py` on the rev3 env.
- **Not verified:** anything about learning, speed on the training machine with 25 vehicles, whether 25 vehicles is too dense, whether the settings produce a gap.
- The monitor writes `PROGRESS.md`/`progress.json` into this folder (smoke: `artifacts/smoke/`) with its own lock file, so it cannot clobber rev2's repo-root files.

## Decisions already made (do not relitigate without the user)
- Reward definition unchanged; lanes unchanged (ε keeps its physical meaning through the y-normalisation).
- 2,000,000 frames, checkpoint + mini-test every 50,000 (final checkpoint = last mini-test); episode length 60 is deliberately not aligned with frames (crashes end episodes early).
- 4 seeds per group (12 runs = all 12 threads of the desktop; costs some speed per run); evaluate two views by default: the final frame of every run (same frame, no selection) and each group's peak-of-mean frame (`evaluate.py --frames final,peak`).
- WocaR: worst-case loss on at `worst_kappa_end` 0.5 (RA-DQN default; upstream README advises 1.0 because of instability), `worst_ratio` 0.1, `worst_eps_end` 0.001. See `deviations.md`.

## Next steps
1. Make sure the rev2 runs on the training machine are finished; the machine pulls `wocar_r3`.
2. Full run: `run_orchestrator.py --concurrent` (all 12 runs at once), `monitoring.py` in a second window. Then `evaluate.py` and `report.py`, and look at the results. Step-by-step: `r3_agent_runbook.md`.
3. Ideas if the results show no gap (the user decides): κ 0.05; one SA-DQN run with `bound_solver: pgd`; if WocaR diverges, `worst_kappa_end` 1.0 (upstream README advice). Deferred: Double DQN.

## Rules carried over from the user
- Never delete or move anything under `artifacts/`; commit only what you produced with explicit pathspecs; do not touch the repo-root `PROGRESS.md` / `progress.json` (rev2's monitor writes them); do not add the large cumulative `.npy` files.
- Commit trailer: `Co-Authored-By: <model name> <noreply@anthropic.com>`. The repo stores LF line endings.
- Report outcomes faithfully (failures included); ask before anything outward-facing (push, publish).

