# Rev3 — changes, status and next steps (`05_wocar_dqn_r3`)

Goal: a setup where robust training shows a visible gap to vanilla DQN, within a smaller compute budget.
Reward definition is **unchanged** (collision −2.0, right lane 0.1, high speed 0.4, `normalize_reward: false`).
Only the three `config_highway_*.json` files and two constants in `run_orchestrator.py` differ from r2 (this folder
is a copy of r2's code; `artifacts/` and the notebook were not copied).

| Setting | r2 | r3 | Config key | Files |
|---|---|---|---|---|
| Training frames | 4,000,000 | 2,000,000 (40 × 50,000) | `num_frames` | all |
| Seeds per group | 6 | 3 (4 was tried: 12 concurrent runs were too slow, hyperthreads added no throughput) | `N_RUNS` in `run_orchestrator.py` | — |
| Other vehicles (env) | 10 | 15 (25 was tried first and aborted, see Compute) | `env_config.vehicles_count` | all |
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
- Evaluation scripts were adapted for r3: `evaluate.py` defaults to seeds 1–3, `--frames peak` (the group's peak-of-mean frame; `final` / integer frames are optional), epsilons 0/0.0025/0.005/0.0075/0.01, attacks PGD + worst-Q + MinBest + random L∞;
  `report.py` uses `TRAIN_EPS = 0.005`, 2M frames, 3 seeds, shows the peak-checkpoint robustness section and skips attacks that were not evaluated.
  Fixed along the way: `evaluate.py` crashed (after finishing the job) when the run folder was outside the entry dir.
- Launcher additions: `run_orchestrator.py --groups --seeds --concurrent`; `monitoring.py` uses 9 runs × 2M frames. `smoke_test.py` (rev2 gate tooling) was not copied.

## Status (prepared, not yet trained)
- Committed on branch `wocar_r3`. Folder = copy of the r2 code with 3 changed configs, 9-run launcher/monitor constants, new orchestrator flags (`--groups --seeds --concurrent`), and rev3 defaults in `evaluate.py` / `report.py`.
- Verified on macOS: `run_orchestrator.py --dry-run`; 4,000-frame smoke runs of all three configs (WocaR with the ramps pulled forward so the worst-case-loss path runs: finite losses); the full pipeline smoke (`--smoke`, 9 runs × 3,000 frames, monitor shows 9/9 done); `evaluate.py` + `report.py` end to end on the smoke checkpoints; `visualize_perturbation.py` on the rev3 env.
- **Not verified:** anything about learning, speed on the training machine, whether the settings produce a gap.
- The monitor writes `PROGRESS.md`/`progress.json` into this folder (smoke: `artifacts/smoke/`) with its own lock file, so it cannot clobber rev2's repo-root files.

## Decisions already made (do not relitigate without the user)
- Reward definition unchanged; lanes unchanged (ε keeps its physical meaning through the y-normalisation).
- 2,000,000 frames, checkpoint + mini-test every 50,000 (final checkpoint = last mini-test); episode length 60 is deliberately not aligned with frames (crashes end episodes early).
- 3 seeds per group; evaluate each group's peak-of-mean frame (the only view in the report; `evaluate.py --frames peak`, the default). The final frame was also evaluated once (cached, not reported).
- WocaR: worst-case loss on at `worst_kappa_end` 0.5 (RA-DQN default; upstream README advises 1.0 because of instability), `worst_ratio` 0.1, `worst_eps_end` 0.001. See `deviations.md`.

## Compute (measured on 2026-11-09)
Simulator speed, one thread, random actions, no network (Mac; episode length does not matter, vehicle count does):

| vehicles | env steps/s | vs rev2 (10 vehicles) |
|---|---|---|
| 10 | 129 | 1.0× |
| 15 | 88 | 1.5× slower |
| 18 | 72 | 1.8× slower |
| 20 | 63 | 2.1× slower |
| 25 | 47 | 2.7× slower |

First snapshot of the real run (12 concurrent runs, 25 vehicles, desktop with 6 cores / 12 threads): about 20 frames/s per run (rev2: 93 / 77 / 72 for vanilla / SA / WocaR with 6 runs), projected ≈ 28 h. The total (~240 frames/s) is about what 6 physical cores give, so hyperthreads added nothing visible and wall time scales with the number of runs.
Rough projections from this (estimates; per-run speed alone ≈ 1 / (simulator time + ~5 ms of training) and capacity ≈ 6 runs' worth):

| runs \ vehicles | 25 | 20 | 18 | 15 | 10 |
|---|---|---|---|---|---|
| 12 | ~29 h | ~23 h | ~21 h | ~18 h | ~14 h |
| 9 | ~22 h | ~17 h | ~16 h | ~13 h | ~10 h |
| 6 | ~15 h | ~12 h | ~11 h | ~9 h | ~7 h |

The earlier plan said 9–12 h for 12 runs; that came from rev2's speeds and ignored the simulator cost of the denser traffic.

**Scheduling (estimates).** With 9 jobs on 12 threads, three cores carry one job and three carry two; a shared core gives each job about half speed. Approximate CPU time per run at 15 vehicles (from rev2 speeds): vanilla ~8 h, SA-DQN ~9.3 h, WocaR ~9.8 h; a pair on one core finishes after the sum of its two jobs.

| Setup | WocaR done | All done |
|---|---|---|
| default OS placement | ~10 h if alone, 17–19 h if paired | ~17–19 h |
| best static pinning (WocaR alone, others paired) | ~9.8 h | ~17.3 h |
| one thread per core (affinity 0x555), fair time-slicing | ~13.8 h | ~13.8 h |
| the same + WocaR at above-normal priority | ~10 h | ~14 h |

The runbook (Step 3b) applies the last row. It assumes hyperthreads add no throughput (inferred from the first snapshot of the 12-run attempt) and adjacent-pair numbering of logical processors.

**Decision:** the first launch (25 vehicles, 12 runs, `artifacts/run_1`) was stopped after about 40 minutes at ~20 frames/s per run (~28 h projected). It was restarted with 15 vehicles and 9 runs (3 seeds) in `artifacts/run_2` (~13 h projected, ~40 frames/s per run expected). `artifacts/run_1` is kept untouched as the aborted attempt and is not used for any result.

## Extra seeds (seeds 4–7)
Why: with 3 seeds per group the seed-to-seed spread of crash rates under attack (5–12 points) dominates the comparison; the group-mean uncertainty is about ±5.5 points (estimate), and more evaluation episodes cannot reduce it. Plan: seeds 4 and 5 on the desktop (6 runs, ~10 h; the agent is pointed at `docs/desktop_extra_seeds.md`, a self-contained task file) and seeds 6 and 7 on the laptop (M1 Pro, 8 performance cores; 6 runs, ~9–11 h estimated, run with `caffeinate`). Expected result: 7 seeds per group, group-mean uncertainty about ±3.6 points.

Laptop (from `entries/05_wocar_dqn_r3/`, plugged in, nothing else heavy running):
```
caffeinate -i ../../.venv/bin/python run_orchestrator.py --concurrent --seeds 6,7 --no-commit
```
Evaluation after both batches are on the laptop (`git pull` first): `evaluate.py artifacts/run_2 --episodes 300 --workers 9` discovers all seeds present (1–7). The peak frame is the argmax of the seed-mean mini-test curve, so adding seeds can move it; results are cached per (group, seed, frame), so seeds 1–3 are re-evaluated only if the peak frame changes (up to 21 jobs, ~2.3 h at 300 episodes; otherwise 12 jobs, ~1.5 h).

## Next steps
1. Make sure the rev2 runs on the training machine are finished; the machine pulls `wocar_r3`.
2. Full run: `run_orchestrator.py --concurrent` (all 9 runs at once), `monitoring.py` in a second window. Then `evaluate.py` and `report.py`, and look at the results. Step-by-step: `r3_agent_runbook.md`.
3. Ideas if the results show no gap (the user decides): κ 0.05; one SA-DQN run with `bound_solver: pgd`; if WocaR diverges, `worst_kappa_end` 1.0 (upstream README advice). Deferred: Double DQN.

## Rules carried over from the user
- Never delete or move anything under `artifacts/`; commit only what you produced with explicit pathspecs; do not touch the repo-root `PROGRESS.md` / `progress.json` (rev2's monitor writes them); do not add the large cumulative `.npy` files.
- Commit trailer: `Co-Authored-By: <model name> <noreply@anthropic.com>`. The repo stores LF line endings.
- Report outcomes faithfully (failures included); ask before anything outward-facing (push, publish).

