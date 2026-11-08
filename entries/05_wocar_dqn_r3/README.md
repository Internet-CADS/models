# 05_wocar_dqn_r3 — rev3 training + evaluation guide

Rev3 of the WocaR-DQN highway project. Goal: **a setup where robust training shows a visible gap to vanilla DQN, with much less compute than rev2.**
This folder is a copy of `04_wocar_dqn_r2` (code) with changed configs; see `../../docs/rev3.md` for every difference and the reasoning.
**Reward definition is unchanged** (collision −2.0, right lane 0.1, high speed 0.4, `normalize_reward: false`).

Status: prepared and smoke-tested, **nothing has been trained yet**. Production runs are started by you (see Launch).

## What runs
9 runs: 3 model types × 3 seeds, 2,000,000 frames each, checkpoint + mini-test every 50,000 frames (40 each).

| model | config | run dirs |
|---|---|---|
| vanilla DQN | `config_highway_vanilla.json` | `artifacts/run_1/vanilla_dqn/model_{1..3}/` |
| SA-DQN | `config_highway_sadqn.json` | `artifacts/run_1/sa_dqn/model_{1..3}/` |
| WocaR-DQN | `config_highway_wocar.json` | `artifacts/run_1/wocar_dqn/model_{1..3}/` |

Key settings (full table in `../../docs/rev3.md`): env = 25 other vehicles (obs shows 10 nearest), 60-step episodes; robust runs use ε = 0.005, κ = 0.02,
ε ramp over frames 100k–600k; WocaR additionally `worst_ratio` 0.1, `worst_eps_end` 0.001, `worst_kappa_end` 0.5 (worst-case loss **on**, RA-DQN default), `worst_schedule` 1,000,000.
Trainers run with `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, cwd = this folder.

## Recommended order (compute is limited)
1. **Dry run** (no training): `python run_orchestrator.py --dry-run`
2. **Pilot** (≈3–4 h, 3 runs in parallel): vanilla + SA-DQN + WocaR, seed 1, 600k frames, kept out of `run_1`:
   ```
   python run_orchestrator.py --groups vanilla_dqn,sa_dqn,wocar_dqn --seeds 1 --concurrent --target-frames 600000 --run-root artifacts/pilot --no-commit
   python evaluate.py artifacts/pilot --seeds 1 --final-frame 600000 --episodes 100
   python report.py artifacts/pilot
   ```
   Go ahead with the full run only if SA-DQN's **or** WocaR's crash rate under PGD / worst-Q at ε = 0.005 and 0.01 is clearly below vanilla's (at least ~10 percentage points; one seed cannot show significance).
   Also look at vanilla's **clean** crash rate in `train.log` / the report: if it is very high (say > 40 %), the traffic is probably too dense; lower `env_config.vehicles_count` (try 18) in all three configs.
3. **Full run**, then evaluate (below).

Gates while training (check `train.log`): after the ε ramp ends (~600k) `last cert reg loss` should fall well below the 1–2 it sat at in rev2;
for WocaR watch `avg batch worst q` (rev2 drifted to −300) and clean return; if either collapses by ~400k frames, restart that run with `worst_kappa_end` 1.0 (the upstream README's advice) and note it in `../../docs/rev3.md`.

## Launch (production)
Run from the repo root inside `tmux`, with the project venv (Windows: `.venv\Scripts\python.exe`, macOS/Linux: `.venv/bin/python`).
```
1) python entries/05_wocar_dqn_r3/run_orchestrator.py                 # window 1: launches the 9 runs (3 per batch, batches one after another)
   python entries/05_wocar_dqn_r3/run_orchestrator.py --concurrent    # …or all 9 at once (needs ≥ 9 free cores)
2) python entries/05_wocar_dqn_r3/monitoring.py                       # window 2: live bars + PROGRESS.md/progress.json (in this folder)
```
Rough wall time from rev2 speeds (vanilla ~93, SA ~77, WocaR ~72 frames/s at 6 concurrent): vanilla ≈ 6 h, SA-DQN ≈ 7 h, WocaR ≈ 8 h per run. Sequential batches ≈ 20 h, `--concurrent` ≈ 8–10 h. The denser traffic may be somewhat slower.

Orchestrator flags: `--dry-run`, `--resume` (reattach / skip finished; always on), `--retry N`, `--smoke` (3k frames into `artifacts/smoke/orch_flat`),
`--groups a,b`, `--seeds 1,2`, `--concurrent`, `--target-frames N`, `--run-root DIR`, `--no-commit`.
**By default the orchestrator commits each batch's `train.log` + TensorBoard files and the monitor commits/pushes `entries/05_wocar_dqn_r3/PROGRESS.md` + `progress.json` (not the repo-root files of rev2; `--progress-dir` overrides) on the branch checked out at start**
(`--no-commit` / `--no-push` to disable; `git push origin HEAD` sends every local commit on the branch). Checkpoints and `.npy` files are never committed by the scripts.
Monitoring flags: `--smoke`, `--publish-minutes M`, `--stall-minutes M`, `--refresh S`, `--no-commit`, `--no-push`, `--once`.

## Evaluate
From this folder, with the repo venv. Evaluation uses the **same frame for every run** (final = 2,000,000), no checkpoint selection.
```
python evaluate.py artifacts/run_1 --workers 6      # defaults: seeds 1-3, frame 2M, 100 episodes, PGD + worst-Q, eps 0/0.0025/0.005/0.0075/0.01, certification
python report.py artifacts/run_1                    # -> artifacts/run_1/report.html: mini-test curves per group + robustness section
python visualize_perturbation.py -o perturbation.html   # model-free view of what epsilon means (reads eps/env from config_highway_sadqn.json)
```
`evaluate.py` caches one JSON per (group, seed, frame, settings) in `artifacts/eval_cache/<run name>/` and skips cached jobs; extra epsilons can be added later
(`--epsilons 0.004,0.007`) and `report.py` merges them. Add `--frames final,peak` to also evaluate each group's peak-of-mean frame; add `--attacks pgd,worst_q,minbest,random_linf`
for more attacks. Per-job cost is dominated by the attacks (about 13 min per run at the old 28-sweep grid; the new default grid is smaller).
`report.py`'s mini-test section is read straight from `train.log` (50 episodes, seeds 1000–1049, every 50k frames).

## Before the real run
- Make sure the rev2 runs on the training machine are finished (shared cores; rev2's monitor owns the repo-root `PROGRESS.md`).
- Work on branch `wocar_r3` (created from `wocar_r2`): the orchestrator and monitor commit/push to whatever branch is checked out.
- `python run_orchestrator.py --dry-run` on the training machine, then the pilot, then the full run.

## Notes
- Use `flat_run_dir=true` (the orchestrator does) so each run writes straight into `artifacts/run_1/<group>/model_<i>/`; a manual `train.py --config …` keeps the nested default.
- `train.py` ramps `worst_eps`, `worst_ratio` and `worst_kappa` together with `coeff = (frame − schedule_start)/worst_schedule`; they cannot be delayed independently without a code change.
- ε-greedy exploration: 1.0 → 0.05 over frames 1k→51k, then linearly to 0 at the end (`act_epsilon_decay_zero: true`, upstream behaviour). Gradient clipping at 10 is active.
- The verification notes for the (shared) training code are in `../../docs/rev2_verification/`; `../../docs/rev3.md` has every rev3 change plus status and next steps; `../../docs/deviations.md` lists the differences from the paper and the authors' code.
- Nothing under `artifacts/` is ever deleted or moved by any script. `artifacts/smoke/` holds the three 4,000-frame smoke runs used to check the rev3 configs.
- Environment: see `requirements.txt` (torch CPU, highway-env, gymnasium, tensorboard, auto_LiRPA vendored in `auto_LiRPA/`).
