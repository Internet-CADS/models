# Rev3 runbook (for the agent on the training desktop)

Goal: train rev3 (`entries/05_wocar_dqn_r3/`, branch `wocar_r3`): **first a small pilot, then — only if the pilot passes — the full run**, then evaluate. Read `docs/rev3.md` for what rev3 is and why; this file is the step-by-step.

In the commands below `python` means the repo venv's interpreter (Windows: `.venv\Scripts\python.exe`, macOS/Linux: `.venv/bin/python`). Run commands from the repo root unless a step says `cd entries/05_wocar_dqn_r3` (evaluation scripts **must** run from inside that folder). Use `tmux`/separate terminals for anything long; a long job must not die with your shell.

## Rules
- Never delete or move anything under any `artifacts/` folder. Never edit `train.py`, `upstream/`, or the configs, except where a step below says so.
- Do not touch the repo-root `PROGRESS.md` / `progress.json` (rev2's monitor wrote them; rev2 is finished).
- Commit with explicit paths only (never `git add -A` / `git add .`). Keep LF line endings. Commit trailer: `Co-Authored-By: <your model name> <noreply@anthropic.com>`.
- **Do not push unless the user tells you to.** At the end, say what is committed and what to push.
- If something fails or the numbers look wrong, stop and report to the user; do not improvise fixes to the setup.

## Step 0 — rev2 must be finished
1. Confirm no `train.py` process is running and rev2's orchestrator/monitor windows have exited (Ctrl-C the rev2 monitor if it is still open: its progress pushes have been failing because `origin/wocar_r2` moved on; that is expected and harmless).
2. All 18 rev2 logs should end with `frame 4000000/4000000`. If not, stop and tell the user.

## Step 1 — get the rev3 branch
```
git fetch origin
git switch wocar_r3        # creates the local branch tracking origin/wocar_r3
git pull
```
If `git switch` complains about local changes to `PROGRESS.md` / `progress.json` (generated files, rev2 is done), run `git checkout -- PROGRESS.md progress.json` and retry. Local unpushed rev2 commits stay on your local `wocar_r2`; do not delete that branch.

## Step 2 — check the setup (no training yet)
```
python -c "import torch, gymnasium, highway_env; print('imports ok')"
python entries/05_wocar_dqn_r3/run_orchestrator.py --dry-run
```
The dry run must end with `DRY RUN OK`. Also note the number of logical CPU cores (`python -c "import os; print(os.cpu_count())"`) for Step 5.

## Step 3 — pilot (≈ 2.5–4 h; vanilla + SA-DQN + WocaR, seed 1, 600k frames, 3 runs in parallel)
```
python entries/05_wocar_dqn_r3/run_orchestrator.py --groups vanilla_dqn,sa_dqn,wocar_dqn --seeds 1 --concurrent --target-frames 600000 --run-root entries/05_wocar_dqn_r3/artifacts/pilot --no-commit
```
(`--run-root` is resolved relative to the folder you run from, so from the repo root it must be the full `entries/05_wocar_dqn_r3/artifacts/pilot`.) Optional live view in a second terminal:
`python entries/05_wocar_dqn_r3/monitoring.py --no-commit --run-root entries/05_wocar_dqn_r3/artifacts/pilot --target-frames 600000`.
Wait for the summary `3/3 runs finished`. If a run failed, show the user the last 40 lines of its `train.log` and stop. If the machine/shell restarts, run the same command again: it reattaches to live runs and skips finished ones.

## Step 4 — evaluate the pilot and decide
Do this only after the pilot training has finished (the evaluation competes for cores).
```
cd entries/05_wocar_dqn_r3
python evaluate.py artifacts/pilot --seeds 1 --final-frame 600000 --episodes 100 --workers 3
python pilot_gate.py artifacts/pilot
python report.py artifacts/pilot
```
`pilot_gate.py` prints a table and a verdict (exit code 0 = GO, 1 = NO-GO, 2 = data missing). Rule: **GO if SA-DQN or WocaR has a crash rate ≥ 10 points below vanilla** (mean of PGD and worst-Q at ε = 0.005), and vanilla's clean crash rate is ≤ 40%.
- **GO** → continue with Step 5. If the verdict also says WocaR looks diverged, first edit `config_highway_wocar.json` (the only config edit you may make): set `"worst_kappa_end": 1.0`, and commit that one-line change.
- **NO-GO** (or any other verdict) → **stop. Do not start the full run.** Give the user the printed table and the path of `artifacts/pilot/report.html`. Do not change settings yourself (planned next steps are κ 0.05, then a `pgd` bound-solver pilot, but that is the user's call).
Commit the pilot logs/results if you like (explicit paths: the pilot `train.log` files and `artifacts/eval_cache/pilot/`), but not checkpoints.

## Step 5 — full run (3 models × 3 seeds, 2,000,000 frames each)
Choose the mode from the core count and what else is running:
- 9 or more free logical cores: all 9 at once, about 8–10 h:
  `python entries/05_wocar_dqn_r3/run_orchestrator.py --concurrent --no-commit`
- otherwise one group of 3 at a time (vanilla → SA-DQN → WocaR), about 20 h:
  `python entries/05_wocar_dqn_r3/run_orchestrator.py --no-commit`

Monitor in a second terminal: `python entries/05_wocar_dqn_r3/monitoring.py --no-commit` (writes `entries/05_wocar_dqn_r3/PROGRESS.md`, not the repo-root file).
Runs land in `entries/05_wocar_dqn_r3/artifacts/run_1/<vanilla_dqn|sa_dqn|wocar_dqn>/model_<1..3>/`. Same restart rule as before: re-run the same command and it reattaches / skips finished runs.

Health checks (look at the last occurrence in each `train.log`):
- Around frame 400k and 700k, for the SA-DQN and WocaR runs: `avg cert reg loss` should be falling, not stuck near 1–2.
- For WocaR: `avg batch worst q` should not be far below about −200, and the mini-test return (`this test avg reward:`) should not collapse (below ~10).
If WocaR violates the second check, **report to the user right away** (include the numbers and the run); do not restart or edit anything on your own.

## Step 6 — evaluate and report the full run
When the orchestrator summary says `9/9 runs finished`:
```
cd entries/05_wocar_dqn_r3
python evaluate.py artifacts/run_1 --workers <number of cores, at most 6>
python report.py artifacts/run_1
```
This evaluates the final checkpoint (frame 2,000,000) of every run: PGD and worst-Q attacks at ε 0, 0.0025, 0.005, 0.0075, 0.01, 100 episodes, plus certification. Output: `artifacts/run_1/report.html`.
Commit with explicit paths: the nine `train.log` files, `artifacts/eval_cache/run_1/`, `artifacts/run_1/report.html`. Do **not** commit checkpoints (`*.pth`) or `.npy` files without asking.

## Step 7 — final message to the user
Send: what ran (pilot verdict table, full-run completion), the headline comparison from the report at ε = 0.005 and 0.01 (vanilla vs SA-DQN vs WocaR: return and crash rate under PGD / worst-Q), anything unusual (failed or restarted runs, divergence), what is committed locally, and what is left to push.

## Where things are
`entries/05_wocar_dqn_r3/README.md` (flags, details) · `docs/rev3.md` (changes, decisions) · `docs/deviations.md` (paper/authors' code differences) · `entries/05_wocar_dqn_r3/pilot_gate.py` (go/no-go).
