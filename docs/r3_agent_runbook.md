# Rev3 runbook (for the agent on the training desktop)

Goal: train rev3 (`entries/05_wocar_dqn_r3/`, branch `wocar_r3`): all 9 runs (vanilla / SA-DQN / WocaR × seeds 1–3, 2M frames each), then evaluate. No pilot and no go/no-go gate: run it, evaluate it, report the results. Read `docs/rev3.md` for what rev3 is and why; this file is the step-by-step.

In the commands below `python` means the repo venv's interpreter (Windows: `.venv\Scripts\python.exe`, macOS/Linux: `.venv/bin/python`). Run commands from the repo root unless a step says `cd entries/05_wocar_dqn_r3` (evaluation scripts **must** run from inside that folder). Use `tmux`/separate terminals for anything long; a long job must not die with your shell.

## Rules
- Never delete or move anything under any `artifacts/` folder. Never edit `train.py`, `upstream/` or the configs.
- Do not touch the repo-root `PROGRESS.md` / `progress.json` (rev2's monitor wrote them; rev2 is finished).
- Commit with explicit paths only (never `git add -A` / `git add .`). Keep LF line endings. Commit trailer: `Co-Authored-By: <your model name> <noreply@anthropic.com>`.
- **Do not push unless the user tells you to.** At the end, say what is committed and what is left to push.
- Do not kill or restart training runs on your own. If something fails or looks wrong, stop and report it to the user.

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
The dry run must end with `DRY RUN OK`.

## Step 3 — run the training (all 9 at once)
```
python entries/05_wocar_dqn_r3/run_orchestrator.py --concurrent --no-commit
```
All nine runs start together. The desktop has 6 cores / 12 threads and every run uses one thread (`OMP_NUM_THREADS=1`), so all nine fit on separate threads; in rev2 six runs used only about half the CPU. Some runs will share a physical core, so each run is probably somewhat slower than in rev2 (typically 15–30% for this kind of load; not measured here); expect roughly 8–11 h in total (an estimate, versus ~20 h for three sequential batches). A few minutes after the start, look at the `speed:` value in a `train.log` and tell the user (rev2 at six concurrent runs: about 93 / 77 / 72 frames/s for vanilla / SA-DQN / WocaR).
Monitor in a second terminal: `python entries/05_wocar_dqn_r3/monitoring.py --no-commit` (writes `entries/05_wocar_dqn_r3/PROGRESS.md`, not the repo-root file).
Runs land in `entries/05_wocar_dqn_r3/artifacts/run_1/<vanilla_dqn|sa_dqn|wocar_dqn>/model_<1..3>/`. If the machine or shell restarts, run the same command again: it reattaches to live runs and skips finished ones.

## Step 4 — evaluate and report
When the orchestrator summary says `9/9 runs finished`:
```
cd entries/05_wocar_dqn_r3
python evaluate.py artifacts/run_1 --workers 9        # one single-threaded job per run; lower it if other programs are using the CPU
python report.py artifacts/run_1
```
This evaluates the final checkpoint (frame 2,000,000) of every run, the same frame for all runs and with no checkpoint selection: PGD and worst-Q at ε 0, 0.0025, 0.005, 0.0075, 0.01, 100 episodes, plus certification. Output: `artifacts/run_1/report.html`.
Commit with explicit paths: the nine `train.log` files, `artifacts/eval_cache/run_1/`, `artifacts/run_1/report.html`. Do **not** commit checkpoints (`*.pth`) or `.npy` files without asking.

## Step 5 — message to the user
Send: the headline comparison from the report at ε = 0.005 and 0.01 (vanilla vs SA-DQN vs WocaR: return and crash rate under PGD / worst-Q) with the per-seed values and not only the means (seeds differ a lot in this kind of training), anything unusual (failed or restarted runs, a run whose mini-test return collapsed late in training), what is committed locally, and what is left to push.

## Where things are
`entries/05_wocar_dqn_r3/README.md` (flags, details) · `docs/rev3.md` (changes, decisions) · `docs/deviations.md` (paper/authors' code differences).
