# Rev3 runbook (for the agent on the training desktop)

This file: https://github.com/Internet-CADS/models/blob/wocar_r3/docs/r3_agent_runbook.md  
If your clone is on another branch (e.g. `wocar_r2`) the file is not in your working tree yet; read it with `git fetch origin && git show origin/wocar_r3:docs/r3_agent_runbook.md`.

Goal: train rev3 (`entries/05_wocar_dqn_r3/`, branch `wocar_r3`): all 12 runs (vanilla / SA-DQN / WocaR × seeds 1–4, 2M frames each), then evaluate. No pilot and no go/no-go gate: run it, evaluate it, report the results. Read `docs/rev3.md` for what rev3 is and why; this file is the step-by-step.

In the commands below `python` means the repo venv's interpreter (Windows: `.venv\Scripts\python.exe`, macOS/Linux: `.venv/bin/python`). Run commands from the repo root unless a step says `cd entries/05_wocar_dqn_r3` (evaluation scripts **must** run from inside that folder). Use `tmux`/separate terminals for anything long; a long job must not die with your shell.

## Rules
- Never delete or move anything under any `artifacts/` folder. Never edit `train.py`, `upstream/` or the configs.
- Do not touch `main`. **Never force-push anything.**
- Commit with explicit paths only (never `git add -A` / `git add .`). Keep LF line endings. Commit trailer: `Co-Authored-By: <your model name> <noreply@anthropic.com>`.
- **Do not push unless the user tells you to**, with two exceptions the user has authorised: the git operations in Step 1 (push `wocar_r3`, delete the branch `wocar_r2`), and the monitor's automatic progress commits/pushes during the run (Step 3). Any other push needs the user's go-ahead; at the end say what is committed and what is left to push.
- Do not kill or restart training runs on your own. If something fails or looks wrong, stop and report it to the user.

## Step 0 — rev2 must be finished
1. Confirm no `train.py` process is running; stop rev2's orchestrator and monitor windows (Ctrl-C). The monitor's progress pushes have been failing because `origin/wocar_r2` moved on; that is expected and harmless.
2. All 18 rev2 logs should end with `frame 4000000/4000000`. If not, stop and tell the user.

## Step 1 — publish the rev2 results into `wocar_r3`, then delete `wocar_r2` (authorised by the user)
Why: the user wants the final rev2 results on the remote (so they can be evaluated from another machine) inside `wocar_r3`, which then holds everything about rev2 and rev3 and is merged into `main` later (by the user; do not touch `main`). `origin/wocar_r3` already contains everything that is on `origin/wocar_r2`; your local `wocar_r2` has diverged from the remote one (its own unpushed progress/data commits), which is why pushes from it fail.

**1a. Commit the rev2 data that is not committed yet (on `wocar_r2`).** Look first: `git status --short entries/04_wocar_dqn_r2/artifacts/run_1`. Add only these kinds of files, with explicit pathspecs (the quotes matter):
```
git add -- "entries/04_wocar_dqn_r2/artifacts/run_1/*/model_*/train.log" "entries/04_wocar_dqn_r2/artifacts/run_1/*/model_*/frame_*.pth" "entries/04_wocar_dqn_r2/artifacts/run_1/*/model_*/frame_4000000_losses.npy" "entries/04_wocar_dqn_r2/artifacts/run_1/*/model_*/frame_4000000_rewards.npy" "entries/04_wocar_dqn_r2/artifacts/run_1/*/model_*/tensorboard/*"
git commit -m "data(rev2): final WocaR results (logs, checkpoints, final losses/rewards)"   # + the Co-Authored-By trailer
```
Do **not** add the other, cumulative `frame_<N>_losses.npy` / `frame_<N>_rewards.npy` files (about 15 GB), and nothing outside `entries/04_wocar_dqn_r2/artifacts/run_1/`. If `git status` shows nothing new to add, skip the commit.

**1b. Merge into `wocar_r3`.**
```
git fetch origin
git switch wocar_r3            # creates the local branch tracking origin/wocar_r3
git pull
git merge wocar_r2 -m "Merge rev2 results (desktop wocar_r2) into wocar_r3"
```
If `git switch` complains about local changes to `PROGRESS.md` / `progress.json` (generated files), run `git checkout -- PROGRESS.md progress.json` and retry. The merge is expected to be conflict-free (tested with a simulated rev2 history). If it conflicts **only** in `PROGRESS.md` / `progress.json`: `git checkout --ours -- PROGRESS.md progress.json`, `git add PROGRESS.md progress.json`, `git commit --no-edit`. Any other conflict: stop and report to the user.

**1c. Check before pushing.** `git diff --stat origin/wocar_r3..HEAD | tail -5` and `git diff --name-only origin/wocar_r3..HEAD`. The changed files should be under `entries/04_wocar_dqn_r2/artifacts/run_1/` plus `PROGRESS.md` / `progress.json`; there must be no `.npy` file other than `frame_4000000_losses.npy` / `frame_4000000_rewards.npy`, and no more than about 1,500 changed files. If anything else shows up, stop and report before pushing.

**1d. Push.** `git push origin wocar_r3` (a normal push, never `--force`).

**1e. Delete `wocar_r2`, only after both checks print OK.**
```
git fetch origin
git merge-base --is-ancestor wocar_r2 origin/wocar_r3 && echo LOCAL_OK
git merge-base --is-ancestor origin/wocar_r2 origin/wocar_r3 && echo REMOTE_OK
```
Only if `LOCAL_OK` and `REMOTE_OK` both appear: `git push origin --delete wocar_r2`, then `git branch -d wocar_r2`. If either check fails: stop, delete nothing, report to the user.

**1f.** Tell the user: the rev2 results are on `origin/wocar_r3` (give the commit hash from `git log -1 --oneline`) and `wocar_r2` is deleted. You are now on `wocar_r3`; continue with Step 2.

## Step 2 — check the setup (no training yet)
```
python -c "import torch, gymnasium, highway_env; print('imports ok')"
python entries/05_wocar_dqn_r3/run_orchestrator.py --dry-run
```
The dry run must end with `DRY RUN OK`.

## Step 3 — run the training (all 12 at once)
```
python entries/05_wocar_dqn_r3/run_orchestrator.py --concurrent --no-commit
```
All twelve runs start together. **Measured at the start of the real run:** about 20 frames/s per run, projected at roughly 28 h for all 12 (the earlier estimate of 9–12 h was wrong: the 25-vehicle traffic makes the simulator ~2.7× slower per step than rev2's 10 vehicles, and the 12 hyperthreaded runs together gave no more throughput than the 6 physical cores; see `docs/rev3.md`, section Compute). A few minutes after the start, look at the `speed:` value in a `train.log` and tell the user. The machine will be fully loaded, so avoid other heavy programs.
Start the monitor in a second terminal: `python entries/05_wocar_dqn_r3/monitoring.py`. Every 15 minutes it writes `entries/05_wocar_dqn_r3/PROGRESS.md` and `progress.json` (not the repo-root files), commits **only those two files** and pushes `wocar_r3` (these progress pushes are authorised). If a push is rejected because the remote branch moved (the user may push from another machine), the monitor rebases onto it and retries by itself; if that ever fails it aborts cleanly and tries again at the next publish. Leave it running until all 12 runs are done. Note that its push sends every commit that exists locally on `wocar_r3` at that moment.
Runs land in `entries/05_wocar_dqn_r3/artifacts/run_1/<vanilla_dqn|sa_dqn|wocar_dqn>/model_<1..4>/`. If the machine or shell restarts, run the same command again: it reattaches to live runs and skips finished ones.

## Step 4 — evaluate and report
When the orchestrator summary says `12/12 runs finished`, first stop the monitor (Ctrl-C; it writes a final snapshot) so that nothing pushes automatically from here on, then:
```
cd entries/05_wocar_dqn_r3
python evaluate.py artifacts/run_1 --workers 12       # one single-threaded job per run; lower it if other programs are using the CPU
python report.py artifacts/run_1
```
This evaluates two views: the final checkpoint (frame 2,000,000) of every run (same frame for all, no selection), and each group's peak-of-mean checkpoint (the frame where the seed-mean mini-test return is highest; the script picks it from the logs). 12 jobs per view (up to 24). PGD and worst-Q at ε 0, 0.0025, 0.005, 0.0075, 0.01, 100 episodes, plus certification. Output: `artifacts/run_1/report.html`.
Commit with explicit paths: the twelve `train.log` files, `artifacts/eval_cache/run_1/`, `artifacts/run_1/report.html`. Do **not** commit checkpoints (`*.pth`) or `.npy` files without asking.

## Step 5 — message to the user
Send: the headline comparison from the report at ε = 0.005 and 0.01 (vanilla vs SA-DQN vs WocaR: return and crash rate under PGD / worst-Q), for both views (final frame and peak frame) with the per-seed values and not only the means (seeds differ a lot in this kind of training), anything unusual (failed or restarted runs, a run whose mini-test return collapsed late in training), what is committed locally, and what is left to push.

## Where things are
`entries/05_wocar_dqn_r3/README.md` (flags, details) · `docs/rev3.md` (changes, decisions) · `docs/deviations.md` (paper/authors' code differences).
