# Rev3 runbook (for the agent on the training desktop)

This file: https://github.com/Internet-CADS/models/blob/wocar_r3/docs/archive/r3_agent_runbook.md  
If your clone is on another branch (e.g. `wocar_r2`) the file is not in your working tree yet; read it with `git fetch origin && git show origin/wocar_r3:docs/archive/r3_agent_runbook.md`.

Goal: train rev3 (`entries/05_wocar_dqn_r3/`, branch `wocar_r3`): all 9 runs (vanilla / SA-DQN / WocaR × seeds 1–3, 2M frames each), then evaluate. No pilot and no go/no-go gate: run it, evaluate it, report the results. Read `docs/rev3.md` for what rev3 is and why; this file is the step-by-step.

In the commands below `python` means the repo venv's interpreter (Windows: `.venv\Scripts\python.exe`, macOS/Linux: `.venv/bin/python`). Run commands from the repo root unless a step says `cd entries/05_wocar_dqn_r3` (evaluation scripts **must** run from inside that folder). Use `tmux`/separate terminals for anything long; a long job must not die with your shell.

## Rules
- Never delete or move anything under any `artifacts/` folder. Never edit `train.py`, `upstream/` or the configs.
- Do not touch `main`. **Never force-push anything.**
- Commit with explicit paths only (never `git add -A` / `git add .`). Keep LF line endings. Commit trailer: `Co-Authored-By: <your model name> <noreply@anthropic.com>`.
- **Do not push unless the user tells you to**, with two exceptions the user has authorised: the git operations in Step 1 (push `wocar_r3`, delete the branch `wocar_r2`), the monitor's automatic progress commits/pushes during a run (Step 3, and the extra-seeds batch in `docs/archive/desktop_extra_seeds.md`), and the final push of the extra-seeds results described in that file. Any other push needs the user's go-ahead; at the end say what is committed and what is left to push.
- Do not kill or restart training runs on your own (except as Step R says). The user has also authorised changing the CPU affinity and priority of the `train.py` processes as described in Step 3b, and nothing else about running processes. If something fails or looks wrong, stop and report it to the user.

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

## Step R — only if you already started the run with 25 vehicles (authorised by the user)
The first launch used 25 vehicles in `artifacts/run_1` and is too slow (about 28 h). **The user has authorised stopping it (all 12 runs, including the seed-4 ones) and restarting once with 15 vehicles and 3 seeds (9 runs).** If your runs are already writing to `entries/05_wocar_dqn_r3/artifacts/run_2` with 15 vehicles, skip this step.
1. Stop the monitor and the orchestrator (Ctrl-C in their windows). Then stop all trainer processes (only rev3's `train.py` processes should exist; rev2 is finished):
   - Windows PowerShell: `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'train\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }`
   - macOS/Linux: `pkill -f train.py`
   Check that no `train.py` process is left (Task Manager / `tasklist`).
2. Leave `entries/05_wocar_dqn_r3/artifacts/run_1` exactly as it is (the aborted attempt): do not delete, move or commit it.
3. `git pull`. Check that the environment `vehicles_count` is now 15 in all three configs: `grep -n vehicles_count entries/05_wocar_dqn_r3/config_highway_*.json` (each file shows 15 for the environment and 10 for the observation block).
4. Go to Step 3: it starts the new run in `artifacts/run_2` (the default run root now), then do Step 3b.

## Step 3 — run the training (all 9 at once)
```
python entries/05_wocar_dqn_r3/run_orchestrator.py --concurrent --no-commit
```
All nine runs start together. Expected: about 40 frames/s per run and about 13 h in total (15-vehicle traffic; the hyperthreaded runs together give about 6 cores' worth of throughput, so time scales with the number of runs). A few minutes after the start, look at the `speed:` value in a `train.log` and tell the user; if it is well below ~35 frames/s, say so. (A first attempt with 25 vehicles and 12 runs ran at only ~20 frames/s per run, ~28 h projected, and was stopped; see `docs/rev3.md`, section Compute.)
Start the monitor in a second terminal: `python entries/05_wocar_dqn_r3/monitoring.py`. Every 15 minutes it writes `entries/05_wocar_dqn_r3/PROGRESS.md` and `progress.json` (not the repo-root files), commits **only those two files** and pushes `wocar_r3` (these progress pushes are authorised). If a push is rejected because the remote branch moved (the user may push from another machine), the monitor rebases onto it and retries by itself; if that ever fails it aborts cleanly and tries again at the next publish. Leave it running until all 9 runs are done. Note that its push sends every commit that exists locally on `wocar_r3` at that moment.
Runs land in `entries/05_wocar_dqn_r3/artifacts/run_2/<vanilla_dqn|sa_dqn|wocar_dqn>/model_<1..3>/`. If the machine or shell restarts, run the same command again: it reattaches to live runs and skips finished ones.

## Step 3b — one thread per core, and WocaR first (Windows; authorised by the user)
Why: the desktop has 6 physical cores / 12 threads, and nine single-threaded jobs on 12 threads leave three cores with two jobs and three with one. A job that shares a core runs at about half speed (hyperthreads add no throughput here), so a WocaR run that lands on a shared core can take 17–19 h instead of ~10 h. Restricting every trainer to one logical processor per physical core lets Windows time-slice the nine jobs fairly across six cores (all finish at about 14 h), and a priority bump for the WocaR runs gets them done earlier (about 10–11 h) at the cost of the others finishing slightly later. Do this within the first minutes after Step 3 (all nine trainers start at once), and again for any trainer that gets relaunched.
1. Check the CPU layout: `Get-CimInstance Win32_Processor | Select-Object NumberOfCores, NumberOfLogicalProcessors` must show 6 and 12 (summed over processors). If not, skip this step and tell the user.
2. Restrict all trainers to logical processors 0, 2, 4, 6, 8, 10 (mask `0x555`; this assumes Windows numbers the two threads of a core as adjacent pairs (0,1), (2,3), ...):
```
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'train\.py' } | ForEach-Object { (Get-Process -Id $_.ProcessId).ProcessorAffinity = 0x555 }
```
3. Give the WocaR trainers above-normal priority:
```
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'train\.py' -and $_.CommandLine -match 'config_highway_wocar' } | ForEach-Object { (Get-Process -Id $_.ProcessId).PriorityClass = 'AboveNormal' }
```
4. Check about 10 minutes later: read the last `speed: ... frames/sec` value in each `train.log` (the monitor's `f/s` column shows it too, smoothed). With all nine runs pinned expect roughly 35–45 frames/s per run (WocaR a bit higher thanks to the priority). If every run is at or below ~25 frames/s, the logical-processor numbering is probably different from the assumption (6 threads on only 3 cores): undo with the commands below and tell the user; do not experiment further.
5. Undo (also if the desktop becomes unusable or the user asks): same commands with `0xFFF` instead of `0x555`, and `'Normal'` instead of `'AboveNormal'`.
Report in your next message what you applied and the speeds you saw. Affinity and priority only affect scheduling, not the training itself.

## Step 4 — evaluate and report
When the orchestrator summary says `9/9 runs finished`, first stop the monitor (Ctrl-C; it writes a final snapshot) so that nothing pushes automatically from here on, then:
```
cd entries/05_wocar_dqn_r3
python evaluate.py artifacts/run_2 --workers 9        # one single-threaded job per run; lower it if other programs are using the CPU
python report.py artifacts/run_2
```
This evaluates each group's peak-of-mean checkpoint (the frame where the seed-mean mini-test return is highest; the script picks it from the logs): 9 jobs. PGD, worst-Q, MinBest and random L∞ noise at ε 0, 0.0025, 0.005, 0.0075, 0.01, 100 episodes, plus certification. Output: `artifacts/run_2/report.html`.
Commit with explicit paths: the nine `train.log` files, `artifacts/eval_cache/run_2/`, `artifacts/run_2/report.html`. Do **not** commit checkpoints (`*.pth`) or `.npy` files without asking.

## Step 5 — message to the user
Send: the headline comparison from the report at ε = 0.005 and 0.01 (vanilla vs SA-DQN vs WocaR: return and crash rate under PGD / worst-Q / MinBest / random noise), for the peak checkpoints, with the per-seed values and not only the means (seeds differ a lot in this kind of training), anything unusual (failed or restarted runs, a run whose mini-test return collapsed late in training), what is committed locally, and what is left to push.

## Extra seeds batch (seeds 4 and 5)
This is a separate, self-contained task with its own file: `docs/archive/desktop_extra_seeds.md` (https://github.com/Internet-CADS/models/blob/wocar_r3/docs/archive/desktop_extra_seeds.md). It is not part of the steps above; if you were pointed at that file, follow only that file.

## Where things are
`entries/05_wocar_dqn_r3/README.md` (flags, details) · `docs/rev3.md` (changes, decisions) · `docs/deviations.md` (paper/authors' code differences).
