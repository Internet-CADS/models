# Task for the agent on the training desktop: run the extra seeds 4 and 5 and push the results

You are an agent on the user's Windows training desktop (6 cores / 12 threads). This file is your complete task description: read it fully, then do it. Nothing else is needed. If you lose context at any point (this takes about 10 hours), re-read this file, look at the current state (`git status`, which `train.py` processes are running, `entries/05_wocar_dqn_r3/PROGRESS.md`), and continue at the first step that is not done yet.

## What and why
- Repository: https://github.com/Internet-CADS/models (already cloned on this machine; use that clone, do not make a new one). Branch: **`wocar_r3`**. Work from the repo root.
- Project: robust DQN on highway-env; vanilla DQN vs SA-DQN vs WocaR-DQN. Experiment "rev3" lives in `entries/05_wocar_dqn_r3/`. The first 9 runs (seeds 1–3, 2M frames each) are finished and already on the remote.
- **Your task:** train **seeds 4 and 5** for all three models (6 runs, 2,000,000 frames each, about 10 hours in total), then commit the results and push `wocar_r3`. The user runs seeds 6 and 7 on a laptop and does the evaluation; you do **not** evaluate. Nothing about the setup changes: same configs, same run folder `entries/05_wocar_dqn_r3/artifacts/run_2`, the seeds only add `model_4` and `model_5` folders next to the existing ones.
- In the commands below `python` means the repo venv's interpreter (`.venv\Scripts\python.exe`). Run everything from the repo root.

## Rules (the user's, non-negotiable)
- Never delete or move anything under any `artifacts/` folder. Never edit `train.py`, `upstream/`, or the configs.
- Commit with **explicit paths only** (never `git add -A` / `git add .`). Keep LF line endings. End every commit message with the trailer `Co-Authored-By: <your model name> <noreply@anthropic.com>`.
- **Never force-push.** Never touch `main`. The only pushes you may do are the two described here: the monitor's automatic progress pushes during the run, and the final push of the results in step 6. Do not run `git pull`/`push` for anything else.
- Do not kill or restart the training runs on your own, except changing their CPU affinity as described in step 3. If something fails or looks wrong, stop and report it to the user with the relevant log lines.

## Steps

**1. Get the right code and check it.**
```
git fetch origin --prune
git switch wocar_r3
git pull --rebase
```
If git complains about local changes to the generated files `entries/05_wocar_dqn_r3/PROGRESS.md` / `progress.json`, run `git checkout -- entries/05_wocar_dqn_r3/PROGRESS.md entries/05_wocar_dqn_r3/progress.json` and retry. Make sure no `train.py` process is running and the machine is otherwise idle (close heavy programs). Then:
```
python -c "import torch, gymnasium, highway_env; print('imports ok')"
python entries/05_wocar_dqn_r3/run_orchestrator.py --dry-run --concurrent --seeds 4,5
```
The dry run must list 6 commands (`model_4` and `model_5` for `vanilla_dqn`, `sa_dqn`, `wocar_dqn`) and end with `DRY RUN OK`. If not, stop and report.

**2. Start the training and the monitor (both must keep running for the whole time).**
```
python entries/05_wocar_dqn_r3/run_orchestrator.py --concurrent --seeds 4,5 --no-commit
python entries/05_wocar_dqn_r3/monitoring.py --seeds 4,5
```
Run each in its own terminal / tmux window or as a detached background process with its output redirected to a log file; they must keep running after your tool call returns. All six runs start together. The monitor writes `entries/05_wocar_dqn_r3/PROGRESS.md` and `progress.json` every 15 minutes, commits only those two files and pushes `wocar_r3` (if a push is rejected because the remote moved, it rebases and retries by itself). If the machine or your session restarts, start the same two commands again: the orchestrator reattaches to live runs and skips finished ones.

**3. Pin the trainers to one thread per physical core (within the first minutes).** Six runs on 12 threads would otherwise pair up on cores and run at half speed. First check the CPU: `Get-CimInstance Win32_Processor | Select-Object NumberOfCores, NumberOfLogicalProcessors` must show 6 and 12 (summed over processors); if not, skip this step and tell the user. Then (PowerShell):
```
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'train\.py' } | ForEach-Object { (Get-Process -Id $_.ProcessId).ProcessorAffinity = 0x555 }
```
(`0x555` = logical processors 0, 2, 4, 6, 8, 10, one per physical core, assuming Windows numbers the two threads of a core as adjacent pairs.) About 10 minutes later read the last `speed: ... frames/sec` line in each of the six `train.log` files (`entries/05_wocar_dqn_r3/artifacts/run_2/<group>/model_<4|5>/train.log`): expect roughly 55–65 frames/s per run and about 10 hours in total. If a run is at or below ~40 frames/s, undo with `0xFFF` instead of `0x555` and tell the user. Redo the affinity command for any trainer that gets relaunched.

**4. Wait.** Check back about every 30–60 minutes by reading `entries/05_wocar_dqn_r3/PROGRESS.md`. Nothing to do meanwhile. If a run fails, stop and report; do not improvise.

**5. Detect completion.** The orchestrator prints `6/6 runs finished` and exits; each of the six `train.log` files ends with `frame 2000000/2000000`. Then stop the monitor (Ctrl-C; it writes a final snapshot).

**6. Publish the results of seeds 4 and 5 and push.**
```
git add -- "entries/05_wocar_dqn_r3/artifacts/run_2/*/model_[45]/train.log" "entries/05_wocar_dqn_r3/artifacts/run_2/*/model_[45]/frame_*.pth" "entries/05_wocar_dqn_r3/artifacts/run_2/*/model_[45]/frame_2000000_losses.npy" "entries/05_wocar_dqn_r3/artifacts/run_2/*/model_[45]/frame_2000000_rewards.npy" "entries/05_wocar_dqn_r3/artifacts/run_2/*/model_[45]/tensorboard/*"
git commit -m "data(rev3): extra seeds 4 and 5 (desktop): logs, tensorboard, checkpoints, final losses/rewards"    # + the Co-Authored-By trailer
git pull --rebase
git diff --name-only origin/wocar_r3..HEAD
git push origin wocar_r3
```
Before pushing, look at the file list: it must contain only files under `entries/05_wocar_dqn_r3/artifacts/run_2/*/model_4/` or `model_5/` (plus the monitor's two progress files). There must be no other `.npy` files (the cumulative `frame_<N>_losses.npy` / `frame_<N>_rewards.npy` files must not be added). If anything else shows up, stop and report instead of pushing. The laptop may have pushed seeds 6 and 7 meanwhile; those are different files, so the rebase should be clean. If it conflicts **only** in `PROGRESS.md` / `progress.json` (generated), keep either version, `git add` them and `git rebase --continue`; any other conflict: stop and report. Use a normal push only.

**7. Report to the user** (final message): the commit hash of the push, the last frame and final mini-test return of each of the six runs (last `this test avg reward:` line of each `train.log`), the speeds you saw, and anything unusual (failed or restarted runs, a run whose return collapsed late in training).

That is the whole task. Do not evaluate, do not start other experiments, do not change anything else in the repo.
