# Rev 4, stage 7 runbook (for the agent on the training desktop)

You are an agent on the user's Windows training desktop (6 cores / 12 threads). This file is your complete task
description: you need no other instructions. Read it fully, then work through the steps in order, starting at Step 0.
It takes about 8 hours of machine time: about 5½ hours of training, about 2 hours of evaluation, and a few minutes for
the rest.

**What the user has authorised** (you do not need to ask for these):
- pulling and pushing branch `rev4-sel` with normal pushes;
- the progress commits and pushes made by `monitoring.py --push`;
- the commits listed in this runbook;
- changing the CPU affinity of the `train.py` processes (step 1c).

Anything else that changes the repository, the training runs or the machine needs the user's go-ahead.

**Never push `main` or `rev4x`.** `rev4x` was deleted on GitHub after rev 4 was merged, and pushing it would bring it
back. Your local `rev4x` branch from the last run stays as it is; you only switch away from it.

**When to contact the user:** only when this runbook says **"stop and ask"**, and at the end (step 4). Always use the
procedure in "Stopping to ask the user" below: the user watches GitHub, not your chat. Everything else you do on your
own and record in `RUNLOG.md`.

**If you lose context at any point:**
1. Read the end of `entries/06_wocar_dqn_r4/RUNLOG.md` (what has been done).
2. Read `entries/06_wocar_dqn_r4/rev4_params.json`: `gate_sel` is set once step 2 is done.
3. Check whether `train.py` processes are running.

Continue at the first step that is not finished. Never start a step twice in parallel.

## What this is
- **Repository:** https://github.com/Internet-CADS/models. It is already cloned on this machine, with its `.venv`; use
  that clone. Branch: **`rev4-sel`**.
- **Project:** robust DQN on highway-env, vanilla DQN vs SA-DQN vs WocaR-DQN. Rev 4 lives in `entries/06_wocar_dqn_r4/`.
  You ran rev 4 stages 1–6 and the final evaluation on this machine (`docs/rev4x_runbook.md`); this is one more stage
  of the same run.
- **Why stage 7 exists:** stage 6 trained an exploratory arm, `wocar_sel_dqn`, with two seeds. It is WocaR-DQN with the
  target of the authors' code. Those two seeds crashed much less without attack than the main WocaR group. Stage 7
  trains seeds 3–6 of that arm to test this.
  - The rule is fixed in advance (`gate.py sel`; `RUNLOG.md` entry "Stage 7 planned").
  - You run the gate and report what it prints. You never decide it yourself and you never change its threshold.
- **Commands:** `python` below means the repo venv's interpreter (`.venv\Scripts\python.exe` on Windows). **Run every
  command from inside `entries/06_wocar_dqn_r4/`**, so on Windows the interpreter is `..\..\.venv\Scripts\python.exe`.

## Rules
- Never delete or move anything under any `artifacts/` folder.
- Never edit `train.py`, `gate.py`, `rev4.py`, `evaluate.py`, `report.py`, the configs, `upstream/` or
  `rev4_params.json`.
- If a script refuses to run with `NOT ALLOWED`, a previous step is missing. Go back to the runbook; do not work around
  it.
- Do not kill or restart training runs on your own, except changing their CPU affinity as described in step 1c. If a
  run fails or something looks wrong, stop and ask.
- **Git:**
  - Stay on `rev4-sel`. Never touch `main` or `rev4x`, and never force-push.
  - Commit with **explicit paths only** (never `git add -A` / `git add .`). Never add `.npy` files.
  - Keep LF line endings.
  - End every commit message with `Co-Authored-By: <your model name> <noreply@anthropic.com>`.
- **Pushing:** after every commit in this runbook, push it with `git -C ../.. push origin rev4-sel`.
  - If the push is rejected because the remote moved, run `git -C ../.. pull --rebase origin rev4-sel` and push again.
  - If the rebase conflicts, run `git -C ../.. rebase --abort`, note it in `RUNLOG.md` and carry on. Push again after
    the next commit.
  - A failed push never stops the training.

## Progress reporting
As in rev 4, there are three files in `entries/06_wocar_dqn_r4/`:
- **`RUNLOG.md`:** the orchestrator, `evaluate.py` and `gate.py` append to it by themselves. You add a short line when
  you start a step, see a problem, or apply the CPU affinity:
  ```
  python -c "import rev4; rev4.log('Step 1 started: stage 7 training launched')"
  ```
- **`PROGRESS.md` / `progress.json`:** `monitoring.py --push` rewrites, commits and pushes these every 15 minutes during
  training.
- **`rev4_params.json`:** the gates' decisions. Read it, never edit it.

## Stopping to ask the user
The user searches `RUNLOG.md` and the commit list for these markers: `BLOCKED: needs the user`, `RESUMED` and
`FINISHED: stage 7 is complete`. Whenever this runbook says **stop and ask**:
1. Leave every running `train.py` process alone (do not kill anything).
2. Add the marker with a short reason (what happened, which step, which files to look at):
   ```
   python -c "import rev4; rev4.blocked('Step 1d: wocar_sel_dqn/model_4 shows failed. Last lines of run.err: ...')"
   ```
   The orchestrator adds this marker by itself when a run fails; in that case skip this point and go to 3.
3. Commit and push it, with the marker in the commit message:
   ```
   git -C ../.. add -- entries/06_wocar_dqn_r4/RUNLOG.md
   git -C ../.. commit -m "BLOCKED: needs the user (<step>)"
   git -C ../.. push origin rev4-sel
   ```
4. Write the same reason in your chat reply, and wait. Do nothing else until the user answers.

When the user tells you to continue, first run `python -c "import rev4; rev4.resumed('<what the user decided>')"`,
then continue as told. The next commit carries it.

If the user sees no new `PROGRESS.md` commit for more than an hour during training, or no `RUNLOG.md` commit for more
than 7 hours, they will assume your session died and come to the desktop.

## Step 0: setup and checks (15 minutes, no real training)
From the repository root:
```
git status --short
git fetch origin --prune
git switch rev4-sel
git log -1 --format=%s
```
- **`git status --short`** must show no modified tracked files (lines starting with ` M`, `M `, `A ` or ` D`). Untracked
  files (`??`) are fine. If there are modified tracked files, stop and ask, and do not switch branches.
- **`git switch rev4-sel`** creates the local branch from `origin/rev4-sel`. If it says the branch does not exist,
  check the fetch, then stop and ask.
- **`git log -1`** must print a subject starting with `docs(rev4-sel)` or later work on this branch.

Then:
```
cd entries/06_wocar_dqn_r4
python -c "import torch, gymnasium, highway_env; print('imports ok')"
python run_orchestrator.py --stage 7 --dry-run
python run_orchestrator.py --stage 7 --smoke
```
- Make sure no `train.py` process is running and the machine is otherwise idle.
- **The dry run** must list exactly 4 commands, for `wocar_sel_dqn/model_3` to `model_6`, and end with `DRY RUN OK`.
  Every command must contain `training_config:worst_target_mode=select` and `training_config:worst_ratio=0.2`.
- **The smoke run** trains the 4 runs for 3,000 frames each (about 2 minutes) into `artifacts/smoke/stage7`. It must end
  with `4/4 runs finished`.
- If any check fails, stop and ask.
- Log it: `python -c "import rev4; rev4.log('Stage 7 step 0 done: branch rev4-sel, imports, dry run and smoke run ok')"`

## Step 1: stage 7 training (about 5½ hours)
**1a. Start the training.** Four runs: `wocar_sel_dqn` seeds 3, 4, 5 and 6.
```
python run_orchestrator.py --stage 7
```
**1b. Start the monitor** in a second terminal:
```
python monitoring.py --stage 7 --push
```
Both must keep running for the whole stage. Use separate terminals, or start them as detached background processes
with their output redirected to a file. If the machine or your session restarts, start the same two commands again:
the orchestrator reattaches to live runs and skips finished ones.

**1c. Pin the trainers to one thread per physical core** (within the first minutes; Windows PowerShell).
1. Check the CPU: `Get-CimInstance Win32_Processor | Select-Object NumberOfCores, NumberOfLogicalProcessors` must show
   6 and 12. If not, skip 1c and note it in `RUNLOG.md`.
2. Pin the trainers:
   ```
   Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'train\.py' } | ForEach-Object { (Get-Process -Id $_.ProcessId).ProcessorAffinity = 0x555 }
   ```
3. About 10 minutes later, read `PROGRESS.md`. Expect roughly 55–65 frames/s per run (about 5 hours for the stage).
   - If a run is at or below about 40 frames/s, undo with `0xFFF` instead of `0x555`, and note it in `RUNLOG.md`.
   - Log the affinity: `python -c "import rev4; rev4.log('Stage 7 / 1c: affinity 0x555 applied to train.py processes')"`

**1d. Wait.** Look at `PROGRESS.md` every 30–60 minutes; there is nothing else to do. If a run shows `failed` or
`stalled`, stop and ask, and put the last 30 lines of that run's `run.err` and `train.log` in the chat reply.

**1e. Done** when the orchestrator prints `4/4 runs finished` and exits. Then stop the monitor (Ctrl-C).

**1f. Commit and push.**
```
git -C ../.. add -- "entries/06_wocar_dqn_r4/artifacts/run_1/*/model_*/train.log" "entries/06_wocar_dqn_r4/artifacts/run_1/*/model_*/frame_*.pth" "entries/06_wocar_dqn_r4/artifacts/run_1/*/model_*/tensorboard/*" entries/06_wocar_dqn_r4/RUNLOG.md entries/06_wocar_dqn_r4/PROGRESS.md entries/06_wocar_dqn_r4/progress.json
git -C ../.. commit -m "data(rev4-sel): stage 7 training"
git -C ../.. push origin rev4-sel
```
`git -C ../..` runs git from the repo root, so the paths start at `entries/`. Before committing,
`git -C ../.. status --short` must show nothing staged outside `entries/06_wocar_dqn_r4/`.

## Step 2: evaluation and gate sel (about 2 hours)
```
python evaluate.py --preset final --workers 6
python gate.py sel
```
- **The evaluation** first prints `N jobs to run`. N must be **4 or 6**: seeds 3–6, plus seeds 1–2 again if adding the
  new seeds moved the arm's peak checkpoint. Every other group is already cached.
  - If N is larger than 6, stop the evaluation (Ctrl-C) and stop and ask: it would re-evaluate groups that should not
    change.
  - If the evaluation is interrupted, run the same command again; finished jobs are cached.
- **The gate** prints a table and one of two verdicts, **CONFIRMED** or **NOT CONFIRMED**. **Both are normal results.**
  Continue to step 3 in either case. Nothing more is trained after this gate.
- **If the gate prints an error instead** (for example that seeds are not evaluated at the peak frame), do what it says
  (usually: run the evaluation command again), then run the gate again. If it still fails, stop and ask.
- **Commit and push:**
  ```
  git -C ../.. add -- entries/06_wocar_dqn_r4/RUNLOG.md entries/06_wocar_dqn_r4/rev4_params.json entries/06_wocar_dqn_r4/artifacts/eval_cache
  git -C ../.. commit -m "data(rev4-sel): gate sel"
  git -C ../.. push origin rev4-sel
  ```

## Step 3: report and finish (5 minutes)
```
python report.py
```
This rewrites `artifacts/run_1/report.html`. The headline table now has six seeds for
`WocaR-DQN, upstream target (mix picks the action only)`, and a note naming the gate sel verdict.

Commit, mark the end and push:
```
git -C ../.. add -- entries/06_wocar_dqn_r4/RUNLOG.md entries/06_wocar_dqn_r4/artifacts/run_1/report.html
git -C ../.. commit -m "data(rev4-sel): report with stage 7"
python -c "import rev4; rev4.finished('Report: artifacts/run_1/report.html. Summary in the chat reply.', rev4.FINISHED_SEL)"
git -C ../.. add -- entries/06_wocar_dqn_r4/RUNLOG.md
git -C ../.. commit -m "FINISHED: stage 7 is complete"
git -C ../.. push origin rev4-sel
```

## Step 4: final message to the user
Send:
1. Which steps ran, and anything unusual: failed or restarted runs, speeds below 50 frames/s, the number of evaluation
   jobs if it was not 4.
2. The complete output of `python gate.py sel`: the verdict, all three tables and the verdict sentence. Copy it from
   `RUNLOG.md`.
3. The headline table from the report at the training ε and at 2×, for every group: seeds, crash without attack, crash
   under gradient attacks, attack-induced crash, certified actions and both t columns.
4. What is committed and pushed: `git -C ../.. log --oneline origin/rev4-sel..HEAD` must be empty.

## If something goes wrong
| Situation | What to do |
|---|---|
| A script prints `NOT ALLOWED` | A previous step is missing. Read `RUNLOG.md` and `rev4_params.json`, find the step, do it. |
| `gate.py sel` says `gate already decided` | Step 2 is done. Continue with step 3. |
| A training run shows `failed` or `stalled` | Stop and ask. Do not delete or relaunch anything by hand. |
| The orchestrator was interrupted | Run the same command again; it reattaches to live runs. |
| `evaluate.py` was interrupted | Run the same command again; finished jobs are cached. |
| `git switch rev4-sel` fails, or a push is rejected twice | Stop and ask. Never push another branch instead. |
| Speeds far below 40 frames/s | Check that no other program uses the CPU and that the affinity is set (1c). Note it in `RUNLOG.md`. |
| Anything else you do not understand | Stop and ask. |
