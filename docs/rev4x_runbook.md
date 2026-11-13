# Rev 4 runbook (for the agent on the training desktop)

You are an agent on the user's Windows training desktop (6 cores / 12 threads). This file is your complete task
description: you need no other instructions. Read it fully, then work through the steps in order, starting at Step 0.
The whole thing takes about 25 hours of machine time (about 5 more for each retry).

**What the user has authorised** (you do not need to ask for these): pulling and pushing branch `rev4x` with normal
pushes, the progress commits and pushes made by `monitoring.py --push`, the commits listed in this runbook, and
changing the CPU affinity of the `train.py` processes (step 1c). Anything else that changes the repository, the
training runs or the machine needs the user's go-ahead.

**When to contact the user:** only when this runbook says **"stop and ask"** (a gate that says STOP, a failed or stalled
run, a check that fails, anything you do not understand), and at the end (Step 8). Always do it with the procedure in
"Stopping to ask the user" below: the user watches GitHub, not your chat. Everything else you do on your own and record
in `RUNLOG.md`.

**If you lose context at any point:** read `entries/06_wocar_dqn_r4/RUNLOG.md` (what has been done), then
`entries/06_wocar_dqn_r4/rev4_params.json` (which gates have passed), then check whether `train.py` processes are
running. Continue at the first step that is not finished. Never start a step twice in parallel.

## What this is
- Repository: https://github.com/Internet-CADS/models (already cloned on this machine, with its `.venv`; use that
  clone). Branch: **`rev4x`**. If you were given a link to this file, it is `docs/rev4x_runbook.md` on that branch;
  Step 0 checks the branch out, after which you work only in the local clone.
- Project: robust DQN on highway-env; vanilla DQN vs SA-DQN vs WocaR-DQN. Rev 4 lives in `entries/06_wocar_dqn_r4/`.
  `entries/06_wocar_dqn_r4/README.md` explains what rev 4 changes and why; you do not need it to do the task.
- Rev 4 is trained in **four stages of six runs each** (one run per core, 1,000,000 frames, about 5 hours per stage).
  After each stage a script called a **gate** decides whether to go on. You run the gate and do what it prints.
  You never decide a gate yourself and you never change a threshold.
- In the commands below `python` means the repo venv's interpreter (`.venv\Scripts\python.exe` on Windows,
  `.venv/bin/python` on macOS/Linux). **Run every command from inside `entries/06_wocar_dqn_r4/`** (so on Windows
  the interpreter is `..\..\.venv\Scripts\python.exe`).

## Rules
- Never delete or move anything under any `artifacts/` folder.
- Never edit `train.py`, `gate.py`, `rev4.py`, the configs, `upstream/` or `rev4_params.json`. If a script refuses to
  run with `NOT ALLOWED`, a previous step is missing: go back to the runbook, do not work around it.
- A gate prints one of three verdicts. **PASS**: go to the step it names. **RETRY**: repeat the step it names, once.
  **FAIL**: do what the `Next:` line says (a line starting with STOP means: stop and ask). Do not train anything after a FAIL
  unless the `Next:` line tells you to.
- Do not kill or restart training runs on your own, except changing their CPU affinity as described in step 1c.
  If a run fails or something looks wrong, stop and ask (see "Stopping to ask the user").
- Git: stay on `rev4x`. Never touch `main`, never force-push. Commit with **explicit paths only** (never
  `git add -A` / `git add .`). Keep LF line endings. End every commit message with
  `Co-Authored-By: <your model name> <noreply@anthropic.com>`.
- **Pushing:** after every commit in this runbook, push it: `git -C ../.. push origin rev4x` (a normal push). If the
  push is rejected because the remote moved, run `git -C ../.. pull --rebase origin rev4x` and push again; if the
  rebase conflicts, run `git -C ../.. rebase --abort`, note it in `RUNLOG.md` and carry on (push again after the next
  commit). A failed push never stops the training.

## Progress reporting
Three files in `entries/06_wocar_dqn_r4/`, all plain text:
- `RUNLOG.md`: the progress file. The orchestrator, `evaluate.py` and `gate.py` append to it by themselves (every
  finished stage, evaluation and gate verdict with its table). **You** add a short line whenever you start a step,
  see a problem, or apply the CPU affinity (use the helper below; it adds the timestamp).
- `PROGRESS.md` / `progress.json`: live training progress (frames, speed and time left per run). `monitoring.py --push`
  rewrites them every 15 minutes during a stage, commits only these two files and pushes `rev4x`, so the user can
  follow along on GitHub.
- `rev4_params.json`: the gates' decisions. Read it, never edit it.

Helper to add a line to `RUNLOG.md`:
```
python -c "import rev4; rev4.log('Step 1 started: stage 1 training launched')"
```

## Stopping to ask the user
The user follows the run on GitHub and searches `RUNLOG.md` and the commit list for three fixed markers:
`BLOCKED: needs the user`, `RESUMED` and `FINISHED: rev 4 is complete`. Whenever this runbook says **stop and ask**:
1. Leave every running `train.py` process alone (do not kill anything).
2. Add the marker with a short reason (what happened, which step, which files to look at):
   ```
   python -c "import rev4; rev4.blocked('Step 1d: vanilla_g090/model_2 shows failed. Last lines of run.err: ...')"
   ```
   `gate.py` (on a STOP verdict) and the orchestrator (when a run fails) add this marker by themselves; then skip this
   point and go to 3.
3. Commit and push it, with the marker in the commit message:
   ```
   git -C ../.. add -- entries/06_wocar_dqn_r4/RUNLOG.md
   git -C ../.. commit -m "BLOCKED: needs the user (<step>)"
   git -C ../.. push origin rev4x
   ```
   (Gate steps: use the gate commit command of that step instead, with `BLOCKED: needs the user (gate N)` as the
   message, so the gate's files go along.)
4. Write the same reason in your chat reply, and wait. Do nothing else until the user answers.

When the user tells you to continue, first add `python -c "import rev4; rev4.resumed('<what the user decided>')"`,
then continue as told (the next commit carries it).

If the user sees no new `PROGRESS.md` commit for more than an hour while a stage is running, or no `RUNLOG.md` commit
for more than 7 hours, they will assume your session died and come to the desktop.

## Step 0 — setup and checks (15 minutes, no real training)
```
git fetch origin --prune
git switch rev4x
git pull
cd entries/06_wocar_dqn_r4
python -c "import torch, gymnasium, highway_env; print('imports ok')"
python run_orchestrator.py --stage 1 --dry-run
python run_orchestrator.py --stage 1 --smoke
```
- Make sure no `train.py` process is running and the machine is otherwise idle.
- The dry run must list 6 commands and end with `DRY RUN OK`.
- The smoke run trains 6 runs for 3,000 frames each (about 2 minutes) into `artifacts/smoke/stage1` and must end with
  `6/6 runs finished`. If either check fails, stop and ask.
- Log it: `python -c "import rev4; rev4.log('Step 0 done: imports, dry run and smoke run ok')"`

## Step 1 — stage 1: vanilla DQN with two discount factors (about 5 hours)
**1a. Start the training.** Six runs: `vanilla_g097` and `vanilla_g090` × seeds 1, 2 and 3.
```
python run_orchestrator.py --stage 1
```
**1b. Start the monitor** in a second terminal:
```
python monitoring.py --stage 1 --push
```
Both must keep running for the whole stage: use separate terminals / tmux windows, or start them as detached
background processes with their output redirected to a file. If the machine or your session restarts, start the same
two commands again: the orchestrator reattaches to live runs and skips finished ones.

**1c. Pin the trainers to one thread per physical core (within the first minutes; Windows PowerShell).** Six runs on
12 threads otherwise pair up on cores and run at half speed. First check the CPU:
`Get-CimInstance Win32_Processor | Select-Object NumberOfCores, NumberOfLogicalProcessors` must show 6 and 12; if not,
skip 1c and note it in `RUNLOG.md`. Then:
```
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'train\.py' } | ForEach-Object { (Get-Process -Id $_.ProcessId).ProcessorAffinity = 0x555 }
```
About 10 minutes later read `PROGRESS.md`: expect roughly 55–65 frames/s per run (about 5 hours for the stage). If a
run is at or below ~40 frames/s, undo with `0xFFF` instead of `0x555` and note it in `RUNLOG.md`. Repeat the affinity
command whenever trainers are (re)started, i.e. at the start of every stage.

**1d. Wait.** Look at `PROGRESS.md` every 30–60 minutes. Nothing else to do. If a run shows `failed` or `stalled`,
stop and ask (put the last 30 lines of that run's `run.err` and `train.log` in the chat reply).

**1e. Done when** the orchestrator prints `6/6 runs finished` and exits. Then stop the monitor (Ctrl-C).

**1f. Commit.**
```
git -C ../.. add -- "entries/06_wocar_dqn_r4/artifacts/run_1/*/model_*/train.log" "entries/06_wocar_dqn_r4/artifacts/run_1/*/model_*/frame_*.pth" "entries/06_wocar_dqn_r4/artifacts/run_1/*/model_*/tensorboard/*" entries/06_wocar_dqn_r4/RUNLOG.md entries/06_wocar_dqn_r4/PROGRESS.md entries/06_wocar_dqn_r4/progress.json
git -C ../.. commit -m "data(rev4): stage 1 training (vanilla, two discount factors)"
```
(`git -C ../..` runs git from the repo root, so the paths start at `entries/`. Never add `.npy` files. This same
command is used after every training stage; only the message changes. Before committing, `git -C ../.. status --short`
must show nothing staged outside `entries/06_wocar_dqn_r4/`.)

## Step 2 — gate 1: pick the vanilla arm (5 minutes)
```
python diagnose.py --stage1 --workers 6
python gate.py stage1
```
- **PASS:** it names the chosen arm and writes γ and `hinge_c` into `rev4_params.json`. Go to step 3.
- **FAIL:** stop and ask (the gate has added the BLOCKED marker; commit with the gate command below and the message
  `BLOCKED: needs the user (gate 1)`). Nothing else is trained.

Commit after every gate (same command each time; `artifacts/eval_cache` does not exist yet after gate 1, leave it out there):
```
git -C ../.. add -- entries/06_wocar_dqn_r4/RUNLOG.md entries/06_wocar_dqn_r4/rev4_params.json entries/06_wocar_dqn_r4/artifacts/diag entries/06_wocar_dqn_r4/artifacts/eval_cache
git -C ../.. commit -m "data(rev4): gate 1"
```

## Step 3 — pick the attack budget ε (about 30 minutes)
```
python evaluate.py --preset epsilon --workers 6
python gate.py epsilon
```
The evaluation attacks the chosen vanilla arm at ε 0.005 … 0.1; the gate picks the smallest ε at which vanilla clearly
breaks.
- **PASS:** go to step 4. **FAIL:** stop and ask (as in step 2).
- Commit as after gate 1 (message `gate epsilon`).

## Step 4 — stage 2: more vanilla seeds + SA-DQN (about 5 hours, then about 40 minutes of evaluation)
```
python run_orchestrator.py --stage 2          # terminal 1
python monitoring.py --stage 2 --push         # terminal 2
```
Do 1c (affinity) again, wait as in 1d, finish as in 1e, commit as in 1f (message `stage 2 training`). Then:
```
python evaluate.py --preset gate2 --workers 6
python gate.py stage2
```
- **PASS:** go to step 5.
- **RETRY:** the gate has changed κ and created a new group `sa_dqn_b`. Repeat this whole step 4 once (the same
  commands; the orchestrator now trains only `sa_dqn_b` seeds 1–3, three runs, and skips the finished vanilla runs).
- **FAIL:** stop and ask (as in step 2).
- Commit as after gate 1 (message `gate 2`).

## Step 5 — stage 3: more SA-DQN seeds + WocaR-DQN (about 5 hours, then about 40 minutes of evaluation)
```
python run_orchestrator.py --stage 3          # terminal 1
python monitoring.py --stage 3 --push         # terminal 2
```
Affinity (1c), wait, finish, commit (message `stage 3 training`). Then:
```
python evaluate.py --preset gate3 --workers 6
python gate.py stage3
```
- **PASS:** go to step 6.
- **RETRY:** the gate has lowered `worst_ratio` and created `wocar_dqn_b`. Repeat this whole step 5 once (only
  `wocar_dqn_b` seeds 1–3 are trained).
- **FAIL:** **skip step 6** and go to step 7. (The final evaluation then covers vanilla, SA-DQN and the three WocaR
  seeds that exist.)
- Commit as after gate 1 (message `gate 3`).

## Step 6 — stage 4: remaining seeds + exploratory arm (about 5 hours)
```
python run_orchestrator.py --stage 4          # terminal 1
python monitoring.py --stage 4 --push         # terminal 2
```
This trains WocaR seeds 4–6 and `wocar_wcl_dqn` seeds 1–3. Affinity (1c), wait, finish, commit
(message `stage 4 training`). There is no gate after this stage.

## Step 7 — final evaluation and report (about 3 hours)
```
python evaluate.py --preset final --workers 6
python report.py
```
- The evaluation runs every group and seed at its group's peak checkpoint, 300 episodes, four attacks, five ε values,
  plus certification. It caches every finished job, so if it is interrupted, run the same command again.
- `report.py` writes `artifacts/run_1/report.html` and prints one line per group.
- Commit:
```
git -C ../.. add -- entries/06_wocar_dqn_r4/RUNLOG.md entries/06_wocar_dqn_r4/rev4_params.json entries/06_wocar_dqn_r4/artifacts/eval_cache entries/06_wocar_dqn_r4/artifacts/run_1/report.html
git -C ../.. commit -m "data(rev4): final evaluation and report"
```
- Then mark the end and push:
```
python -c "import rev4; rev4.finished('Final report: artifacts/run_1/report.html. Summary in the chat reply.')"
git -C ../.. add -- entries/06_wocar_dqn_r4/RUNLOG.md
git -C ../.. commit -m "FINISHED: rev 4 is complete"
git -C ../.. push origin rev4x
```

## Step 8 — final message to the user
Send:
1. Which steps ran and every gate verdict (copy the verdict lines from `RUNLOG.md`), including retries.
2. The contents of `rev4_params.json`.
3. The headline table from the report at the training ε and at 2× (per group: seeds, crash without attack, crash under
   gradient attacks, attack-induced crash, certified actions), and the per-seed values for crash without attack and
   attack-induced crash (seeds differ a lot in this kind of training).
4. Anything unusual: failed or restarted runs, runs slower than expected, a run whose mini-test return collapsed late.
5. What is committed locally (`git log --oneline origin/rev4x..HEAD`) and whether it is pushed.

## If something goes wrong
| Situation | What to do |
|---|---|
| A script prints `NOT ALLOWED` | A previous step or gate is missing. Read `RUNLOG.md` and `rev4_params.json`, find the step, do it. |
| A gate says `gate already decided` | That gate is done. Continue with the next step. |
| A training run shows `failed` or `stalled` | Stop and ask. Do not delete or relaunch anything by hand. |
| The orchestrator was interrupted | Run the same command again; it reattaches to live runs. |
| `evaluate.py` was interrupted | Run the same command again; finished jobs are cached. |
| A gate or `evaluate.py` says an evaluation or checkpoint is missing | The step before it did not finish. Go back to it. |
| Speeds far below 40 frames/s | Check that no other program uses the CPU and that the affinity is set (1c). Note it in `RUNLOG.md`. |
| Anything else you do not understand | Stop and ask. |

## Time plan (estimates from the rev 3 measurements on this desktop)
| Step | Time |
|---|---|
| 0 setup | 15 min |
| 1 stage 1 | ~5 h |
| 2 gate 1 | 5 min |
| 3 ε sweep + gate | ~30 min |
| 4 stage 2 + evaluation + gate | ~5 h + 40 min |
| 5 stage 3 + evaluation + gate | ~5 h + 40 min |
| 6 stage 4 | ~5 h |
| 7 final evaluation + report | ~3 h |
| Each retry (steps 4 or 5) | + ~5 h 40 min |
