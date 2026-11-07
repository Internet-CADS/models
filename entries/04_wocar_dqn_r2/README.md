Rev 2 of wocar, mainly for training 6x vanilla dqn, sa dqn, wocar dqn

# 04_wocar_dqn_r2 — training launch guide

Training side only. Evaluation and checkpoint selection are out of scope (run separately after training).
Production runs are **not** launched by this repo's tooling automatically: you start them, from two terminals, as described below.

Verification and smoke-test results: `docs/01_eval_changes.md` (safety-score removal), `docs/02_implementation_verification.md` (paper checks), `docs/03_smoke_test.md` (smoke runs + Gate 4 orchestrator/monitor results).

## What runs
18 runs in three batches, 6 concurrent per batch (one per core):

| batch | model | config | run dirs |
|---|---|---|---|
| 1 | vanilla DQN | `config_highway_vanilla.json` | `artifacts/run_1/vanilla_dqn/model_{1..6}/` |
| 2 | SA-DQN | `config_highway_sadqn.json` | `artifacts/run_1/sa_dqn/model_{1..6}/` |
| 3 | WocaR-DQN | `config_highway_wocar.json` | `artifacts/run_1/wocar_dqn/model_{1..6}/` |

Each run: `TRAIN_SEED = model index` (1..6), 4,000,000 frames. Each run dir holds `train.log`, `frame_<N>.pth` every 50,000 frames, `tensorboard/`, and the `.npy`/`.pdf` plots. Runs of one model type differ only in seed and output dir.

Trainers are started with `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, cwd = this entry dir.

ε-greedy: 1.0 → 0.05 linearly over frames 1k→51k, then linearly to 0 at 4M (`act_epsilon_decay_zero: true`, upstream behaviour).
Gradient clipping at 10 is active (see `docs/02_implementation_verification.md` F1).

## Launch (production)
Run from the repo root, inside `tmux`, with the project venv (`.venv`, see below).

```
tmux new -s train
  1) .venv\Scripts\python.exe entries/04_wocar_dqn_r2/run_orchestrator.py          # session/window 1: launches and sequences the 18 runs
  2) .venv\Scripts\python.exe entries/04_wocar_dqn_r2/monitoring.py                # session/window 2: live bars + PROGRESS.md/progress.json
```
Linux/macOS: use `.venv/bin/python` in place of `.venv\Scripts\python.exe`.

Orchestrator prints only lifecycle events (launching, finished, failed, batch starts, summary). Monitoring is display-only and publishes `PROGRESS.md` + `progress.json` at the repo root every 15 minutes, committing only those two files and then pushing the branch to `origin` (a failed push is retried on the next publish).

Ctrl-C in the orchestrator leaves trainers running; use `--resume` to reattach. Ctrl-C in monitoring exits the monitor only.

### Orchestrator flags
- `--dry-run`: precreates the run folders, validates the configs and the `frame N/M` regex against a real `train.log`, prints every per-run command, launches nothing.
- `--resume`: reattaches to live runs (log still growing) and skips runs already complete on disk.
- `--retry N`: retries a failed run N times into `model_<i>/retry_<k>/`. Existing logs are never overwritten.
- `--smoke`: 3,000 frames per run into `artifacts/smoke/orch_flat/`. Use it to check the pipeline end to end before production.

### Monitoring flags
- `--smoke`: reads `artifacts/smoke/orch_flat/` with 3,000 frames.
- `--publish-minutes M` (default 15), `--stall-minutes M` (default 30), `--refresh S` (default 2).
- `--no-commit`: write progress files without committing (implies no push). `--no-push`: commit progress locally but do not push. `--once`: render and publish once, then exit.
- Pushing: `git push origin HEAD` sends **all** local commits on `wocar_r2`, not only progress commits, so batch log commits from the orchestrator go out with the next progress push. Use `--no-push` if you want to push by hand.

## Commands to run first
```
.venv\Scripts\python.exe entries/04_wocar_dqn_r2/run_orchestrator.py --dry-run      # validate, no launch
.venv\Scripts\python.exe entries/04_wocar_dqn_r2/run_orchestrator.py --smoke        # ~3 min, 18 x 3k-frame runs
.venv\Scripts\python.exe entries/04_wocar_dqn_r2/smoke_test.py --skip-runs          # re-verify the Gate 3 artifacts
```

## Environment
`.venv/` (gitignored) at the repo root. Installed for this rev: torch 2.15.0 CPU, torchvision 0.29.2 CPU, numpy, gymnasium, highway-env 1.12.1, tensorboard, ipython, opencv-python, setproctitle, termcolor, packaging, appdirs, oslo.concurrency, matplotlib, pygame. `gym[atari]` is not needed for the highway runs and was not installed.

## Config keys touched in this rev
- `path_prefix`: `artifacts/run_1` (was `artifacts/3rd_run`).
- `flat_run_dir: true` (new, opt-in in `train.py`): run dir = `path_prefix` exactly, no `highway-v0_<suffix>` folder.
- `safety_max_crash_rate` removed; all `best_frame_*` saving removed.
- Hyperparameters otherwise unchanged (verified against `agents/wocar_context.md` §3).

## Artifacts policy
Nothing under `artifacts/` is deleted or moved by any script. Checkpoints, `train.log`, `.npy` and TensorBoard files are kept in place. `*.log` is tracked (the `.gitignore` rule was removed), so each batch's `train.log` and TensorBoard event files are committed once when the batch completes. Checkpoints and `.npy` files are not committed by the scripts.

## Known limits
- A run that dies without a Python traceback shows `stalled` in monitoring (after 30 min without log growth), not `failed`.
- Wall-clock: at smoke-test speeds, a wocar run at 4M frames takes on the order of 18 h when sharing the machine with 5 other trainers. The total is roughly 2.3–2.5 days; plan accordingly.
- `test.py` still prefers `best_frame_*.pth` if present (evaluation script, out of scope; no such files are written any more).
