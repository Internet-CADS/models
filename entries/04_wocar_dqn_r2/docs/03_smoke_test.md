# 03 — Smoke test (Gate 3)

First real execution of the entry. Script: `smoke_test.py` (entry dir). Artifacts left in place under `artifacts/smoke/`.

## Environment
Repo-root `.venv` (gitignored): Python 3.12, torch 2.15.0+cpu, torchvision 0.29.2+cpu, numpy 2.5.4, gymnasium 1.4.0, highway-env 1.12.1, tensorboard, ipython, opencv-python, setproctitle, termcolor, packaging, appdirs, oslo.concurrency, matplotlib, pygame. `gym[atari]` was **not** installed (not needed: `import train` works). No code changes were needed to run on this stack.

## What ran
| Step | Command shape | Result |
|---|---|---|
| Clip check | `compute_td_loss` driven with a stub net/buffer, `grad_clip` 10 and 0.5 | **PASS**: `grad_clip=0.5` -> post-clip norm 0.5000; `grad_clip=10` -> 3.29 (below the cap, untouched) |
| Smoke runs | `train.py --config config_highway_<m>.json --path_prefix artifacts/smoke/<m> training_config:seed=1 training_config:num_frames=60000`, cwd = entry dir, `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` | all three exit 0, reach `frame 60000/60000` |
| Seed check | vanilla, seed 1, 12,000 frames, twice (`seedcheck_a`, `seedcheck_b`) | **PASS**: normalised `train.log` identical (121 = 121 lines) |

Output dirs follow the known quirk: `artifacts/smoke/<m>/highway-v0_<m>/`.

## Pass criteria
| Criterion | Result |
|---|---|
| `train.log` exists; `^frame N/M` regex parses | PASS (68 / 76 / 76 perf lines for vanilla / sadqn / wocar; last line `60000/60000`) |
| `frame_<N>.pth` on the 50k cadence | PASS: `frame_50000.pth` + final `frame_60000.pth` in each run |
| No `best_frame_*` files | PASS: 0 found |
| mini_test row `this test avg reward: <r>, crash rate: <c>` at 50k | PASS in all three; no `safety` text anywhere in the logs |
| TensorBoard event files written | PASS |
| Grad clip engages when norm > cap | PASS (unit-level; not expected to engage in 60k frames, see handoff) |
| Same seed twice -> identical log | PASS |
| Orchestrator batch transition vanilla -> sa -> wocar | **DEFERRED to Gate 4** (needs `run_orchestrator.py`) |
| monitoring shows bars, publishes PROGRESS files | **DEFERRED to Gate 4** (needs `monitoring.py`) |
| Git progress commit touches only the progress files | **DEFERRED to Gate 4** |

Expected console tail of `python smoke_test.py`:
```
[exit]   ('smoke', 'vanilla'): returncode 0   (x5)
[PASS] vanilla  frames=60000/60000 ckpts=['frame_50000.pth', 'frame_60000.pth'] best_frame=[] tb=True mini_test(reward,crash)=(...)
[PASS] sadqn ... / [PASS] wocar ...
[PASS] seed repeatability (vanilla, 12000 frames, seed 1): {... 'identical': True ...}
SMOKE TEST: ALL PASS
```

## Observations worth knowing
1. **Identical mini_test rows across all three models** (`7.06079`, crash `0.720`). Not a bug. Verified: `frame_50000.pth` weights are **bit-identical** for vanilla / sadqn / wocar. With one seed, the three runs are the same computation until the SA-reg and worst-critic ramps start at frame 50k (`schedule_start 50000`, eps = 0 before). By `frame_60000.pth` they have diverged (max |dW| 0.032-0.036), so the robust code paths are live. Consequence: the smoke test exercises SA/WocaR machinery only for frames 50k-60k (eps up to ~0.00013 of the 0.0025 ramp); late-ramp behaviour is untested.
2. **Epsilon in smoke runs** is 0.0056 at frame 59,000: `decay_zero` anchors the second-stage decay to `num_frames`, so ε hits 0 at 60,000 here (production: 4M). Expected per F2; just not representative of production ε.
3. **Speed** (5 trainers on 6 cores): vanilla ~101 fps, sadqn ~66 fps, wocar ~60 fps, 12k vanilla ~74 fps. At 60 fps, 4M frames is ~18.5 h per wocar run before contention from 6 concurrent runs; plan for the 2.5-day estimate being tight.
4. `smoke_test.py` writes each run's stdout to `artifacts/smoke/<dir>/stdout.txt` (kept).
5. `*.log` is still gitignored (`.gitignore` l.60), so smoke `train.log` files are untracked-and-ignored. Removal is a Gate 4/5 item.
6. A 3k-frame trial run during setup wrote to `C:\Users\Frederik\Temp\x` (relative-path slip, outside the repo). Left in place, not deleted.

## Open for Gate 4
Run the three deferred criteria with `--smoke` (about 3k frames per run, lowered `TARGET_FRAMES`) once `run_orchestrator.py` and `monitoring.py` exist; append results here.

## Gate 4 addendum: deferred criteria (orchestrator + monitoring)
Run: `run_orchestrator.py --smoke` (3 batches x 6 runs, 3,000 frames each, `artifacts/smoke/orch/`) with `monitoring.py --smoke` polling alongside, publishing to a **throwaway scratch git repo** (`--repo-root`) so no test commits land on `wocar_r2`.

| Criterion | Result |
|---|---|
| Batch transition vanilla -> sa -> wocar | PASS: batch 1 launched 04:29:42, each batch started only after all six runs finished or failed; 18/18 finished in ~2.5 min |
| monitoring shows bars for the runs | PASS: 18 grouped rows, `queued -> training -> done` transitions observed; header `batch 3/3 · 18/18 runs done · elapsed 00:02` |
| monitoring publishes PROGRESS.md + progress.json | PASS: 10 progress commits in the scratch repo (publish interval shortened to 18 s for the test) |
| Progress commit touches only the progress files | PASS: every commit lists exactly `PROGRESS.md`, `progress.json`; a pre-staged `other.txt` and an untracked `scratch.txt` were not swept in |
| Batch-log commit (`commit_batch`) | PASS (scratch repo, copies of the smoke logs): 12 files = 6 `train.log` + 6 TensorBoard event files, nothing else; second call is a no-op |

Failure-path checks (scratch fakes): stalled (log 3 h old), failed (Traceback in `run.err`), queued, done and training rows all render correctly; a second monitor instance is refused while the first runs; a stale lock is reclaimed after the first is killed; the Ctrl-C path (simulated by an interrupt in the loop) wrote its final snapshot and removed the lock; a failing trainer with `--retry 1` retried into `model_<i>/retry_1/`, and a pre-existing incomplete `train.log` was left byte-for-byte intact.

Known limits: the monitor's `failed` status comes from a Traceback in the per-run `run.err` that the orchestrator writes; a run that dies without one shows `stalled` after 30 minutes. A real console Ctrl-C was simulated, not typed.

## Gate 4 addendum 2: flat run directories (user decision)
Gate 3 and the first Gate 4 pass used `train.py`'s nested layout (`<path_prefix>/highway-v0_<suffix>/`). On request the layout is now flat: `artifacts/run_1/<model>/model_<i>/train.log` (+ `frame_*.pth`, `tensorboard/`).
- **Code change (training code, opt-in):** `train.py` main(), new top-level config key `flat_run_dir`. When true, the run dir is exactly `path_prefix`; when absent/false the old `env_id + name_suffix` folder is used. The three configs set `flat_run_dir: true`. No hyperparameter or training logic touched.
- **Scripts:** `run_orchestrator.py`, `monitoring.py`, `smoke_test.py` read/write `<run dir>/train.log` directly. `--smoke` root is now `artifacts/smoke/orch_flat`. `smoke_test.py` reads a nested Gate 3 run (`<prefix>/highway-v0_<m>/`) when present and a flat one (`<prefix>/`) otherwise.
- **Verification:** (1) vanilla, seed 1, 12,000 frames, flat (`artifacts/smoke/flatcheck/`): normalised `train.log` is **identical** (121 lines) to the earlier nested `seedcheck_a` run, so training is unchanged. (2) `run_orchestrator.py --smoke` on the flat layout: 3 batches x 6 runs, 18/18 finished (04:38:52 -> 04:41:32). (3) monitoring, failure paths and batch commit re-run on flat fakes/copies: same results as above.
- Not re-run: the 60k-frame Gate 3 runs (training code identical apart from the output path). The original nested Gate 3 artifacts under `artifacts/smoke/<model>/highway-v0_<model>/`, `seedcheck_*` and `orch/` are kept as they were.
