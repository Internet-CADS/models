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
