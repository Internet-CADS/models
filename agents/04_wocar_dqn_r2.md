# Agent Task: `04_wocar_dqn_r2` — Cleanup, Verification, Smoke Test & Training Infrastructure

**EXECUTION MODE — READ THIS FIRST**

Do **not** work through all four parts in one continuous session. Execute strictly as separate phases with user approval gates between each. Stop and report at every gate:

- **Gate 1** → Part 1 only: remove the safety-score/best-frame machinery (incl. config keys), write `docs/01_eval_changes.md`. Report the diff summary, then STOP and wait for my go-ahead.
- **Gate 2** → Part 2 only: implementation verification against the paper, write `docs/02_implementation_verification.md`. Report pass/fail findings — especially any failed checks — then STOP and wait.
- **Gate 3** → Part 3 only: run the smoke test of all three model types, write `docs/03_smoke_test.md`. Report pass criteria results, then STOP and wait.
- **Gate 4** → Part 4 only: build `run_orchestrator.py` and `monitoring.py` (adapted from `watch_progress.py`, renamed — see Part 4b). Show me both scripts, then STOP and wait.
- **Gate 5** → Final review pass: checklist verification, README with launch commands. Report, then STOP. **Do not launch anything** — production launch is mine.

Never begin the next gate's work without my explicit go-ahead. If any gate reveals a problem (a failed verification check, a smoke-test failure), report it and stop — do not attempt to fix unrelated things. Token budget matters to me: keep each gate's response compact, no sprawling explanations, no re-printing entire files in your responses — cite file/function/line references instead.

---

## Scope — two hard limits

1. **Training-side only.** Evaluation, checkpoint selection, and post-training analysis are explicitly out of scope. The user handles those separately after the runs finish.
2. **You do not launch or babysit the 18-run production training.** You build and smoke-test everything, then hand the user the exact command. The production run (~2.5 days) is launched and supervised by the user. Do not start it, do not monitor it on your own initiative — only look at run progress if the user explicitly asks.

You are working in the `models` repo (branch `wocar-eval`; create a feature branch off it). The entry `entries/04_wocar_dqn_r2/` already exists as a copy of `03_wocar_dqn` — **work inside it directly**; do not create new entries. Never modify vendored code in `upstream/`.

## Model configs (the training interface)

Each model type is trained via its JSON config in the entry dir: **`config_highway_vanilla.json`, `config_highway_sadqn.json`, `config_highway_wocar.json`** (currently `"num_frames": 4000000`, `"path_prefix": "artifacts/3rd_run"`, `"seed": 1`). The orchestrator (Part 4a) invokes `train.py --config <file>` — these configs are its interface to training. Required config changes:

- **Part 1**: remove all safety-score-related keys (e.g. `safety_max_crash_rate`) and any config entries feeding the deleted `best_frame` logic. Leave all training hyperparameters untouched — they are verified in Part 2 and must not drift.
- **Part 4a prep**: repoint `"path_prefix"` away from `artifacts/3rd_run` to the new `artifacts/run_1/` layout, and plumb **`TRAIN_SEED` and per-run output dir as CLI/config overrides** (so six runs of the same model type differ only in seed and output directory, never in config substance). Prefer minimal plumbing: e.g. `train.py --config <file> --train_seed <i> --out-dir <dir>`, falling back to per-run generated configs (copies differing only in `seed`/`path_prefix`) only if train.py lacks CLI overrides.
- **Smoke mode**: the smoke test (Part 3) uses the same configs with an overridden small `num_frames` and `path_prefix: artifacts/smoke/`.

## Required reading (do this first)

| File | Why |
|---|---|
| `wocar_context.md` | Project context: three methods, configs, the 4 past bugs (§4 — do not reintroduce them) |
| `/Users/frederikbach/Downloads/paper.pdf` | WocaR-RL paper (Liang et al., NeurIPS 2022): Algorithm 2 (WocaR-DQN, p. 19) + SA-DQN description for implementation verification |
| The legacy 3rd-run runbook (user will paste/provide) | Operational details: cwd requirements, env vars, log format, checkpoint cadence, config invocation (`python train.py --config config_highway_<model>.json`) |

## Legacy runbook: keep vs. discard

**KEEP:**
- `train.py` must run with the entry dir (`entries/04_wocar_dqn_r2/`) as **cwd** — `read_config.py` opens `./defaults.json`; `train.py` appends `./common`, `./auto_LiRPA`, `./upstream` to `sys.path`. The orchestrator sets cwd per subprocess accordingly.
- Env vars per subprocess: `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.
- train.log format: progress lines look like `frame 4000000/4000000`; per-checkpoint eval rows contain `this test avg reward:`. Pin parsers to this and **calibrate against a real `artifacts/3rd_run/.../train.log`**.
- Checkpoints: `frame_<N>.pth` every 50,000 frames, plus the existing 50-episode mini_test every 50k with its train.log lines. Leave that logging untouched.
- TensorBoard logging: keep as-is (writes alongside checkpoints; event files tracked/committed per batch like train.log).
- **The existing `watch_progress.py`**: it already does bars/FPS/ETA parsing of train.log — it is the **starting point for Part 4b**, not a competitor. Read it first; adapt and extend it rather than writing a new monitor from scratch. Rename the adapted file to `monitoring.py`.

**DISCARD:**
- Saving `best_frame_<N>.pth` on new `safety_score = test_reward − 5*crash_rate` highs — removed in Part 1.
- The legacy "Post-Run Checkpoint Selection" step (keep top-3 by lowest crash / highest safety / final). **No checkpoint curation of any kind.**

**Artifact retention (explicit):** never delete or move any training artifact. No cleanup scripts, no backup-to-Desktop steps. All checkpoints, `.npy`s, logs, and tensorboard event files stay in place.

**Git:** remove `*.log` from `.gitignore` so train.log is tracked.

---

## Part 1 (Gate 1) — Remove the mini_test safety-score machinery (write `docs/01_eval_changes.md`)

Delete the composite `safety = return − 5 × crash_rate` mechanism and all `best_frame_*.pth` saving logic (find every occurrence; delete, don't stub) — in code **and in the model config files** (see "Model configs" above). Replace with nothing — unconditional 50k checkpoint saving already exists. Leave the mini_test evaluation and its train.log output untouched. Document: which files/functions/and config keys contained the score logic, what was deleted, confirmation the 50k cadence is unchanged, and that `TRAIN_SEED = model index (1..6)` is plumbed through as a CLI/config parameter for training runs.

## Part 2 (Gate 2) — Implementation verification (write `docs/02_implementation_verification.md`)

Pass/fail checklist with **evidence (file + function + line references; config key values)** per item.

**Vanilla DQN** (`config_highway_vanilla.json`): standard DQN (replay, target net, ε-greedy 1.0→0.05, Huber, grad clip 10, γ=0.99, lr=1.25e-4), argmax action. No robustness machinery — confirm its config carries no SA/WocaR keys.

**SA-DQN** (`config_highway_sadqn.json`): Q-network + `auto_LiRPA` CROWN-IBP state-regularization keeping argmax stable over L∞ box `[s−ε, s+ε]`: `epsilon: 0.0025`, linear ramp 50k→250k, CROWN beta 1.0→0.0, hinge `hinge_c: 1`, `bound_solver: cov`, `kappa: 0.001`. Equal weight on all states (no w(s)).

**WocaR-DQN** (`config_highway_wocar.json`, per paper Algorithm 2, p. 19): SA-DQN plus:
- Worst-attack critic Q_φ via TD on the worst-attack Bellman backup: IBP bounds of robust Q for s′, build Â_adv (a ∈ Â_adv iff its upper bound exceeds every other action's lower bound), `â = argmin_{a∈Â_adv} Q_φ(s′, a)`, bootstrap from **`worst_target_model`** (past bug #3c — verify fix survived the copy).
- Robust-Q blended target `y_r = r + γ·max_a′[κ_wst·Q_v′(s′,a′) + (1−κ_wst)·Q_φ(s′,a′)]`, `worst_ratio` ramped 0→0.1 over 50k→250k, running **every step** (past bug #3: `worst_rate: 1000` guard made it 1-in-1000 — verify unconditional dispatch).
- State-importance weight `w(s) = max_a Q(s,a) − min_a Q(s,a)`, batch-normalized, **detached**.
- `worst_kappa_end: 1.0` intentionally disables upstream's `worst_case_loss` (per upstream README advice) — document as an intentional deviation.
- Verify past bugs #1 and #2 stayed fixed through the copy: `QNetwork.bound_forward` single `compute_bounds` call, correct `(lower, upper)` order; `ibp.py` `worst_action_select` uses `>=` with greedy fallback, no 1e9 injection.
- Verify **total seeding**: one `TRAIN_SEED` drives network init, ε-greedy RNG, replay sampling, and training-env resets.
- Cross-check each config's values against `wocar_context.md` §3 exactly — any hyperparameter drift between the configs and the documented baseline is a failed check.

## Part 3 (Gate 3) — Smoke test (write `docs/03_smoke_test.md` + a `smoke_test` script)

Run all three model types ~60k frames each (crosses the 50k ramp start), single seed, into `artifacts/smoke/` (via config overrides — see "Model configs"), and verify:

- train.log exists; monitoring.py's regex parses the `frame N/M` lines;
- `frame_<N>.pth` saves on cadence; **no `best_frame_*` files appear** (confirms Part 1);
- orchestrator's batch transition works (with `TARGET_FRAMES` lowered to smoke length — advances vanilla → sa → wocar);
- monitoring.py shows bars for smoke runs and publishes PROGRESS files;
- git progress commit touches **only** the progress files.
Leave smoke artifacts in place. Document pass criteria + expected console output.

## Part 4 (Gate 4) — Deliverables: TWO separate scripts

### 4a. `run_orchestrator.py` — launching & sequencing only (terminal 1)

Standard library only. Run from repo root inside `tmux`. Responsibilities:

- Precreate (recreate missing, never overwrite existing train.log):
  ```
  artifacts/run_1/
  ├── vanilla_dqn/model_{1..6}/   → train.log, frame_*.pth
  ├── sa_dqn/     /model_{1..6}/
  └── wocar_dqn/  /model_{1..6}/
  ```
- Batch order **vanilla_dqn → sa_dqn → wocar_dqn**, 6 concurrent runs per batch. Each run invokes `train.py --config config_highway_<model>.json` with **`TRAIN_SEED = model index (1..6)`** and the per-run output dir override (per "Model configs"), cwd + env vars per legacy runbook. Machine has 6 cores; never exceed 6 concurrent trainers.
- A run is **finished** ⇔ exited 0 AND log reached `frame 4000000/4000000`; otherwise **FAILED**. Start the next batch automatically when all six are finished-or-failed. Failed runs marked, batch proceeds, `--retry 1` optional.
- At batch completion, commit that batch's six train.log files (and tensorboard event files) — tracked now; commit each once, here — not on any progress cadence.
- **No display duties.** It prints only lifecycle events (launching run X, run X finished/failed, starting batch N, final summary). The visual monitoring lives entirely in 4b.
- Flags: `--dry-run` (precreate folders, validate paths + regex against a real 3rd_run train.log, print the exact per-run command lines, launch nothing), `--resume` (reattach to existing runs via log state), `--smoke` (low TARGET_FRAMES + `artifacts/smoke/` mode for Part 3).

### 4b. `monitoring.py` — display & progress publishing only (terminal 2)

**Built by adapting and renaming the existing `watch_progress.py`** (rename the file; the original is preserved via git history — never lose its logic). Standard library only; independent of the orchestrator (polls train.logs, doesn't talk to the orchestrator process). Required changes/additions to the existing script:

- **Display, grouped in order: vanilla_dqn, then sa_dqn, then wocar_dqn** — all 18 rows visible from the start; unstarted batches show `queued` at 0%.
- Row: run name · `[####------]` · % · `frames / 4,000,000` · smoothed frames/sec · ETA · status ∈ {queued, training, stalled, done, failed}.
- Tail each train.log **incrementally** (byte offset, never rescan); regex pinned to `frame N/M`; stall detection: no log growth >30 min while training ⇒ `stalled`.
- Header: `batch <n>/3 · <k>/18 runs · elapsed hh:mm` (batch inferred from which runs have logs).
- **Progress publishing:** every 15 min (configurable), write and commit **two files at the repo root** — `PROGRESS.md` and `progress.json`. `PROGRESS.md` must render the **same view the terminal shows**: an ASCII-progress-bar column (in a fenced code block so bars survive Markdown), followed by a compact table with run / % / frames / rate / ETA / status, plus a timestamp. `progress.json`: same fields, machine-readable. Commit via `git commit --only PROGRESS.md progress.json -m "chore(progress): <timestamp>"` — nothing else ever gets swept in. Skip commit when nothing changed. PID/lock file prevents two monitoring instances.
- Ctrl-C exits monitoring.py only (never touches training processes); writes a final progress snapshot on exit.

## Gate 5 — Handoff

Write `README.md` in the entry with the exact two-terminal launch sequence, e.g.:
```
tmux new -s train
  1) python entries/04_wocar_dqn_r2/run_orchestrator.py          # terminal/session 1
  2) python entries/04_wocar_dqn_r2/monitoring.py                # terminal/session 2
```
plus the dry-run and smoke commands. **Then stop.** Do not launch the production run. Summarize: what was changed (Part 1), verification results (Part 2), smoke results (Part 3), and the exact command the user should run. If the user later asks you to "check progress," read `PROGRESS.md`/logs and report — nothing more.

## Deliverables checklist

- [ ] Safety-score/best-frame code AND config keys removed from `entries/04_wocar_dqn_r2/`; `upstream/` untouched
- [ ] Configs repointed (`path_prefix` → `artifacts/run_1/` layout) with TRAIN_SEED/output-dir plumbing; hyperparameters otherwise unchanged (verified in Part 2)
- [ ] `docs/01_eval_changes.md`, `02_implementation_verification.md`, `03_smoke_test.md`
- [ ] `smoke_test` script + passing smoke run for all 3 model types
- [ ] `run_orchestrator.py` (launch/sequence only, event-log output)
- [ ] `monitoring.py` — `watch_progress.py` adapted & renamed (18 grouped bars, stall detection, PROGRESS.md/json publishing with monitor-mirroring format); no duplicate monitor scripts left behind
- [ ] All four past-bug fixes verified intact; all three configs match `wocar_context.md` §3 exactly
- [ ] TensorBoard logging kept as-is
- [ ] `.gitignore`: `*.log` removed
- [ ] README with two-terminal launch instructions — **production run NOT launched by you**

Stop and ask before changing anything outside `entries/04_wocar_dqn_r2/`, the two progress files, and the `.gitignore` line.
