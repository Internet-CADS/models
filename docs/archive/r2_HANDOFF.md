> ARCHIVED: rev2 handoff from the Gate 3 era; the gates are done and the paths/branch names below are stale.

# HANDOFF — `04_wocar_dqn_r2` (continue from Gate 3)

You are picking this up in a **local** Claude Code session (real shell on the user's machine, torch + highway-env available). Earlier gates were done in a cloud sandbox with no torch, so **nothing has ever been executed** — Gate 3 is the first real run.

## Read first
1. `agents/04_wocar_dqn_r2.md` — the full brief. **Its EXECUTION MODE section rules**: strict gates, stop and report after each, never start the next gate without the user's explicit go-ahead, compact responses (cite file/function/line, don't reprint files), never launch or babysit the 18-run production, never delete/move artifacts, never edit `upstream/`, ask before touching anything outside `entries/04_wocar_dqn_r2/`, the two progress files, and the `.gitignore` `*.log` line.
2. `agents/wocar_context.md` (project context; its branch/path names are stale — see below).
3. `entries/04_wocar_dqn_r2/docs/01_eval_changes.md`, `docs/02_implementation_verification.md`.
4. `agents/train-highway-3rd-run-cpu.md` is the legacy runbook (cwd/env-var/log details). Its best-frame / checkpoint-selection steps are DISCARDED.

## Where things stand
- Branch **`wocar_r2`** (the brief says `wocar-eval` and context says `entries/02_wocar_dqn/`; both stale — work is in `entries/04_wocar_dqn_r2/` on `wocar_r2`). The brief asks for a feature branch off it; none was created. Ask the user whether to branch or commit on `wocar_r2` (so far commits went straight to `wocar_r2`).
- **Gate 1 DONE** (`8e7c792`): safety-score / `best_frame_*` logic and `safety_max_crash_rate` removed from `train.py` and the 3 configs. Mini_test row is now `this test avg reward: <r>, crash rate: <c>`.
- **Gate 2 DONE** (`d32ca75`): static verification, no runtime. All four past bugs intact; configs match `wocar_context.md` §3.
- **Gate 3 NEXT** (user said not to start a gate before they ask — wait for their go).

## Decisions already made by the user (do not relitigate)
- **F1 fixed:** gradient clip was a no-op (generator exhausted). Now `list(model.parameters())` at `train.py` l.290, 533, 544; clip at 10 is **active** (a deliberate deviation from upstream/3rd run). 3rd-run pre-clip grad norms: median ≈3, p99 ≈26–29, ~15–16% of points >10 after frame ~540k, so it should engage in later training, **not** in a 60k smoke run. Worth a tiny runtime check that clipping actually scales grads (norm after = 10).
- **F2 kept:** ε-greedy = 1.0→0.05 over frames 1k→51k, **then linearly to 0 at 4M** (`act_epsilon_decay_zero: true`, upstream behaviour). Use that exact wording in the README; no config change.
- Deviations D1–D4 in doc 02 are accepted/documented (blended target uses Q_v′ at blended argmax; worst critic every 4th step; one Q-net + hard target copy every 2000 frames; `worst_case_loss` disabled by `worst_kappa_end: 1.0`).
- Line endings: the repo stores **LF**. Don't introduce CRLF (a CRLF copy once turned a 15-line change into a 2,300-line diff).

## Facts the later gates depend on
- **Entry dir is cwd**: `read_config.py` opens `./defaults.json`; `train.py` appends `./common`, `./auto_LiRPA`, `./upstream` to `sys.path` (`argparser` lives in `upstream/`, not modified). Env vars per run: `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.
- **CLI already supports what's needed — no train.py change**: `python train.py --config config_highway_<m>.json --path_prefix <dir> training_config:seed=<i> training_config:num_frames=<n>` (`--path_prefix` overrides config; positional `key:subkey=value` overrides via `read_config.load_config`). `training_config.seed` is the single TRAIN_SEED (network init, ε-greedy, replay sampling, training-env resets).
- **Output dir quirk (Gate 4):** `train.py` writes to `<path_prefix>/highway-v0_<name_suffix>/` (suffix `_vanilla|_sadqn|_wocar`), not directly into `<path_prefix>`. The brief's layout is `artifacts/run_1/<model>/model_<i>/`, so per-run `--path_prefix artifacts/run_1/<model>/model_<i>` will yield `.../model_<i>/highway-v0_<suffix>/train.log`. Decide with the user whether to accept that extra folder or pass a different mechanism (don't edit `upstream/`; editing `train.py`'s prefix logic is inside scope but needs the user's OK since it's training code). Configs' `path_prefix` is still `artifacts/3rd_run` (repoint is part of Gate 4 prep).
- **Log format** (from `train.py`): progress line `\nframe {N}/{M}, learning rate: …, buffer beta: …, action epsilon: …` (begins a line with `frame `); a perf line starting `total time: …, speed: … frames/sec, … avg 10 episode reward: …`; mini_test row `this test avg reward: …, crash rate: …` every `mini_test: 50000` frames; `frame_<N>.pth` every `save_frame: 50000`; TensorBoard in `<prefix>/tensorboard/` (scalars incl. `train/grad_norm`, `test/return`, `test/crash_rate`).
- **`watch_progress.py` is stale after Gate 1:** its `TEST_RE` requires `safety score:` in the test row, which no longer exists. `monitoring.py` (Gate 4) must drop that and the safety column. It currently targets `artifacts/3rd_run/highway-v0_*` and 3 rows; Gate 4 makes it 18 grouped rows.
- **No sample `train.log` exists in the repo or on the user's disk** for the brief's "calibrate against a real `artifacts/3rd_run/.../train.log`" step (`*.log` is gitignored; the 3rd-run folder has only `.pth`/`.npy`/TensorBoard). Ask the user for one, or calibrate against the Gate 3 smoke logs. 3rd-run TensorBoard event files exist and were parsed for grad norms.
- `.gitignore` line 60 `*.log` still needs removing (brief's Git section; in scope).
- `test.py` l.97–100 still prefers `best_frame_*.pth` when present (evaluation script, out of scope, harmless now).
- `requirements.txt` is unpinned (`torch`, `numpy`, `gym[atari]`, `highway-env`); `highway_adapter.py` expects gymnasium-style `reset()`/no `env.seed` (seed is applied on the next reset).

## Gate 3 — what to do
Run all 3 model types ≈60k frames, single seed, into `artifacts/smoke/`, via config overrides (same configs, `training_config:num_frames=60000`, `--path_prefix artifacts/smoke/...`). Write a `smoke_test` script and `docs/03_smoke_test.md`. Pass criteria are in the brief (train.log + regex parse, `frame_<N>.pth` on the 50k cadence, **no `best_frame_*`**, batch transition vanilla→sa→wocar with lowered `TARGET_FRAMES`, monitoring shows bars + publishes PROGRESS files, git progress commit touches only the progress files). Note the last three need `run_orchestrator.py`/`monitoring.py`, which are Gate 4 deliverables — raise with the user how to sequence this before building them early (the brief's Gate order puts Gate 3 before Gate 4). Also check: same seed twice → identical `train.log` (seeding), and a quick check that grad clipping engages. Leave smoke artifacts in place. Report, then STOP.
