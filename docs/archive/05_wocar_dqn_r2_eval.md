> ARCHIVED: intro for the rev2 notebook-based evaluation (the notebook was removed; evaluation is now `evaluate.py` + `report.py` in the entry folders).

# 05 — Evaluation of the `04_wocar_dqn_r2` runs (intro for the eval agent)

You are picking up the **evaluation side** of `entries/04_wocar_dqn_r2/` (branch `wocar_r2`). Training was set up and launched by another agent/session; do not change training code or configs. Read `entries/04_wocar_dqn_r2/README.md` and `docs/HANDOFF.md` for the training setup. This file is what you need for evaluation.

## What was trained
18 DQN runs on `highway-v0`, 4,000,000 frames each, seeds 1–6 per model type:

| group | config | run dirs |
|---|---|---|
| vanilla DQN | `config_highway_vanilla.json` | `artifacts/run_1/vanilla_dqn/model_{1..6}/` |
| SA-DQN | `config_highway_sadqn.json` | `artifacts/run_1/sa_dqn/model_{1..6}/` |
| WocaR-DQN | `config_highway_wocar.json` | `artifacts/run_1/wocar_dqn/model_{1..6}/` |

Each run dir (flat layout, no `highway-v0_*` subfolder) contains `train.log`, `frame_<N>.pth` every 50,000 frames (80 per run), `tensorboard/`, `frame_<N>_{losses,rewards}.npy`. There are **no** `best_frame_*.pth` files and no safety score; the old "safety = return − 5×crash" selection was deliberately removed.

## Status of the data in git (as of the commit that added this file)
- **Vanilla and SA-DQN: complete.** All 80 checkpoints per run are committed. Only `frame_4000000_losses.npy` / `frame_4000000_rewards.npy` are committed per run (the per-frame `.npy` files are cumulative, ~15 GB total; the final pair contains the full history). `train.log` files are tracked too.
- **Per-group mean return curves:** `artifacts/run_1/vanilla_dqn/mean_return.csv` and `.../sa_dqn/mean_return.csv`. Columns: `frame, mean_return, std_return, min_return, max_return, mean_crash_rate, std_crash_rate`, 80 rows (50k…4M), mean/std over the 6 seeds (std uses ddof=1). Source: the `this test avg reward: <r>, crash rate: <c>` rows in each `train.log` (mini_test: 50 episodes with seeds 1000–1049, every 50k frames). Not a fresh evaluation.
- **WocaR-DQN: still training** when this was written (about 65 % done, expected to finish around 2026-11-09 01:00 +0100). Committed so far: checkpoints up to 2.6M, a snapshot of the six `train.log`s, and `wocar_dqn/mean_return_partial.csv` (52 points). The final state arrives as more commits on `wocar_r2`; `git pull origin wocar_r2` and check `PROGRESS.md` (updated by the monitor every 15 min) to see whether `wocar_dqn` shows 6/6 done. Do not evaluate WocaR as final before then. When the runs are done, regenerate the WocaR curve as `mean_return.csv` with the same method (parse `train.log`, `^frame N/M` for the frame, `this test avg reward:\s*(\S+),\s*crash rate:\s*(\S+)` for the row; note the number can be padded with extra spaces).

## Evaluation tooling (already adapted to the flat layout)
- `entries/04_wocar_dqn_r2/test.py`: single-run evaluator. Now supports `flat_run_dir`, and no longer looks for `best_frame_*`. Example (run from the entry dir):
  `python test.py --config config_highway_sadqn.json --path_prefix artifacts/run_1/sa_dqn/model_1 flat_run_dir=true test_config:load_model_path=artifacts/run_1/sa_dqn/model_1/frame_4000000.pth test_config:num_episodes=100`
  Without `load_model_path` it takes the latest `frame_<N>.pth` in the run dir. It writes `test.log` into the run dir (use another `--path_prefix` plus `load_model_path` if you do not want files inside `artifacts/run_1/`). Attack/certify options live in `test_config` (see the config files).
- `entries/04_wocar_dqn_r2/02_wocar_evaluation.ipynb`: the main evaluation notebook (clean rollouts, noise/PGD/MinBest/worst-Q sweeps over ε, certified action ratio via auto_LiRPA). It now builds `MODELS` for all 18 runs from `RUN_ROOT`/`GROUPS`/`SEEDS`, evaluates `CHECKPOINT_FRAME` (default 4,000,000, the same frame for every run), skips runs that have not reached 4M, plots seed-mean training/mini_test curves per group, and has no safety-score parts. Output cells were cleared. Evaluation results are cached per checkpoint in `artifacts/eval_cache/`. Cost is high (per run: 300 episodes × 4 attacks × 8 ε), so subset the runs while iterating.
- Both were smoke-checked headless (test.py on a finished SA-DQN checkpoint; notebook cells run against the real logs and one 2-episode eval). They have **not** been run in full.

## Environment
- Needs torch (CPU is enough), gymnasium, `highway-env`, `tensorboard`, `ipywidgets`, `pandas`, `matplotlib`, `ipython`, `setproctitle`, `termcolor`, `appdirs`, `oslo.concurrency`, `opencv-python`. The author's setup was a repo-root `.venv` (Python 3.12, torch 2.15 CPU, highway-env 1.12.1); `.venv` is gitignored, so recreate it. See `entries/04_wocar_dqn_r2/requirements.txt` and the README "Environment" section.
- Run scripts with cwd = `entries/04_wocar_dqn_r2/` (they use relative `./common`, `./auto_LiRPA`, `./upstream`). `auto_LiRPA` is a submodule/vendored dir; check it is present.
- Use `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `CUDA_VISIBLE_DEVICES=""` if you want to match training.

## Rules from the user
- Training-side code, configs and hyperparameters stay unchanged. Never delete or move anything under `artifacts/`.
- No checkpoint curation was done during training. Do not silently introduce a selection rule: evaluate the same frame for every run (default 4M) unless the user says otherwise.
- Commit only what you produce (explicit pathspecs, no `git add -A`), avoid touching `PROGRESS.md`/`progress.json` (written by the monitor), and do not add the large cumulative `.npy` files.
- Commit trailers used in this repo: `Co-Authored-By: <model name> <noreply@anthropic.com>` and `Claude-Session: <url>`. The repo-local git identity is set per machine; if `git commit` says the author is unknown, ask the user which name/email to use.

## Open decisions (ask the user, do not guess)
1. Which checkpoint(s) per run get the full attack/certification evaluation: final 4M frame for all (the current default), or something else (e.g. a fixed set of frames for learning-curve-vs-robustness plots).
2. Attack budgets and the set of attacks: the notebook's defaults are ε ∈ {0, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05}, attacks random_linf / pgd / minbest / worst_q, 300 episodes (seeds 0–299). Training ε is 0.0025.
3. How to aggregate across seeds for the report (mean ± std over 6 seeds is what the CSVs use) and which figures/tables the user wants for each group comparison (vanilla vs SA-DQN vs WocaR-DQN).
4. Whether the per-group results should be written back into the repo (e.g. under `entries/04_wocar_dqn_r2/docs/` or `artifacts/`) and in what form.

## Known quirks
- Unfinished runs: `INCLUDE_IN_PROGRESS_RUNS=False` in the notebook means WocaR runs without `frame_4000000.pth` are skipped.
- mini_test return/crash numbers in `train.log` come from the training-time evaluation (50 episodes, seeds 1000–1049) with the greedy policy; the notebook uses its own seeds and 300 episodes, so values will differ a little.
- At frame 50,000 all three groups of the same seed have identical weights (same computation until the SA/WocaR ramps start), so the first curve point is identical across groups.
- The monitor's elapsed-time header in `PROGRESS.md` is wrong on this setup; ignore it.
