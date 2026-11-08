> ARCHIVED: legacy 3rd-run runbook; its best-frame / checkpoint-selection steps were discarded.

# Train 3rd Run Highway Models on CPU (Vanilla DQN / SA-DQN / WocaR-DQN)

Use this prompt to run one or more of the **3rd run** `highway-v0` models in `entries/02_wocar_dqn/` on **CPU** (forced via `CUDA_VISIBLE_DEVICES=""` and single-thread BLAS `OMP_NUM_THREADS=1`).

## Why CPU is forced
The Q-network for `highway-v0` is a small 3-layer MLP (`50 -> 128 -> 128 -> 5`, ~23k parameters). On a discrete GPU, PCIe tensor transfers and tiny CUDA kernel launch overheads are slower than running the matrix math directly in CPU L1/L2 cache. Locking `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` avoids OpenMP multi-thread contention and allows running multiple models in parallel.

## Models & Output Folders in `entries/02_wocar_dqn/artifacts/3rd_run/`
All three configs already set `"path_prefix": "artifacts/3rd_run"` and `"num_frames": 4000000`, so `train.py` automatically writes into the corresponding nested folder:

| Model | Config file | Output directory |
|---|---|---|
| **Vanilla DQN** | `config_highway_vanilla.json` | `entries/02_wocar_dqn/artifacts/3rd_run/highway-v0_vanilla/` |
| **SA-DQN** | `config_highway_sadqn.json` | `entries/02_wocar_dqn/artifacts/3rd_run/highway-v0_sadqn/` |
| **WocaR-DQN** | `config_highway_wocar.json` | `entries/02_wocar_dqn/artifacts/3rd_run/highway-v0_wocar/` |

---

## Setup & Run Instructions

> **Important:** `train.py` must be executed with `entries/02_wocar_dqn` as the working directory (because `read_config.py` opens `./defaults.json` and `train.py` appends `./common`, `./auto_LiRPA`, and `./upstream` to `sys.path`).

### Option A: Windows (PowerShell — Desktop `AtlasOS`)

```powershell
# 1) Pull latest changes on branch wocar-eval
git checkout wocar-eval
git pull

# 2) Activate venv (create if needed)
if (!(Test-Path .venv)) {
    python -m venv .venv
}
.\.venv\Scripts\Activate.ps1
pip install -r entries\02_wocar_dqn\requirements.txt
pip install torch torchvision opencv-python appdirs oslo.concurrency highway-env gymnasium tensorboard

# 3) Move into entries/02_wocar_dqn and force single-threaded CPU
cd entries\02_wocar_dqn
$env:CUDA_VISIBLE_DEVICES=""
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"

# Verify CPU mode (must print: cuda False)
python -c "import torch; print('cuda', torch.cuda.is_available())"

# 4) Run the assigned model (default for Desktop: SA-DQN; change config if instructed)
python train.py --config config_highway_sadqn.json
```

### Option B: macOS / Linux (zsh / bash)

```bash
# 1) Pull latest changes on branch wocar-eval
git checkout wocar-eval
git pull

# 2) Move into entries/02_wocar_dqn
cd entries/02_wocar_dqn

# 3) Run Vanilla DQN and/or WocaR-DQN (can run in 2 parallel terminals)
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../../.venv/bin/python train.py \
  --config config_highway_vanilla.json

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../../.venv/bin/python train.py \
  --config config_highway_wocar.json
```

## Live Progress Bar (Run in Another Terminal)
Once one or more training runs have started, run `watch_progress.py` in a separate terminal to see live progress bars, FPS, ETA, and latest `mini_test` scores for all active runs in `artifacts/3rd_run/`:

```bash
# macOS / Linux (from repo root)
python entries/02_wocar_dqn/watch_progress.py
```

```powershell
# Windows PowerShell (from repo root)
python entries\02_wocar_dqn\watch_progress.py
```

---

## Post-Run Checkpoint Selection (Same as `2nd_run`)
During the 4M-frame run, `train.py` runs a 50-episode `mini_test` every `50,000` frames and saves `frame_<N>.pth` (each `.pth` is only ~98 KB, so all 80 checkpoints take <8 MB total) as well as `best_frame_<N>.pth` whenever `safety_score = test_reward - 5 * crash_rate` reaches a new high.

**Do NOT delete `frame_<N>.pth` files during training.** When training finishes (`frame 4000000/4000000` in `train.log`):
1. Parse all `this test avg reward:` rows in `train.log` (just like `artifacts/2nd_run/README.md`) and identify the **top 3 checkpoints**:
   - The checkpoint with the **lowest crash rate** (breaking ties by highest reward)
   - The checkpoint with the **highest safety score** (`test_reward - 5 * crash_rate`)
   - The **final checkpoint** (`frame_4000000.pth`)
2. Keep inside `artifacts/3rd_run/highway-v0_<model>/`:
   - Those top `frame_<N>.pth` checkpoints (and `best_frame_<N>.pth`)
   - Their corresponding `frame_<N>_{rewards,losses}.npy` files
   - `rewards_losses_so_far.pdf`
   - `train.log` (note: `*.log` is in `.gitignore`, so use `git add -f` when staging `train.log`)
   - `tensorboard/` event file
   - A short `README.md` table (matching `artifacts/2nd_run/README.md`) listing the kept checkpoints and their `mini_test` reward, crash rate, and safety score.
3. **Never delete any checkpoints or `.npy` files.** Instead, move the remaining intermediate `frame_<N>.pth` and `frame_<N>_{rewards,losses}.npy` files (the ones not kept in the repo for git commit) into a backup folder on the user's Desktop (`~/Desktop/wocar_3rd_run_backup/highway-v0_<model>/` on macOS, or `$HOME\Desktop\wocar_3rd_run_backup\highway-v0_<model>\` on Windows PowerShell) before committing to git, so nothing is ever lost.

## Agent Instructions
1. Check with the user (or use **SA-DQN** `config_highway_sadqn.json` if on the Desktop and no model was specified) which of the 3 configs to run.
2. Ensure `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS="1"`, and `MKL_NUM_THREADS="1"` are set in the environment and `torch.cuda.is_available()` is `False`.
3. Run `train.py` with the working directory set to `entries/02_wocar_dqn`.
4. Monitor `entries/02_wocar_dqn/artifacts/3rd_run/highway-v0_<model>/train.log` until `frame 4000000/4000000` completes, then perform the post-run cleanup above.
