> ARCHIVED: legacy GPU runbook for the 3rd run.

# Train WocaR highway on GPU

Use this prompt to start a full robust training run on **GPU** (any CUDA device). On Mac (CPU, `torch.cuda.is_available()==False`) the same command falls back to CPU automatically — no change needed.

## One-shot setup + run (from repo root `models/`)

```bash
cd /Users/frederikbach/PycharmProjects/models

# 1) venv + deps (run once)
python3 -m venv .venv && source .venv/bin/activate
pip install -r entries/02_wocar_dqn/requirements.txt
pip install torch torchvision opencv-python appdirs oslo.concurrency highway-env gymnasium

# 2) verify GPU visible (True on CUDA box, False on Mac)
python -c "import torch; print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# 3) full robust run — 3M frames, ~4-8h on GPU, ~8-15h on Mac CPU
#    Artifacts go to artifacts/highway-v0_wocar/ like entry 01 (frame_*.pth + train.log + plot)
python entries/02_wocar_dqn/train.py \
  --config entries/02_wocar_dqn/config_highway_wocar.json \
  training_config:num_frames=3000000

# 4) quick smoke test instead (3k frames, ~30s, writes to .tmp/test_run/)
# python entries/02_wocar_dqn/train.py \
#   --config entries/02_wocar_dqn/config_highway_wocar.json \
#   training_config:num_frames=3000 training_config:save_frame=1500 path_prefix=.tmp/test_run


```

## Notes

- `upstream/` is pristine `umd-huang-lab/WocaR-RL/WocaR-DQN/SA-DQN` (commit 2856174). Patched files at `entries/02_wocar_dqn/` (`train.py`, `models.py`, `test.py`, etc.) already do `torch.cuda.is_available()` checks and `bias.to(device)` — no code change needed for GPU.
- `auto_LiRPA` at entry root is the patched copy for newer torch; `upstream/auto_LiRPA` is pristine.
- To force CPU on a GPU box: `CUDA_VISIBLE_DEVICES="" python entries/02_wocar_dqn/train.py --config ...`
- Logs/plots: `entries/02_wocar_dqn/artifacts/highway-v0_wocar/train.log` + `rewards_losses_so_far.pdf` (from `train.py:641`/`1070`).

## Agent instructions

When the user says "train highway wocar on gpu", run the setup + full run above, ensure `torch.cuda.is_available()` is True on the GPU host before starting, and tail `train.log`.
