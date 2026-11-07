# WocaDQN / SA-DQN highway model

This entry trains a WocaR-DQN agent on `highway-v0` using the SA-DQN robust training path.

## Provenance — what is upstream vs. what we wrote

- **Upstream (pristine, not edited):** `upstream/` is an exact copy of
  `https://github.com/umd-huang-lab/WocaR-RL/tree/main/WocaR-DQN/SA-DQN`
  (commit `2856174b2c649ee377a63f1b15808cab74c6c699`, cloned 2023-09-13 to
  `/Users/frederikbach/extras/wocar_dqn_source`). That folder contains the WocaR SA-DQN
  implementation (`train.py`, `models.py`, `ibp.py`, `worst_action.py`, `environment.py`,
  `common/`, `config/` Atari JSONs, `auto_LiRPA/` submodule, etc.). Do not edit files under
  `upstream/` — they are the reference.

- **Our highway adaptation (new files, not in upstream):**
  - `highway_adapter.py` — `LegacyHighwayEnv` bridge: flattens `highway-v0` `Kinematics`
    observations to 1-D and exposes the 4-value `(obs, reward, done, info)` API SA-DQN expects;
    also surfaces `info["crashed"]` for evaluation.
  - `config_highway_wocar.json` (robust WocaR), `config_highway_sadqn.json` (SA-DQN baseline),
    `config_highway_vanilla.json` (non-robust vanilla DQN baseline) — `env_type: highway`,
    `DiscreteMetaAction`, normalized `Kinematics` (10 vehicles, 5 features).
  - `artifacts/highway-v0_wocar/frame_2450000.pth` — 2.45 M-frame pretrained checkpoint, stored like
    `entries/01_highway_dqn/artifacts/model.zip` (not under `upstream/`).

- **Patched files (same name as upstream, with header `# PATCHED vs upstream/...`):**
  `train.py`, `models.py`, `test.py`, `utils.py`, `my_replay_buffer.py`, `common/wrappers.py`,
  `ibp.py`, `auto_LiRPA/auto_LiRPA/{bound_general,bound_ops,parse_graph}.py`, `requirements.txt`.
  Changes are portability fixes (optional `auto_LiRPA`/`cpprb`/`Async*`/`IPython`, `gym`→`gymnasium`,
  `torch.device` CPU handling, `worst_target_model` fix, `pin_memory`/`masked_fill` fixes) plus the
  highway branch in train/test. See each file's header for the exact delta.
  `diff -rq upstream . --exclude=upstream --exclude=artifacts --exclude=README.md` shows exactly this set.

- **`auto_LiRPA`:** `upstream/auto_LiRPA/` is the upstream's submodule checkout; `auto_LiRPA/` at the
  entry root is the same code patched for newer PyTorch (`to(device)` vs `.cuda()`, `get_node_attribute`
  compat). Keep `upstream/auto_LiRPA/` pristine; the root copy is the patched one actually imported.

The upstream SA-DQN/WocaR docs warn that Atari wrapper differences can change observations and the
available action set, so results should identify the wrappers and config used — the same applies here
for the highway `Kinematics` wrapper.

## Setup

Run these commands from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The vendored `auto_LiRPA` package is used by the robust training path and does
not require a separate checkout (patched copy at `auto_LiRPA/`, pristine in `upstream/auto_LiRPA/`).

## Train

The robust highway configuration is:

```powershell
python train.py --config config_highway_wocar.json
```

Training checkpoints and metric arrays are written below the configured
`path_prefix`. The configuration uses normalized kinematic observations and
the `DiscreteMetaAction` action space; checkpoints from another environment
configuration must not be reused.

## Evaluate the pretrained model

Evaluate the included 2.45-million-frame checkpoint without rendering:

```powershell
python test.py --config config_highway_wocar.json `
  test_config:load_model_path=artifacts\highway-v0_wocar\frame_2450000.pth
```

To open the highway visualization, add:

```text
test_config:render=true
```

The robustness radius is an input-perturbation radius over the normalized
observation vector; it is not a direct collision-safety guarantee.
