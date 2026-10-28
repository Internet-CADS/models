# WocaDQN / SA-DQN highway model

This entry contains the WocaDQN implementation based on the SA-DQN code path,
including the highway environment adapter, training configuration, evaluation
script, and the 2.45-million-frame pretrained checkpoint.

## Setup

Run these commands from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The vendored `auto_LiRPA` package is used by the robust training path and does
not require a separate checkout.

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

## Source note

This entry tracks the local WocaR-RL SA-DQN implementation used to produce the
checkpoint. The upstream implementation warns that wrapper differences can
change observations and the available action set, so results should identify
the environment wrappers and configuration used.
