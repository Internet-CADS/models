# Vanilla DQN 4M — teammate run (`highway-v0_vanilla`) — trained elsewhere

Vanilla (non-robust) DQN trained for 4M frames on the shared `highway-v0` env config, by teammate Casper S (commit `471c867` on branch `origin/casstef-vanilla`, "Add 4m vanilla model").

- Checkpoint: `best_frame_2450000.pth` (best safety score at frame 2,450,000; final frame 4,000,000 was not the best)
- TensorBoard: `tensorboard/events.out.tfevents.1793710046.Casper.25128.0`
- No `train.log` was committed for this run — the eval notebook falls back to parsing the TensorBoard event file for training curves.

TB highlights (frame / return / crash / safety):

- Best safety: frame `2,450,000` → return `18.00`, crash `0.06`, safety `17.92`
- Final frame `4,000,000` → return `17.62`, crash `0.10`, safety `17.12`

Each `.pth` holds `current_model.features.state_dict()` (`torch.save`).
Load with the same `model_setup()` / `QNetwork` from `entries/02_wocar_dqn/`.
