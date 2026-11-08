# Models for our p7 project

Robust DQN on `highway-env` (vanilla DQN vs SA-DQN vs WocaR-DQN). Each experiment is an *entry* under `entries/`; project-wide documents are in `docs/`.

## Entries
| entry | what it is | state |
|---|---|---|
| `entries/01_highway_dqn` | first highway DQN notebook + noise evaluation | done |
| `entries/02_dqn_obs_noise_training` | DQN trained with observation noise, comparison notebook | done |
| `entries/03_wocar_dqn` | first vanilla / SA-DQN / WocaR-DQN baseline (single seed per model, 4M frames) | done |
| `entries/04_wocar_dqn_r2` | rev2: 3 models × 6 seeds, 4M frames, plus the script-based evaluation (`evaluate.py`, `report.py`, `visualize_perturbation.py`) and its cached results | trained on the training machine (WocaR was still finishing when last checked); peak-checkpoint evaluation done |
| `entries/05_wocar_dqn_r3` | rev3: 3 models × 3 seeds, 2M frames, denser traffic, stronger robust training; same evaluation scripts with rev3 defaults | prepared and smoke-tested, **not yet trained** — start with its `README.md` |

Branches: `wocar_r2` is rev2 (the training machine's monitor pushes progress commits there); `wocar_r3` carries rev3 work.

## Docs (`docs/`)
| file | content |
|---|---|
| `docs/paper.pdf` | the WocaR-RL paper (Liang et al., NeurIPS 2022) |
| `docs/wocar_context.md` | background on the three methods, the highway config and the four past bugs (historical paths) |
| `docs/rev2_verification/` | static verification of the training code against the paper, removal of the safety-score selection, smoke-test results (the training code is shared by rev2 and rev3) |
| `docs/archive/` | old agent briefs, handoffs and runbooks, kept for history |
| `docs/r3_agent_runbook.md` | step-by-step instructions for the agent that runs rev3 on the training desktop (pilot, go/no-go, full run, evaluation) |
| `docs/rev3.md` | every rev3 change, status, decisions and next steps |
| `docs/deviations.md` | where our setup differs from the paper and the authors' code, and the paper's notation slips |
| `PROGRESS.md`, `progress.json` | written by rev2's training monitor; do not edit |
