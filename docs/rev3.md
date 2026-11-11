# Rev 3: changes, decisions, compute and deviations (`entries/05_wocar_dqn_r3`)

Goal: a setup in which robust training shows a visible gap to vanilla DQN, within a smaller compute budget. Results and the reasoning per change are in the top-level `README.md`; this file keeps the details. The reward definition is **unchanged** (collision −2.0, right lane 0.1, high speed 0.4, `normalize_reward: false`). Code is a copy of rev2's (`04_wocar_dqn_r2`); only the three `config_highway_*.json` files and the run constants/flags differ.

## 1. Changes relative to rev2

| Setting | rev2 | rev3 | Config key | Files |
|---|---|---|---|---|
| Training frames | 4,000,000 | 2,000,000 (40 checkpoints of 50,000) | `num_frames` | all |
| Seeds per group | 6 | 7 (3 first; seeds 4–7 added, see section 4) | `--seeds` of `run_orchestrator.py` | — |
| Other vehicles | 10 | 15 (25 was tried first and aborted, see section 3) | `env_config.vehicles_count` | all |
| Episode length | 40 | 60 | `env_config.duration` | all |
| Training epsilon | 0.0025 | 0.005 | `epsilon` | sadqn, wocar |
| kappa | 0.001 | 0.02 | `kappa` | sadqn, wocar |
| Epsilon ramp | frames 50k–250k | 100k–600k | `schedule_start`, `schedule_length` | sadqn, wocar |
| Worst-critic mixing ratio | 0.1 | 0.1 (unchanged; ramped with the eps ramp) | `worst_ratio` | wocar |
| Worst-critic bound radius | 0.0025 | 0.001 (RA-DQN uses a fixed 1e-3) | `worst_eps_end` | wocar |
| Worst ramp length | 200k | 1,000,000 | `worst_schedule` | wocar |
| Worst-case loss weight | 1.0 (off) | 0.5 | `worst_kappa_end` | wocar |
| Evaluation | 100 episodes | 300 episodes, peak-of-mean frame, 4 attacks, certification | `evaluate.py`, `report.py` | — |

Notes
- `train.py` ramps `worst_eps`, `worst_ratio` and `worst_kappa` together: `coeff = (frame − schedule_start) / worst_schedule`, so they start with the epsilon ramp (frame 100k) and reach their end values at 1.1M frames.
- Checkpoint and mini-test interval stay at 50,000, so the final checkpoint `frame_2000000.pth` is also the last mini-test. The episode length (60) is deliberately not aligned with frames: crashes end episodes early.
- Evaluation: `evaluate.py` takes the peak-of-mean frame of each group (the argmax of the seed-mean mini-test curve over all seeds present), PGD + worst-Q + MinBest + random L∞, eps 0 / 0.0025 / 0.005 / 0.0075 / 0.01, certification; `report.py` writes one HTML report. Launcher flags: `run_orchestrator.py --groups --seeds --concurrent`.
- Decisions already made: reward and lanes unchanged (ε keeps its physical meaning through the y-normalisation); WocaR runs with the worst-case loss on (RA-DQN default; the upstream README advises 1.0 because of instability), `worst_ratio` 0.1, `worst_eps_end` 0.001.

## 2. Evaluated frames
Rev3 evaluates each group's peak-of-mean frame only (vanilla 1.5M, SA-DQN 1.6M, WocaR 1.45M with 7 seeds). The final frame was evaluated once for seeds 1–3 (cached, not reported): the robust models are visibly worse there than at their peak, because they dip late in training.

## 3. Compute (measured)
Simulator speed, one thread, random actions, no network (Mac; episode length does not matter, vehicle count does):

| vehicles | env steps/s | vs 10 vehicles |
|---|---|---|
| 10 | 129 | 1.0× |
| 15 | 88 | 1.5× slower |
| 18 | 72 | 1.8× slower |
| 20 | 63 | 2.1× slower |
| 25 | 47 | 2.7× slower |

- **What happened:** the first launch (25 vehicles, 12 concurrent runs) ran at ~20 frames/s per run (~28 h projected) and was stopped after ~40 minutes. The 12 hyperthreaded runs together gave no more throughput than the 6 physical cores (~240 frames/s), so wall time scales with the number of runs. The restart used 15 vehicles and 3 seeds (9 runs): measured ~14–15 h on the desktop (vanilla 14.0 h, SA-DQN 14.9–15.2 h, WocaR 14.2 h), ~329 frames/s in total, 37–42 frames/s per run.
- **Scheduling lesson:** with 9 jobs on 12 threads, three cores carry one job and three carry two, and a shared core gives each job about half speed. Restricting the trainers to one logical processor per physical core (affinity mask 0x555 on the desktop) lets Windows time-slice fairly; a priority bump for WocaR gets it done earlier. Approximate CPU time per run at 15 vehicles: vanilla ~8 h, SA-DQN ~9.3 h, WocaR ~9.8 h.
- **Rough projection (estimate; capacity ≈ 6 runs' worth, per-run speed alone ≈ 1 / (simulator time + ~5 ms)):**

| runs \ vehicles | 25 | 20 | 15 | 10 |
|---|---|---|---|---|
| 12 | ~29 h | ~23 h | ~18 h | ~14 h |
| 9 | ~22 h | ~17 h | ~13 h | ~10 h |
| 6 | ~15 h | ~12 h | ~9 h | ~7 h |

- Mini-tests cost about 3% of training time (rev2) and probably ~4% in rev3. Evaluating at 300 episodes takes about 45–55 minutes per job (all four attacks plus certification); 21 jobs took 2 h 16 min on 9 workers.
- The aborted 25-vehicle attempt used `artifacts/run_1` (not part of the repo, not used for any result); all rev3 results are in `artifacts/run_2`.

## 4. Extra seeds (4–7)
With 3 seeds per group the seed-to-seed spread of crash rates under attack (5–12 points) dominates the comparison, and more evaluation episodes cannot reduce it. Seeds 4 and 5 were trained on the desktop (6 runs) and seeds 6 and 7 on the laptop (M1 Pro, 6 runs, 74–80 frames/s per run before the epsilon ramp, ~50 after, `caffeinate`). The peak frame is chosen from the mean over all seeds, so adding seeds moved it for all three groups and all 21 runs were re-evaluated.

## 5. Deviations from the paper and the authors' code
Sources: the paper (`docs/paper.pdf`, Algorithm 2 on p.20, D.1.2 on p.22), the vendored rewrite in `upstream/` (SA-DQN-based; its README says it differs from the paper), and the authors' other implementation `WocaR-DQN/RA-DQN/` in https://github.com/umd-huang-lab/WocaR-RL (Atari-only). **The RA-DQN facts below were read through a page-summarising tool, not verbatim; spot-check before relying on exact numbers.** Deviations that merely come from using the rewrite (single acting network + target + worst critic, blended-argmax bootstrap, hard target copies, ε-greedy decaying to 0) are not listed: RA-DQN shares them.

### A. Remaining deviations (ours), ranked by expected importance
| # | Deviation | Status |
|---|---|---|
| 1 | Robust signal. RA-DQN trains `κ·TD + (1−κ)·bound-based worst-case loss` (κ 1→0.5 on the ε ramp), with **no** SA-style regulariser and **no** w(s). Ours: SA-style regulariser (`kappa`) for SA-DQN and WocaR, plus the worst-case loss at `worst_kappa_end` 0.5 for WocaR only | In rev3 WocaR has the authors' main mechanism and SA-DQN only the weaker regulariser; rev3 shows WocaR clearly more robust than SA-DQN's regulariser alone |
| 2 | Regulariser weight κ was 0.001 (paper: 0.005 / 0.01 / 0.02) | Fixed in rev3: 0.02 |
| 3 | `worst_ratio`: rev2 ramped 0→0.1; paper (D.1.2) delays and reaches 0.5 (ambiguous, see B1); RA-DQN a constant 0.05; upstream configs 0.01–0.1 | Rev3: 0.1 (ramped with the ε ramp) |
| 4 | Worst-critic bound radius: RA-DQN fixed 1e-3; ours ramped to `worst_eps_end` (rev2: 0.0025). A wide radius makes the worst critic more pessimistic (rev2: worst-Q drifted to −325 … −2300) | Rev3: 0.001 |
| 5 | `bound_solver` `cov` (convex relaxation). The authors' rewrite configs / pretrained models use `pgd`; RA-DQN uses interval bounds | Open: `pgd` would remove certification (rev 4 plan, direction C) |
| 6 | No Double DQN (paper and RA-DQN use it). Applies to all groups | Open; small code change (rev 4 plan, direction C) |
| 7 | Worst critic updated every 4th step (`worst_rate` 4); authors: 10 (RA-DQN), 1 and 100 (rewrite configs); the paper says every step | Kept at 4 |
| 8 | w(s) is in the paper but absent from RA-DQN; our code has it (added by the project) | Informational |
| 9 | Gradient clipping at 10 is active (rev2 fix; upstream's was a no-op) | Kept |
| 10 | Domain adaptation from Atari (batch 64, lr 1.25e-4, buffer 50k, replay start 1k, ε-greedy decay over 50k frames; Atari defaults 32 / 6.25e-5 / 200k / 50k / 1.5M). Relevance not examined | Informational |
| 11 | 7 seeds × 300 episodes (paper: median of 30 agents, 1,000 episodes) | Compute-limited |
| 12 | ε ramp: RA-DQN ramps over the whole run (≥ 3M frames); ours 100k–600k | Kept |

### B. Slips in the paper (the authors' intended meaning)
1. **Algorithm 2, line 17.** Printed: `y^r = r + γ max_a′ [κ_wst·Q_v′ + (1−κ_wst)·Q_φ]`, i.e. κ_wst on the natural Q. **Intended (moderate-high confidence):** κ_wst weights the worst critic. Reasons: D.1.2(c) starts κ_wst at 0 and raises it to 0.5 (otherwise the first 2M steps would train on pure worst-case values); the PPO version weights the worst-case term by κ_wst; the authors' code ramps the worst-critic weight from ~0. (`docs/archive/rev2_verification/02_implementation_verification.md` D1 restates the formula literally as κ_wst = 1 − `worst_ratio`; that is only a doc note, the code uses `worst_ratio` as the worst-critic weight.)
2. **Line 14.** `Q_φ(s′, â; θ)`: θ should be φ (typo). High confidence.
3. **Line 12.** `Q_u(a) > Q_l(a′), ∀a′` includes a′ = a and can leave the set empty; intended ≥ excluding a′ = a. High confidence (rev2 bug #2 found the same failure in upstream code; our `ibp.py` uses ≥ plus a greedy fallback).
4. Upstream README typo: "RS-DQN" is the folder `RA-DQN`.
