# Deviations of the rev3 WocaR-DQN setup from the paper and the authors' code

Sources: the paper (`docs/paper.pdf` at the repo root, Algorithm 2 on p.20, D.1.2 on p.22), the vendored rewrite in `upstream/` (SA-DQN-based; its README says it differs from the paper),
and the authors' other implementation `WocaR-DQN/RA-DQN/` in https://github.com/umd-huang-lab/WocaR-RL (Atari-only). **The RA-DQN facts below were read through a page-summarising
tool (main.py, train.py), not verbatim; spot-check before relying on exact numbers.** Deviations that merely come from using the rewrite (single acting network + target + worst critic,
blended-argmax bootstrap, hard target copies, ε-greedy decaying to 0) are not listed: RA-DQN shares the network trio and the blended-argmax bootstrap.

## A. Remaining deviations (ours), ranked by expected importance
| # | Deviation | Status |
|---|---|---|
| 1 | Robust signal. RA-DQN trains `κ·TD + (1−κ)·bound-based worst-case loss` (κ 1→0.5 on the ε ramp), with **no** SA-style regulariser and **no** w(s). Ours: SA-style regulariser (`kappa`) for SA-DQN and WocaR, plus the worst-case loss at `worst_kappa_end` 0.5 for WocaR only. | Rev3 WocaR now has the authors' main mechanism; SA-DQN only the weaker regulariser, so a WocaR gap over vanilla is more likely than an SA-DQN gap |
| 2 | Regulariser weight κ was 0.001 (paper: 0.005 / 0.01 / 0.02). | Fixed in rev3: 0.02 |
| 3 | `worst_ratio`: rev2 ramped 0→0.1; paper (D.1.2) delays and reaches 0.5 (ambiguous, see B1); RA-DQN uses a constant 0.05; upstream configs 0.01–0.1. | Rev3: 0.1 (ramped with the ε ramp) |
| 4 | Worst-critic bound radius: RA-DQN fixed 1e-3; ours ramps to `worst_eps_end` (rev2: 0.0025). A wide radius makes the worst critic more pessimistic (rev2: worst-Q drifted to −325). | Rev3: `worst_eps_end` 0.001 |
| 5 | `bound_solver` `cov` (convex relaxation). Authors' rewrite configs / pretrained models use `pgd`; RA-DQN uses interval bounds. | Open: try `pgd` on one SA-DQN pilot run only if cov shows no gap (it removes certification) |
| 6 | No Double DQN (paper and RA-DQN use it). Applies to all groups. | Open; small code change, deferred |
| 7 | Worst critic updated every 4th step (`worst_rate` 4). Authors: 10 (RA-DQN default), 1 and 100 (rewrite configs); the paper says every step. | Kept at 4 (no clear authors' value) |
| 8 | w(s) is in the paper but absent from RA-DQN; our code has it (added by the project). | Informational |
| 9 | Gradient clipping at 10 is active (F1 fix; upstream's was a no-op). | Kept |
| 10 | Domain adaptation from Atari (batch 64, lr 1.25e-4, buffer 50k, replay start 1k, ε-greedy decay over 50k frames; Atari defaults 32 / 6.25e-5 / 200k / 50k / 1.5M). Relevance not examined. | Informational |
| 11 | 3 seeds × 100 episodes (paper: median of 30 agents, 1,000 episodes). | Compute-limited |
| 12 | ε ramp: RA-DQN ramps over the whole run (≥3M frames); ours 100k–600k. | Kept (gives the regulariser time without a huge budget) |

## B. Slips in the paper (the authors' intended meaning)
1. **Algorithm 2, line 17.** Printed: `y^r = r + γ max_a′ [κ_wst·Q_v′ + (1−κ_wst)·Q_φ]`, i.e. κ_wst on the natural Q. **Intended (moderate-high confidence):** κ_wst weights the worst critic. Reasons: D.1.2(c) starts κ_wst at 0 and raises it to 0.5 (otherwise the first 2M steps would train on pure worst-case values); the PPO version weights the worst-case term by κ_wst; the authors' code ramps/uses the worst-critic weight from ~0. (`docs/rev2_verification/02_implementation_verification.md` D1 restates the formula literally as κ_wst = 1 − `worst_ratio`; that is only a doc note, the code uses `worst_ratio` as the worst-critic weight.)
2. **Line 14.** `Q_φ(s′, â; θ)`: θ should be φ (typo). High confidence.
3. **Line 12.** `Q_u(a) > Q_l(a′), ∀a′` includes a′ = a and can leave the set empty; intended ≥ excluding a′ = a. High confidence (rev2 bug #2 found the same failure in upstream code; our `ibp.py` uses ≥ plus a greedy fallback).
4. Upstream README typo: "RS-DQN" is the folder `RA-DQN`.

## C. Recommended changes (applied in rev3 where marked)
1. `worst_ratio` 0.1, `worst_eps_end` 0.001 — **applied**. Keep `worst_kappa_end` 0.5 (RA-DQN default) and `worst_rate` 4.
2. Run it and look at the results (no pilot, no gate). Fallback if WocaR diverges: `worst_kappa_end` 1.0.
3. If there is no gap: κ 0.05 first, then one SA-DQN run with `bound_solver: pgd`.
4. Deferred code changes: Double DQN for all groups; (if WocaR adds nothing over SA-DQN) read RA-DQN's `ibp.worst_action_select` and compare with ours.
