# Rev 4: a stronger regulariser (stages 4–6)

2026-11-14. Status: approved and implemented (`rev4.py`, `gate.py`, `report.py`, `docs/rev4x_runbook.md` steps 5–10).

## 1. Problem

Gate 2 passed, but SA-DQN is barely more robust than vanilla where it matters, which is crashes:

| at ε 0.005 | Vanilla (6 seeds) | SA-DQN (3 seeds) |
|---|---|---|
| Attack-induced crashes (gradient attacks) | +34.8 points | +32.1 points |
| Actions flipped by PGD | 55.8% | 9.9% |
| Certified actions | 0% | 66% |
| Noise-induced crashes (random noise) | +1.9 points | +6.0 points |

The gap between vanilla and SA-DQN in attack-induced crashes was also small in rev 3 at the training ε (6.6 points at
ε 0.005) and large only at ε 0.01 (18.7 points). Rev 4 has 2.7 and 7.2 points.

## 2. Diagnosis

- **Why a few flips are enough:** each forced action on SA-DQN costs about 6 points of crash probability per episode
  (the same at ε 0.00125 and 0.005). At a 9.9% flip rate that is about five forced actions per 55-step episode.
  Vanilla's flips are far more numerous (about 30 per episode) but mostly swap near-equivalent actions (about 1.4
  points each).
- **What it takes:** a flip rate of about 5% should give roughly +17 points, 1–2% roughly +5 (vanilla: +35). That
  needs 90–95% of actions certified, the regime of the SA-DQN Atari results, not the 66% of the first setting.
- **Which knob:** the regulariser is `-min(lower bound of the margin over the ε-ball, hinge_c)` (`train.py`, lines
  265–270). A state whose certified margin is below `hinge_c` is pushed with a constant gradient, weighted by κ against
  the TD loss. The uncertified states have a margin below zero, so they are pushed equally hard at any `hinge_c`; how
  hard is set by **κ**. A larger `hinge_c` acts indirectly: it keeps pushing states that are already certified, which
  rev 3 (`hinge_c` 1, 5.7% flips) suggests carries over to the others.
- **The reasoning error:** gate 1 set `hinge_c` to vanilla's median action gap (0.031). That measures how close
  vanilla's preferences are, not how large a margin has to be. Gate 2 asked for only 50% certified, so it passed.
- **WocaR in stage 3 has the same setting**, so it may be held back the same way.

Confounders: γ, the loss and the PGD step size also changed between rev 3 and rev 4, so the regulariser is tested
directly.

## 3. What changes

| Stage | Runs (6 cores) | Gate |
|---|---|---|
| 3 (was running) | SA-DQN first setting seeds 4–6 + WocaR first setting seeds 1–3 | `stage3`, now informational |
| 4 | SA-DQN arms × seeds 1–2: `hinge_c` 0.3; κ 0.1; both | `reg` |
| 5 | chosen SA-DQN seeds 3–5 + WocaR with the same regulariser seeds 1–3 | `stage5` |
| 6 | chosen SA-DQN seed 6 + WocaR seeds 4–6 + WocaR with the worst-case loss seeds 1–2 | — |
| final | 300 episodes: vanilla, chosen SA-DQN, WocaR, exploratory arm; ablation table from the gate evaluations | |

- **2×2 design:** with the first setting (`hinge_c` 0.031, κ 0.02), the three arms cover both knobs and their
  combination. Two seeds per arm are enough because the gate selects on the certified share, which varied by only
  three points between the first setting's seeds.
- **Gate `reg`:** among the arms that drive about as well as vanilla (clean crash ≤ vanilla + 8 points, clean return
  ≥ 85%), the highest certified share wins (within 2 points: the lower clean crash rate). It must be at least 5 points
  above the first setting, otherwise BLOCKED. Target 90%; below it the gate still picks and adds a note.
- **Gate `stage5`:** the former gate 3 on the new WocaR group, with certified ≥ the chosen SA-DQN's share − 10 points.
  One retry with `worst_ratio` 0.2; a second failure is BLOCKED.
- **Same settings for both:** WocaR gets the chosen `hinge_c` and κ, so the only difference to SA-DQN stays the
  worst-case value, as in the paper.
- **First setting kept:** `sa_dqn` (6 seeds) and `wocar_dqn` (3 seeds) appear in the report's ablation table.

## 4. Timing (estimates)

| Step | Ends about |
|---|---|
| Stage 3 + gate 3 evaluation | Sat 14 Nov 18:45 |
| Stage 4 + evaluation | Sun 15 Nov 01:30 |
| Stage 5 + evaluation | Sun 15 Nov 09:30 |
| Stage 6 | Sun 15 Nov 16:30 |
| Final evaluation | Sun 15 Nov 22:00 |

A WocaR retry adds about 7 hours.

## 5. What this does and does not promise

- **The gate selects on certification, not on crashes.** Choosing the regulariser by "whichever gives SA-DQN the largest crash advantage" on seeds 1–3 would be tuning towards the result we hope for. The rule asks only that SA-DQN does what SA-DQN is meant to do: certify most states, as it does in the paper's Atari experiments.
- **What a remaining small gap would mean:** if SA-DQN at 90% or more certified still barely beats vanilla in crashes, that is a real finding, backed by the ablation: on highway-env, stable actions alone do not prevent crashes.
- **Harm per flip may rise:** it already rose from about 1.4 points (vanilla) to about 6 (SA-DQN) as the policy got crisper, so fewer flips may buy less than the arithmetic in section 2 suggests.
- **Selection bias:** seeds 1–3 of the chosen arm also count in the final numbers. The bias is small, because the selection uses certification, not the reported crash metric. The article mentions it.
- **Cost of a high `hinge_c`:** if an arm is too strong, it shows up as worse clean driving (the rev 3 pattern: 6 of 7 seeds never chose IDLE). The health check catches that, and the gate leaves that arm out.
