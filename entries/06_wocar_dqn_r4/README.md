# 06_wocar_dqn_r4 — rev 4: fix the base learner, then rebuild the comparison in gated stages

Status: **prepared and smoke-tested, nothing has been trained yet.** The step-by-step for the training agent is
`../../docs/rev4x_runbook.md`. This file explains what rev 4 is and why.

## Why rev 4
Rev 3 showed WocaR ahead of vanilla, but with small gaps and a robust-model handicap in normal driving. Looking at the
rev 3 checkpoints and at the authors' code gave three reasons:

1. **The base learner barely registered crashes and episode ends.** Rev 3 vanilla, seed 1: mean Q 43.5 against a
   realised discounted return of 11.6; Q 42.6 one step before the time limit; Q 31.7 one step before a crash (true
   value about −2). The Huber loss caps every error at 1 while Q sits near 45, and the 60-step time limit was stored as
   a real ending. All three models share this, and WocaR's worst-case critic inherits it.
2. **`hinge_c` = 1 was an Atari value.** SA-DQN's action gaps sat at about 1.0 (the hinge), against natural gaps of
   about 0.04, so the regulariser overrode real preferences.
3. **`upstream/` is the authors' fast rewrite, not the code behind the paper's tables.** In it the worst-case value
   only picks the next action and never enters the target as a number, and state importance w(s) is missing.

## What changed against rev 3
Code is a copy of `05_wocar_dqn_r3`. Environment and reward are unchanged (15 vehicles, 60 steps, collision −2).

| Change | Where | Default in rev 4 |
|---|---|---|
| The time limit is no longer stored as an ending; only a crash or leaving the road is | `highway_adapter.py` (`info["terminated"]`), `train.py` (`bootstrap_on_timeout`) | on |
| Squared-error loss instead of Huber, also for the worst-case critic and the bound-based worst-case loss | `natural_loss_fn: "mse"` | on |
| Discount factor chosen between 0.97 and 0.9 (laptop pilot ruled out 0.99) | stage 1 arms in `rev4.py` | gate 1 decides |
| `hinge_c` set from vanilla's measured action gaps | `gate.py stage1` | gate 1 decides (first setting; turned out too weak, see below) |
| Regulariser strength (`hinge_c`, κ) chosen by the certified share | stage 4 arms in `rev4.py`, `gate.py reg` | gate decides |
| Training and attack ε set where vanilla clearly breaks | `gate.py epsilon` | gate decides (≤ 0.05) |
| WocaR: the mixed natural / worst-case value is bootstrapped (paper, Algorithm 2 line 17), not only used to pick the action | `train.py` (`worst_target_mode: "value"`) | on, ratio 0.5 |
| WocaR: worst-case critic radius = training ε (paper) | `worst_eps_end` | on |
| WocaR: bound-based worst-case loss off (the paper has none; rev 3 had it at 0.5) | `worst_kappa_end: 1.0` | off; the exploratory arm `wocar_wcl_dqn` turns it on |
| Training length 1M frames (rev 3: 2M); ε ramp 100k→500k; worst-case ramp 100k→700k | configs | — |
| PGD step 2.5·ε/steps (rev 3: ε/steps, which cannot cross the ball from a random start) | `evaluate.py` | on |
| Report shows crashes without attack, total crashes under attack and the attack-induced part, with t-statistics | `report.py` | on |
| Logging of the mean number of actions the attacker can force | `ibp.py`, `train.py` | on |

Everything in `train.py` is switched by config; with the rev 3 configs it behaves as in rev 3.

**Correction to the earlier plan.** The first idea was "squared error and γ ≈ 0.9". A lower γ also makes a crash
cheaper (the lost future is 0.45/(1−γ): about 45 at γ 0.99, 4.5 at γ 0.9), so it may make the policy more reckless.
That is why γ is a pilot arm and not a fixed choice.

## Stages and gates (every stage is 6 runs, one per core, about 5–7 h)
| Step | Trains | Gate (`gate.py`) | Rule |
|---|---|---|---|
| Stage 1 | vanilla × γ 0.97 / 0.9 × seeds 1–3 | `stage1` | mean of the last 3 mini-tests: crash ≤ 15%, return ≥ 18; mean Q within 0.6–1.5 × the discounted return that followed (guards against diverging values). Lowest crash rate among passing arms wins. Sets γ and `hinge_c` (median action gap, clipped to 0.02–0.5). |
| ε sweep | — (attacks on the chosen arm, ε 0.005, 0.0075, 0.01, 0.015, 0.02, 0.05, 0.1) | `epsilon` | smallest ε ≤ 0.05 at which gradient attacks add ≥ 20 points to vanilla's crash rate |
| Stage 2 | vanilla seeds 4–6, SA-DQN seeds 1–3 | `stage2` | SA-DQN: clean crash ≤ vanilla + 8 points, clean return ≥ 85% of vanilla, certified actions ≥ 50%, PGD flip rate ≤ half of vanilla's. One retry: κ ×5 if too weak, κ ÷4 if too strong. |
| Stage 3 | SA-DQN seeds 4–6, WocaR seeds 1–3 (both with the first regulariser setting) | `stage3` | Informational (PASS or NOTED): the same checks as gate 5, but the run continues either way. |
| Stage 4 | SA-DQN × three regulariser arms × seeds 1–2: `hinge_c` 0.3; κ 0.1; both | `reg` | Among the arms with clean crash ≤ vanilla + 8 points and clean return ≥ 85% of vanilla, the highest certified share wins (within 2 points: the lower clean crash rate). It must certify at least 5 points more than the first setting. Target 90%. Crash rates under attack are not part of the rule. |
| Stage 5 | chosen SA-DQN seeds 3–5, WocaR (same regulariser) seeds 1–3 | `stage5` | WocaR: clean crash and return as above, certified ≥ the chosen SA-DQN's share − 10 points (at least 50%), worst-case critic bounded. One retry with `worst_ratio` 0.2. |
| Stage 6 | chosen SA-DQN seed 6, WocaR seeds 4–6, `wocar_wcl_dqn` seeds 1–2 | — | — |
| Final | — (300 episodes, all groups and seeds) | — | `report.py` |

Why these numbers:
- **Crash ≤ 15%, return ≥ 18.** Rev 3 vanilla had 23% and 23. "Always slow down" crashes 1% but earns about 3
  (measured, 80 episodes), and the cautious early phase of the γ 0.99 pilot earned 7–9. The pilot reached 10–14%
  at γ 0.97 / 0.9, so the bar is realistic. The bar is "clearly fewer
  crashes than rev 3 without giving up on driving".
- **Q within 0.6–1.5 × the discounted return that followed.** A guard against diverging or collapsed values, which
  WocaR's worst-case critic would inherit. It does not test the crash signal (see the pilot section).
- **ε where vanilla breaks.** With a better learner vanilla may shrug off ε 0.005 (about 1 m). The comparison needs an
  attack that hurts vanilla; 0.05 (10 m, 0.8 m sideways, 4 m/s) is the largest budget accepted.
- **Gates 2, 3 and 5 check health, not the result.** Three seeds are too few to gate on crash rates under attack, and
  whether WocaR beats SA-DQN is the question of the experiment.

The gates write `rev4_params.json`; the orchestrator, evaluation and report read it. A stage cannot start before its
gate has passed.

## Laptop pilot (one seed per γ, not part of any result)
Vanilla DQN with the rev 4 learner, seed 1, the full 1M-frame schedule; the mini-tests are 50 episodes (±5–6 points on
a crash rate), the last column 100 fresh episodes run with `diagnose.py`.

| γ | Crash, last 3 mini-tests | Return, last 3 | Crash, 100 fresh episodes | Q one step before a crash / mean Q | Crash sensitivity |
|---|---|---|---|---|---|
| 0.99 | 21% | 23.2 | 17% | 26.0 / 38.1 | 0.30 |
| 0.97 | 11% | 25.4 | 10% | 8.8 / 17.0 | 0.43 |
| 0.9 | 12% | 25.8 | 14% | 1.8 / 4.9 | 0.45 |
| Rev 3 vanilla (seed 1; 7-seed mean in brackets) | 19% (21%) | 23.4 (23.5) | 19% | 34.4 / 43.5 | 0.20 |

Crash sensitivity = (mean Q − Q one step before a crash) / (mean Q + 2): the share of a crash's cost that the value
function shows just before it.

- γ 0.97 and 0.9 crash about half as often as rev 3 and earn more; all three drive near top speed (29.7 m/s).
- γ 0.99 learns slowly and would fail gate 1, so stage 1 compares only 0.97 and 0.9, with three seeds each.
- The consistency check mean Q / mean discounted return (bootstrapped at the time limit) is 0.98–1.05 for all
  three, and 1.05 for rev 3 as well: at γ ≥ 0.97 the bootstrapped tail dominates a 60-step episode, so it only guards
  against diverging values. Gate 1 therefore decides on crash rate and return, and shows crash sensitivity for
  information.
- Reward kept: γ 0.9 was not more reckless than 0.97, so a larger collision penalty is not needed.

## Files
| File | Role |
|---|---|
| `rev4.py` | stages, groups, parameters, run log (shared by all scripts) |
| `run_orchestrator.py` | launches the runs of a stage (`--stage N`, `--dry-run`, `--smoke`) |
| `monitoring.py` | live view; writes `PROGRESS.md` and `progress.json` |
| `diagnose.py` | Q-values against realised returns, action gaps (gate 1 reads it) |
| `evaluate.py` | attacks and certification (`--preset epsilon / gate2 / gate3 / reg / gate5 / ablation / final`) |
| `gate.py` | the go/no-go checks; appends its tables to `RUNLOG.md` |
| `report.py` | HTML report (`artifacts/run_1/report.html`) |
| `config_highway_{vanilla,sadqn,wocar,wocar_wcl}.json` | fixed settings; gate-decided values are passed as overrides |
| `RUNLOG.md` | progress file of the training agent |

## Open points
- The gate thresholds are choices, not measurements. If gate 1 fails narrowly, that is a decision for the project, not
  for the training agent.
- `wocar_wcl_dqn` has only three seeds: a first look, not a result.

## Why stages 4–6 were added (2026-11-14)
Gate 2 passed, but SA-DQN with the first setting (`hinge_c` 0.031 from vanilla's action gaps, κ 0.02) was barely more
robust than vanilla where it counts: attack-induced crashes +32.1 against +34.8 points at ε 0.005, with 9.9% against
55.8% of actions flipped and 66% of actions certified. About five forced actions per episode remain, and one forced
lane change is enough for a crash. Setting `hinge_c` from vanilla's action gaps was a reasoning error: those gaps say
how close vanilla's preferences are, not how large a margin has to be. Stage 4 therefore tries a larger `hinge_c`
(the margin up to which a state keeps being pushed), a larger κ (the weight of the regulariser, which is what acts on
the states that are still uncertified) and both; `gate.py reg` picks by certified share, not by crash rates, so the
choice is not tuned towards the hoped-for result. The groups of the first setting (`sa_dqn`, `wocar_dqn`) stay in the
report as an ablation. Details: `docs/rev4x_regulariser_plan.md`.
