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
| Discount factor chosen by a pilot (0.99, 0.95 or 0.9) | stage 1 arms in `rev4.py` | gate 1 decides |
| `hinge_c` set from vanilla's measured action gaps | `gate.py stage1` | gate 1 decides |
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

## Stages and gates (every stage is 6 runs, one per core, about 5 h)
| Step | Trains | Gate (`gate.py`) | Rule |
|---|---|---|---|
| Stage 1 | vanilla × γ 0.99 / 0.95 / 0.9 × seeds 1–2 | `stage1` | mean of the last 3 mini-tests: crash ≤ 15%, return ≥ 18; mean Q within 0.6–1.5 × the realised discounted return. Lowest crash rate among passing arms wins. Sets γ and `hinge_c` (median action gap, clipped to 0.02–0.5). |
| ε sweep | — (attacks on the chosen arm, ε 0.005–0.1) | `epsilon` | smallest ε ≤ 0.05 at which gradient attacks add ≥ 20 points to vanilla's crash rate |
| Stage 2 | vanilla seeds 3–5, SA-DQN seeds 1–3 | `stage2` | SA-DQN: clean crash ≤ vanilla + 8 points, clean return ≥ 85% of vanilla, certified actions ≥ 50%, PGD flip rate ≤ half of vanilla's. One retry: κ ×5 if too weak, κ ÷4 if too strong. |
| Stage 3 | SA-DQN seeds 4–6, WocaR seeds 1–3 | `stage3` | WocaR: same driving and certification checks, worst-case critic bounded. One retry with `worst_ratio` 0.2. If it still fails, the final evaluation runs without stage 4. |
| Stage 4 | WocaR seeds 4–6, vanilla seed 6, `wocar_wcl_dqn` seeds 1–2 | — | — |
| Final | — (300 episodes, all groups and seeds) | — | `report.py` |

Why these numbers:
- **Crash ≤ 15%, return ≥ 18.** Rev 3 vanilla had 23% and 23. "Always slow down" crashes 1% but earns about 3
  (measured, 80 episodes), and the cautious early phase of the γ 0.99 pilot earned 7–9. So the bar is "clearly fewer
  crashes than rev 3 without giving up on driving".
- **Q within 0.6–1.5 × realised.** Rev 3 was at 3.7. WocaR's worst-case critic needs values that mean something.
- **ε where vanilla breaks.** With a better learner vanilla may shrug off ε 0.005 (about 1 m). The comparison needs an
  attack that hurts vanilla; 0.05 (10 m, 0.8 m sideways, 4 m/s) is the largest budget accepted.
- **Gates 2 and 3 check health, not the result.** Three seeds are too few to gate on crash rates under attack, and
  whether WocaR beats SA-DQN is the question of the experiment.

The gates write `rev4_params.json`; the orchestrator, evaluation and report read it. A stage cannot start before its
gate has passed.

## Local sanity pilot (laptop, one seed per γ, 200k frames, not part of any result)
Mini-test return / crash rate (50 episodes, so ±6 points of noise on the crash rate):

| Frames | γ 0.99 | γ 0.95 | γ 0.9 | Rev 3 vanilla (7 seeds) |
|---|---|---|---|---|
| 50k | 6.9 / 2% | 9.0 / 44% | 8.6 / 78% | 8.0 / 81% |
| 100k | 9.5 / 26% | 12.4 / 68% | 13.5 / 62% | 10.5 / 68% |
| 150k | 10.7 / 54% | 16.3 / 42% | 20.3 / 26% | 12.9 / 58% |
| 200k | 12.2 / 24% | 18.5 / 40% | 20.2 / 34% | 14.8 / 54% |

- γ 0.99 first learns to be safe and slow (0–6% crashes at return 7–9), then starts to speed up.
- γ 0.9 comes close to rev 3's 1M-frame level (return 22, 27% crashes) after about 150k frames (20 / 26%).
- 200k frames say nothing about where the arms end up; stage 1 answers that. The ε-greedy schedule of these pilot
  runs also ended at 200k, unlike a 1M run.

## Files
| File | Role |
|---|---|
| `rev4.py` | stages, groups, parameters, run log (shared by all scripts) |
| `run_orchestrator.py` | launches the runs of a stage (`--stage N`, `--dry-run`, `--smoke`) |
| `monitoring.py` | live view; writes `PROGRESS.md` and `progress.json` |
| `diagnose.py` | Q-values against realised returns, action gaps (gate 1 reads it) |
| `evaluate.py` | attacks and certification (`--preset epsilon / gate2 / gate3 / final`) |
| `gate.py` | the go/no-go checks; appends its tables to `RUNLOG.md` |
| `report.py` | HTML report (`artifacts/run_1/report.html`) |
| `config_highway_{vanilla,sadqn,wocar,wocar_wcl}.json` | fixed settings; gate-decided values are passed as overrides |
| `RUNLOG.md` | progress file of the training agent |

## Open points
- The gate thresholds are choices, not measurements. If gate 1 fails narrowly, that is a decision for the project, not
  for the training agent.
- `wocar_wcl_dqn` has only two seeds: a first look, not a result.
- If gate 3 fails twice, vanilla ends with five seeds (seed 6 is trained in stage 4).
