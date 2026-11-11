# Rev 4: possible plan (draft)

Status: a proposal, not a decision. Drafted from `README.md`, `docs/deviations.md`, `docs/rev3.md`, `config_highway_wocar.json` and parts of `train.py` / `evaluate.py`. All hour figures are derived from the rev 3 estimates (single-core CPU times), not measured for rev 4 configs. Items marked (unverified) were not checked in the repo.

## Facts used for the arithmetic
- CPU time per run at 15 vehicles and 2M frames (`docs/rev3.md`): vanilla ~8 h, SA-DQN ~9.3 h, WocaR ~9.8 h. A run uses one core.
- Desktop capacity is about 6 runs' worth (hyperthreads add nothing). Wall time ≈ max(longest run, total core-hours / 6).
- Check against rev 3: 3 × (8 + 9.3 + 9.8) = 81 core-h, / 6 = 13.5 h; measured ~13.8 h for 9 runs. (The later 9-run batches on the desktop took about 15 h.)
- Evaluation: 21 jobs of 300 episodes take ~2.3 h, 12 jobs ~1.5 h (`docs/rev3.md`).
- Double DQN is not implemented. In `train.py` the commented-out block (~l.185-193) is the unused Double-DQN variant; all groups use `tgt_next_q_logits.max(1)`, so adding it is a clean change for all groups.

## Candidate directions

| | A. Cut the clean-driving cost (WocaR with lower kappa) | B. Evaluation only: stronger attacks, per-feature eps, density transfer | C. Algorithm: Double DQN, then `bound_solver: pgd` | D. Retrain at another density or more seeds |
|---|---|---|---|---|
| Question | Is the clean cost (WocaR 31% vs vanilla 23% crash) a direct effect of the regulariser strength, and can it shrink without losing most of the robustness gap? | Does the rev 3 ranking (WocaR > SA-DQN ≈ vanilla) survive stronger or adaptive attacks and other conditions? | Do the authors' choices (Double DQN, pgd solver) change the SA-DQN result or the late dip? | Does the result hold at another traffic density, or with 10+ seeds? |
| Changes | `config_highway_wocar.json`: `kappa` 0.02 → 0.01 (single lever). Optional second arm: 0.005. Everything else unchanged (`epsilon` 0.005, `worst_kappa_end` 0.5). | Only `evaluate.py` / `report.py`. PGD with `PGD_STEPS` 10 → 50 plus restarts; per-feature eps (smaller on x/y, larger on vx/vy) or attacks on one feature group only; a transfer attack (gradient from vanilla applied to the robust models); rev 3 checkpoints evaluated at `vehicles_count` 10 / 25 (how the env config is passed in evaluate is unverified). | Small change in `compute_td_loss` and `compute_td_loss_worst` (gather target Q at the online argmax). All 3 groups must be retrained, otherwise the comparison is confounded. pgd needs a pilot, because `cov` also gives the certificates. | Config only (`vehicles_count` 20, or more seeds via `run_orchestrator.py --seeds`). |
| Success | WocaR clean crash ≤ 28% (3+ points lower) with gradient crash at eps 0.01 ≤ 60% (rev 3: 53 WocaR, 74 vanilla; keeps ≥ 2/3 of the 21-point gap). | A clear statement either way. Ranking unchanged under 50-step PGD plus the transfer attack = stronger claim; ranking flips = the main finding gets a caveat. | Late dip smaller, or the SA-DQN gap to vanilla becomes significant (rev 3: t = 0.2 at eps 0.005). | Same ordering at 20 vehicles. |
| Cost (desktop) | 6 runs × 9.8 h on 6 cores: ~10 h wall (58.8 core-h / 6). Second arm on the laptop in parallel: ~9–11 h. | ~4–6 h on laptop or desktop. Existing 4 attacks × 5 eps ≈ 2.3 h for 21 jobs; 50-step PGD ≈ 5× the PGD share plus new variants and 2 extra densities gives ~2–3× that. Estimate, not measured. | Double DQN for 9 runs: 81 core-h / 6 = 13.5 h. Plus a pgd pilot (speed unmeasured, probably slower than `cov`). | 20 vehicles: per step ~1/63 s + 5 ms = 20.9 ms (15 vehicles: 11.4 + 5 = 16.4 ms), ×1.27: 9 runs ≈ 17 h. 10 seeds × 3 groups = 30 runs = 270 core-h / 6 = 45 h. |
| Risks | Seed noise: with 6 seeds the SE of the clean crash is ~5/√6 ≈ 2 points, so a 3-point shift is only borderline detectable. The gap may shrink more than the clean cost. The cost may come from eps rather than kappa. | Eval cost may exceed the estimate. Adaptive attacks can have bugs, so they need a sanity check (must be ≥ PGD-10 in strength). | Two changes confound each other if combined. Double DQN may alter vanilla more than the robust models. pgd breaks certification. | Costly and answers a secondary question. Density 20 is slow. |
| If it fails | Lower kappa loses robustness without helping clean driving: the cost is likely tied to eps or to the robust loss on state importance. Next lever: eps or w(s). | If the ranking holds under all attacks, that is a result; nothing in the training pipeline is at risk. | No effect: the rev 3 conclusions are not sensitive to these deviations (useful to say in the report). Dip persists: it is not a bootstrap-overestimate artefact. | Without effect we only get a replication. |

## Recommendation: ordered pair, 4a then 4b (run in parallel)

**4a** is evaluation only on the rev 3 checkpoints (direction B): cheap, no risk to the training pipeline, and it tests whether the rev 3 claim holds. **4b** is the one training batch (direction A), with a single question. C and D are deferred: C changes all three groups and confounds any comparison, D is expensive and secondary. Two items go into the report as open threats without testing: eps scale per feature (partly covered by 4a) and the paper's mixing-formula notation slip.

### Run matrix for 4b
| Model | Setting | Seeds | Frames | Machine | Wall |
|---|---|---|---|---|---|
| WocaR kappa 0.01 | only `kappa` changed vs rev 3 | 1–6 | 2M | desktop | ~10 h |
| WocaR kappa 0.005 (optional, dose response) | only `kappa` changed | 1–6 | 2M | laptop (M1 Pro) | ~9–11 h |
| Baselines | rev 3 vanilla and WocaR kappa 0.02, seeds 1–7, reused, not retrained | | | | 0 |

Same seeds as rev 3 (1–6); RL seeds do not control the outcome, so there is no variance reduction from pairing. Use `run_orchestrator.py --groups wocar_dqn --seeds 1,2,3,4,5,6 --concurrent` (the flags exist in rev 3; unverified for this combination). Use a new run root (e.g. `artifacts/run_3`) so `run_2` stays untouched.

### Evaluation plan (consistent with rev 3)
- Same protocol: peak-of-mean checkpoint per group, 300 episodes, eps 0 / 0.0025 / 0.005 / 0.0075 / 0.01, PGD + worst-Q + MinBest + random L∞, certification at eps 0.005. Cost for 12 new jobs: ~1.5 h.
- Compare with rev 3 vanilla and WocaR kappa 0.02: clean crash, gradient crash at eps 0.005 and 0.01, certified fraction, mini-test return curve (late dip yes/no). Report mean ± std and t-statistics as in the README.
- Additional 4a part: re-evaluate the rev 3 peak checkpoints under 50-step PGD with restarts, the transfer attack and per-feature eps, plus 10 / 25 vehicles. Use the saved mini-test curves to quantify the drop from peak to final per group (free, no new runs).
- Fix the rules in advance: success as in the table above. If kappa 0.01 gets clean ≤ 28% but the eps 0.01 crash is > 60%, report a trade-off curve (kappa 0.005 / 0.01 / 0.02), not a failure.

### Go / no-go
1. Before launch (~30 min): a 4,000-frame smoke run with the changed config, as in rev 3; confirm the config diff is only `kappa`.
2. At 1M frames (~5 h in): compare the mean mini-test return and crash rate of the kappa 0.01 runs with rev 3 WocaR at the same frame. No-go (stop and free the desktop) if most seeds have collapsed (mini-test return below about half of rev 3's, as in rev 2) or the run is slower than ~35 frames/s per run (projection beyond 16 h). Otherwise continue to 2M. The 50-episode mini-tests are noisy; do not stop for a few points of difference.
3. After the final evaluation: write the result either way, then decide on C (Double DQN) as rev 5.

### Timeline (rough, working time)
| Step | Time |
|---|---|
| Config + smoke test, launch 4b on the desktop (and optional laptop arm) | day 1 morning |
| 4a on the laptop while the desktop trains (if the laptop arm is used, run 4a first and the arm after; this delays it by ~5 h) | day 1 |
| 1M-frame look (go/no-go) | day 1 evening |
| 4b finishes | day 2 morning to afternoon |
| Evaluation of 12 new jobs (1.5 h) + 4a analysis, report, README row for rev 4 | day 2–3 |

## Uncertainties
- All hour figures derive from rev 3 estimates (single-core CPU times); nothing was measured for the rev 4 configs.
- The evaluation cost of the new attacks and densities is a guess; the real number depends on the job structure in `evaluate.py` (not read in full).
- Whether the clean cost comes from kappa, eps or the worst-case loss is not known; 4b varies only kappa, so a negative result does not exclude eps.
- The cause of the late dip is unknown; 4b only tests whether a lower kappa reduces it.
