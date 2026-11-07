# WocaR-DQN on highway-env — Project Context for Discussion

Use this file as background when discussing this project. The WocaR-RL paper is in `paper.pdf` (Liang et al., NeurIPS 2022). The executed evaluation notebook (tables + figures) is in `02_wocar_evaluation.pdf`. Code lives in `entries/02_wocar_dqn/` of the `models` repo (branch `wocar-eval`).

## 1. What this project is

7th-semester software-engineering project: adapt **WocaR-DQN** (Worst-case-aware Robust DQN) from Atari pixel games to the **highway-env** autonomous-driving simulator (`highway-v0`), establish it as a validated baseline, then iterate on it. Upstream code (pristine reference) is vendored in `entries/02_wocar_dqn/upstream/` (WocaR-RL repo, `WocaR-DQN/SA-DQN`, commit `2856174`); our highway adaptation + fixes are top-level files in `entries/02_wocar_dqn/`.

## 2. The three methods (Vanilla DQN → SA-DQN → WocaR-DQN)

- **Vanilla DQN**: one Q-network predicting total future return per action; acts with `argmax`. No robustness.
- **SA-DQN** (Zhang et al. 2020): same Q-network plus an `auto_LiRPA` convex-relaxation (CROWN-IBP) regularizer (`cert reg loss`) forcing the network to keep the same argmax action for every observation in the L-inf box `[s−eps, s+eps]`. Treats all states equally.
- **WocaR-DQN** (the paper): SA-DQN **plus** (1) a second "worst-attack critic" network estimating worst-case long-term return via the worst-attack Bellman operator (`min` over the IBP-reachable action set instead of `max`), blended into the TD target with weight `worst_ratio`; and (2) a **state-importance weight** `w(s) = max_a Q(s,a) − min_a Q(s,a)` weighting the regularizer toward critical states. The point of Figure 1 in the paper: smoothness alone isn't enough, the policy must also avoid intrinsically dangerous trajectories.

## 3. Environment: `highway-v0` config (same for all three models)

4 lanes, 10 vehicles, `duration: 40` steps, `simulation_frequency: 5` / `policy_frequency: 1`, `DiscreteMetaAction` (5 actions: LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER), normalized `Kinematics` obs of 10 vehicles × 5 features `[presence, x, y, vx, vy]` flattened to a 50-dim vector, `collision_reward: -2.0`, `right_lane_reward: 0.1`, `high_speed_reward: 0.4`, `normalize_reward: false` (crashes are explicitly −2.0, which is what makes the worst-case critic meaningful), `offroad_terminal: true`. Network for all models: MLP `50 → 128 → 128 → 5`. `gamma: 0.99`, `lr: 1.25e-4`, `batch_size: 64`, Huber loss, `grad_clip: 10`, replay `50k`/`1k`, epsilon-greedy `1.0 → 0.05` over 50k frames. SA/WocaR robustness: `bound_solver: cov`, hinge loss (`hinge_c: 1`), `epsilon: 0.0025`, linear ramp frames 50k→250k, CROWN beta `1.0 → 0.0`, `kappa: 0.001`. WocaR extras: `worst_ratio: 0.1` (ramped 0→0.1 over frames 50k→250k), `worst_rate: 4` (worst-critic update every 4th frame; the main policy uses the worst-target on *every* frame), `worst_eps_end: 0.0025`, `worst_kappa_end: 1.0` (disables upstream's unstable `worst_case_loss`, per upstream README advice), `safety_max_crash_rate: 0.10`. All trained 4M frames, `seed: 1`, CPU (`OMP_NUM_THREADS=1`).

## 4. Bugs found during the audit (all fixed, separate commits)

1. `models.py` `QNetwork.bound_forward`: double `compute_bounds` call returned `(upper, −lower)` instead of `(upper, lower)` because auto_LiRPA caches node intervals until the next `forward()`. Fixed to a single `lower, upper = compute_bounds(...)` call (verified `ub ≥ q ≥ lb`).
2. `ibp.py` `worst_action_select`: strict `>` check plus no fallback meant an all-False mask injected `1e9` into the Bellman target (`td worst loss ~1e8` visible in old logs). Fixed to `>=` + greedy-action fallback + gather from unmasked Q.
3. `train.py` WocaR dispatch: `compute_td_loss_worst` only ran 1 in every 1000 steps (`worst_rate: 1000` + outer `if` skipping the call, originally added to dodge an upstream `IndexError` when unpacking `res`). Fixed to run every step with correct unpacking; added the paper's $w(s)$ weighting (batch-normalized, detached); fixed the `worst_eps`/`worst_ratio` ramp to run over `worst_schedule` after `schedule_start`; added `worst_target_model` usage (upstream wrongly bootstrapped from the online worst net).
4. `highway_adapter.py` + `train.py`/`test.py` seeding: Gymnasium `seed()`+`reset()` double-reset discarded the seeded episode and eval used random seeds. Fixed to deterministic per-episode seeds (`mini_test`: 1000..1049; eval notebook: 0..99).

Consequence: old runs (1st_run `casstef_4m.pth`, 2nd_run) trained with bugs 1–3 active and `worst_rate: 1000`, so they are effectively **99.9% SA-DQN** (SA regularizer ran on 100% of steps and is bug-free; the broken worst-critic ran on 0.1% of steps with clipped gradients). They drive well and are valid SA-DQN reference points, but they are not true WocaR-DQN.

## 5. Training runs (3rd_run = the matched ablation, 4M frames each)

| Run | Model | Best checkpoint (by safety = return − 5×crash) | mini_test (50 eps) at best |
|---|---|---|---|
| 3rd_run `highway-v0_vanilla` | Vanilla DQN | `best_frame_4000000.pth` (final) | return 18.54, crash 0.08, safety 18.14 |
| 3rd_run `highway-v0_sadqn` | SA-DQN | `best_frame_2450000.pth` | return 18.41, crash 0.04, safety 18.21 |
| 3rd_run `highway-v0_wocar` | WocaR-DQN (project baseline) | `best_frame_3850000.pth` | return 18.00, crash 0.06, safety 17.70 |
| 1st_run (legacy, ~SA-DQN) | `casstef_4m.pth` (frame 2.45M) | return 18.36, crash 0.06, safety 18.06 | — |
| 2nd_run (legacy, ~SA-DQN) | `frame_4000000.pth` (final) | return 18.07, crash 0.08, safety 17.67 | — |

Note: 4M frames was chosen because both legacy runs peaked at 2.2–2.5M and 4M captures that peak with margin; `best_frame_*.pth` is saved on every new safety-score high. Theoretical max clean return is 20.0 (40 steps × 0.5).

## 6. Evaluation methodology (`02_wocar_evaluation.ipynb`, results cached in `artifacts/eval_cache/`)

100 deterministic episodes (`env.reset(seed=ep)`, ep 0..99) per model/attack/epsilon. Perturbation budgets `eps ∈ {0, 0.0005, 0.001, 0.0025 (=eps_train), 0.005, 0.01, 0.02, 0.05}`. Four test-time perturbations: (a) uniform random L-inf noise in `[s−eps, s+eps]`; (b) 10-step untargeted PGD (cross-entropy vs clean argmax); (c) MinBest (PGD on the logit margin `Q(a*) − max Q(a≠a*)`); (d) Worst-Action "Min-Q" targeted PGD (forces the argmin-Q, i.e. most dangerous, action). Plus `auto_LiRPA` certification (fraction of visited states provably stable under *all* eps-perturbations) and critical-state flip analysis (top-25% `w(s)` states vs rest). Adding a future model = one dict entry; old results load from cache.

## 7. Headline results (100 episodes, seeds 0..99)

Clean (eps=0): Vanilla 16.67±5.65 / 19% crash; SA-DQN 16.46±6.02 / 17%; WocaR 16.60±5.65 / **14%**. Action styles: Vanilla spams LANE_RIGHT (61%), SA-DQN spams FASTER (67%), WocaR is balanced (LEFT 18 / IDLE 22 / RIGHT 25 / FASTER 33 / SLOWER 2%) — cf. paper Figure 2.

At train budget (eps=0.0025) — WocaR best crash rate on all four attacks: random 11% (vs 20–22%), PGD 18% (vs 30–33%), MinBest 19% (vs 24–26%), Worst-Q 21% (vs 28–31%).
At 2× budget (eps=0.005) — WocaR still best crash rate on all four (15/35/33/36% vs 22–25/36–42/33–40/44–45%).
At ~3/255 equivalent (eps=0.01) — all models degrade to 55–70% crash (see discussion point 3 below).

Flip-rate-vs-safety paradox: Run 1 flips *less* than Run 3 models (e.g. PGD@0.0025: 16.3% vs 30–37%) but crashes *more* (30% vs 18% for WocaR), because WocaR's $w(s)$ tolerates flips in safe states and protects critical ones (WocaR critical-state flip 14.5% vs Run 1 15.3% on PGD@0.0025; 5.8% vs 6.4% on Worst-Q@0.0025). Flip rate alone is a misleading metric; crash rate under attack is the objective. Certified ratios at eps=0.0025 are tiny for all models (1–5%), so IBP certificates don't discriminate here.

## 8. Suggested discussion topics

1. Is crash-rate-under-attack (rather than flip rate or certified ratio) the right primary metric for this highway baseline, and is `safety = return − 5×crash` a sensible model-selection score?
2. The paper tests Atari at 3/255 (≈0.0118) but trains at 1/255; we train at 0.0025 and our eps=0.01 (≈3/255 scale) collapses all models. Is that a fair comparison given that one highway feature unit is semantically denser (≈half a lane width) than one pixel step, and one wrong highway action ends the episode while Atari tolerates single mistakes?
3. Given tiny certified ratios (1–5%), is IBP certification informative on this 50-dim MLP, or should certification effort go elsewhere?
4. What should the *next* iteration after this baseline try (e.g. larger `worst_ratio`, larger train eps, different $w(s)$ normalization, longer training, different env traffic density)?
