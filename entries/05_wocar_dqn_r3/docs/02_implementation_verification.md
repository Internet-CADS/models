# 02 — Implementation verification (Gate 2)

Scope: static review of `entries/04_wocar_dqn_r2/` at commit `8e7c792` against the paper (Alg. 2, p. 19–20), `agents/wocar_context.md` §3/§4, and the brief. Line refs are to `train.py` unless stated.
**Not run at runtime** (torch / highway-env not installable here); runtime confirmation is Gate 3. Two behaviours were reproduced in plain Python (items F1, F2).

## Summary
**2 failed checks (F1, F2), 4 documented deviations (D1–D4), everything else passes.** Resolutions (user decision): **F1 fixed** (deliberate deviation from upstream, see below); **F2 kept as is**, description corrected. No other code changed.

| # | Result | Item |
|---|---|---|
| F1 | **FAIL → FIXED** | Gradient clipping (`grad_clip: 10`) was never applied, in all three models |
| F2 | **KEPT (documented)** | ε-greedy does not stay at 0.05; it decays to 0 by frame 4M (upstream `decay_zero` behaviour) |

## Failed checks

### F1 — `grad_clip: 10` was a no-op (vanilla, SA-DQN, WocaR) — FIXED
- Evidence: l.290–296 (`compute_td_loss`), l.533–540 (worst critic) and l.544–551 (main model) in `compute_td_loss_worst`. Pattern: `parameters = model.parameters()` (a **generator**); first `for p in parameters` computes `grad_norm` and exhausts it; the second `for p in parameters` (`p.grad.data.mul_(clip_coef)`) iterates nothing.
- Reproduced with a mock of the same loop: grad norm 141.4, `clip_coef` 0.07, **0 params scaled**, grads unchanged.
- Origin: identical code in `upstream/train.py` (l.262–269, 498–513) — inherited, not introduced by the project. `grad_norm` is still computed and logged, so logs look like clipping happens.
- **Resolution:** `parameters = list(model.parameters())` at l.290, 533, 544 (3 one-line edits, no line shifts). Verified with the same mock: clip now scales all params, post-clip norm = 10.0. **This is a deliberate deviation from upstream** (and from the 3rd run / paper results, which ran unclipped): the new runs clip at 10. Evidence it matters: 3rd-run logged pre-clip `train/grad_norm` (every 1000 frames) had median ≈3, p99 ≈26–29, max 40–50, with **~15–16% of points >10** in all three models (first exceeding 10 around frame 540k–840k), so clipping will engage on roughly one update in six in later training. Applies equally to all three models. Not yet exercised at runtime (norms are <10 in the first 60k frames, so the smoke test cannot show it engaging).
- Impact (before the fix): config says 10; effective clip is none. The 3rd-run baseline was trained the same way, so runs stay comparable with it, but the documented hyperparameter is not what runs.

### F2 — ε-greedy goes 1.0 → 0.05 → 0 (not "1.0→0.05")
- Evidence: config `act_epsilon_start 1.0 / final 0.05 / decay 50000 / decay_zero true`; `utils.py` `ActEpsilonScheduler.get` l.43–54; `train.py` l.832 sets `decay_zero = num_frames`; l.835 passes `start_frame = replay_initial (1000)`.
- Computed values: ε = 1.0 @1k, 0.525 @26k, **0.05 @51k**, 0.0475 @250k, 0.0380 @1M, 0.0253 @2M, 0.0127 @3M, **0 @4M**.
- Config keys match §3 exactly; the second-stage decay-to-zero is upstream behaviour (`decay_zero`) and is what the 3rd run used. The brief/§3 prose ("1.0→0.05 over 50k") describes only the first stage. **Resolution (user decision): kept** — matches upstream and the 3rd run; `act_epsilon_decay_zero` stays `true`. Correct description: *ε = 1.0 → 0.05 linearly over frames 1k→51k, then linearly to 0 at frame 4M* (use this wording in the README). Holding 0.05 would be the one-key change `act_epsilon_decay_zero: false`.

## Documented deviations (not failures)
- **D1 — blended target.** Paper/brief: `y_r = r + γ·max_a′[κ·Q_v′ + (1−κ)·Q_φ]`. Code l.399–402: `argmax` of the blend, then **bootstraps `Q_v′` at that action** (not the blended value). Identical to `upstream/train.py` l.370–375. `worst_ratio` = weight on `Q_φ` (κ_wst = 1 − worst_ratio).
- **D2 — worst-critic cadence.** `worst_rate: 4` (§3): `Q_φ` TD update on every 4th step (l.407–410, 983); paper updates every step. The blended target and robust-Q update use `worst_target_model` on **every** step (see pass item).
- **D3 — network structure.** Paper has separate `Q_v`, `Q_r`. Code has one `current_model` (acts, receives SA reg + TD) with `target_model` as its target; plus `worst_model`/`worst_target_model`. Target nets are **hard copies every `update_target_frame: 2000`** (`utils.py` `update_target`; train.py l.1123–1125; worst nets l.1125), not the paper's soft τ update. Same as upstream.
- **D4 — `worst_kappa_end: 1.0`** (intentional, per upstream README): `worst_kappa` = 1 for all frames (l.969), so `worst_case_loss` is the zero tensor (l.431–432) and `loss = 1·loss + 0` (l.452). Upstream's unstable `worst_case_loss` is disabled.
- Also: `safety_max_crash_rate` is absent from the configs vs §3 — intentional, removed in Gate 1.

## Vanilla DQN — `config_highway_vanilla.json`
| Check | Result | Evidence |
|---|---|---|
| Replay, target net, Huber, γ, lr | PASS | `buffer_capacity 50000`, `replay_initial 1000`, `natural_loss_fn huber`, `gamma 0.99`, `lr 0.000125`, `batch_size 64`; `compute_td_loss` l.112; `update_target_frame 2000` |
| ε-greedy 1.0→0.05 | see F2 | |
| Grad clip 10 | see F1 | |
| argmax action | PASS | `models.py` `QNetwork.act` l.88–97 (`q_value.max(1)[1]`) |
| No SA/WocaR keys | PASS | `robust:false`, `worst:false`; no `kappa/epsilon/schedule_*/convex_*/worst_*` in `training_config` (the `epsilon 0.0025` at l.84 is `test_config.attack_config`, inert for training). `hinge/hinge_c/bound_solver` present but unused when `robust:false` (`robust_model=False`, no `BoundedModule`) |

## SA-DQN — `config_highway_sadqn.json`
| Check | Result | Evidence |
|---|---|---|
| Q-net + auto_LiRPA CROWN-IBP reg | PASS | `models.py` l.79–83 wraps in `BoundedModule`; `get_logits_lower_bound` l.65–77 mixes `β·backward(CROWN) + (1−β)·IBP` |
| L∞ box `[s−ε, s+ε]` | PASS | l.255–256 `state_ub/lb = state ± eps_v`, `PerturbationLpNorm(norm=inf, x_L, x_U)` l.67 |
| `epsilon 0.0025`, ramp 50k→250k | PASS | `epsilon 0.0025`, `schedule_start 50000`, `schedule_length 200000`, `linear` → `EpsilonScheduler(…, start, start+length−1, 0, 0.0025)` l.715 |
| CROWN β 1.0→0.0 | PASS | `convex_start_beta 1.0`, `convex_final_beta 0.0`; l.976: `β = (max_eps − eps·(1−final))/max_eps·start` → 1.0 at ε=0, 0.0 at ε=max |
| hinge `hinge_c 1`, `cov`, `kappa 0.001` | PASS | `hinge true, hinge_c 1`, `bound_solver cov`, `kappa 0.001`; hinge reg l.262–265: `−clamp(min(lb), max=hinge_c)`; `loss += kappa·reg_loss` l.282 |
| Equal weight on all states | PASS | `compute_td_loss` (l.112–297) has no `w(s)`; `worst:false` routes here (l.988–989) |

## WocaR-DQN — `config_highway_wocar.json`
| Check | Result | Evidence |
|---|---|---|
| SA-DQN base identical | PASS | sorted-key diff sadqn↔wocar: only `name_suffix` and `worst*` keys differ |
| IBP bounds of robust Q at s′ | PASS | `models.py` `bound_forward` l.99–114 (`IBP=True, method=None`) called on `current_model` with `next_state`, l.416 |
| Â_adv: a ∈ set iff upper(a) ≥ every lower(a′) | PASS | `ibp.py` `worst_action_select` l.53–57 |
| `â = argmin_{Â_adv} Q_φ(s′,·)` | PASS | `ibp.py` l.64–66 (`masked_fill(~mask, inf)` on a copy, `min(1)`) |
| Bootstrap from **`worst_target_model`** (bug #3c) | PASS | l.388–389 `tgt_worst_q_logits = worst_target_model(next_state)`, used at l.399 and l.417 |
| Blended target per step, unconditional (bug #3) | PASS (see D1) | l.977–980 call `compute_td_loss_worst` on every step once `buffer > replay_initial`; no `worst_rate` guard around the call; `worst_rate` only gates the `Q_φ` update inside (D2) |
| `worst_ratio` ramp 0→0.1, 50k→250k | PASS | l.963–979: `coeff = clip((frame−schedule_start)/worst_schedule, 0, 1)` with `schedule_start 50000`, `worst_schedule 200000`; `current_worst_ratio = coeff·0.1`; `worst_eps = coeff·0.0025` |
| `w(s)` = max_a Q − min_a Q, batch-normalized, detached | PASS | l.515–518: `torch.no_grad()`, `/(mean+1e-8)`, multiplies `reg_loss` only when `eps>0` and solver ≠ pgd |
| `worst_kappa_end 1.0` disables `worst_case_loss` | PASS, intentional (D4) | see D4 |
| Bug #1: single `compute_bounds`, `(lower, upper)` order | PASS | `models.py` l.113 `lower, upper = …compute_bounds(C=spec, IBP=True, method=None)`; l.114 returns `(upper, lower)`; callers unpack `ub, lb` (train.py l.416) |
| Bug #2: `>=`, greedy fallback, no 1e9 | PASS | `ibp.py` l.56 `>=`; l.59–62 empty-row fallback to `upper_q.argmax`; l.66 gathers from **unmasked** `worst_q`; no `1e9` anywhere in `ibp.py` (the `1e10` at train.py l.106 is the unrelated pgd-margin helper) |
| Bug #4: deterministic per-episode seeds | PASS | `mini_test` l.598–604, 629–633 (seeds 1000..1049, `env.seed`+`reset`); `highway_adapter.py` l.43–47, 62–66 consumes the pending seed on the next `reset` (single reset) |

## Total seeding (one `TRAIN_SEED` = `training_config.seed`)
Static PASS. Set via `training_config:seed=<i>` override; `main()` l.685–689 (before any model is built, l.714+):
- network init → `torch.manual_seed(seed)` (l.689)
- ε-greedy RNG → `np.random` (`models.py` `act` l.95–96), `np.random.seed(seed)` (l.687)
- replay sampling → `random.sample` (`my_replay_buffer.py` l.41), `random.seed(seed)` (l.688); PER not used (`per:false`)
- training env resets → `env.seed(seed)` (l.686) stores a pending seed consumed by the first `env.reset()` (l.691); later `env.reset()` (l.1030) continues from the env's seeded `np_random`. `mini_test` uses its own env and fixed seeds, so it doesn't disturb the training env.
- Caveat: `argparser` also seeds with 2019 before `main`; irrelevant because `main` reseeds first. `mini_test` calls `act` which draws from the global `np` stream (deterministic, but shifts it). Bit-for-bit repeatability to be checked in Gate 3 (same seed twice).

## Config values vs `wocar_context.md` §3
All **match exactly** (wocar config checked; vanilla/sadqn are strict subsets per the diffs above): env (4 lanes, 10 vehicles, duration 40, 5/1 Hz, collision −2.0, right-lane 0.1, high-speed 0.4, `normalize_reward false`, `offroad_terminal true`, 10×5 Kinematics → 50-d), MLP 50→128→128→5 (`model_width 1`, name `DQN`; `dueling:true` is ignored for non-Atari ids, `models.py` l.281–287), γ 0.99, lr 1.25e-4, batch 64, Huber, grad_clip 10 (see F1, now effective), replay 50k/1k, ε 1.0→0.05/50k (see F2), `cov`, `hinge_c 1`, ε 0.0025, ramp 50k→250k, β 1→0, κ 0.001, `worst_ratio 0.1`, `worst_rate 4`, `worst_eps_end 0.0025`, `worst_kappa_end 1.0`, 4M frames, `seed 1`. Only difference: `safety_max_crash_rate` (removed, Gate 1). `path_prefix` still `artifacts/3rd_run` (Gate 4).
