# Models for our p7 project

Robust DQN for autonomous driving: **vanilla DQN vs SA-DQN vs WocaR-DQN** on `highway-env` (`highway-v0`). We adapted the authors' code ([WocaR-RL](https://github.com/umd-huang-lab/WocaR-RL), paper in `docs/paper.pdf`) from Atari to driving and ran three revisions of the experiment. Each revision is an entry under `entries/` with its own evaluation report (one HTML file with mini-test curves per model and the robustness results).

| Rev | Entry | Runs | Evaluation file | In one line |
|---|---|---|---|---|
| 1 | `entries/03_wocar_dqn` | 1 seed per model, 4M frames | [`artifacts/report.html`](entries/03_wocar_dqn/artifacts/report.html) | WocaR looked best, but with one seed we could not tell luck from effect |
| 2 | `entries/04_wocar_dqn_r2` | 6 seeds per model, 4M frames | [`artifacts/run_1/report.html`](entries/04_wocar_dqn_r2/artifacts/run_1/report.html) | With six seeds there was no real difference: the robust term was far too weak |
| 3 | `entries/05_wocar_dqn_r3` | 7 seeds per model, 2M frames | [`artifacts/run_2/report.html`](entries/05_wocar_dqn_r3/artifacts/run_2/report.html) | Stronger robust training works: WocaR is clearly the most robust, SA-DQN is not reliably better than vanilla |

Throughout, **crash rate under attack** is the main metric. Attacks: PGD, worst-Q (forces the lowest-Q action), MinBest (logit-margin attack) and random L∞ noise, all with budget ε on the normalised observation. "Gradient attacks" below means the mean over PGD, worst-Q and MinBest. Tables show the evaluated checkpoint (see each rev).

---

## Rev 1: first baseline (`entries/03_wocar_dqn`)

Goal: get the authors' WocaR-DQN running on highway-env and compare it with its two ablations, trained identically. The three models are a matched ablation: **vanilla** (no robustness) → **SA-DQN** (adds a regulariser that keeps the chosen action stable inside an ε-ball) → **WocaR-DQN** (adds a worst-case critic and state-importance weighting). Two earlier legacy runs are kept in `artifacts/` but were effectively SA-DQN (three bugs, fixed before the matched run).

| Setting | Value | Why |
|---|---|---|
| **Environment** | | |
| Task | `highway-v0`, 4 lanes, 5 meta-actions (left, idle, right, faster, slower), simulation 5 Hz, one decision per second | The project goal: adapt Atari WocaR-DQN to driving |
| Traffic | 10 other vehicles (highway-env default 50), simulation 5 Hz (default 15) | Reason not recorded |
| Episode | 40 steps, ends on a crash or leaving the road (off-road terminal; default off) | 40 is highway-env's default; the off-road rule: reason not recorded |
| Observation | 10 nearest vehicles × (presence, x, y, vx, vy), normalised, flattened to **50 inputs** | The authors' code reads Atari pixels; `highway_adapter.py` turns the kinematic state into a flat vector |
| Reward | collision **−2** (default −1), right lane 0.1, high speed 0.4, **no reward normalisation** (default on) | Documented: crashes must stay explicit (−2 against at most 0.5 per step), which is what makes the worst-case critic meaningful |
| **Training (all models)** | | |
| Network, loss | MLP 50→128→128→5 (dueling), γ 0.99, Huber loss, gradient clip 10 | Network sized for a 50-input state; γ, loss, clip are the authors' defaults. (The clip never actually ran: found in rev2.) |
| Optimiser, replay, exploration | lr 1.25e-4, batch 64, replay buffer 50k (learning starts at 1k), ε-greedy 1.0→0.05 over 50k frames, target copy every 2000 frames, no prioritised replay | Scaled down from the authors' Atari defaults (lr 6.25e-5, batch 32, buffer 200k with 50k warm-up, ε decay 1.5M); lr equals the authors' RA-DQN default. Individual reasons not recorded |
| Length | **4M frames**, 1 seed per model | Documented: the two legacy runs peaked at 2.2–2.5M frames, 4M captures that with margin |
| Checkpoint choice | every 50k frames a 50-episode mini-test; best checkpoint = highest **safety score = return − 5 × crash rate** (only if crash ≤ 10%) | Pick a checkpoint that drives well and crashes rarely |
| **Robust training (SA-DQN, WocaR)** | | |
| Regulariser | CROWN-IBP convex relaxation (`auto_LiRPA`) with hinge loss; keeps the greedy action unchanged for all inputs within ε | This is the SA-DQN method; it also allows certification |
| Training ε | **0.0025**, linear ramp over frames 50k→250k | The paper trains at 1/255 ≈ 0.0039; our value and the short ramp: reason not recorded |
| Weight κ | **0.001** | Authors' default 0.005, paper tried 0.005–0.02; our lower value: reason not recorded |
| **WocaR only** | | |
| Worst-case critic | blended into the TD target with ratio 0 → 0.1 (ramp over 200k frames), updated every 4th step, bound radius → 0.0025 | Documented: the legacy runs updated it only every 1000th step (a bug) so it hardly trained; fixed to 4. Ratio 0.1 is the top of the authors' config values (0.01–0.1) |
| State importance w(s) | max Q − min Q, batch-normalised, weights the regulariser | The paper's mechanism 3, added by us |
| Worst-case loss | off (`worst_kappa_end` 1.0) | Documented: the authors' README warns that it makes training unstable |
| **Evaluation** | the checkpoint with the best safety score; 300 episodes; 4 attacks; certification | Crash rate under attack is the objective (flip rate alone misleads) |

**Main findings (one seed per model, 300 episodes; crash rate %)**

| | Vanilla | SA-DQN | WocaR |
|---|---|---|---|
| Clean | 20 | 19 | **15** |
| Gradient attacks, ε 0.0025 (training) | 30 | 29 | **20** |
| Gradient attacks, ε 0.005 | 44 | 42 | **35** |
| Gradient attacks, ε 0.01 | 71 | 63 | 64 |
| Certified actions at ε 0.0025 | 1% | 1% | 1% |

WocaR looked clearly best at the training budget and at 2×, and SA-DQN behaved like vanilla. But this is one run per model and the checkpoints were picked by a crash-penalised score, so a lucky seed could not be ruled out. Certificates were nearly empty for everyone. This motivated rev2: **more seeds and no hand-made checkpoint selection**.

---

## Rev 2: replication with 6 seeds (`entries/04_wocar_dqn_r2`)

Same setup as rev1 on purpose (**hyperparameters untouched**), so that rev2 tests whether the rev1 result holds up. Changes are in bold.

| Setting | Value in rev2 | Why |
|---|---|---|
| Environment, reward, network, optimiser, replay, ε-greedy | as rev1 | — |
| Seeds | **6 per model (18 runs)** | One seed cannot separate effect from luck; RL results vary a lot between seeds |
| Checkpoint choice | **safety score and best-checkpoint saving removed**; each group is evaluated at the frame where its seed-mean mini-test return peaks | Rule: no hand-made checkpoint selection. The old score used the crash rate that the evaluation then measures |
| Gradient clipping | **now actually applied** (clip 10) | Verification found it never ran in rev1 (the parameter generator was exhausted before the clip) |
| Robust training: ε, ramp, κ | as rev1: ε 0.0025, ramp 50k→250k, κ 0.001 | — |
| WocaR: critic ratio, rate, worst-case loss | as rev1: ratio 0.1, every 4th step, `worst_kappa_end` 1.0 | — |
| Length | 4M frames | — |
| Evaluation | **scripts instead of a notebook** (`evaluate.py`, `report.py`); peak-of-mean frame; 100 episodes; 4 attacks; certification | One reproducible HTML report per rev |

**Main findings (6 seeds, 100 episodes; crash rate %, mean ± std over seeds)**

| | Vanilla | SA-DQN | WocaR |
|---|---|---|---|
| Clean | 19 ± 4 | 20 ± 7 | 21 ± 7 |
| Gradient attacks, ε 0.0025 (training) | 26 ± 5 | 24 ± 7 | 26 ± 7 |
| Gradient attacks, ε 0.005 | 39 ± 5 | 38 ± 6 | 36 ± 8 |
| Gradient attacks, ε 0.01 | 65 ± 5 | 62 ± 7 | 59 ± 6 |
| Certified actions at ε 0.0025 | 0% | 3% | 1% |

- **No real difference between the three models** (every t-test t < 2). The rev1 advantage of WocaR was luck.
- **Why the robust training did so little:** the robust term (κ × regulariser) was about 0.1% of the loss; ε = 0.0025 barely hurts vanilla; Q-value margins between the best and second-best action were tiny (median 0.02–0.04 against Q ≈ 38), so many actions flip but few flips are harmful in the sparse traffic (≈ 25% of actions flip, crash rate rises only about 6 points).
- **WocaR was unstable:** its worst-case critic drifted (worst-Q from −50 to −300 … −2300), and **4 of 6 WocaR seeds collapsed late in training** (mini-test return from about 18 down to 1–8 at 4M frames); vanilla and SA-DQN did not.

---

## Rev 3: stronger robust training, 7 seeds (`entries/05_wocar_dqn_r3`)

Goal: a setup in which robust training can show a gap, with much less compute. Reward definition unchanged. Details of every difference from the authors' code are in `docs/deviations.md`; the plan and decisions are in `docs/rev3.md`. Changes are in bold.

| Setting | Rev3 value (rev2) | Why |
|---|---|---|
| Traffic | **15 other vehicles** (10) | Sparse traffic made most flipped actions harmless. 25 vehicles was tried first and aborted after 40 minutes: it makes the simulator 2.7× slower per step |
| Episode length | **60 steps** (40) | More time for an attack to turn a wrong action into a crash; no cost in speed |
| Reward, observation, network, optimiser, replay, ε-greedy | as rev2 | — |
| Length | **2M frames** (4M) | Rev2 curves plateau at about 1M frames and WocaR got worse later; halves the compute |
| Seeds | **7 per model (21 runs)** (6) | Started with 3; four more were added because the seed-to-seed spread dominated the comparison |
| Training ε, ramp | **ε 0.005, ramp 100k→600k** (0.0025, 50k→250k) | Vanilla barely suffered at 0.0025; a slower ramp lets the Q-function form first |
| Weight κ | **0.02** (0.001) | Rev2's robust term was 0.1% of the loss and certificates were empty; 0.02 is the top of the paper's range |
| WocaR: critic bound radius | **0.001** (0.0025) | The authors' RA-DQN uses 1e-3; rev2's worst-case critic drifted to −300 … −2300 |
| WocaR: critic ramp | **1M frames** (200k) | Let the critic converge before it influences the targets |
| WocaR: worst-case loss | **on, weight 0.5** (off) | The authors' RA-DQN default; their README warns about instability (a risk we accepted) |
| WocaR: critic ratio, update rate | 0.1, every 4th step (unchanged) | In line with the authors' values (≤ 0.1) |
| Evaluation | **300 episodes** (100), 7 seeds, peak-of-mean frame, 4 attacks, certification | Tighter per-run estimates. (More seeds help more than more episodes: the seed spread dominates.) |

**Main findings (7 seeds, 300 episodes; crash rate %, mean ± std over seeds)**

| | Vanilla | SA-DQN | WocaR |
|---|---|---|---|
| Clean | **23 ± 3** | 29 ± 8 | 31 ± 5 |
| Gradient attacks, ε 0.005 (training) | 44 ± 5 | 43 ± 14 | **38 ± 5** |
| Gradient attacks, ε 0.01 | 74 ± 5 | 61 ± 16 | **53 ± 8** |
| Random noise, ε 0.005 | 26 ± 5 | 31 ± 10 | 32 ± 4 |
| Certified actions at ε 0.005 | 0% | **82%** | 75% |

- **WocaR is the most robust against gradient attacks**, and consistent between seeds: better than vanilla by 6.5 points at ε 0.005 (t = 2.3) and 21 points at ε 0.01 (t = 5.7).
- **SA-DQN is not reliably better than vanilla** (1 point at ε 0.005, t = 0.2; 13 points at ε 0.01, t = 2.1) and its seed spread is large (14–16 points): its good 3-seed result was luck. SA-DQN and WocaR are not significantly different from each other.
- **The price:** the robust models drive worse without attack (WocaR 31% vs vanilla 23% crashes), they do not help against random noise, and they dip late in training, which is why the peak checkpoint and not the last one is evaluated.
- **Certificates now work** (75–82% of actions provably stable at ε 0.005; rev2: about 0%).

---

## More
- Evaluation files: open the three `report.html` files above. Rebuild: `python3 entries/05_wocar_dqn_r3/report.py <run folder> --eval-dir <cache> ...` (rev1: `python3 entries/03_wocar_dqn/make_report.py`). Rev3 evaluation: `python evaluate.py artifacts/run_2` inside `entries/05_wocar_dqn_r3`.
- `docs/rev3.md` (changes and decisions of rev3), `docs/deviations.md` (differences from the paper and the authors' code), `docs/rev2_verification/` (code verification), `docs/wocar_context.md` (rev1 background), `docs/archive/` (old briefs), `docs/desktop_extra_seeds.md` and `docs/r3_agent_runbook.md` (agent task files used to run rev3).
- Earlier, non-WocaR experiments: `entries/01_highway_dqn`, `entries/02_dqn_obs_noise_training` (SB3 DQN notebooks).
